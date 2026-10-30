"""Contrastive heal for a CodeBrain embedder: InfoNCE on (question, code unit, negatives).

The forked heal step of the BlackHole-Dev recipe (``recipe_codebrain_embedder.py``),
and a script on its own, for example to fine-tune a small model as the control:

    .venv-eval/bin/python eval/embedder/heal.py --student Qwen/Qwen3-Embedding-0.6B \\
        --data embedder-data --out models/cb-qwen3-emb-0.6b

``--data`` is the output of ``python -m codebrain.eval.pairs``.

The model is trained through its own SentenceTransformer ``tokenize`` and ``forward``,
so the pooling (last token for Qwen3-Embedding), normalization and prompts it ships
with are what gets trained and saved. ``prune.py`` keeps them by editing the backbone
inside the original pipeline. A language-model heal (KL on next-token logits) would
train none of that, which is why this step is forked rather than the built-in one.

Loss for a batch of B questions, each with its positive and ``--negatives`` hard
negatives:

* InfoNCE over all B * (1 + h) documents in the batch, temperature ``--temperature``.
  A document that is the same unit as a row's positive (another question about the
  same code) is masked out of that row instead of counted as a negative.
* Matryoshka: the same loss on the first d dimensions, renormalized, for each
  ``--matryoshka d``. CodeBrain stores 1,024 dimensions and prefilters at 256, and
  pgvector cannot index more than 4,000, so the 4,096 of Qwen3-Embedding-8B are cut.
* ``--distill w`` with ``--teacher``: KL between the teacher's and the student's
  softmax over the same candidates. It compares scores, not vectors, so the teacher's
  dimension does not have to match the student's.

Big batches fit through GradCache: every text is embedded once without gradients,
the loss gradient with respect to the embeddings is taken, then each chunk is encoded
again with gradients and backpropagated with that gradient. Exact for models without
dropout (Qwen3 has none).

Dev metrics (held-out repos): each dev question ranks every dev unit; Recall@1, @10 and
MRR of the positive, for the code view and the summary view.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

try:
    from codebrain.eval.codebase_qa import QUERY_INSTRUCTION
except ImportError:  # running outside the CodeBrain checkout
    QUERY_INSTRUCTION = (
        "Given a question about a codebase, retrieve the code and code summaries that answer it"
    )

_MASKED = -1e4
"""Finite stand-in for -inf in masked logits, so the KL term stays finite."""


@dataclass
class HealConfig:
    epochs: float = 1.0
    max_steps: int | None = None
    batch_size: int = 128
    negatives: int = 3
    chunk: int = 32
    """Texts per forward pass with gradients; only memory depends on it."""
    lr: float = 2e-5
    warmup: float = 0.05
    weight_decay: float = 0.01
    temperature: float = 0.02
    matryoshka: tuple[int, ...] = ()
    distill: float = 0.0
    distill_temperature: float = 0.02
    summary_share: float = 0.3
    """Share of rows whose documents are the units' summaries (CodeBrain's summary view)
    instead of their code; one view per row, so its negatives are as hard as its positive."""
    max_len: int = 1024
    query_instruction: str = QUERY_INSTRUCTION
    eval_every: int = 200
    dev_questions: int = 1000
    grad_checkpointing: bool = True
    bf16_weights: bool = False
    """Keep the student's weights in bf16 instead of fp32 master weights (saves memory on
    an 8B student, at some precision cost in the optimizer)."""
    seed: int = 0


@dataclass
class HealResult:
    steps: int
    seconds: float
    history: list[dict[str, Any]] = field(default_factory=list)
    dev_before: dict[str, float] = field(default_factory=dict)
    dev_after: dict[str, float] = field(default_factory=dict)


# Data


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def load_pairs(
    data: Path,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    units = {u["id"]: u for u in read_jsonl(data / "units.jsonl")}
    return units, read_jsonl(data / "train.jsonl"), read_jsonl(data / "dev.jsonl")


def doc_text(unit: dict[str, Any], view: str) -> str:
    return (unit.get("summary") or unit["text"]) if view == "summary" else unit["text"]


def query_prefix(model: Any, instruction: str) -> str:
    """The query prompt CodeBrain's eval uses for this model (see ``STEmbedder.embed``)."""
    prompts = getattr(model, "prompts", None) or {}
    if "query" not in prompts:
        return ""
    return f"Instruct: {instruction}\nQuery:" if instruction else str(prompts["query"])


@dataclass
class Batch:
    queries: list[str]
    docs: list[str]
    labels: list[int]
    """Index in ``docs`` of each query's positive."""
    doc_units: list[str]
    pos_units: list[str]


def batches(
    pairs: Sequence[dict[str, Any]],
    units: dict[str, dict[str, Any]],
    cfg: HealConfig,
    prefix: str,
    rng: random.Random,
) -> Iterator[Batch]:
    order = list(range(len(pairs)))
    while True:
        rng.shuffle(order)
        for start in range(0, len(order) - cfg.batch_size + 1, cfg.batch_size):
            b = Batch([], [], [], [], [])
            for i in order[start : start + cfg.batch_size]:
                p = pairs[i]
                pos = units[p["positive"]]
                view = (
                    "summary" if pos.get("summary") and rng.random() < cfg.summary_share else "code"
                )
                negs = [n["id"] for n in p["negatives"] if n["id"] in units]
                negs = rng.sample(negs, min(cfg.negatives, len(negs)))
                b.queries.append(prefix + p["query"])
                b.labels.append(len(b.docs))
                b.pos_units.append(pos["id"])
                for uid in [pos["id"], *negs]:
                    b.docs.append(doc_text(units[uid], view))
                    b.doc_units.append(uid)
            yield b


# Model


def tokenize(model: Any, texts: Sequence[str]) -> dict[str, Any]:
    """``preprocess`` in sentence-transformers 6, ``tokenize`` before."""
    fn = getattr(model, "preprocess", None) or model.tokenize
    return dict(fn(list(texts)))


def encode(model: Any, texts: Sequence[str], device: torch.device) -> torch.Tensor:
    feats = tokenize(model, texts)
    feats = {k: v.to(device) if hasattr(v, "to") else v for k, v in feats.items()}
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        out = model(feats)["sentence_embedding"]
    return out.float()


def encode_no_grad(
    model: Any, texts: Sequence[str], chunk: int, device: torch.device
) -> torch.Tensor:
    with torch.no_grad():
        return torch.cat(
            [encode(model, texts[i : i + chunk], device) for i in range(0, len(texts), chunk)]
        )


def info_nce(
    q: torch.Tensor,
    d: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    cfg: HealConfig,
    teacher_scores: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """InfoNCE at full size and each Matryoshka size, plus the distillation term."""
    dims = [None] + [m for m in cfg.matryoshka if m < q.shape[1]]
    total = q.new_zeros(())
    parts: dict[str, float] = {}
    full_scores = None
    for dim in dims:
        qq = F.normalize(q[:, :dim] if dim else q, dim=-1)
        dd = F.normalize(d[:, :dim] if dim else d, dim=-1)
        s = (qq @ dd.T / cfg.temperature).masked_fill(mask, _MASKED)
        if dim is None:
            full_scores = s
        loss = F.cross_entropy(s, labels)
        parts[f"nce@{dim or 'full'}"] = loss.item()
        total = total + loss / len(dims)
    if teacher_scores is not None and cfg.distill > 0 and full_scores is not None:
        target = F.softmax(teacher_scores.masked_fill(mask, _MASKED), dim=-1)
        kl = F.kl_div(F.log_softmax(full_scores, dim=-1), target, reduction="batchmean")
        parts["kl"] = kl.item()
        total = total + cfg.distill * kl
    return total, parts


def same_unit_mask(b: Batch, device: torch.device) -> torch.Tensor:
    """True where a document is the row's positive unit but not the row's own label."""
    mask = torch.tensor(
        [[u == pos for u in b.doc_units] for pos in b.pos_units], dtype=torch.bool, device=device
    )
    rows = torch.arange(len(b.labels), device=device)
    mask[rows, torch.tensor(b.labels, device=device)] = False
    return mask


def train_step(
    model: Any,
    b: Batch,
    cfg: HealConfig,
    device: torch.device,
    teacher: Any = None,
    teacher_prefix: str = "",
    prefix: str = "",
) -> dict[str, float]:
    """One GradCache step; the caller steps the optimizer."""
    labels = torch.tensor(b.labels, device=device)
    mask = same_unit_mask(b, device)
    q0 = encode_no_grad(model, b.queries, cfg.chunk, device).requires_grad_()
    d0 = encode_no_grad(model, b.docs, cfg.chunk, device).requires_grad_()
    t_scores = None
    if teacher is not None and cfg.distill > 0:
        tq_texts = [teacher_prefix + q[len(prefix) :] for q in b.queries]
        tq = F.normalize(encode_no_grad(teacher, tq_texts, cfg.chunk, device), dim=-1)
        td = F.normalize(encode_no_grad(teacher, b.docs, cfg.chunk, device), dim=-1)
        t_scores = tq @ td.T / cfg.distill_temperature
    loss, parts = info_nce(q0, d0, labels, mask, cfg, t_scores)
    loss.backward()
    for texts, grads in ((b.queries, q0.grad), (b.docs, d0.grad)):
        assert grads is not None
        for i in range(0, len(texts), cfg.chunk):
            emb = encode(model, texts[i : i + cfg.chunk], device)
            emb.backward(grads[i : i + cfg.chunk])
    parts["loss"] = loss.item()
    return parts


# Dev


def lineage(units: dict[str, dict[str, Any]]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {uid: set() for uid in units}
    for uid, u in units.items():
        seen, p = set(), u.get("parent_id")
        while p and p not in seen:
            seen.add(p)
            out[uid].add(p)
            out.setdefault(p, set()).add(uid)
            p = units[p].get("parent_id") if p in units else None
    return out


def evaluate(
    model: Any,
    units: dict[str, dict[str, Any]],
    dev: Sequence[dict[str, Any]],
    cfg: HealConfig,
    prefix: str,
    device: torch.device,
) -> dict[str, float]:
    """Rank all dev units for each dev question (positive's own lineage excluded)."""
    if not dev:
        return {}
    was_training = model.training
    model.eval()
    rng = random.Random(cfg.seed)
    qs = list(dev) if len(dev) <= cfg.dev_questions else rng.sample(list(dev), cfg.dev_questions)
    repos = {p["repo"] for p in qs}
    ids = sorted(uid for uid, u in units.items() if u["repo"] in repos)
    col = {uid: i for i, uid in enumerate(ids)}
    related = lineage(units)
    elsewhere = {
        repo: torch.tensor([col[u] for u in ids if units[u]["repo"] != repo], dtype=torch.long)
        for repo in repos
    }
    qv = F.normalize(
        encode_no_grad(model, [prefix + p["query"] for p in qs], cfg.chunk, device), dim=-1
    )
    out: dict[str, float] = {}
    for view in ("code", "summary"):
        dv = F.normalize(
            encode_no_grad(model, [doc_text(units[u], view) for u in ids], cfg.chunk, device),
            dim=-1,
        )
        scores = qv @ dv.T
        for r, p in enumerate(qs):
            for other in related.get(p["positive"], ()):
                if other in col:
                    scores[r, col[other]] = -2.0
            # A question is about one repo; units of the other dev repos never compete.
            scores[r, elsewhere[p["repo"]]] = -2.0
        target = torch.tensor([col[p["positive"]] for p in qs], device=device)
        pos = scores[torch.arange(len(qs), device=device), target]
        rank = (scores > pos[:, None]).sum(dim=1) + 1
        out[f"{view}_r@1"] = float((rank <= 1).float().mean())
        out[f"{view}_r@10"] = float((rank <= 10).float().mean())
        out[f"{view}_mrr"] = float((1.0 / rank.float()).mean())
    out["dev_questions"] = float(len(qs))
    if was_training:
        model.train()
    return out


# Loop


def load_model(name: str, device: torch.device, dtype: torch.dtype) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name, device=str(device), model_kwargs={"torch_dtype": dtype})


def heal(
    model: Any,
    data: Path,
    cfg: HealConfig,
    *,
    teacher: Any = None,
    log_path: Path | None = None,
) -> HealResult:
    """Train ``model`` (a SentenceTransformer) in place on the pairs in ``data``."""
    device = next(model.parameters()).device
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    units, train, dev = load_pairs(data)
    if not train:
        raise SystemExit(f"no training pairs in {data}/train.jsonl")
    model.max_seq_length = cfg.max_len
    prefix = query_prefix(model, cfg.query_instruction)
    t_prefix = query_prefix(teacher, cfg.query_instruction) if teacher is not None else ""
    if teacher is not None:
        teacher.max_seq_length = cfg.max_len
        teacher.eval()
    backbone = model[0].auto_model
    if cfg.grad_checkpointing and hasattr(backbone, "gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(backbone.config, "use_cache"):
            backbone.config.use_cache = False
    steps = cfg.max_steps or max(1, math.ceil(len(train) / cfg.batch_size * cfg.epochs))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    warm = max(1, int(steps * cfg.warmup))

    def lr_at(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    result = HealResult(steps=steps, seconds=0.0)
    result.dev_before = evaluate(model, units, dev, cfg, prefix, device)
    log(f"dev before: {result.dev_before}", log_path)
    model.train()
    t0 = time.perf_counter()
    for step, b in enumerate(batches(train, units, cfg, prefix, rng)):
        if step >= steps:
            break
        parts = train_step(model, b, cfg, device, teacher, t_prefix, prefix)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        row: dict[str, Any] = {"step": step + 1, "lr": sched.get_last_lr()[0], **parts}
        if (step + 1) % 10 == 0 or step == 0:
            row["s_per_step"] = (time.perf_counter() - t0) / (step + 1)
            log(json.dumps(row), log_path)
        if cfg.eval_every and (step + 1) % cfg.eval_every == 0 and step + 1 < steps:
            row["dev"] = evaluate(model, units, dev, cfg, prefix, device)
            log(f"dev at {step + 1}: {row['dev']}", log_path)
        result.history.append(row)
    result.seconds = time.perf_counter() - t0
    model.eval()
    result.dev_after = evaluate(model, units, dev, cfg, prefix, device)
    log(f"dev after: {result.dev_after}", log_path)
    return result


def log(msg: str, path: Path | None) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if path is not None:
        with path.open("a") as f:
            f.write(line + "\n")


def config_from_args(args: argparse.Namespace) -> HealConfig:
    return HealConfig(
        epochs=args.epochs,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        negatives=args.negatives,
        chunk=args.chunk,
        lr=args.lr,
        temperature=args.temperature,
        matryoshka=tuple(args.matryoshka),
        distill=args.distill if args.teacher else 0.0,
        summary_share=args.summary_share,
        max_len=args.max_len,
        query_instruction=args.query_instruction,
        eval_every=args.eval_every,
        bf16_weights=args.bf16_weights,
        seed=args.seed,
    )


def add_heal_args(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("heal")
    g.add_argument("--data", required=True, help="output of python -m codebrain.eval.pairs")
    g.add_argument(
        "--teacher", default=None, help="distill from this model (e.g. the unpruned one)"
    )
    g.add_argument(
        "--distill", type=float, default=1.0, help="weight of the KL term with --teacher"
    )
    g.add_argument("--epochs", type=float, default=1.0)
    g.add_argument("--max-steps", type=int, default=None)
    g.add_argument("--batch-size", type=int, default=128)
    g.add_argument("--negatives", type=int, default=3)
    g.add_argument("--chunk", type=int, default=32)
    g.add_argument("--lr", type=float, default=2e-5)
    g.add_argument("--temperature", type=float, default=0.02)
    g.add_argument("--matryoshka", type=int, action="append", default=[], help="e.g. 1024, 256")
    g.add_argument("--summary-share", type=float, default=0.3)
    g.add_argument("--max-len", type=int, default=1024)
    g.add_argument("--query-instruction", default=QUERY_INSTRUCTION)
    g.add_argument("--eval-every", type=int, default=200)
    g.add_argument("--bf16-weights", action="store_true")
    g.add_argument("--seed", type=int, default=0)


def save(model: Any, out: Path, record: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    (out / "codebrain-heal.json").write_text(json.dumps(record, indent=2, default=str))


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--student", required=True, help="SentenceTransformer model id or folder")
    ap.add_argument("--out", required=True)
    add_heal_args(ap)
    args = ap.parse_args(argv)
    cfg = config_from_args(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if cfg.bf16_weights else torch.float32
    model = load_model(args.student, device, dtype)
    teacher = load_model(args.teacher, device, torch.bfloat16) if args.teacher else None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result = heal(model, Path(args.data), cfg, teacher=teacher, log_path=out / "heal.log")
    save(
        model,
        out,
        {
            "student": args.student,
            "teacher": args.teacher,
            "config": asdict(cfg),
            "result": asdict(result),
        },
    )
    log(f"saved {out}", out / "heal.log")
    return 0


if __name__ == "__main__":
    sys.exit(main())
