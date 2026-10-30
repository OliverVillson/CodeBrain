"""Shrink a decoder embedder (Qwen3-Embedding) by dropping layers and MLP channels.

    .venv-eval/bin/python eval/embedder/prune.py --model Qwen/Qwen3-Embedding-8B \\
        --data embedder-data --keep-layers 18 --mlp-keep 0.75 --out models/q3e8b-l18-m75

The edits happen inside the model's SentenceTransformer pipeline, so the tokenizer,
pooling, normalization and prompts are unchanged and the saved folder loads with plain
``SentenceTransformer(path)`` (and so with CodeBrain's eval, ``--embed-model <path>``).

* Layers (``--keep-layers``): greedy on what an embedder has to keep. Starting from the
  full model, drop the layer whose removal moves the pooled embeddings of calibration
  texts (questions and code from the pairs) the least, measured as mean cosine to the
  full model's embeddings; repeat. ``--method influence`` is the one-pass ShortGPT
  variant (1 - cos between a layer's input and output hidden states), much cheaper.
* MLP channels (``--mlp-keep``): Minitron-style activation importance, the mean
  |act(gate(x)) * up(x)| per intermediate channel over calibration tokens; each layer
  keeps its top share. One share for every layer, so the config stays uniform.

Size of Qwen3-Embedding-8B (36 layers, hidden 4096, MLP 12288): about 0.62B in the
token embedding table, which costs memory but no compute, and 0.19B per layer. 18
layers is about 4.1B, 18 layers with 75% of the MLP about 3.4B, 12 layers with 50%
about 2.0B. The pruned model needs the heal step (``heal.py --teacher <original>``)
before it is any good.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from heal import QUERY_INSTRUCTION, load_pairs, query_prefix  # noqa: E402


def backbone(model: Any) -> Any:
    """The HF decoder inside a SentenceTransformer (``Qwen3Model`` for Qwen3-Embedding)."""
    return model[0].auto_model


def param_count(model: Any) -> int:
    return sum(p.numel() for p in model.parameters())


def calibration_texts(data: Path, n: int, instruction: str, model: Any, seed: int) -> list[str]:
    """Half questions (with the query prompt), half unit code, from the training pairs."""
    units, train, _ = load_pairs(data)
    rng = random.Random(seed)
    pairs = rng.sample(train, min(n // 2, len(train)))
    prefix = query_prefix(model, instruction)
    docs = rng.sample(list(units.values()), min(n - len(pairs), len(units)))
    return [prefix + p["query"] for p in pairs] + [u["text"] for u in docs]


@torch.no_grad()
def embed(model: Any, texts: Sequence[str], batch: int) -> torch.Tensor:
    return model.encode(
        list(texts), batch_size=batch, convert_to_tensor=True, normalize_embeddings=True
    )


class LayerView:
    """Swaps which decoder layers run, without copying weights."""

    def __init__(self, model: Any) -> None:
        self.net = backbone(model)
        self.all = list(self.net.layers)
        self.types = list(getattr(self.net.config, "layer_types", None) or [])

    def use(self, keep: Sequence[int]) -> None:
        self.net.layers = nn.ModuleList(self.all[i] for i in keep)
        self.net.config.num_hidden_layers = len(keep)
        if self.types:
            self.net.config.layer_types = [self.types[i] for i in keep]
        for new, i in enumerate(keep):
            attn = getattr(self.all[i], "self_attn", None)
            if attn is not None and hasattr(attn, "layer_idx"):
                attn.layer_idx = new


def greedy_layers(
    model: Any, texts: Sequence[str], keep: int, batch: int, protect: Sequence[int]
) -> tuple[list[int], list[dict[str, Any]]]:
    view = LayerView(model)
    current = list(range(len(view.all)))
    ref = embed(model, texts, batch)
    trace = []
    while len(current) > keep:
        best, best_cos = -1, -2.0
        for i in current:
            if i in protect:
                continue
            view.use([j for j in current if j != i])
            cos = float((embed(model, texts, batch) * ref).sum(-1).mean())
            if cos > best_cos:
                best, best_cos = i, cos
        current.remove(best)
        trace.append({"dropped": best, "cos_to_full": best_cos, "layers": len(current)})
        print(f"drop layer {best}: {len(current)} left, cos to full {best_cos:.4f}", flush=True)
    view.use(current)
    return current, trace


@torch.no_grad()
def influence_layers(
    model: Any, texts: Sequence[str], keep: int, batch: int, protect: Sequence[int]
) -> tuple[list[int], list[dict[str, Any]]]:
    """ShortGPT block influence: 1 - cos(hidden in, hidden out), averaged over tokens."""
    net = backbone(model)
    n = len(net.layers)
    total = torch.zeros(n, dtype=torch.float64)
    count = 0
    device = next(net.parameters()).device
    for i in range(0, len(texts), batch):
        feats = model.tokenize(list(texts[i : i + batch]))
        feats = {k: v.to(device) for k, v in feats.items() if hasattr(v, "to")}
        out = net(
            input_ids=feats["input_ids"],
            attention_mask=feats["attention_mask"],
            output_hidden_states=True,
        )
        mask = feats["attention_mask"].bool()
        hs = out.hidden_states
        for layer in range(n):
            cos = F.cosine_similarity(hs[layer][mask].float(), hs[layer + 1][mask].float(), dim=-1)
            total[layer] += float((1 - cos).sum())
        count += int(mask.sum())
    scores = (total / max(count, 1)).tolist()
    order = sorted((i for i in range(n) if i not in protect), key=lambda i: scores[i])
    drop = set(order[: n - keep])
    current = [i for i in range(n) if i not in drop]
    LayerView(model).use(current)
    return current, [{"layer": i, "influence": s} for i, s in enumerate(scores)]


@torch.no_grad()
def prune_mlp(model: Any, texts: Sequence[str], share: float, batch: int) -> int:
    """Keep the top ``share`` of each layer's MLP channels by mean |act(gate) * up|."""
    net = backbone(model)
    sums: dict[int, torch.Tensor] = {}
    counts: dict[int, int] = {}
    hooks = []
    for li, layer in enumerate(net.layers):

        def hook(_: nn.Module, inputs: tuple[torch.Tensor, ...], li: int = li) -> None:
            x = inputs[0].detach().float().abs()
            flat = x.reshape(-1, x.shape[-1])
            sums[li] = sums.get(li, 0) + flat.sum(0)
            counts[li] = counts.get(li, 0) + flat.shape[0]

        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(hook))
    try:
        embed(model, texts, batch)
    finally:
        for h in hooks:
            h.remove()
    width = net.layers[0].mlp.down_proj.in_features
    k = max(1, int(round(width * share / 128)) * 128)  # stays a multiple of 128 for kernels
    for li, layer in enumerate(net.layers):
        idx = torch.topk(sums[li] / counts[li], k).indices.sort().values
        mlp = layer.mlp
        for name in ("gate_proj", "up_proj"):
            old = getattr(mlp, name)
            new = nn.Linear(
                old.in_features,
                k,
                bias=old.bias is not None,
                device=old.weight.device,
                dtype=old.weight.dtype,
            )
            new.weight.copy_(old.weight[idx])
            if old.bias is not None:
                new.bias.copy_(old.bias[idx])
            setattr(mlp, name, new)
        old = mlp.down_proj
        new = nn.Linear(
            k,
            old.out_features,
            bias=old.bias is not None,
            device=old.weight.device,
            dtype=old.weight.dtype,
        )
        new.weight.copy_(old.weight[:, idx])
        if old.bias is not None:
            new.bias.copy_(old.bias)
        mlp.down_proj = new
        if hasattr(mlp, "intermediate_size"):
            mlp.intermediate_size = k
    net.config.intermediate_size = k
    return k


def prune(
    model: Any,
    data: Path,
    *,
    keep_layers: int | None,
    mlp_keep: float | None,
    method: str = "greedy",
    calib: int = 256,
    batch: int = 32,
    instruction: str = QUERY_INSTRUCTION,
    protect: Sequence[int] = (0,),
    seed: int = 0,
) -> dict[str, Any]:
    """Edit ``model`` in place; returns what was done, for the run record."""
    texts = calibration_texts(data, calib, instruction, model, seed)
    n = len(backbone(model).layers)
    record: dict[str, Any] = {"params_before": param_count(model), "layers_before": n}
    full = embed(model, texts, batch)
    if keep_layers and keep_layers < n:
        protect = [p if p >= 0 else n + p for p in protect]
        pick = greedy_layers if method == "greedy" else influence_layers
        kept, trace = pick(model, texts, keep_layers, batch, protect)
        record.update(kept_layers=kept, layer_trace=trace, layer_method=method)
    if mlp_keep and mlp_keep < 1:
        record["intermediate_size"] = prune_mlp(model, texts, mlp_keep, batch)
    record["params_after"] = param_count(model)
    record["cos_to_full"] = float((embed(model, texts, batch) * full).sum(-1).mean())
    return record


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="pairs from python -m codebrain.eval.pairs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-layers", type=int, default=None)
    ap.add_argument("--mlp-keep", type=float, default=None, help="share of MLP channels kept")
    ap.add_argument("--method", choices=["greedy", "influence"], default="greedy")
    ap.add_argument(
        "--protect", type=int, action="append", default=None, help="layer never dropped (default 0)"
    )
    ap.add_argument("--calib", type=int, default=256, help="calibration texts")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(
        args.model, device=device, model_kwargs={"torch_dtype": torch.bfloat16}
    )
    record = prune(
        model,
        Path(args.data),
        keep_layers=args.keep_layers,
        mlp_keep=args.mlp_keep,
        method=args.method,
        calib=args.calib,
        batch=args.batch,
        protect=args.protect if args.protect is not None else (0,),
        seed=args.seed,
    )
    out = Path(args.out)
    model.save(str(out))
    (out / "codebrain-prune.json").write_text(json.dumps({"model": args.model, **record}, indent=2))
    print(
        f"{record['params_before'] / 1e9:.2f}B -> {record['params_after'] / 1e9:.2f}B params, "
        f"cos to full {record['cos_to_full']:.4f}; saved {out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
