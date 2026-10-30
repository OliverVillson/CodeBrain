"""BlackHole-Dev recipe (draft): Qwen3-Embedding-8B -> a smaller CodeBrain embedder.

Stages, in BlackHole-Harness order (Read -> REAP -> Heal -> Quantize -> Ship):

1. ``load``           Read: the model as a SentenceTransformer, so pooling and prompts
                       travel with it.
2. ``prune_layers``   model.edit, in place of REAP (REAP prunes MoE experts; this model is
                       dense): greedy layer drop scored on embedding drift.
3. ``prune_mlp``      model.edit: Minitron-style MLP channel pruning.
4. ``heal``           job.fn, forked from the built-in Heal: contrastive InfoNCE on
                       CodeBrain question -> code pairs, distilled from the unpruned model,
                       instead of KL on next-token logits (an embedder has no logits to
                       match, and the LM heal would not train the pooled vector).
5. ``quantize``       The built-in Quantize is LM-oriented (GPTQ, REAQ, HEX tables). For an
                       embedder the first thing to try is FP8 at serve time (vLLM pooling
                       runner, ``--quantization fp8``), measured with the same recall-only
                       eval; not done in this file.
6. ``package``        Ship: a SentenceTransformer folder (safetensors, bf16) plus a record of
                       every stage, loadable by CodeBrain's eval as ``--embed-model <folder>``.

About the BlackHole-Dev API: the repo is private, so this is written from the
architecture map. What the map says: a recipe is a Python file of named functions;
``model.edit`` stages change the model and ``job.fn`` stages run a job; both replace or
add pipeline stages, keep the harness's stage order, and are found through the
``blackhole.ops`` and ``blackhole.jobfn`` entry points. What it does not say: the
decorator's name and signature, how parameters reach a step, and what a step receives
(model object or folder). So every stage here is a plain function
``stage(model, ctx) -> model`` and ``STAGES`` lists them with their kind; ``register``
is the one place that needs the real decorator. Until then, run the same stages without
the app:

    .venv-eval/bin/python eval/embedder/recipe_codebrain_embedder.py \\
        --data embedder-data --keep-layers 18 --mlp-keep 0.75 --out models/cb-emb-l18-m75
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import heal as heal_mod  # noqa: E402
import prune as prune_mod  # noqa: E402

BASE_MODEL = "Qwen/Qwen3-Embedding-8B"


@dataclass
class Context:
    """What the stages share: settings in, records out."""

    data: Path
    out: Path
    base_model: str = BASE_MODEL
    keep_layers: int | None = 18
    mlp_keep: float | None = 0.75
    layer_method: str = "greedy"
    heal: heal_mod.HealConfig = field(
        default_factory=lambda: heal_mod.HealConfig(matryoshka=(1024, 256), distill=1.0)
    )
    teacher: Any = None
    records: dict[str, Any] = field(default_factory=dict)

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


Stage = Callable[[Any, Context], Any]


def load(model: Any, ctx: Context) -> Any:
    if model is not None:
        return model
    dtype = torch.bfloat16 if ctx.heal.bf16_weights else torch.float32
    return heal_mod.load_model(ctx.base_model, ctx.device, dtype)


def prune_layers(model: Any, ctx: Context) -> Any:
    ctx.records["prune_layers"] = prune_mod.prune(
        model, ctx.data, keep_layers=ctx.keep_layers, mlp_keep=None, method=ctx.layer_method
    )
    return model


def prune_mlp(model: Any, ctx: Context) -> Any:
    ctx.records["prune_mlp"] = prune_mod.prune(
        model, ctx.data, keep_layers=None, mlp_keep=ctx.mlp_keep
    )
    return model


def heal(model: Any, ctx: Context) -> Any:
    teacher = ctx.teacher
    if teacher is None and ctx.heal.distill > 0:
        teacher = heal_mod.load_model(ctx.base_model, ctx.device, torch.bfloat16)
    result = heal_mod.heal(
        model, ctx.data, ctx.heal, teacher=teacher, log_path=ctx.out / "heal.log"
    )
    ctx.records["heal"] = {"config": asdict(ctx.heal), "result": asdict(result)}
    del teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model


def package(model: Any, ctx: Context) -> Any:
    ctx.out.mkdir(parents=True, exist_ok=True)
    model.save(str(ctx.out))
    record = {"base_model": ctx.base_model, "params": prune_mod.param_count(model), **ctx.records}
    (ctx.out / "codebrain-recipe.json").write_text(json.dumps(record, indent=2, default=str))
    return model


STAGES: list[tuple[str, str, Stage]] = [
    ("load", "read", load),
    ("prune_layers", "model.edit", prune_layers),
    ("prune_mlp", "model.edit", prune_mlp),
    ("heal", "job.fn", heal),
    ("package", "ship", package),
]


def register(bh: Any) -> None:  # pragma: no cover - needs BlackHole-Dev
    """Hook the stages into BlackHole-Dev. ASSUMED API: replace with the real decorator.

    Expected shape, per the architecture map: each ``model.edit`` / ``job.fn`` stage is a
    named function the pipeline loads through the ``blackhole.ops`` / ``blackhole.jobfn``
    entry points, and ``heal`` replaces the built-in Heal stage.
    """
    raise NotImplementedError(
        "blackhole_dev's recipe decorator is not known here (private repo); wrap each "
        "function in STAGES with it, mapping kind 'model.edit' and 'job.fn' to the "
        "matching stage types, and point the heal stage's data at the pairs folder."
    )


def run(ctx: Context, only: Sequence[str] | None = None) -> Any:
    """Run the stages in order without the app. The output folder gets each stage's record."""
    ctx.out.mkdir(parents=True, exist_ok=True)
    model = None
    for name, kind, fn in STAGES:
        if only and name not in only and name != "load":
            continue
        if name == "prune_layers" and not ctx.keep_layers:
            continue
        if name == "prune_mlp" and not (ctx.mlp_keep and ctx.mlp_keep < 1):
            continue
        t0 = time.perf_counter()
        heal_mod.log(f"stage {name} ({kind})", ctx.out / "heal.log")
        model = fn(model, ctx)
        ctx.records.setdefault("seconds", {})[name] = time.perf_counter() - t0
    return model


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-layers", type=int, default=18, help="0 keeps them all")
    ap.add_argument("--mlp-keep", type=float, default=0.75, help="1 keeps them all")
    ap.add_argument("--layer-method", choices=["greedy", "influence"], default="greedy")
    heal_mod.add_heal_args(ap)
    args = ap.parse_args(argv)
    if not args.teacher:
        args.teacher = args.base_model  # distill from the unpruned model by default
    cfg = heal_mod.config_from_args(args)
    if not args.matryoshka:
        cfg.matryoshka = (1024, 256)
    ctx = Context(
        data=Path(args.data),
        out=Path(args.out),
        base_model=args.base_model,
        keep_layers=args.keep_layers or None,
        mlp_keep=args.mlp_keep,
        layer_method=args.layer_method,
        heal=cfg,
    )
    model = run(ctx)
    print(f"{prune_mod.param_count(model) / 1e9:.2f}B params; saved {ctx.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
