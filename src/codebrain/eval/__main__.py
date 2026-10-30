"""``python -m codebrain.eval {retrieval,paging,qa,all}``: run the eval and write a report."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import gc
import json
import platform
import random
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from codebrain.config import get_settings
from codebrain.eval import codebase_qa as qa
from codebrain.eval import paging, retrieval
from codebrain.eval.backends import LLM, CachedEmbedder, Embedder, FakeLLM, HashEmbedder, OpenAILLM
from codebrain.eval.mine import Git as _Git
from codebrain.eval.mine import GoldTask, Snapshot, assign_splits, mine_repo, snapshot, task_to_json
from codebrain.eval.units import is_test_path

DEFAULT_LLM = "Qwen/Qwen3.8-27B"
DEFAULT_EMBED = ["Qwen/Qwen3-Embedding-8B"]


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def resolve_repo(spec: str, cache: Path) -> tuple[str, Path]:
    """A local path is used as is; a URL is cloned once into the cache with full history."""
    p = Path(spec).expanduser()
    if p.exists():
        return p.resolve().name, p.resolve()
    name = "/".join(spec.rstrip("/").removesuffix(".git").split("/")[-2:])
    dest = cache / name.replace("/", "__")
    if not dest.exists():
        log(f"cloning {spec}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--quiet", spec, str(dest)], check=True)
    return name, dest


def make_llm(args: argparse.Namespace) -> LLM:
    if args.fake:
        return FakeLLM()
    return OpenAILLM(
        args.llm_model,
        args.llm_base_url,
        concurrency=args.concurrency,
        disable_thinking=not args.allow_thinking,
        tokenizer=args.tokenizer,
    )


def make_embedder(name: str, args: argparse.Namespace) -> Embedder:
    if args.fake:
        # One fake model per distinct --embed-model, so fake runs can compare embedders.
        first = not args.embed_model or name == args.embed_model[0]
        return HashEmbedder(salt="" if first else name.replace("/", "-"))
    from codebrain.eval.backends import STEmbedder

    return STEmbedder(name, batch_size=args.embed_batch, trust_remote_code=args.trust_remote_code)


def free_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def fmt(x: object, ci: object = None) -> str:
    if isinstance(x, float):
        s = f"{x:.3f}"
        if isinstance(ci, tuple | list):
            s += f" ({ci[0]:.2f}–{ci[1]:.2f})"
        return s
    return str(x)


async def cmd_retrieval(args: argparse.Namespace, out: Path, llm: LLM) -> dict[str, Any]:
    t0 = time.perf_counter()
    tasks: list[tuple[GoldTask, Snapshot]] = []
    for spec in args.repo:
        name, path = resolve_repo(spec, Path(args.cache).expanduser())
        found = mine_repo(
            path,
            name,
            max_tasks=args.tasks_per_repo,
            max_scan=args.max_scan,
            include_tests=args.include_tests,
            seed=args.seed,
        )
        log(f"{name}: {len(found)} tasks mined")
        tasks.extend(found)
    if not tasks:
        raise SystemExit("no usable commits found; try --max-scan higher or another repo")
    assign_splits([t for t, _ in tasks], seed=args.seed)
    mine_s = time.perf_counter() - t0
    units = {u.content_hash: u for _, s in tasks for u in s.units}
    log(f"{len(tasks)} tasks, {len(units)} distinct units across snapshots ({mine_s:.1f}s)")

    variants = ["original"]
    if not args.no_paraphrase:
        await retrieval.paraphrase_queries(llm, [t for t, _ in tasks])
        variants.append("paraphrase")

    with (out / "goldset.jsonl").open("w") as f:
        for t, _ in tasks:
            f.write(json.dumps(task_to_json(t)) + "\n")

    summaries: dict[str, str] | None = None
    summary_s = 0.0
    if not args.no_summaries:
        log(f"summarizing {len(units)} units with {llm.name}")
        summaries, summary_s = await retrieval.summarize_units(llm, list(units.values()))
        log(f"summaries done in {summary_s:.1f}s ({len(units) / max(summary_s, 1e-9):.1f}/s)")

    hyde_texts: dict[str, str] | None = None
    if not args.no_hyde:
        keys, queries = [], []
        for t, _ in tasks:
            for v in variants:
                keys.append(f"{t.id}|{v}")
                queries.append(t.query if v == "original" else (t.query_paraphrase or t.query))
        hyde_texts = dict(zip(keys, await retrieval.hyde(llm, queries), strict=True))

    per_model: dict[str, Any] = {}
    for model in args.embed_model:
        emb = CachedEmbedder(make_embedder(model, args))
        log(f"embedding with {emb.name}")
        sums = None
        if summaries is not None:
            sums = {h: summaries[h] for h in units}
        results = retrieval.run_retrieval(tasks, emb, sums, hyde_texts, variants)
        rows = retrieval.score(results, tasks, llm.count_tokens, budget=args.budget)
        per_model[emb.name] = {
            "rows": rows,
            "embed_seconds": emb.seconds,
            "embedded_texts": emb.count,
            "embed_per_s": emb.count / max(emb.seconds, 1e-9),
        }
        safe = emb.name.replace("/", "__")
        with (out / f"rankings-{safe}.jsonl").open("w") as f:
            for r in results:
                f.write(
                    json.dumps(
                        {
                            "task": r.task_id,
                            "variant": r.variant,
                            "config": r.config,
                            "top20": r.ranked[:20],
                        }
                    )
                    + "\n"
                )
        del emb
        free_gpu()

    return {
        "tasks": len(tasks),
        "tasks_by_repo": {
            r: sum(1 for t, _ in tasks if t.repo == r) for r in sorted({t.repo for t, _ in tasks})
        },
        "distinct_units": len(units),
        "mean_relevant": sum(len(t.relevant) for t, _ in tasks) / len(tasks),
        "mine_seconds": mine_s,
        "summary_seconds": summary_s,
        "summaries_per_s": (len(units) / summary_s) if summary_s else None,
        "models": per_model,
    }


async def cmd_paging(args: argparse.Namespace, out: Path, llm: LLM) -> dict[str, Any]:
    name, path = resolve_repo(args.repo[0], Path(args.cache).expanduser())
    git = _Git(path)
    try:
        head = git.run("rev-parse", "HEAD").strip()
        pool = [u for u in snapshot(git, head, False).units if not is_test_path(u.path)]
    finally:
        git.close()
    pool = [u for u in pool if 200 <= len(u.text) <= 5000] or pool
    log(f"paging: {len(pool)} filler units from {name}")
    rng = random.Random(args.seed)
    sessions = [
        paging.build_session(
            f"s{i}", pool, llm, rng, target_tokens=args.session_tokens, n_needles=args.needles
        )
        for i in range(args.sessions)
    ]
    emb = CachedEmbedder(make_embedder(args.embed_model[0], args))
    t0 = time.perf_counter()
    outs = await asyncio.gather(
        *(
            paging.run_session(
                s,
                llm,
                emb,
                window=args.window,
                high=args.high_water,
                low=args.low_water,
                memory_budget=args.memory_budget,
                max_context=args.max_context,
            )
            for s in sessions
        )
    )
    secs = time.perf_counter() - t0
    probes = [r for rs, _ in outs for r in rs]
    with (out / "paging-probes.jsonl").open("w") as f:
        for r in probes:
            f.write(json.dumps(paging.probe_to_json(r)) + "\n")
    del emb
    free_gpu()
    return {
        "sessions": [st for _, st in outs],
        "probes": len(probes),
        "seconds": secs,
        "embed_model": args.embed_model[0],
        **paging.summarize_probes(probes),
    }


def check_database(url: str) -> None:
    """Fail before hours of other work if the store the Q&A benchmark needs is not there."""
    from codebrain.store import migrate

    try:
        migrate(url)
    except Exception as e:
        raise SystemExit(
            f"codebase Q&A needs Postgres with pgvector at {url} ({e}). "
            "Start one with eval/b300/db.sh or set CODEBRAIN_DATABASE_URL."
        ) from e


async def cmd_qa(args: argparse.Namespace, out: Path, llm: LLM) -> dict[str, Any]:
    root = Path(args.qa_codebase).expanduser().resolve()
    questions = qa.load_questions(Path(args.qa_questions).expanduser())
    name = root.name
    libs: set[str] = set()
    if args.qa_haystack:
        root = qa.build_haystack_codebase(
            root,
            args.qa_haystack,
            Path(args.cache).expanduser(),
            args.qa_haystack_tokens,
            # Not the model's tokenizer: the same files must be picked whatever the model,
            # because questions are written against them.
            lambda text: len(text) // 4,
            out / f"codebase-{name}-haystack",
        )
        libs = {p.name for p in (root / "libs").iterdir() if p.is_dir()}
    asked = [q for q in questions if q.package is None or q.package in libs]
    if len(asked) < len(questions):
        log(f"qa: skipping {len(questions) - len(asked)} questions about libraries not mixed in")
    questions = asked
    if args.qa_haystack:
        # Other libraries now share the tree, so say which package the question is about.
        questions = [
            dataclasses.replace(q, question=f"About the {q.package or name} package: {q.question}")
            for q in questions
        ]
    files = qa.codebase_files(root)
    codebase = qa.render_codebase(files)
    codebase_tokens = llm.count_tokens(codebase)
    if not args.qa_recall_only and codebase_tokens + args.qa_max_tokens + 1000 > args.max_context:
        raise SystemExit(
            f"{root.name} is {codebase_tokens:,} tokens, too big for --max-context "
            f"{args.max_context:,} in normal mode"
        )
    log(f"qa: {len(questions)} questions, {root.name}: {len(files)} files, {codebase_tokens:,} tok")

    # Untimed setup: index the codebase into the store.
    repo = f"qa-fixture/{root.name}"
    model = args.embed_model[0]
    inner = make_embedder(model, args)
    summarizer: Any
    if args.fake or args.qa_summarizer == "heuristic":
        from codebrain.ingest.summarize import HeuristicSummarizer

        summarizer = HeuristicSummarizer()
    else:
        summarizer = qa.openai_summarizer(
            args.llm_model,
            args.llm_base_url,
            concurrency=min(args.concurrency, 64),
            disable_thinking=not args.allow_thinking,
        )
    log(f"qa: ingesting {root.name} with {model} (not timed)")
    stats = qa.ingest_codebase(
        args.database_url, root, repo, qa.QueryEmbedder(inner, query=False), summarizer
    )
    log(f"qa: ingest took {stats.seconds:.1f}s, {stats.units} units")
    if args.qa_query_instruction != "none" and hasattr(inner, "query_instruction"):
        inner.query_instruction = args.qa_query_instruction
    reranker = None
    if args.qa_rerank and not args.fake:
        reranker = qa.LLMReranker(
            args.llm_model, args.llm_base_url, disable_thinking=not args.allow_thinking
        )
    recall = qa.StoreRecall(
        args.database_url,
        repo,
        qa.QueryEmbedder(inner, query=True),
        args.qa_budget,
        args.qa_k,
        reranker=reranker,
    )

    def whole_codebase(_: str) -> qa.Context:
        return qa.Context(codebase)

    # Timed: answering only, one question at a time.
    answers: list[qa.Answer] = []
    modes: tuple[tuple[str, Callable[[str], qa.Context]], ...] = (
        ("normal", whole_codebase),
        ("codebrain", recall),
    )
    embedders: list[dict[str, Any]] = []
    dense: list[qa.DenseHit] = []
    try:
        if args.qa_recall_only:
            modes = ()
            answers, hits = probe_embedder(questions, recall, model, llm.count_tokens)
            dense += hits
            embedders.append(qa.embedder_row(model, stats, answers, hits))
        for mode, context_for in modes:
            for q in questions:
                a = await qa.ask(
                    llm,
                    q,
                    mode,
                    context_for,
                    llm.count_tokens,
                    max_tokens=args.qa_max_tokens,
                    warm_cache=args.qa_warm_cache,
                )
                answers.append(a)
                log(
                    f"qa {mode} {q.id}: {'correct' if a.correct else 'missing ' + str(a.missing)} "
                    f"{a.seconds:.2f}s {a.prompt_tokens + a.completion_tokens:,} tokens"
                )
    finally:
        recall.close()
    del inner
    free_gpu()
    others: list[qa.Answer] = []
    if args.qa_recall_only:
        # The same units and summaries, vectors from each further --embed-model.
        for other in args.embed_model[1:]:
            log(f"qa: adding vectors from {other} (not timed)")
            inner = make_embedder(other, args)
            st = qa.ingest_codebase(
                args.database_url,
                root,
                repo,
                qa.QueryEmbedder(inner, query=False),
                summarizer,
                fresh=False,
            )
            if args.qa_query_instruction != "none" and hasattr(inner, "query_instruction"):
                inner.query_instruction = args.qa_query_instruction
            rec = qa.StoreRecall(
                args.database_url,
                repo,
                qa.QueryEmbedder(inner, query=True),
                args.qa_budget,
                args.qa_k,
                reranker=reranker,
            )
            try:
                ans, hits = probe_embedder(questions, rec, other, llm.count_tokens)
            finally:
                rec.close()
            others += ans
            dense += hits
            embedders.append(qa.embedder_row(other, st, ans, hits))
            del inner
            free_gpu()
    elif len(args.embed_model) > 1:
        log("qa: answers use the first --embed-model; add --qa-recall-only to compare embedders")

    if not args.fake and not args.no_judge and not args.qa_recall_only:
        by_id = {q.id: q for q in questions}
        verdicts = await asyncio.gather(*(qa.judge(llm, by_id[a.question_id], a) for a in answers))
        for a, v in zip(answers, verdicts, strict=True):
            a.judge = v

    with (out / "qa-answers.jsonl").open("w") as f:
        for a in answers + others:
            f.write(json.dumps(qa.answer_to_json(a)) + "\n")
    if dense:
        with (out / "qa-dense.jsonl").open("w") as f:
            for h in dense:
                f.write(json.dumps(dataclasses.asdict(h)) + "\n")
    return {
        "codebase": root.name,
        "files": len(files),
        "codebase_tokens": codebase_tokens,
        "budget": args.qa_budget,
        "k": args.qa_k,
        "warm_cache": args.qa_warm_cache,
        "recall_only": args.qa_recall_only,
        "rerank": bool(reranker),
        "rerank_prompt_tokens": reranker.prompt_tokens if reranker else 0,
        "questions": [q.id for q in questions],
        "packages": {q.id: q.package or name for q in questions},
        "ingest": dataclasses.asdict(stats),
        "summary": {m: qa.summarize(answers, m) for m in qa.MODES},
        "answers": [qa.answer_to_json(a) for a in answers],
        "embedders": embedders,
    }


def probe_embedder(
    questions: list[qa.Question],
    recall: qa.StoreRecall,
    model: str,
    count_tokens: Callable[[str], int],
) -> tuple[list[qa.Answer], list[qa.DenseHit]]:
    """Recall-only probes for one embedder, plus its dense-only ranking per question."""
    answers, hits = [], []
    for q in questions:
        a = qa.probe(q, recall, count_tokens)
        a.embedder = model
        answers.append(a)
        ranked, secs = recall.dense(q.question)
        hits.append(qa.DenseHit(q.id, model, list(q.files), ranked, secs))
        log(
            f"qa recall {q.id} [{model}]: gold {len(a.files_hit)}/{len(a.expected_files)}, "
            f"facts {a.context_facts}/{a.facts_total}, {a.recall_seconds * 1000:.0f} ms"
        )
    return answers, hits


def render(meta: dict[str, Any], res: dict[str, Any]) -> str:
    lines = [
        "# Memory eval report",
        "",
        f"- Run: {meta['started']} · host {meta['host']}",
        f"- LLM: `{meta['llm']}` · repos: {', '.join(meta['repos']) or '–'}",
        f"- LLM tokens: {meta['llm_input_tokens']:,} in / {meta['llm_output_tokens']:,} out",
        "",
    ]
    r = res.get("retrieval")
    if r:
        lines += [
            "## Q1 · Retrieval",
            "",
            f"{r['tasks']} tasks {r['tasks_by_repo']}, {r['distinct_units']} distinct units, "
            f"{r['mean_relevant']:.1f} relevant units per task on average. Labels are mined "
            "automatically; see `goldset.jsonl`. Intervals are 95% bootstrap over tasks.",
            "",
        ]
        for model, m in r["models"].items():
            lines += [
                f"### `{model}`  ·  {m['embed_per_s']:.0f} texts/s embedded",
                "",
                "| query | config | R@10 | R@20 | MRR | nDCG@10 | BudgetR@8k | p50 ms | p95 ms |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
            for row in m["rows"]:
                lines.append(
                    f"| {row['variant']} | {row['config']} | {fmt(row['r@10'], row['r@10_ci'])} "
                    f"| {fmt(row['r@20'])} | {fmt(row['mrr'], row['mrr_ci'])} "
                    f"| {fmt(row['ndcg@10'])} | {fmt(row['br'])} "
                    f"| {row['p50_ms']:.1f} | {row['p95_ms']:.1f} |"
                )
            lines.append("")
        if r.get("summaries_per_s"):
            lines += [f"Summaries: {r['summaries_per_s']:.1f} units/s.", ""]
    p = res.get("paging")
    if p:
        arms = list(p["overall"])
        lines += [
            "## Q2 · Paging vs compaction",
            "",
            f"{len(p['sessions'])} sessions, {p['probes']} probes, {p['seconds']:.0f}s. "
            f"Mean session {sum(s['tokens'] for s in p['sessions']) / len(p['sessions']):,.0f}"
            " tokens.",
            "",
            "| | " + " | ".join(arms) + " |",
            "| --- | " + " | ".join("---" for _ in arms) + " |",
            "| accuracy | "
            + " | ".join(f"{p['overall'][a]['acc']:.3f} (n={p['overall'][a]['n']})" for a in arms)
            + " |",
            "| prompt tokens per probe | "
            + " | ".join(f"{p['overall'][a]['tokens']:,.0f}" for a in arms)
            + " |",
        ]
        for group in ("by_kind", "by_distance"):
            for k, by_arm in p[group].items():
                cells = [f"{by_arm[a]['acc']:.2f}" if by_arm[a]["n"] else "–" for a in arms]
                label = k if group == "by_kind" else f"Δ {k} turns"
                lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")
    if res.get("qa"):
        render = qa.render_probe_section if res["qa"].get("recall_only") else qa.render_section
        lines += render(res["qa"])
        lines += qa.render_embedder_section(res["qa"].get("embedders") or [])
    return "\n".join(lines)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m codebrain.eval")
    ap.add_argument("command", choices=["retrieval", "paging", "qa", "all"])
    ap.add_argument("--repo", action="append", default=[], help="git URL or local path")
    ap.add_argument("--out", default="eval-results", help="results directory")
    ap.add_argument("--cache", default="~/.cache/codebrain-eval/repos")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fake", action="store_true", help="fake models, for smoke tests")
    g = ap.add_argument_group("models")
    g.add_argument("--llm-model", default=DEFAULT_LLM)
    g.add_argument("--llm-base-url", default="http://localhost:8000/v1")
    g.add_argument("--tokenizer", default=None, help="HF tokenizer for counting, default model")
    g.add_argument("--allow-thinking", action="store_true")
    g.add_argument("--concurrency", type=int, default=256)
    g.add_argument(
        "--embed-model",
        action="append",
        default=None,
        help="repeat to compare (retrieval, and qa with --qa-recall-only; a local folder works)",
    )
    g.add_argument("--embed-batch", type=int, default=64)
    g.add_argument("--trust-remote-code", action="store_true")
    g = ap.add_argument_group("retrieval (Q1)")
    g.add_argument("--tasks-per-repo", type=int, default=20)
    g.add_argument("--max-scan", type=int, default=3000)
    g.add_argument("--include-tests", action="store_true")
    g.add_argument("--budget", type=int, default=8000)
    g.add_argument("--no-summaries", action="store_true")
    g.add_argument("--no-hyde", action="store_true")
    g.add_argument("--no-paraphrase", action="store_true")
    g = ap.add_argument_group("paging (Q2)")
    g.add_argument("--sessions", type=int, default=10)
    g.add_argument("--needles", type=int, default=20)
    g.add_argument("--session-tokens", type=int, default=100_000)
    g.add_argument("--window", type=int, default=16_000)
    g.add_argument("--high-water", type=float, default=0.70)
    g.add_argument("--low-water", type=float, default=0.50)
    g.add_argument("--memory-budget", type=int, default=2_000)
    g.add_argument("--max-context", type=int, default=131_072)
    g = ap.add_argument_group("codebase Q&A")
    g.add_argument("--qa-codebase", default=str(qa.DEFAULT_CODEBASE))
    g.add_argument("--qa-questions", default=str(qa.DEFAULT_QUESTIONS))
    g.add_argument("--qa-budget", type=int, default=4_000, help="recall token budget")
    g.add_argument("--qa-k", type=int, default=None, help="pack only the top k recall results")
    g.add_argument(
        "--qa-haystack",
        action="append",
        default=[],
        metavar="URL@REF",
        help="add this repo's code under libs/ (repeatable; 'default' = rq, huey, arq)",
    )
    g.add_argument("--qa-haystack-tokens", type=int, default=130_000, help="cap on added code")
    g.add_argument("--qa-max-tokens", type=int, default=600, help="answer length cap")
    g.add_argument("--qa-summarizer", choices=["llm", "heuristic"], default="llm")
    g.add_argument("--qa-warm-cache", action="store_true", help="let the prefix cache be reused")
    g.add_argument("--no-judge", action="store_true")
    g.add_argument(
        "--qa-rerank",
        action="store_true",
        help="let the served LLM pick among the top recall candidates (one extra call each)",
    )
    g.add_argument(
        "--qa-query-instruction",
        default=qa.QUERY_INSTRUCTION,
        help="query instruction for embedders that take one (Qwen3-Embedding); 'none' keeps "
        "the model's own",
    )
    g.add_argument(
        "--qa-recall-only",
        action="store_true",
        help="ingest and recall only: report gold-file hits and facts in context, no answers",
    )
    g.add_argument("--database-url", default=None, help="default CODEBRAIN_DATABASE_URL")
    return ap


def expand_haystack(specs: list[str]) -> list[str]:
    """``default`` anywhere in ``--qa-haystack`` stands for the built-in haystack repos."""
    out: list[str] = []
    for spec in specs:
        out.extend(qa.DEFAULT_HAYSTACK if spec == "default" else [spec])
    return list(dict.fromkeys(out))


async def amain(args: argparse.Namespace) -> int:
    args.embed_model = args.embed_model or DEFAULT_EMBED
    if args.command in ("retrieval", "paging", "all") and not args.repo:
        raise SystemExit(f"{args.command} needs at least one --repo")
    if args.qa_haystack and "default" in args.qa_haystack:
        args.qa_haystack = expand_haystack(args.qa_haystack)
    if args.command in ("qa", "all"):
        args.database_url = args.database_url or get_settings().database_url
        check_database(args.database_url)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out = Path(args.out) / stamp
    out.mkdir(parents=True, exist_ok=True)
    llm = make_llm(args)
    meta: dict[str, Any] = {
        "started": stamp,
        "host": platform.node(),
        "llm": llm.name,
        "repos": args.repo,
        "argv": sys.argv[1:],
    }
    res: dict[str, Any] = {}
    if args.command in ("retrieval", "all"):
        res["retrieval"] = await cmd_retrieval(args, out, llm)
    if args.command in ("paging", "all"):
        res["paging"] = await cmd_paging(args, out, llm)
    if args.command in ("qa", "all"):
        res["qa"] = await cmd_qa(args, out, llm)
    meta["llm_input_tokens"] = llm.input_tokens
    meta["llm_output_tokens"] = llm.output_tokens
    (out / "results.json").write_text(json.dumps({"meta": meta, **res}, indent=2, default=str))
    report = render(meta, res)
    (out / "report.md").write_text(report)
    print("\n" + report)
    log(f"wrote {out}")
    return 0


def main() -> int:
    return asyncio.run(amain(parser().parse_args()))


if __name__ == "__main__":
    sys.exit(main())
