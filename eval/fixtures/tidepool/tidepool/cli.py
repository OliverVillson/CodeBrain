"""Command line: ``tidepool <command>``.

    tidepool worker [--queue NAME ...] [--concurrency N]
    tidepool scheduler --schedules schedules.toml
    tidepool enqueue TASK [--arg JSON ...] [--delay SECONDS] [--priority N]
    tidepool stats
    tidepool dlq list QUEUE [--limit N]
    tidepool dlq requeue QUEUE (--id JOB_ID ... | --all)
    tidepool dlq purge

Handlers must be importable: pass ``--import myapp.tasks`` (repeatable) so the
``@task`` decorators run before the worker starts.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
import tomllib
from collections.abc import Sequence

from tidepool import deadletter, metrics
from tidepool.config import load_settings
from tidepool.queue import Queue
from tidepool.scheduler import Schedule, Scheduler
from tidepool.storage import open_storage
from tidepool.worker import Worker


def _queue(args: argparse.Namespace) -> Queue:
    overrides = {}
    if getattr(args, "concurrency", None):
        overrides["concurrency"] = args.concurrency
    s = load_settings(args.config, **overrides)
    return Queue(open_storage(s.database), s)


def cmd_worker(args: argparse.Namespace) -> int:
    q = _queue(args)
    metrics.serve(q.settings.metrics_port)
    Worker(q, args.queue or ["default"]).run()
    return 0


def cmd_scheduler(args: argparse.Namespace) -> int:
    q = _queue(args)
    with open(args.schedules, "rb") as fh:
        data = tomllib.load(fh)
    schedules = [Schedule(**entry) for entry in data.get("schedule", [])]
    Scheduler(q, schedules).run_forever()
    return 0


def cmd_enqueue(args: argparse.Namespace) -> int:
    q = _queue(args)
    job = q.enqueue(
        args.task, *[json.loads(a) for a in args.arg], delay=args.delay, priority=args.priority
    )
    print(job.id)
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    print(json.dumps(_queue(args).stats(), indent=2, sort_keys=True))
    return 0


def cmd_dlq(args: argparse.Namespace) -> int:
    q = _queue(args)
    if args.dlq_command == "list":
        for job in deadletter.list_dead(q, args.queue, args.limit):
            print(f"{job.id}\t{job.task}\tattempts={job.attempts}\t{(job.last_error or '')[:80]}")
        return 0
    if args.dlq_command == "requeue":
        if args.all:
            done = deadletter.requeue_all(q, args.queue)
        elif args.id:
            done = deadletter.requeue(q, args.id)
        else:
            print("give --id JOB_ID (repeatable) or --all", file=sys.stderr)
            return 2
        print(f"requeued {len(done)} job(s)")
        return 0
    if args.dlq_command == "purge":
        n = deadletter.purge_expired(q)
        print(f"purged {n} dead job(s) older than {q.settings.dlq_retention_days} days")
        return 0
    return 2


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tidepool")
    p.add_argument("--config", default=None, help="path to tidepool.toml")
    p.add_argument("--import", dest="imports", action="append", default=[], metavar="MODULE")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    w = sub.add_parser("worker")
    w.add_argument("--queue", action="append")
    w.add_argument("--concurrency", type=int)
    w.set_defaults(fn=cmd_worker)

    s = sub.add_parser("scheduler")
    s.add_argument("--schedules", required=True)
    s.set_defaults(fn=cmd_scheduler)

    e = sub.add_parser("enqueue")
    e.add_argument("task")
    e.add_argument("--arg", action="append", default=[], help="JSON value, repeatable")
    e.add_argument("--delay", type=float, default=0.0)
    e.add_argument("--priority", type=int)
    e.set_defaults(fn=cmd_enqueue)

    st = sub.add_parser("stats")
    st.set_defaults(fn=cmd_stats)

    d = sub.add_parser("dlq")
    dsub = d.add_subparsers(dest="dlq_command", required=True)
    dl = dsub.add_parser("list")
    dl.add_argument("queue")
    dl.add_argument("--limit", type=int, default=50)
    dr = dsub.add_parser("requeue")
    dr.add_argument("queue")
    dr.add_argument("--id", action="append")
    dr.add_argument("--all", action="store_true")
    dsub.add_parser("purge")
    d.set_defaults(fn=cmd_dlq)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for mod in args.imports:
        importlib.import_module(mod)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
