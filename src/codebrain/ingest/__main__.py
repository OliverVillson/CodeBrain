"""``python -m codebrain.ingest ROOT [PATH ...]``: ingest a checkout into the store.

With paths, only those files or directories are re-ingested, which is what a git
post-commit hook wants::

    git diff --name-only HEAD~1 | xargs python -m codebrain.ingest .
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path

from codebrain.ingest.pipeline import Ingestor
from codebrain.ingest.store import PostgresStore
from codebrain.store import connection


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m codebrain.ingest")
    p.add_argument("root", help="checkout to ingest")
    p.add_argument("paths", nargs="*", help="only these files or directories (relative to root)")
    p.add_argument("--repo", help="repo name used in unit ids (default: the root's folder name)")
    p.add_argument("--force", action="store_true", help="re-summarize and re-embed everything")
    p.add_argument("--database-url", default=None, help="defaults to CODEBRAIN_DATABASE_URL")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    root = Path(args.root).resolve()
    with connection(args.database_url) as conn:
        ingestor = Ingestor(PostgresStore(conn), args.repo or root.name)
        report = ingestor.ingest(root, args.paths or None, force=args.force)
    print(json.dumps(dataclasses.asdict(report), indent=2))
    return 1 if report.errors else 0


if __name__ == "__main__":
    sys.exit(main())
