"""Command line entry point: ``codebrain <command>``."""

from __future__ import annotations

import argparse
import sys


def _migrate(args: argparse.Namespace) -> int:
    from codebrain.store.migrate import migrate

    applied = migrate(args.database_url)
    print("\n".join(applied) if applied else "schema is up to date")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="codebrain")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("migrate", help="apply pending database migrations")
    p.add_argument("--database-url", default=None, help="defaults to CODEBRAIN_DATABASE_URL")
    p.set_defaults(func=_migrate)

    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
