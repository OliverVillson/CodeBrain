"""Run the MCP server over stdio: ``python -m codebrain.server``.

stdout carries the protocol, so all logging goes to stderr.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys

from codebrain.server.app import build_server
from codebrain.server.backend import MemoryBackend
from codebrain.server.memory import InMemoryBackend

BACKENDS = {"memory": InMemoryBackend}


def load_backend(spec: str) -> MemoryBackend:
    """``memory``, or ``package.module:factory`` for any callable returning a backend."""
    if spec in BACKENDS:
        return BACKENDS[spec]()
    module_name, sep, attr = spec.partition(":")
    if not sep or not attr:
        raise ValueError(f"backend must be one of {sorted(BACKENDS)} or 'module:factory'")
    backend = getattr(importlib.import_module(module_name), attr)()
    if not isinstance(backend, MemoryBackend):
        raise TypeError(f"{spec} did not return a MemoryBackend")
    return backend


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m codebrain.server")
    parser.add_argument(
        "--backend",
        default="memory",
        help="'memory' (in-process fake, the default for now) or 'module:factory'",
    )
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args(argv)

    logging.basicConfig(stream=sys.stderr, level=args.log_level.upper())
    build_server(load_backend(args.backend)).run("stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
