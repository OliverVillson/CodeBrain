"""Payload encoding: JSON with a few extra types, compressed when large.

Encoded payloads start with one header byte:

* ``b"j"``: the rest is UTF-8 JSON;
* ``b"z"``: the rest is zlib-compressed JSON (level 6).

Compression kicks in when the JSON is longer than ``COMPRESS_THRESHOLD`` bytes and
only if it actually saves space. Payloads above ``MAX_PAYLOAD_BYTES`` after
compression are refused at enqueue time; store large blobs elsewhere and pass a key.
"""

from __future__ import annotations

import base64
import json
import zlib
from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from tidepool.errors import PayloadError

RAW = b"j"
ZLIB = b"z"
COMPRESS_THRESHOLD = 4096
COMPRESS_LEVEL = 6
MAX_PAYLOAD_BYTES = 1_048_576

__all__ = ["PayloadError", "decode", "encode"]


def _default(o: Any) -> Any:
    if isinstance(o, datetime):
        return {"__dt__": o.isoformat()}
    if isinstance(o, Decimal):
        return {"__dec__": str(o)}
    if isinstance(o, UUID):
        return {"__uuid__": str(o)}
    if isinstance(o, bytes):
        return {"__b64__": base64.b64encode(o).decode()}
    if isinstance(o, set | frozenset):
        return {"__set__": sorted(o, key=repr)}
    raise PayloadError(f"cannot encode {type(o).__name__} in a job payload")


def _hook(d: dict[str, Any]) -> Any:
    if len(d) == 1:
        (k, v), = d.items()
        if k == "__dt__":
            return datetime.fromisoformat(v)
        if k == "__dec__":
            return Decimal(v)
        if k == "__uuid__":
            return UUID(v)
        if k == "__b64__":
            return base64.b64decode(v)
        if k == "__set__":
            return set(v)
    return d


def encode(args: list[Any], kwargs: dict[str, Any]) -> bytes:
    raw = json.dumps({"a": args, "k": kwargs}, default=_default, separators=(",", ":")).encode()
    out = RAW + raw
    if len(raw) > COMPRESS_THRESHOLD:
        packed = zlib.compress(raw, COMPRESS_LEVEL)
        if len(packed) < len(raw):
            out = ZLIB + packed
    if len(out) > MAX_PAYLOAD_BYTES:
        raise PayloadError(f"payload is {len(out)} bytes, limit {MAX_PAYLOAD_BYTES}")
    return out


def decode(data: bytes) -> tuple[list[Any], dict[str, Any]]:
    if not data:
        raise PayloadError("empty payload")
    header, body = data[:1], data[1:]
    if header == ZLIB:
        try:
            body = zlib.decompress(body)
        except zlib.error as e:
            raise PayloadError(f"corrupt compressed payload: {e}") from e
    elif header != RAW:
        raise PayloadError(f"unknown payload header {header!r}")
    obj = json.loads(body, object_hook=_hook)
    return list(obj["a"]), dict(obj["k"])
