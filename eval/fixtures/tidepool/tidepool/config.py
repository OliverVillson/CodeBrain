"""Settings and where they come from.

Values are resolved in layers, each overriding the one before it:

1. the defaults on ``Settings``;
2. the ``[tidepool]`` table of ``tidepool.toml`` in the working directory (or the file
   named by ``TIDEPOOL_CONFIG``);
3. environment variables named ``TIDEPOOL_`` plus the field name in upper case, for
   example ``TIDEPOOL_LEASE_SECONDS=120``;
4. keyword arguments passed to ``load_settings`` by the embedding application.

Rate limits are a table in the file (``[tidepool.rate_limits]``) or a comma list in
``TIDEPOOL_RATE_LIMITS`` such as ``emails=50/s,reports=10/m``.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

from tidepool.errors import ConfigError

ENV_PREFIX = "TIDEPOOL_"
DEFAULT_CONFIG_FILE = "tidepool.toml"


@dataclass(frozen=True)
class Settings:
    database: str = "tidepool.db"
    """SQLite path, or ``:memory:`` for the in-memory backend."""

    # Leases and heartbeats
    lease_seconds: int = 60
    heartbeat_divisor: int = 3
    """Workers heartbeat every ``lease_seconds / heartbeat_divisor`` seconds."""
    reap_interval_seconds: int = 15
    """How often a worker looks for expired leases held by dead workers."""

    # Retries
    max_attempts: int = 7
    backoff_base: float = 2.0
    backoff_cap: float = 900.0

    # Workers
    concurrency: int = 4
    poll_interval: float = 0.5
    shutdown_grace_seconds: int = 30

    # Enqueue
    dedupe_window_seconds: int = 86_400
    default_priority: int = 5

    # Dead letters
    dlq_retention_days: int = 14

    # Scheduler
    misfire_grace_seconds: int = 60

    # Integrations
    rate_limits: dict[str, str] = field(default_factory=dict)
    webhook_url: str | None = None
    webhook_secret: str | None = None
    admin_token: str | None = None
    metrics_port: int = 9464

    @property
    def heartbeat_interval(self) -> float:
        return self.lease_seconds / self.heartbeat_divisor

    def validate(self) -> Settings:
        if self.lease_seconds < 10:
            raise ConfigError("lease_seconds must be at least 10")
        if self.heartbeat_divisor < 2:
            raise ConfigError("heartbeat_divisor must be at least 2, or leases expire mid-beat")
        if not 1 <= self.max_attempts <= 100:
            raise ConfigError("max_attempts must be between 1 and 100")
        if self.backoff_base <= 0 or self.backoff_cap < self.backoff_base:
            raise ConfigError("need 0 < backoff_base <= backoff_cap")
        if not 0 <= self.default_priority <= 9:
            raise ConfigError("default_priority must be 0-9")
        if self.webhook_url and not self.webhook_secret:
            raise ConfigError("webhook_url needs webhook_secret so receivers can verify calls")
        return self


def _coerce(name: str, raw: Any, current: Any) -> Any:
    if name == "rate_limits":
        if isinstance(raw, dict):
            return {str(k): str(v) for k, v in raw.items()}
        out: dict[str, str] = {}
        for part in str(raw).split(","):
            if not part.strip():
                continue
            queue, _, limit = part.partition("=")
            if not limit:
                raise ConfigError(f"rate limit {part!r} should look like queue=50/s")
            out[queue.strip()] = limit.strip()
        return out
    if isinstance(current, bool):
        return str(raw).lower() in ("1", "true", "yes", "on")
    if isinstance(current, int):
        try:
            return int(raw)
        except ValueError as e:
            raise ConfigError(f"{name} must be an integer, got {raw!r}") from e
    if isinstance(current, float):
        return float(raw)
    return None if raw in ("", None) else str(raw)


def _from_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    return dict(data.get("tidepool", {}))


def _from_env(environ: dict[str, str]) -> dict[str, Any]:
    names = {f.name for f in fields(Settings)}
    out: dict[str, Any] = {}
    for key, value in environ.items():
        if key.startswith(ENV_PREFIX) and key != "TIDEPOOL_CONFIG":
            name = key[len(ENV_PREFIX) :].lower()
            if name in names:
                out[name] = value
    return out


def load_settings(
    path: str | Path | None = None, environ: dict[str, str] | None = None, **overrides: Any
) -> Settings:
    """Resolve settings from defaults, file, environment and ``overrides``, in that order."""
    environ = dict(os.environ if environ is None else environ)
    path = Path(path or environ.get("TIDEPOOL_CONFIG", DEFAULT_CONFIG_FILE))
    base = Settings()
    merged: dict[str, Any] = {}
    for layer in (_from_file(path), _from_env(environ), overrides):
        for name, raw in layer.items():
            if not hasattr(base, name):
                raise ConfigError(f"unknown setting {name!r}")
            merged[name] = _coerce(name, raw, getattr(base, name))
    return replace(base, **merged).validate()
