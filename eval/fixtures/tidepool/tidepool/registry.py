"""Task registration. ``@task`` records how a function should be queued and retried."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from tidepool.errors import TaskNotFound


@dataclass(frozen=True)
class TaskSpec:
    name: str
    fn: Callable[..., Any]
    queue: str = "default"
    max_attempts: int | None = None
    """Overrides ``Settings.max_attempts`` for this task's jobs."""
    timeout: float | None = None
    """Seconds before the worker abandons the handler and raises ``JobTimeout``."""
    priority: int | None = None
    dedupe: bool = False
    """Derive a dedupe key from the arguments so duplicate enqueues collapse."""


_TASKS: dict[str, TaskSpec] = {}


def task(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    queue: str = "default",
    max_attempts: int | None = None,
    timeout: float | None = None,
    priority: int | None = None,
    dedupe: bool = False,
) -> Any:
    """Register a handler. Usable bare (``@task``) or with options (``@task(queue=...)``)."""

    def register(f: Callable[..., Any]) -> Callable[..., Any]:
        spec = TaskSpec(
            name=name or f"{f.__module__}.{f.__qualname__}",
            fn=f,
            queue=queue,
            max_attempts=max_attempts,
            timeout=timeout,
            priority=priority,
            dedupe=dedupe,
        )
        if spec.name in _TASKS and _TASKS[spec.name].fn is not f:
            raise ValueError(f"task name {spec.name!r} is already registered")
        _TASKS[spec.name] = spec
        f.tidepool_task = spec  # type: ignore[attr-defined]
        return f

    return register(fn) if fn is not None else register


def get_task(name: str) -> TaskSpec:
    try:
        return _TASKS[name]
    except KeyError:
        raise TaskNotFound(name) from None


def spec_for(fn_or_name: Callable[..., Any] | str) -> TaskSpec:
    if isinstance(fn_or_name, str):
        return get_task(fn_or_name)
    spec = getattr(fn_or_name, "tidepool_task", None)
    if spec is None:
        raise TaskNotFound(getattr(fn_or_name, "__qualname__", repr(fn_or_name)))
    return spec


def registered() -> list[TaskSpec]:
    return sorted(_TASKS.values(), key=lambda s: s.name)


def clear() -> None:
    """For tests."""
    _TASKS.clear()
