# tidepool

A small background job queue for Python services. Jobs are plain functions registered
with `@task`, stored in SQLite (or memory for tests), leased by workers, retried with
backoff, and dead-lettered when they keep failing.

```python
from tidepool import Queue, task

@task(queue="emails", max_attempts=5)
def send_welcome(user_id: int) -> None:
    ...

q = Queue.from_settings()
q.enqueue(send_welcome, 42)
```

Run a worker with `tidepool worker --queue emails`. See `docs/operations.md` for the
admin API, metrics and dead-letter handling.

This codebase is a test fixture for CodeBrain's codebase Q&A benchmark. It is not a
real package and is not published anywhere.
