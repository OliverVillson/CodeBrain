# Operating tidepool

## Settings

Put settings in `tidepool.toml` under `[tidepool]`, or in `TIDEPOOL_*` environment
variables. See `tidepool/config.py` for every field and the order in which sources
override each other.

```toml
[tidepool]
database = "/var/lib/tidepool/jobs.db"
concurrency = 8
admin_token = "change-me"

[tidepool.rate_limits]
emails = "50/s"
reports = "10/m"
```

## Workers

Run one `tidepool worker` per CPU or so. Workers on the same host share the SQLite
file. Send SIGTERM to stop a worker; it finishes what it is running first (within a
grace period).

## Metrics

Each worker serves Prometheus metrics on port 9464 at `/metrics`. Alert on
`tidepool_jobs_total{outcome="dead"}` rising and on `tidepool_queue_wait_seconds`
p95.

## Dead letters

`tidepool dlq list emails` shows dead jobs with their last error. Fix the cause, then
`tidepool dlq requeue emails --all`. Dead jobs that nobody requeues are purged
automatically after the retention period.

## Admin API

`gunicorn 'tidepool.http_api:make_app()'` serves the admin API. It needs
`admin_token`.
