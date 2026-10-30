"""A small admin API (WSGI), for dashboards and runbooks.

Routes:

* ``GET  /healthz``                       no auth, returns ``ok``
* ``GET  /queues``                        counts per queue and state
* ``GET  /jobs/<id>``                     one job, without its payload
* ``GET  /dlq/<queue>?limit=50``          dead jobs in a queue
* ``POST /dlq/<queue>/requeue``           body ``{"ids": [...]}`` or ``{"all": true}``
* ``POST /jobs/<id>/cancel``              cancel a queued job

Every route except ``/healthz`` needs ``Authorization: Bearer <admin_token>``. With no
``admin_token`` configured the API refuses to start, so it is never open by accident.
Run it with any WSGI server: ``gunicorn 'tidepool.http_api:make_app()'``.
"""

from __future__ import annotations

import hmac
import json
from collections.abc import Callable, Iterable
from typing import Any
from urllib.parse import parse_qs

from tidepool import deadletter
from tidepool.config import ConfigError
from tidepool.job import Job
from tidepool.queue import Queue

StartResponse = Callable[[str, list[tuple[str, str]]], Any]
Environ = dict[str, Any]


def job_json(job: Job) -> dict[str, Any]:
    return {
        "id": job.id,
        "task": job.task,
        "queue": job.queue,
        "state": job.state.value,
        "priority": job.priority,
        "attempts": job.attempts,
        "max_attempts": job.max_attempts,
        "run_at": job.run_at.isoformat() if job.run_at else None,
        "last_error": job.last_error,
    }


class AdminApp:
    def __init__(self, queue: Queue) -> None:
        if not queue.settings.admin_token:
            raise ConfigError("set admin_token (TIDEPOOL_ADMIN_TOKEN) to run the admin API")
        self.queue = queue
        self.token = queue.settings.admin_token

    def __call__(self, environ: Environ, start_response: StartResponse) -> Iterable[bytes]:
        method = environ["REQUEST_METHOD"]
        path = environ.get("PATH_INFO", "/").rstrip("/") or "/"
        if path == "/healthz":
            return self._send(start_response, "200 OK", "ok")
        if not self._authorized(environ):
            return self._send(start_response, "401 Unauthorized", {"error": "unauthorized"})
        parts = path.strip("/").split("/")
        try:
            if method == "GET" and parts == ["queues"]:
                return self._send(start_response, "200 OK", self.queue.stats())
            if method == "GET" and len(parts) == 2 and parts[0] == "jobs":
                job = self.queue.storage.get(parts[1])
                if job is None:
                    return self._send(start_response, "404 Not Found", {"error": "no such job"})
                return self._send(start_response, "200 OK", job_json(job))
            if method == "POST" and len(parts) == 3 and parts[0] == "jobs" and parts[2] == "cancel":
                ok = self.queue.cancel(parts[1])
                return self._send(start_response, "200 OK" if ok else "409 Conflict", {"cancelled": ok})
            if method == "GET" and len(parts) == 2 and parts[0] == "dlq":
                qs = parse_qs(environ.get("QUERY_STRING", ""))
                limit = min(int(qs.get("limit", ["50"])[0]), 500)
                jobs = deadletter.list_dead(self.queue, parts[1], limit)
                return self._send(start_response, "200 OK", [job_json(j) for j in jobs])
            if method == "POST" and len(parts) == 3 and parts[0] == "dlq" and parts[2] == "requeue":
                body = self._json_body(environ)
                if body.get("all"):
                    done = deadletter.requeue_all(self.queue, parts[1])
                else:
                    done = deadletter.requeue(self.queue, [str(i) for i in body.get("ids", [])])
                return self._send(start_response, "200 OK", {"requeued": done})
        except (ValueError, KeyError) as e:
            return self._send(start_response, "400 Bad Request", {"error": str(e)})
        return self._send(start_response, "404 Not Found", {"error": "no such route"})

    def _authorized(self, environ: Environ) -> bool:
        header = environ.get("HTTP_AUTHORIZATION", "")
        scheme, _, token = header.partition(" ")
        return scheme.lower() == "bearer" and hmac.compare_digest(token.strip(), self.token or "")

    @staticmethod
    def _json_body(environ: Environ) -> dict[str, Any]:
        length = int(environ.get("CONTENT_LENGTH") or 0)
        if length == 0:
            return {}
        if length > 1_000_000:
            raise ValueError("body too large")
        data = json.loads(environ["wsgi.input"].read(length))
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object")
        return data

    @staticmethod
    def _send(start_response: StartResponse, status: str, body: Any) -> list[bytes]:
        if isinstance(body, str):
            data, ctype = body.encode(), "text/plain"
        else:
            data, ctype = json.dumps(body).encode(), "application/json"
        start_response(status, [("Content-Type", ctype), ("Content-Length", str(len(data)))])
        return [data]


def make_app(**overrides: Any) -> AdminApp:
    return AdminApp(Queue.from_settings(**overrides))
