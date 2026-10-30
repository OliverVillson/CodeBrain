"""Counters and histograms, exposed in Prometheus text format on ``metrics_port``.

Metric names:

* ``tidepool_jobs_enqueued_total{queue}``
* ``tidepool_enqueue_deduplicated_total{queue}``
* ``tidepool_jobs_total{queue,outcome}`` with outcome succeeded, retried or dead
* ``tidepool_leases_reaped_total{queue}``
* ``tidepool_dlq_purged_total``
* ``tidepool_job_duration_seconds{queue,task}`` (histogram)
* ``tidepool_queue_wait_seconds{queue}`` (histogram)
* ``tidepool_rate_limited_seconds_total{queue}``
"""

from __future__ import annotations

import bisect
import threading
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BUCKETS = (0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 15.0, 60.0, 300.0, 900.0)

Labels = tuple[tuple[str, str], ...]


def _labels(kw: dict[str, str]) -> Labels:
    return tuple(sorted((k, str(v)) for k, v in kw.items()))


class Registry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.counters: dict[str, dict[Labels, float]] = defaultdict(lambda: defaultdict(float))
        self.hists: dict[str, dict[Labels, list[float]]] = defaultdict(dict)

    def inc(self, name: str, value: float = 1.0, **labels: str) -> None:
        with self._lock:
            self.counters[name][_labels(labels)] += value

    def observe(self, name: str, value: float, **labels: str) -> None:
        key = _labels(labels)
        with self._lock:
            h = self.hists[name].setdefault(key, [0.0] * (len(BUCKETS) + 2))
            h[bisect.bisect_left(BUCKETS, value)] += 1
            h[-1] += value

    def render(self) -> str:
        lines: list[str] = []
        with self._lock:
            for name, series in sorted(self.counters.items()):
                lines.append(f"# TYPE {name} counter")
                for labels, v in sorted(series.items()):
                    lines.append(f"{name}{_fmt(labels)} {v:g}")
            for name, series in sorted(self.hists.items()):
                lines.append(f"# TYPE {name} histogram")
                for labels, h in sorted(series.items()):
                    cum = 0.0
                    for le, n in zip((*BUCKETS, float("inf")), h[:-1], strict=True):
                        cum += n
                        le_s = "+Inf" if le == float("inf") else f"{le:g}"
                        lines.append(f"{name}_bucket{_fmt((*labels, ('le', le_s)))} {cum:g}")
                    lines.append(f"{name}_count{_fmt(labels)} {cum:g}")
                    lines.append(f"{name}_sum{_fmt(labels)} {h[-1]:g}")
        return "\n".join(lines) + "\n"


def _fmt(labels: Labels) -> str:
    if not labels:
        return ""
    return "{" + ",".join(f'{k}="{v}"' for k, v in labels) + "}"


METRICS = Registry()


def serve(port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/metrics":
                self.send_error(404)
                return
            body = METRICS.render().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True, name="metrics").start()
    return server
