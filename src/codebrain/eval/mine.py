"""Mine retrieval gold tasks from git history (plan §2).

Each kept commit is a past "build X" task. The index is the repo at the commit's parent,
so the answer is never in the index. The relevant set is the existing units the change
called or subclassed (grade 2) or edited (grade 1). Labels are automatic and noisy;
``goldset.jsonl`` is written out so a human can prune it.
"""

from __future__ import annotations

import random
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import IO

from codebrain.eval.units import CodeUnit, is_test_path, parse_units

SKIP_SUBJECT = re.compile(
    r"\b(bump|release|version|typo|revert|merge|changelog|lint|format|style|"
    r"pre-commit|ci|docs?|readme|deps?|dependabot|update\s+copyright)\b",
    re.IGNORECASE,
)
TRAILER = re.compile(r"^[A-Za-z-]+-by:.*$|^Change-Id:.*$", re.MULTILINE)
PR_REF = re.compile(r"\(#\d+\)|#\d+|https?://\S+")


@dataclass
class GoldTask:
    id: str
    repo: str
    sha: str
    parent: str
    query: str
    relevant: dict[str, int]
    new_units: list[str]
    query_paraphrase: str | None = None
    split: str = "dev"


@dataclass
class Snapshot:
    """All indexable units at one commit."""

    sha: str
    units: list[CodeUnit] = field(default_factory=list)


class Git:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._cat: subprocess.Popen[bytes] | None = None
        self._blob_units: dict[tuple[str, str], list[CodeUnit]] = {}

    def run(self, *args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(self.path), *args], check=True, capture_output=True, text=True
        ).stdout

    def blob(self, sha: str) -> str:
        if self._cat is None:
            self._cat = subprocess.Popen(
                ["git", "-C", str(self.path), "cat-file", "--batch"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
            )
        stdin: IO[bytes] = self._cat.stdin  # type: ignore[assignment]
        stdout: IO[bytes] = self._cat.stdout  # type: ignore[assignment]
        stdin.write(sha.encode() + b"\n")
        stdin.flush()
        header = stdout.readline().split()
        if len(header) < 3 or header[1] == b"missing":
            return ""
        data = stdout.read(int(header[2]))
        stdout.read(1)
        return data.decode("utf-8", errors="replace")

    def units_of_blob(self, path: str, blob: str) -> list[CodeUnit]:
        key = (path, blob)
        if key not in self._blob_units:
            self._blob_units[key] = parse_units(path, self.blob(blob))
        return self._blob_units[key]

    def py_blobs(self, sha: str, paths: list[str] | None = None) -> dict[str, str]:
        out = self.run("ls-tree", "-r", sha, "--", *(paths or []))
        blobs: dict[str, str] = {}
        for line in out.splitlines():
            meta, path = line.split("\t", 1)
            _, kind, blob = meta.split()
            if kind == "blob" and path.endswith(".py"):
                blobs[path] = blob
        return blobs

    def close(self) -> None:
        if self._cat is not None:
            self._cat.kill()
            self._cat = None


def clean_message(subject: str, body: str) -> str:
    text = f"{subject}\n\n{body}"
    text = TRAILER.sub("", text)
    text = PR_REF.sub("", text)
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return re.sub(r"\n{3,}", "\n\n", text).strip()[:2000]


def snapshot(git: Git, sha: str, include_tests: bool) -> Snapshot:
    snap = Snapshot(sha)
    for path, blob in git.py_blobs(sha).items():
        if include_tests or not is_test_path(path):
            snap.units.extend(git.units_of_blob(path, blob))
    return snap


def _task_for_commit(
    git: Git,
    repo: str,
    sha: str,
    parent: str,
    subject: str,
    body: str,
    *,
    include_tests: bool,
    max_relevant: int,
    max_ambiguity: int,
) -> tuple[GoldTask, Snapshot] | None:
    if SKIP_SUBJECT.search(subject):
        return None
    query = clean_message(subject, body)
    if len(query.split()) < 4:
        return None

    changed: dict[str, str] = {}  # new path -> old path
    for line in git.run("diff-tree", "-r", "-M", "--name-status", parent, sha).splitlines():
        cols = line.split("\t")
        status = cols[0][0]
        if status in "MA" and cols[-1].endswith(".py"):
            changed[cols[1]] = cols[1]
        elif status == "R" and cols[2].endswith(".py"):
            changed[cols[2]] = cols[1]
    changed = {n: o for n, o in changed.items() if include_tests or not is_test_path(n)}
    if not changed:
        return None

    after_blobs = git.py_blobs(sha, list(changed))
    before_blobs = git.py_blobs(parent, list(set(changed.values())))
    new: list[CodeUnit] = []
    modified: list[CodeUnit] = []
    added_refs: list[tuple[CodeUnit, frozenset[str]]] = []
    modified_keys: set[str] = set()
    for new_path, old_path in changed.items():
        after_blob = after_blobs.get(new_path)
        before_blob = before_blobs.get(old_path)
        after = git.units_of_blob(new_path, after_blob) if after_blob else []
        before_units = git.units_of_blob(old_path, before_blob) if before_blob else []
        before = {u.qualname: u for u in before_units}
        for u in after:
            old = before.get(u.qualname)
            if old is None:
                new.append(u)
                added_refs.append((u, u.refs))
            elif old.ast_hash != u.ast_hash:
                modified.append(u)
                modified_keys.add(old.key)
                # Only what the change started using, not everything the unit already used.
                added_refs.append((u, u.refs - old.refs))
    if not new:
        return None

    snap = snapshot(git, parent, include_tests)
    by_name: dict[str, list[str]] = {}
    for u in snap.units:
        by_name.setdefault(u.name, []).append(u.key)

    live = {u.key for u in snap.units}
    relevant: dict[str, int] = {k: 1 for k in modified_keys if k in live}
    for u, refs in added_refs:
        for ref in refs:
            if len(ref) < 3 or ref.startswith("__"):
                continue
            keys = by_name.get(ref, [])
            if 0 < len(keys) <= max_ambiguity:
                for k in keys:
                    if k != u.key:
                        relevant[k] = 2
    if not 1 <= len(relevant) <= max_relevant:
        return None
    task = GoldTask(
        id=f"{repo}@{sha[:10]}",
        repo=repo,
        sha=sha,
        parent=parent,
        query=query,
        relevant=dict(sorted(relevant.items())),
        new_units=[u.key for u in new],
    )
    return task, snap


def mine_repo(
    path: Path,
    repo: str,
    *,
    max_tasks: int,
    max_scan: int = 2000,
    include_tests: bool = False,
    max_relevant: int = 25,
    max_ambiguity: int = 2,
    seed: int = 0,
) -> list[tuple[GoldTask, Snapshot]]:
    """Scan recent non-merge commits and keep an even random sample of usable ones."""
    git = Git(path)
    try:
        log = git.run(
            "log", "--no-merges", f"-n{max_scan}", "--format=%H%x00%P%x00%s%x00%b%x1e", "HEAD"
        )
        candidates: list[tuple[str, str, str, str]] = []
        for rec in log.split("\x1e"):
            rec = rec.strip("\n")
            if not rec:
                continue
            sha, parents, subject, body = (rec.split("\x00") + ["", "", "", ""])[:4]
            if len(parents.split()) == 1:
                candidates.append((sha, parents, subject, body))
        rng = random.Random(seed)
        rng.shuffle(candidates)
        found: list[tuple[GoldTask, Snapshot]] = []
        for sha, parent, subject, body in candidates:
            got = _task_for_commit(
                git,
                repo,
                sha,
                parent,
                subject,
                body,
                include_tests=include_tests,
                max_relevant=max_relevant,
                max_ambiguity=max_ambiguity,
            )
            if got:
                found.append(got)
                if len(found) >= max_tasks:
                    break
        return found
    finally:
        git.close()


def assign_splits(tasks: list[GoldTask], held_out: float = 0.25, seed: int = 0) -> None:
    rng = random.Random(seed)
    for t in tasks:
        t.split = "test" if rng.random() < held_out else "dev"


def task_to_json(t: GoldTask) -> dict[str, object]:
    return asdict(t)
