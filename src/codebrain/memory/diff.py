"""Unified diff parsing: which lines a finished task changed, and what it added.

Write-back uses this to link an episode to the code units in its diff and to tell
whether a recalled unit was called, imported or edited (sketch §8).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


@dataclass
class FileChange:
    """One file in a diff. ``lines`` are 1-based line numbers in the file as the diff leaves it,
    or in the old file when the file was deleted."""

    path: str
    lines: set[int] = field(default_factory=set)
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    deleted: bool = False

    def ranges(self) -> list[tuple[int, int]]:
        """Changed lines as sorted, merged inclusive ranges."""
        out: list[tuple[int, int]] = []
        for n in sorted(self.lines):
            if out and n == out[-1][1] + 1:
                out[-1] = (out[-1][0], n)
            else:
                out.append((n, n))
        return out


def _strip_prefix(path: str) -> str:
    path = path.split("\t", 1)[0]
    return path[2:] if path[:2] in ("a/", "b/") else path


def _is_file_header(lines: list[str], i: int) -> bool:
    """``--- old`` + ``+++ new`` + ``@@``: a new file even if the hunk counts said otherwise."""
    return (
        lines[i].startswith("--- ")
        and i + 2 < len(lines)
        and lines[i + 1].startswith("+++ ")
        and lines[i + 2].startswith("@@")
    )


def parse_diff(diff: str) -> list[FileChange]:
    files: list[FileChange] = []
    current: FileChange | None = None
    old_path: str | None = None
    old_line = new_line = 0
    # Lines still owed to the current hunk. Inside a hunk, ``--- x`` and ``+++ x`` are a
    # removed ``-- x`` and an added ``++ x``, not file headers.
    old_left = new_left = 0
    lines = diff.splitlines()
    for i, raw in enumerate(lines):
        in_hunk = current is not None and (old_left > 0 or new_left > 0)
        if in_hunk and (raw.startswith(("diff ", "@@")) or _is_file_header(lines, i)):
            # Hand-written diffs often get the counts wrong; a header ends the hunk anyway.
            in_hunk = False
            old_left = new_left = 0
        if in_hunk and current is not None:
            if raw.startswith("\\"):
                continue  # "\ No newline at end of file"
            if raw.startswith("+"):
                current.added.append(raw[1:])
                current.lines.add(new_line)
                new_line += 1
                new_left -= 1
            elif raw.startswith("-"):
                current.removed.append(raw[1:])
                # A deletion is recorded where it happened: the old line for a deleted file,
                # otherwise the surrounding line in the new file.
                current.lines.add(old_line if current.deleted else max(new_line, 1))
                old_line += 1
                old_left -= 1
            elif raw.startswith(" ") or raw == "":
                # Some tools strip the space from blank context lines.
                old_line += 1
                new_line += 1
                old_left -= 1
                new_left -= 1
            else:
                old_left = new_left = 0
            continue
        if raw.startswith("--- "):
            old_path = _strip_prefix(raw[4:])
            continue
        if raw.startswith("+++ "):
            new_path = _strip_prefix(raw[4:])
            deleted = new_path == "/dev/null"
            path = old_path if deleted and old_path else new_path
            current = FileChange(path=path, deleted=deleted)
            files.append(current)
            continue
        if current is None:
            continue
        m = _HUNK.match(raw)
        if m:
            old_line, new_line = int(m.group(1)), int(m.group(3))
            old_left = int(m.group(2)) if m.group(2) is not None else 1
            new_left = int(m.group(4)) if m.group(4) is not None else 1
    return [f for f in files if f.lines]
