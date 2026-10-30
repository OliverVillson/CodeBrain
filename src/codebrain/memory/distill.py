"""A distiller that needs no model, used until an LLM distiller is wired in.

It only rearranges what the record already says. The full text is always kept on the
unit, so a weak summary costs recall quality, never information.
"""

from __future__ import annotations

from collections.abc import Sequence

from codebrain.memory.records import DistilledTask, Outcome, TaskRecord, Turn


def _first_line(text: str, limit: int) -> str:
    line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    return line if len(line) <= limit else line[: limit - 1].rstrip() + "…"


class PlainDistiller:
    def __init__(self, line_limit: int = 160, plan_limit: int = 800) -> None:
        self.line_limit = line_limit
        self.plan_limit = plan_limit

    def distill_task(self, task: TaskRecord) -> DistilledTask:
        goal = _first_line(task.prompt, self.line_limit)
        title = goal if task.outcome is Outcome.SUCCEEDED else f"{goal} ({task.outcome})"
        how = task.plan.strip()[: self.plan_limit] if task.plan else None
        return DistilledTask(title=title, built=goal, how=how, pitfalls=list(task.pitfalls))

    def distill_turns(self, turns: Sequence[Turn]) -> str:
        if not turns:
            return ""
        head = f"Turns {turns[0].index} to {turns[-1].index}"
        lines = [f"{t.role}: {_first_line(t.content, self.line_limit)}" for t in turns]
        return "\n".join([head, *lines])
