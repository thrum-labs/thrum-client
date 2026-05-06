"""Plan-boundary detector — FR-215c.

Reads Claude Code transcript JSONL, groups turns into plans, and
returns one `PlanAttribution` per turn that touches a plan:

  * a plan starts at the first `TaskCreate` of a contiguous batch
    (batch = 1+ `TaskCreate` calls before any `TaskUpdate` fires);
  * a plan ships when every `taskId` created in the batch has
    received a `TaskUpdate { status: "completed" }`, with no full
    wipe (`deleted` for every taskId) in between;
  * a plan is abandoned when (a) every taskId received `deleted`
    before any was `completed`, or (b) a fresh `TaskCreate` batch
    opens a new plan while the previous one is still incomplete —
    the previous plan's `abandoned_at` is the new batch's first
    `TaskCreate` timestamp.

`plan_id` is deterministic: `uuidv5(_PLAN_NAMESPACE,
"<session_id>|<first_taskcreate_ts>")` so retries (transcript pass
runs again on the next Stop) produce the same identifier without
backend coordination.

Privacy posture (NFR-318/319): only `tool_use.name`,
`tool_use.input.taskId` (short string), and `tool_use.input.status`
(enum `in_progress|completed|deleted`) are bound. `subject`,
`description`, `activeForm` are free text — ijson observes the
events but the prefix is not in the captured set, so the value is
never assigned to a variable in this module.
"""

from __future__ import annotations

import io
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import ijson


# Stable namespace for plan_id minting. Frozen — never regenerate; doing
# so would silently break dedupe across older client versions.
_PLAN_NAMESPACE = uuid.UUID("3f1c3a7e-7b1f-5b2d-8b6e-7a7268752d31")


_TASK_CREATE = "TaskCreate"
_TASK_UPDATE = "TaskUpdate"
_TASK_TOOLS = frozenset({_TASK_CREATE, _TASK_UPDATE})

_STATUS_COMPLETED = "completed"
_STATUS_DELETED = "deleted"
_STATUS_IN_PROGRESS = "in_progress"


@dataclass(frozen=True)
class ToolUseEvent:
    timestamp: str
    name: str  # "TaskCreate" | "TaskUpdate"
    task_id: str | None  # only set on TaskUpdate; TaskCreate auto-numbers in order
    status: str | None  # only set on TaskUpdate


@dataclass
class _Plan:
    plan_id: uuid.UUID
    session_id: str
    first_taskcreate_ts: str
    task_ids: list[str] = field(default_factory=list)  # auto-numbered in TaskCreate order
    completed: set[str] = field(default_factory=set)
    deleted: set[str] = field(default_factory=set)
    completed_at: str | None = None
    abandoned_at: str | None = None
    # Timestamps of every event (TaskCreate or TaskUpdate) attributed to
    # this plan — used to map plans onto turn windows.
    event_timestamps: list[str] = field(default_factory=list)

    def all_completed(self) -> bool:
        return bool(self.task_ids) and all(t in self.completed for t in self.task_ids)

    def all_deleted_before_completion(self) -> bool:
        return (
            not self.completed
            and bool(self.task_ids)
            and all(t in self.deleted for t in self.task_ids)
        )


@dataclass(frozen=True)
class PlanAttribution:
    """Attribution for one assistant turn that touched a plan.

    `completed_at` is non-null only on the turn whose window contains
    the plan's final `TaskUpdate{completed}` event.
    """

    plan_id: uuid.UUID
    completed_at: str | None


def plan_id_for(session_id: str, first_taskcreate_ts: str) -> uuid.UUID:
    """Public — exposed for tests / future cross-process reconstruction."""
    return uuid.uuid5(_PLAN_NAMESPACE, f"{session_id}|{first_taskcreate_ts}")


def _parse_line_tool_uses(raw: bytes) -> Iterable[ToolUseEvent]:
    """Yield TaskCreate/TaskUpdate events from one JSONL line.

    ijson streams the line; only the four bounded paths
    (`message.content.item.type`, `.name`, `.input.taskId`,
    `.input.status`) are bound. `subject`, `description`, and
    `activeForm` flow through ijson and are discarded — never bound.
    """
    timestamp: str | None = None
    cur_type: str | None = None
    cur_name: str | None = None
    cur_task_id: str | None = None
    cur_status: str | None = None
    for prefix, event, value in ijson.parse(io.BytesIO(raw)):
        if prefix == "timestamp" and event == "string":
            timestamp = value
        elif prefix == "message.content.item" and event == "start_map":
            cur_type = None
            cur_name = None
            cur_task_id = None
            cur_status = None
        elif prefix == "message.content.item.type" and event == "string":
            cur_type = value
        elif prefix == "message.content.item.name" and event == "string":
            cur_name = value
        elif prefix == "message.content.item.input.taskId" and event == "string":
            cur_task_id = value
        elif prefix == "message.content.item.input.status" and event == "string":
            cur_status = value
        elif prefix == "message.content.item" and event == "end_map":
            if (
                cur_type == "tool_use"
                and cur_name in _TASK_TOOLS
                and timestamp is not None
            ):
                yield ToolUseEvent(
                    timestamp=timestamp,
                    name=cur_name,
                    task_id=cur_task_id,
                    status=cur_status,
                )


def iter_tool_use_events(path: Path) -> Iterable[ToolUseEvent]:
    """Stream TaskCreate/TaskUpdate events across a transcript file."""
    with path.open("rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield from _parse_line_tool_uses(line)
            except (ijson.JSONError, ijson.IncompleteJSONError):
                continue


def _replay_state_machine(
    events: list[ToolUseEvent], session_id: str
) -> list[_Plan]:
    """Run the plan state machine over an event list (already in
    transcript order). Returns every plan ever opened, including
    abandoned ones."""
    plans: list[_Plan] = []
    current: _Plan | None = None
    saw_update_in_current = False
    pending_taskcreate_count = 0  # auto-numbered taskIds for the active batch

    def _close_active(reason_ts: str | None) -> None:
        nonlocal current, saw_update_in_current, pending_taskcreate_count
        if current is None:
            return
        if (
            not current.all_completed()
            and reason_ts is not None
            and current.abandoned_at is None
        ):
            current.abandoned_at = reason_ts
        current = None
        saw_update_in_current = False
        pending_taskcreate_count = 0

    for ev in events:
        if ev.name == _TASK_CREATE:
            # A TaskCreate after any TaskUpdate marks the start of a new
            # batch. The previous plan is abandoned at this event's ts
            # if not yet shipped.
            if current is not None and saw_update_in_current:
                _close_active(ev.timestamp)
            if current is None:
                current = _Plan(
                    plan_id=plan_id_for(session_id, ev.timestamp),
                    session_id=session_id,
                    first_taskcreate_ts=ev.timestamp,
                )
                plans.append(current)
                pending_taskcreate_count = 0
                saw_update_in_current = False
            pending_taskcreate_count += 1
            current.task_ids.append(str(pending_taskcreate_count))
            current.event_timestamps.append(ev.timestamp)
            continue

        if ev.name == _TASK_UPDATE:
            if current is None:
                # Orphan TaskUpdate (transcript captured mid-flight).
                # Ignore — no plan to attribute to.
                continue
            saw_update_in_current = True
            current.event_timestamps.append(ev.timestamp)
            tid = ev.task_id
            if tid is None:
                continue
            if ev.status == _STATUS_COMPLETED:
                current.completed.add(tid)
                if current.all_completed() and current.completed_at is None:
                    current.completed_at = ev.timestamp
            elif ev.status == _STATUS_DELETED:
                current.deleted.add(tid)
                if current.all_deleted_before_completion():
                    current.abandoned_at = ev.timestamp
                    _close_active(None)  # already abandoned, just clear pointer
            elif ev.status == _STATUS_IN_PROGRESS:
                # bookkeeping — no state change
                pass

    return plans


def _ts_in_window(
    ts: str, turn_start_ts: str | None, turn_end_ts: str | None
) -> bool:
    if turn_start_ts is not None and ts < turn_start_ts:
        return False
    if turn_end_ts is not None and ts > turn_end_ts:
        return False
    return True


def detect_plan_for_turn(
    transcript_path: Path,
    session_id: str,
    turn_start_ts: str | None,
    turn_end_ts: str | None,
) -> PlanAttribution | None:
    """Run the full-session detector and return the attribution for the
    turn whose window is `[turn_start_ts, turn_end_ts]`.

    A turn is attributed to a plan iff at least one of the plan's
    `TaskCreate` / `TaskUpdate` events lands inside the turn's window.
    `completed_at` is set only on the turn whose window contains the
    plan's `completed_at` timestamp (the final completing event).

    If a turn touches more than one plan (rare — e.g. final completed
    of plan A and first TaskCreate of plan B share the same turn), the
    first plan whose events fall in the window wins. The activity row
    for plan B will still get its plan_id on the *next* turn that
    touches it.
    """
    events = list(iter_tool_use_events(transcript_path))
    if not events:
        return None
    plans = _replay_state_machine(events, session_id)
    if not plans:
        return None

    for plan in plans:
        if any(
            _ts_in_window(ts, turn_start_ts, turn_end_ts)
            for ts in plan.event_timestamps
        ):
            completed_at = (
                plan.completed_at
                if (
                    plan.completed_at is not None
                    and _ts_in_window(plan.completed_at, turn_start_ts, turn_end_ts)
                )
                else None
            )
            return PlanAttribution(
                plan_id=plan.plan_id, completed_at=completed_at
            )
    return None


def detect_all_plans(
    transcript_path: Path, session_id: str
) -> list[_Plan]:
    """Backfill helper — return every plan reconstructed from a
    transcript, including abandoned ones. Backfill (FR-217) walks
    historical transcripts and assigns plan_id per turn the same way
    a live Stop hook would."""
    events = list(iter_tool_use_events(transcript_path))
    return _replay_state_machine(events, session_id)
