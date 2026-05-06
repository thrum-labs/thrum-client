"""Cursor plan-boundary detector — FR-215c (Cursor parity with Claude / Codex).

Cursor exposes plans via two tool_use blocks in the transcript JSONL:

1. **`CreatePlan`** — opens a plan. Input shape:
       {name, overview, plan, todos: [{id, content, dependencies?}, ...]}
   Free-text fields (`name`, `overview`, `plan`, `todos[].content`) are
   NEVER bound (NFR-318/319) — only the seed todo `id` strings are kept.

2. **`TodoWrite`** — updates per-todo statuses. Input shape:
       {merge: bool, todos: [{id, status}, ...]}
   `merge: true` patches (only listed ids update). `merge: false` replaces
   the whole list — when the new id set is disjoint from the previous,
   treat as a fresh plan; when overlapping, treat as in-place edit.
   (`merge: false` semantics are an inferred best-effort — empirical
   Cursor traces in the spike used `merge: true` exclusively. The
   disjoint-replacement case has not been observed in the wild.)

Per FR-215c shape — same contract as Claude (`plan_detector.py`) and
Codex (`codex_plan_detector.py`):

* `plan_id` is deterministic. **Unlike Claude/Codex which seed off
  timestamps, Cursor's transcript JSONL carries no per-line timestamp**
  — so we seed off the conversation_id + the plan's 0-indexed position
  in transcript order: `uuidv5(NS, "<conversation_id>|cursor-plan-<N>")`.
  Stable across re-runs; identical for the live emit + backfill paths.
* A plan **ships** when every seed todo `id` has reached `status:"completed"`,
  AND no full-batch wipe (every id `cancelled`/`deleted`) intervened.
* A plan is **abandoned** when (a) all seed ids reach a terminal
  non-completed status, or (b) a fresh `CreatePlan` opens a new plan
  while the previous one is still incomplete.
* The detector pairs plans to **generations** (0-indexed `role:user →
  role:assistant` boundaries in the transcript). The handler tracks
  generation_index by counting `beforeSubmitPrompt` hook events; the
  detector reports which generation_index each plan touches and which
  one closes it.

Status enum observed empirically: `in_progress`,
`completed`. The spec mentions `cancelled`/`deleted` as terminal-non-
completed values; we treat any status not in {pending, in_progress,
completed} as terminal-non-completed for abandonment-rule (a). This
deliberately fail-closed: an unknown status → treated as terminal →
the plan can abandon, never ship. Better than mis-shipping.
"""

from __future__ import annotations

import io
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import ijson

from thrum_client.parsers.plan_detector import plan_id_for


_CREATE_PLAN = "CreatePlan"
_TODO_WRITE = "TodoWrite"
_PLAN_TOOLS = frozenset({_CREATE_PLAN, _TODO_WRITE})

_STATUS_COMPLETED = "completed"
_STATUS_PENDING = "pending"
_STATUS_IN_PROGRESS = "in_progress"
_OPEN_STATUSES = frozenset({_STATUS_PENDING, _STATUS_IN_PROGRESS})


@dataclass(frozen=True)
class CursorPlanEvent:
    """One CreatePlan or TodoWrite call extracted from the transcript.

    For CreatePlan, `seed_ids` lists the initial todo ids and `updates`
    is empty. For TodoWrite, `seed_ids` is empty and `updates` is the
    `[{id, status}]` list (or list of mappings preserving order).

    `generation_index` is the 0-indexed count of `role:user` blocks
    that preceded this event in the transcript — the handler pairs it
    to its own generation_id list (one per beforeSubmitPrompt).
    """

    name: str  # "CreatePlan" | "TodoWrite"
    generation_index: int
    seed_ids: tuple[str, ...] = ()
    updates: tuple[tuple[str, str], ...] = ()  # ((id, status), ...)
    merge: bool | None = None  # only set on TodoWrite


@dataclass
class _CursorPlan:
    plan_id: uuid.UUID
    position: int  # 0-indexed across all CreatePlans in the conversation
    seed_ids: list[str] = field(default_factory=list)
    statuses: dict[str, str] = field(default_factory=dict)  # id → latest status
    open_generation_index: int = 0
    closing_generation_index: int | None = None
    touched_generations: list[int] = field(default_factory=list)
    abandoned: bool = False
    # Why this plan ended in its current state — recorded for future
    # observability surfacing (debug logs, telemetry). Set when
    # _close_active fires; the empty default applies to plans that
    # ship cleanly through TodoWrite all-completed.
    abandoned_reason: str | None = None

    def all_completed(self) -> bool:
        return bool(self.seed_ids) and all(
            self.statuses.get(t) == _STATUS_COMPLETED for t in self.seed_ids
        )

    def all_terminal_non_completed(self) -> bool:
        """Every seed id has reached a terminal status that is NOT
        `completed` — fail-closed treatment of unknown enums."""
        if not self.seed_ids:
            return False
        for t in self.seed_ids:
            s = self.statuses.get(t)
            if s is None or s in _OPEN_STATUSES or s == _STATUS_COMPLETED:
                return False
        return True


@dataclass(frozen=True)
class CursorPlanAttribution:
    """One generation's plan attribution, mirror of Claude's
    `PlanAttribution` and Codex's `CodexPlanAttribution`."""

    plan_id: uuid.UUID
    completed_at: str | None


def _parse_line(raw: bytes) -> tuple[str | None, list[CursorPlanEvent]]:
    """Single-pass scan of one transcript JSONL line. Returns (role,
    events).

    Allowlist (everything else discarded by the parser):
      role
      message.content.item.type
      message.content.item.name
      message.content.item.input.merge
      message.content.item.input.todos.item.id
      message.content.item.input.todos.item.status

    Free-text never bound: message.content.item.input.{name,overview,plan},
    message.content.item.input.todos.item.content. ijson observes those
    scalars but the prefixes aren't in the captured set, so the values
    are never assigned to a Python variable in this module.

    `generation_index` is filled in by the caller (it depends on prior
    lines, not the current one).
    """
    role: str | None = None
    events: list[CursorPlanEvent] = []

    cur_type: str | None = None
    cur_name: str | None = None
    cur_seed_ids: list[str] = []
    cur_updates: list[tuple[str, str]] = []
    cur_merge: bool | None = None
    cur_todo_id: str | None = None
    cur_todo_status: str | None = None

    for prefix, event, value in ijson.parse(io.BytesIO(raw)):
        if prefix == "role" and event == "string":
            role = value
            continue
        if prefix == "message.content.item" and event == "start_map":
            cur_type = None
            cur_name = None
            cur_seed_ids = []
            cur_updates = []
            cur_merge = None
            continue
        if prefix == "message.content.item.type" and event == "string":
            cur_type = value
            continue
        if prefix == "message.content.item.name" and event == "string":
            cur_name = value
            continue
        if prefix == "message.content.item.input.merge" and event == "boolean":
            cur_merge = value
            continue
        if (
            prefix == "message.content.item.input.todos.item"
            and event == "start_map"
        ):
            cur_todo_id = None
            cur_todo_status = None
            continue
        if (
            prefix == "message.content.item.input.todos.item.id"
            and event == "string"
        ):
            cur_todo_id = value
            continue
        if (
            prefix == "message.content.item.input.todos.item.status"
            and event == "string"
        ):
            cur_todo_status = value
            continue
        if (
            prefix == "message.content.item.input.todos.item"
            and event == "end_map"
        ):
            if cur_todo_id is None:
                continue
            if cur_name == _CREATE_PLAN:
                cur_seed_ids.append(cur_todo_id)
            elif cur_name == _TODO_WRITE and cur_todo_status is not None:
                cur_updates.append((cur_todo_id, cur_todo_status))
            continue
        if prefix == "message.content.item" and event == "end_map":
            if cur_type != "tool_use" or cur_name not in _PLAN_TOOLS:
                continue
            if cur_name == _CREATE_PLAN:
                events.append(
                    CursorPlanEvent(
                        name=_CREATE_PLAN,
                        generation_index=-1,  # filled in by caller
                        seed_ids=tuple(cur_seed_ids),
                    )
                )
            else:  # TodoWrite
                events.append(
                    CursorPlanEvent(
                        name=_TODO_WRITE,
                        generation_index=-1,
                        updates=tuple(cur_updates),
                        merge=cur_merge,
                    )
                )

    return role, events


def iter_plan_events(
    transcript_path: Path,
) -> Iterable[CursorPlanEvent]:
    """Stream plan events with their generation_index attached.

    Generation index = 0-indexed count of `role:user` lines seen so far.
    Events on assistant lines are tagged with the index of the most
    recent user line.
    """
    current_generation: int = -1  # -1 until the first user line opens generation 0

    with transcript_path.open("rb") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                role, events = _parse_line(raw)
            except (ijson.JSONError, ijson.IncompleteJSONError):
                continue
            if role == "user":
                current_generation += 1
                continue
            if role != "assistant" or current_generation < 0:
                continue
            for ev in events:
                # Frozen dataclass — rebuild with the resolved generation_index.
                yield CursorPlanEvent(
                    name=ev.name,
                    generation_index=current_generation,
                    seed_ids=ev.seed_ids,
                    updates=ev.updates,
                    merge=ev.merge,
                )


def _replay_state_machine(
    events: list[CursorPlanEvent], conversation_id: str
) -> list[_CursorPlan]:
    """Run the plan state machine over the event list (transcript order).

    Returns every plan ever opened, including abandoned ones. The
    handler picks the relevant attribution per-generation via
    `detect_plan_for_generation`.
    """
    plans: list[_CursorPlan] = []
    current: _CursorPlan | None = None
    plan_position = 0

    def _close_active(reason: str | None = None) -> None:
        nonlocal current
        if current is None:
            return
        if not current.all_completed() and current.closing_generation_index is None:
            current.abandoned = True
            current.abandoned_reason = reason
        current = None

    for ev in events:
        if ev.name == _CREATE_PLAN:
            # A fresh CreatePlan while a previous plan is still open
            # abandons the previous (FR-215c case b).
            _close_active(reason="fresh_create_plan")

            current = _CursorPlan(
                plan_id=plan_id_for(
                    conversation_id, f"cursor-plan-{plan_position}"
                ),
                position=plan_position,
                seed_ids=list(ev.seed_ids),
                statuses={tid: _STATUS_PENDING for tid in ev.seed_ids},
                open_generation_index=ev.generation_index,
                touched_generations=[ev.generation_index],
            )
            plans.append(current)
            plan_position += 1
            continue

        if ev.name != _TODO_WRITE or current is None:
            # TodoWrite without an open plan is malformed — Cursor
            # generally won't emit it, but tolerate gracefully.
            continue

        # Track which generation touched the plan.
        if (
            not current.touched_generations
            or current.touched_generations[-1] != ev.generation_index
        ):
            current.touched_generations.append(ev.generation_index)

        # Apply per-todo updates.
        if ev.merge is False:
            # `merge: false` semantics: empirical Cursor traces use
            # merge:true exclusively. The behaviour here is INFERRED —
            # verify against real Cursor traces before treating as
            # load-bearing:
            #
            #  * Disjoint (new ids share no overlap with current seed) →
            #    fresh plan. Old plan abandoned with reason
            #    `merge_false_disjoint`.
            #  * Overlapping (any shared ids) → IN-PLACE EDIT. Critically:
            #    we DO NOT shrink `seed_ids`. Earlier code did
            #    `current.seed_ids = list(new_ids)`, which dropped any
            #    seed id NOT in this update batch — review caught the
            #    bug: a TodoWrite{merge:false, todos:[2-of-5]} would
            #    leave only those 2 in the seed and `all_completed`
            #    would prematurely ship the plan when those 2 reached
            #    completed, even though the original 5-todo seed wasn't
            #    actually finished. New rule: ADD any new ids, UPDATE
            #    statuses for everything in the batch, NEVER REMOVE.
            new_ids = [tid for tid, _ in ev.updates]
            if new_ids and not (set(new_ids) & set(current.seed_ids)):
                _close_active(reason="merge_false_disjoint")
                current = _CursorPlan(
                    plan_id=plan_id_for(
                        conversation_id, f"cursor-plan-{plan_position}"
                    ),
                    position=plan_position,
                    seed_ids=list(new_ids),
                    statuses={tid: status for tid, status in ev.updates},
                    open_generation_index=ev.generation_index,
                    touched_generations=[ev.generation_index],
                )
                plans.append(current)
                plan_position += 1
                continue
            # Overlapping or partial-overlap path: extend seed with any
            # new ids; patch statuses for everything in the batch.
            for tid, _ in ev.updates:
                if tid not in current.seed_ids:
                    current.seed_ids.append(tid)
                    current.statuses.setdefault(tid, _STATUS_PENDING)
            for tid, status in ev.updates:
                current.statuses[tid] = status
        else:
            # merge: True (default) — patch in updates.
            for tid, status in ev.updates:
                current.statuses[tid] = status

        if current.closing_generation_index is None and current.all_completed():
            current.closing_generation_index = ev.generation_index
            current = None
        elif current is not None and current.all_terminal_non_completed():
            # Abandonment case (a) — every seed id terminal-non-completed.
            current.abandoned = True
            current = None

    return plans


def iter_plans(
    transcript_path: Path, conversation_id: str
) -> list[_CursorPlan]:
    """Replay the transcript once, return every plan ever opened."""
    events = list(iter_plan_events(transcript_path))
    return _replay_state_machine(events, conversation_id)


def detect_plan_for_generation(
    transcript_path: Path,
    conversation_id: str,
    generation_index: int,
    generation_ts: str | None = None,
) -> CursorPlanAttribution | None:
    """Return the plan attribution for one generation, or None.

    `completed_at` is set ONLY when `generation_index` is the plan's
    closing generation AND `generation_ts` is provided (live-emit path:
    handler passes the hook's stop timestamp). Backfill callers that
    pass `generation_ts=None` get `completed_at=None` even on the
    closing generation — backfill plan attribution is participation-only,
    consistent with how the FR-215c semantics treat re-attribution.

    Returns None if `generation_index` doesn't touch any plan.
    """
    if generation_index < 0:
        return None
    plans = iter_plans(transcript_path, conversation_id)
    for plan in plans:
        if generation_index not in plan.touched_generations:
            continue
        completed_at: str | None = None
        if (
            plan.closing_generation_index == generation_index
            and generation_ts is not None
        ):
            completed_at = generation_ts
        return CursorPlanAttribution(
            plan_id=plan.plan_id, completed_at=completed_at
        )
    return None


def detect_all_plans(
    transcript_path: Path, conversation_id: str
) -> Iterable[_CursorPlan]:
    """Backfill helper — yield every plan reconstructed from a transcript."""
    yield from iter_plans(transcript_path, conversation_id)
