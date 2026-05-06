"""Codex plan-boundary detector — FR-215c (Codex parity with Claude).

Codex CLI exposes two distinct plan signals in the rollout JSONL. Both
project to the same `plan_id` / `completed_at` attribution surface as
Claude's `TaskCreate` / `TaskUpdate` plans:

1. **`update_plan` tool** — a checklist (TODOs) function tool. The agent
   calls `function_call.name == "update_plan"` with `arguments` carrying
   `{"plan": [{"step": "...", "status": "pending"|"in_progress"|"completed"},
   ...], "explanation": "..."}`. We treat the FIRST `update_plan` call as
   the plan's open event and the call whose checklist is `all completed`
   (≥1 step, no `pending` or `in_progress` left) as its ship event.

2. **Plan Mode** (collaboration mode) — `turn_context.payload.collaboration_mode.mode
   == "plan"` opens a planning conversation; the agent ships by emitting a
   `<proposed_plan>...</proposed_plan>` block in an assistant message. Plan
   Mode plans abandon when `collaboration_mode.mode` shifts away from
   `"plan"` without a `<proposed_plan>` ever appearing.

Privacy posture (NFR-318/319, mirrors `parsers/codex_rollout.py`): neither
the `update_plan.arguments` payload nor any `<proposed_plan>` body is bound
to a Python variable. The arguments JSON-string scalar is observed by ijson,
status counts are derived via regex inside the parse loop, and only the
counts (three integers) are kept. Same classify-and-drop pattern for
assistant `payload.content.item.text` — we record only the boolean
`saw_proposed_plan` flag for the current message.

`plan_id` is deterministic via `uuidv5(_PLAN_NAMESPACE, "<session_id>|<first_ts>")`
shared with `plan_detector.py` so retries (live Stop hook racing the
rollout writer, then the FR-218b backfill landing later) produce the
same id without backend coordination.

Out-of-scope for v1:
- `request_user_input` (Plan Mode's clarifying-question tool) is NOT a
  plan signal — it's content. We ignore it.
- Per-step status diffs across `update_plan` calls — we look only at the
  most recent call's aggregate status counts.
"""

from __future__ import annotations

import io
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import ijson

from thrum_client.parsers.plan_detector import plan_id_for


# Status counts come from regex over the raw `arguments` string. Codex emits
# the canonical statuses verbatim; tolerate insignificant whitespace around
# the colon so a future Codex serializer change ("status" : "completed")
# doesn't silently turn every plan into "abandoned".
_STATUS_RE = re.compile(r'"status"\s*:\s*"(pending|in_progress|completed)"')

# Newline-bounded match for `<proposed_plan>` per the Codex Plan Mode
# finalization rule ("opening tag must be on its own line"). A bare
# substring match would false-positive on the developer instructions
# themselves (which embed `<proposed_plan>` examples inside backticks).
_PROPOSED_PLAN_RE = re.compile(r"(^|\n)\s*<proposed_plan>\s*\n")

_UPDATE_PLAN = "update_plan"

_SCALAR_EVENTS: frozenset[str] = frozenset({"string", "number", "boolean", "null"})


@dataclass(frozen=True)
class CodexPlanAttribution:
    """One turn's plan attribution. Mirror of Claude's `PlanAttribution`
    so the emit path stays uniform across source tools."""

    plan_id: uuid.UUID
    completed_at: str | None


# Internal record of one detected plan. Source distinguishes which signal
# (update_plan vs plan_mode) opened it; future debug/telemetry could surface
# this, but it's not part of the attribution contract.
@dataclass
class _CodexPlan:
    plan_id: uuid.UUID
    source: str  # "update_plan" | "plan_mode"
    first_ts: str
    completed_at: str | None = None
    abandoned_at: str | None = None
    # Every turn_id that touched this plan. A turn touches the plan iff at
    # least one of: an update_plan call within the turn, the turn opened in
    # plan mode, or (for plan_mode) the turn carried the closing
    # proposed_plan message.
    touched_turn_ids: list[str] = field(default_factory=list)


def _add_turn(plan: _CodexPlan, turn_id: str | None) -> None:
    if turn_id and turn_id not in plan.touched_turn_ids:
        plan.touched_turn_ids.append(turn_id)


def _scan_record(raw: bytes) -> dict[str, object]:
    """Single-pass scan of one JSONL line. Returns a small captures dict
    populated only with bounded scalars + derived booleans/counts.

    Bound paths (allowlist):
      type, timestamp,
      payload.id, payload.type, payload.role, payload.name, payload.turn_id,
      payload.collaboration_mode.mode

    Classify-and-drop paths (value observed, only derivative kept):
      payload.arguments               → status counts via _STATUS_RE
      payload.content.item.text       → saw_proposed_plan boolean

    Free-text bodies (`payload.content.item.text` content,
    `payload.arguments` JSON, `instructions`, `cmd`, `last_agent_message`,
    `reasoning.text`, ...) are never bound to a variable.
    """
    cap: dict[str, object] = {}
    # Aggregated derivatives (multi-text messages combine):
    completed_count = 0
    in_progress_count = 0
    pending_count = 0
    saw_proposed_plan = False

    for prefix, event, value in ijson.parse(io.BytesIO(raw)):
        if event not in _SCALAR_EVENTS:
            continue
        if prefix == "payload.arguments":
            if isinstance(value, str) and value:
                for status in _STATUS_RE.findall(value):
                    if status == "completed":
                        completed_count += 1
                    elif status == "in_progress":
                        in_progress_count += 1
                    elif status == "pending":
                        pending_count += 1
            continue
        if prefix == "payload.content.item.text":
            if (
                isinstance(value, str)
                and value
                and _PROPOSED_PLAN_RE.search(value)
            ):
                saw_proposed_plan = True
            continue
        if prefix in (
            "type",
            "timestamp",
            "payload.id",
            "payload.type",
            "payload.role",
            "payload.name",
            "payload.turn_id",
            "payload.collaboration_mode.mode",
        ):
            cap[prefix] = value

    cap["__update_plan_status_counts__"] = (
        completed_count,
        in_progress_count,
        pending_count,
    )
    cap["__saw_proposed_plan__"] = saw_proposed_plan
    return cap


def iter_plans(path: Path, session_id: str) -> list[_CodexPlan]:
    """Walk the rollout once, replay both plan state machines, return every
    plan that was ever opened (including abandoned ones).

    State (per source):
    - `update_plan`:
        * first call → open plan, record first_ts
        * subsequent call: if previous plan is still open, treat the new
          call as a *replacement* — abandon the previous plan at the new
          call's timestamp, open a new plan
        * any call whose status counts are `(C>0, in_progress=0, pending=0)`
          ships the active plan at that call's timestamp
    - `plan_mode`:
        * turn_context with mode="plan" while no plan_mode plan is open →
          open plan, record first_ts
        * assistant message containing `<proposed_plan>` while a plan_mode
          plan is open → ship at that message's timestamp
        * turn_context with mode != "plan" while a plan_mode plan is open
          (and not yet shipped) → abandon at that turn_context's timestamp
    """
    plans: list[_CodexPlan] = []
    active_update_plan: _CodexPlan | None = None
    active_plan_mode: _CodexPlan | None = None
    current_turn_id: str | None = None
    pending_turn_ids: set[str] = set()  # tracks live turns for current_turn_id pop

    with path.open("rb") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                cap = _scan_record(raw)
            except (ijson.JSONError, ijson.IncompleteJSONError):
                continue

            top_type = cap.get("type")
            ts = cap.get("timestamp") if isinstance(cap.get("timestamp"), str) else None
            payload_type = cap.get("payload.type")
            payload_turn = cap.get("payload.turn_id")
            payload_turn_str = (
                payload_turn if isinstance(payload_turn, str) else None
            )

            # Track the most-recently-opened turn so response_item records
            # (which lack turn_id) can be attributed.
            if top_type == "event_msg":
                if payload_type == "task_started" and payload_turn_str:
                    pending_turn_ids.add(payload_turn_str)
                    current_turn_id = payload_turn_str
                elif payload_type == "task_complete" and payload_turn_str:
                    pending_turn_ids.discard(payload_turn_str)
                    if current_turn_id == payload_turn_str:
                        current_turn_id = (
                            next(iter(pending_turn_ids), None)
                            if pending_turn_ids
                            else None
                        )

            # ---- Plan Mode signal: turn_context with mode="plan" ----
            if top_type == "turn_context":
                mode = cap.get("payload.collaboration_mode.mode")
                if mode == "plan":
                    if active_plan_mode is None and ts:
                        active_plan_mode = _CodexPlan(
                            plan_id=plan_id_for(session_id, ts),
                            source="plan_mode",
                            first_ts=ts,
                        )
                        plans.append(active_plan_mode)
                    if active_plan_mode is not None:
                        _add_turn(active_plan_mode, payload_turn_str)
                elif mode is not None:
                    # Mode shifted to an EXPLICIT non-plan value (e.g.
                    # "default") without a proposed_plan being emitted —
                    # abandon the active plan_mode plan.
                    #
                    # We deliberately do NOT abandon when `mode is None`
                    # (i.e. the turn_context omitted collaboration_mode
                    # entirely). codex-cli 0.125.0 always emits the field,
                    # but a future Codex version that sends turn_context
                    # records without it on internal/system turns would
                    # otherwise spuriously abandon every active plan-mode
                    # plan at the next such turn — silently turning every
                    # planned-but-not-yet-shipped plan into an abandoned
                    # one. Absence of signal is not the same as "left plan
                    # mode."
                    if active_plan_mode is not None and ts:
                        active_plan_mode.abandoned_at = ts
                        active_plan_mode = None

            # ---- Plan Mode ship: assistant message with <proposed_plan> ----
            if (
                top_type == "response_item"
                and payload_type == "message"
                and cap.get("payload.role") == "assistant"
                and cap.get("__saw_proposed_plan__")
                and active_plan_mode is not None
                and ts
            ):
                active_plan_mode.completed_at = ts
                _add_turn(active_plan_mode, current_turn_id)
                active_plan_mode = None

            # ---- update_plan tool ----
            #
            # We can't tell "agent revised the plan with different steps"
            # apart from "agent updated step statuses on the same plan"
            # without binding step text (NFR-318/319). So every
            # `update_plan` call while a plan is active is treated as a
            # refresh of that plan; ship is the only state transition that
            # closes it. An update_plan plan that never reaches all-completed
            # stays open until session end (and is reported with
            # `completed_at=None` — attributed but not shipped).
            if (
                top_type == "response_item"
                and payload_type == "function_call"
                and cap.get("payload.name") == _UPDATE_PLAN
                and ts
            ):
                completed, in_progress, pending = cap[
                    "__update_plan_status_counts__"
                ]  # type: ignore[misc]

                if active_update_plan is None:
                    active_update_plan = _CodexPlan(
                        plan_id=plan_id_for(session_id, ts),
                        source=_UPDATE_PLAN,
                        first_ts=ts,
                    )
                    plans.append(active_update_plan)

                _add_turn(active_update_plan, current_turn_id)

                if (
                    active_update_plan.completed_at is None
                    and completed > 0
                    and in_progress == 0
                    and pending == 0
                ):
                    active_update_plan.completed_at = ts
                    active_update_plan = None

    return plans


def detect_plan_for_turn(
    rollout_path: Path,
    session_id: str,
    target_turn_id: str | None,
) -> CodexPlanAttribution | None:
    """Return the plan attribution for the Codex turn `target_turn_id`.

    Walks the full rollout (cheap — Codex sessions are bounded), assembles
    every plan, and picks the first plan whose `touched_turn_ids` contains
    the target. `completed_at` is set only when the plan's ship event lands
    inside this same turn (i.e. it was the closing turn of the plan).

    Returns None if the rollout is empty, no plan was opened, or the target
    turn never touched any plan.
    """
    if target_turn_id is None:
        return None
    plans = iter_plans(rollout_path, session_id)
    if not plans:
        return None
    for plan in plans:
        if target_turn_id not in plan.touched_turn_ids:
            continue
        # `completed_at` belongs only on the turn that shipped the plan —
        # otherwise an earlier touched-but-not-shipping turn would be
        # mis-stamped as the completion event.
        completed_at: str | None = None
        if plan.completed_at and plan.touched_turn_ids:
            if plan.touched_turn_ids[-1] == target_turn_id:
                completed_at = plan.completed_at
        return CodexPlanAttribution(
            plan_id=plan.plan_id, completed_at=completed_at
        )
    return None


def detect_all_plans(rollout_path: Path, session_id: str) -> Iterable[_CodexPlan]:
    """Backfill helper — yield every plan reconstructed from a rollout."""
    yield from iter_plans(rollout_path, session_id)
