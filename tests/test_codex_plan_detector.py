"""Codex plan detector tests — FR-215c (Codex parity).

Synthetic JSONL fixtures for the two Codex plan signals:

  * `update_plan` checklist tool — open at first call, ship when all step
    statuses are `completed` (≥1 step, no pending or in_progress).
  * Plan Mode (`turn_context.collaboration_mode.mode == "plan"`) — open
    at the first plan-mode turn, ship when an assistant message contains
    `<proposed_plan>` (newline-bounded), abandon if mode shifts away.

Also covers determinism (re-running the detector returns the same plan_id),
privacy posture (step text and proposed_plan body never bound), and the
real-world rollout path (cross-checked against the captured 2026-05-01
rollout in the user's home dir).
"""

from __future__ import annotations

import json
from pathlib import Path

from thrum_client.parsers.codex_plan_detector import (
    CodexPlanAttribution,
    detect_plan_for_turn,
    iter_plans,
)
from thrum_client.parsers.plan_detector import plan_id_for


SESSION = "019dda49-bb2c-7de2-9a1a-1e5ca5fb0465"


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def _session_meta(ts: str = "2026-04-29T17:30:29.943Z") -> dict:
    return {
        "timestamp": ts,
        "type": "session_meta",
        "payload": {
            "id": SESSION,
            "model_provider": "openai",
            "originator": "codex-tui",
        },
    }


def _task_started(turn_id: str, ts: str) -> dict:
    return {
        "timestamp": ts,
        "type": "event_msg",
        "payload": {"type": "task_started", "turn_id": turn_id},
    }


def _task_complete(turn_id: str, ts: str) -> dict:
    return {
        "timestamp": ts,
        "type": "event_msg",
        "payload": {"type": "task_complete", "turn_id": turn_id},
    }


def _turn_context(turn_id: str, ts: str, mode: str = "default") -> dict:
    return {
        "timestamp": ts,
        "type": "turn_context",
        "payload": {
            "turn_id": turn_id,
            "model": "gpt-5.5",
            "collaboration_mode": {"mode": mode},
        },
    }


def _turn_context_without_mode(turn_id: str, ts: str) -> dict:
    """A turn_context that omits the `collaboration_mode` field entirely.
    Used to model future-Codex internal/system turn_contexts that may
    not carry mode info."""
    return {
        "timestamp": ts,
        "type": "turn_context",
        "payload": {
            "turn_id": turn_id,
            "model": "gpt-5.5",
        },
    }


def _update_plan_call(ts: str, statuses: list[str], explanation: str = "") -> dict:
    """Build an `update_plan` function_call record. `statuses` is a list of
    canonical status strings — the test never inspects the step text so it
    can be any opaque value."""
    plan = [
        {"step": f"SECRET_STEP_{i}", "status": s} for i, s in enumerate(statuses)
    ]
    args_obj = {"plan": plan, "explanation": explanation or "SECRET_EXPLANATION"}
    return {
        "timestamp": ts,
        "type": "response_item",
        "payload": {
            "type": "function_call",
            "name": "update_plan",
            "arguments": json.dumps(args_obj),
            "call_id": "call_" + ts,
        },
    }


def _assistant_message(ts: str, text: str) -> dict:
    return {
        "timestamp": ts,
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text}],
        },
    }


def _developer_message(ts: str, text: str) -> dict:
    return {
        "timestamp": ts,
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": "developer",
            "content": [{"type": "input_text", "text": text}],
        },
    }


# ---- update_plan ----


def test_update_plan_single_call_all_completed_ships_immediately(tmp_path: Path):
    rollout = tmp_path / "rollout.jsonl"
    ts = "2026-05-01T10:00:00.000Z"
    _write_jsonl(
        rollout,
        [
            _session_meta(),
            _task_started("turn_a", "2026-05-01T09:59:50.000Z"),
            _turn_context("turn_a", "2026-05-01T09:59:50.001Z"),
            _update_plan_call(ts, ["completed", "completed"]),
            _task_complete("turn_a", "2026-05-01T10:00:01.000Z"),
        ],
    )

    plans = iter_plans(rollout, SESSION)
    assert len(plans) == 1
    plan = plans[0]
    assert plan.plan_id == plan_id_for(SESSION, ts)
    assert plan.completed_at == ts
    assert plan.touched_turn_ids == ["turn_a"]


def test_update_plan_in_progress_then_completed_across_two_turns(tmp_path: Path):
    rollout = tmp_path / "rollout.jsonl"
    open_ts = "2026-05-01T10:00:00.000Z"
    ship_ts = "2026-05-01T10:05:00.000Z"
    _write_jsonl(
        rollout,
        [
            _session_meta(),
            # Turn 1 — opens the plan with pending+in_progress steps.
            _task_started("turn_a", "2026-05-01T09:59:50.000Z"),
            _turn_context("turn_a", "2026-05-01T09:59:50.001Z"),
            _update_plan_call(open_ts, ["in_progress", "pending"]),
            _task_complete("turn_a", "2026-05-01T10:00:01.000Z"),
            # Turn 2 — refreshes with all completed → ships.
            _task_started("turn_b", "2026-05-01T10:04:50.000Z"),
            _turn_context("turn_b", "2026-05-01T10:04:50.001Z"),
            _update_plan_call(ship_ts, ["completed", "completed"]),
            _task_complete("turn_b", "2026-05-01T10:05:01.000Z"),
        ],
    )

    plans = iter_plans(rollout, SESSION)
    assert len(plans) == 1
    plan = plans[0]
    expected_id = plan_id_for(SESSION, open_ts)
    assert plan.plan_id == expected_id
    assert plan.completed_at == ship_ts
    assert plan.touched_turn_ids == ["turn_a", "turn_b"]

    # detect_plan_for_turn returns the plan on both turns; completed_at is
    # set ONLY on the closing turn (`turn_b`), not the opener.
    a = detect_plan_for_turn(rollout, SESSION, "turn_a")
    assert a == CodexPlanAttribution(plan_id=expected_id, completed_at=None)
    b = detect_plan_for_turn(rollout, SESSION, "turn_b")
    assert b == CodexPlanAttribution(plan_id=expected_id, completed_at=ship_ts)


def test_update_plan_open_but_never_completed_returns_attribution_without_completed(
    tmp_path: Path,
):
    """An open-but-never-shipped plan still attributes its turns; only
    `completed_at` is None. This matches Claude's behaviour for a plan that
    abandoned at session end."""
    rollout = tmp_path / "rollout.jsonl"
    open_ts = "2026-05-01T10:00:00.000Z"
    _write_jsonl(
        rollout,
        [
            _session_meta(),
            _task_started("turn_a", "2026-05-01T09:59:50.000Z"),
            _turn_context("turn_a", "2026-05-01T09:59:50.001Z"),
            _update_plan_call(open_ts, ["in_progress", "pending"]),
            _task_complete("turn_a", "2026-05-01T10:00:01.000Z"),
        ],
    )
    attr = detect_plan_for_turn(rollout, SESSION, "turn_a")
    assert attr is not None
    assert attr.plan_id == plan_id_for(SESSION, open_ts)
    assert attr.completed_at is None


def test_update_plan_subsequent_call_refreshes_same_plan(tmp_path: Path):
    """Without binding step text we cannot tell `replace` from `refresh`,
    so multiple update_plan calls always refresh the active plan. Only an
    all-completed call ships it."""
    rollout = tmp_path / "rollout.jsonl"
    open_ts = "2026-05-01T10:00:00.000Z"
    refresh_ts = "2026-05-01T10:01:00.000Z"
    ship_ts = "2026-05-01T10:02:00.000Z"
    _write_jsonl(
        rollout,
        [
            _session_meta(),
            _task_started("turn_a", "2026-05-01T09:59:50.000Z"),
            _turn_context("turn_a", "2026-05-01T09:59:50.001Z"),
            _update_plan_call(open_ts, ["pending"]),
            _update_plan_call(refresh_ts, ["in_progress"]),
            _update_plan_call(ship_ts, ["completed"]),
            _task_complete("turn_a", "2026-05-01T10:02:30.000Z"),
        ],
    )
    plans = iter_plans(rollout, SESSION)
    assert len(plans) == 1
    assert plans[0].plan_id == plan_id_for(SESSION, open_ts)
    assert plans[0].completed_at == ship_ts


def test_update_plan_after_ship_opens_new_plan_with_distinct_id(tmp_path: Path):
    """Once a plan ships, the next `update_plan` call opens a brand-new
    plan with a different (deterministic) id."""
    rollout = tmp_path / "rollout.jsonl"
    p1_ts = "2026-05-01T10:00:00.000Z"
    p2_ts = "2026-05-01T11:00:00.000Z"
    _write_jsonl(
        rollout,
        [
            _session_meta(),
            _task_started("turn_a", "2026-05-01T09:59:50.000Z"),
            _turn_context("turn_a", "2026-05-01T09:59:50.001Z"),
            _update_plan_call(p1_ts, ["completed"]),
            _task_complete("turn_a", "2026-05-01T10:00:30.000Z"),
            _task_started("turn_b", "2026-05-01T10:59:50.000Z"),
            _turn_context("turn_b", "2026-05-01T10:59:50.001Z"),
            _update_plan_call(p2_ts, ["pending", "pending"]),
            _task_complete("turn_b", "2026-05-01T11:00:30.000Z"),
        ],
    )
    plans = iter_plans(rollout, SESSION)
    assert len(plans) == 2
    assert plans[0].plan_id != plans[1].plan_id
    assert plans[0].plan_id == plan_id_for(SESSION, p1_ts)
    assert plans[1].plan_id == plan_id_for(SESSION, p2_ts)
    assert plans[0].completed_at == p1_ts
    assert plans[1].completed_at is None


# ---- Plan Mode (collaboration_mode + proposed_plan) ----


def test_plan_mode_with_proposed_plan_emission_ships(tmp_path: Path):
    rollout = tmp_path / "rollout.jsonl"
    open_ts = "2026-05-01T17:35:04.125Z"
    ship_ts = "2026-05-01T17:36:43.007Z"
    _write_jsonl(
        rollout,
        [
            _session_meta(),
            _task_started("turn_plan", "2026-05-01T17:35:04.122Z"),
            _turn_context("turn_plan", open_ts, mode="plan"),
            _assistant_message(
                ship_ts,
                "<proposed_plan>\n# A Plan\n\nSummary: do the thing.\n</proposed_plan>",
            ),
            _task_complete("turn_plan", "2026-05-01T17:36:43.031Z"),
        ],
    )
    attr = detect_plan_for_turn(rollout, SESSION, "turn_plan")
    assert attr is not None
    assert attr.plan_id == plan_id_for(SESSION, open_ts)
    assert attr.completed_at == ship_ts


def test_plan_mode_without_proposed_plan_then_mode_shift_abandons(tmp_path: Path):
    rollout = tmp_path / "rollout.jsonl"
    open_ts = "2026-05-01T17:35:00.000Z"
    leave_ts = "2026-05-01T17:40:00.000Z"
    _write_jsonl(
        rollout,
        [
            _session_meta(),
            _task_started("turn_plan", "2026-05-01T17:34:59.000Z"),
            _turn_context("turn_plan", open_ts, mode="plan"),
            _assistant_message(
                "2026-05-01T17:36:00.000Z",
                "Still thinking — no plan yet, just exploring.",
            ),
            _task_complete("turn_plan", "2026-05-01T17:36:30.000Z"),
            _task_started("turn_exec", "2026-05-01T17:39:59.000Z"),
            _turn_context("turn_exec", leave_ts, mode="default"),
            _task_complete("turn_exec", "2026-05-01T17:41:00.000Z"),
        ],
    )
    plans = iter_plans(rollout, SESSION)
    assert len(plans) == 1
    plan = plans[0]
    assert plan.completed_at is None
    assert plan.abandoned_at == leave_ts
    # The plan-mode plan still attributes its open turn even though it was
    # abandoned — useful for backend "plans started vs shipped" counts.
    attr = detect_plan_for_turn(rollout, SESSION, "turn_plan")
    assert attr is not None
    assert attr.plan_id == plan.plan_id
    assert attr.completed_at is None


def test_plan_mode_developer_proposed_plan_example_does_not_false_ship(
    tmp_path: Path,
):
    """The developer instructions injected on every plan-mode turn embed an
    `<proposed_plan>` example block. That role=developer message must NOT
    be treated as the ship signal — only role=assistant counts."""
    rollout = tmp_path / "rollout.jsonl"
    open_ts = "2026-05-01T17:35:04.125Z"
    _write_jsonl(
        rollout,
        [
            _session_meta(),
            _task_started("turn_plan", "2026-05-01T17:35:04.122Z"),
            _developer_message(
                open_ts,
                "Use the format:\n\n<proposed_plan>\nplan content\n</proposed_plan>\n",
            ),
            _turn_context("turn_plan", open_ts, mode="plan"),
            _task_complete("turn_plan", "2026-05-01T17:36:00.000Z"),
        ],
    )
    plans = iter_plans(rollout, SESSION)
    assert len(plans) == 1
    assert plans[0].completed_at is None


def test_plan_mode_survives_turn_context_without_collaboration_mode(
    tmp_path: Path,
):
    """Regression — a turn_context that omits the `collaboration_mode`
    field entirely (possible on internal/system turns in a hypothetical
    future Codex version) MUST NOT abandon an active plan_mode plan.
    Absence of signal is not the same as "left plan mode".

    Without this guard, every still-open plan would silently flip to
    abandoned at the next mode-less turn_context — wiping otherwise valid
    plan attribution.
    """
    rollout = tmp_path / "rollout.jsonl"
    open_ts = "2026-05-01T17:35:04.125Z"
    ship_ts = "2026-05-01T17:38:00.000Z"
    _write_jsonl(
        rollout,
        [
            _session_meta(),
            _task_started("turn_a", "2026-05-01T17:35:04.122Z"),
            _turn_context("turn_a", open_ts, mode="plan"),
            # Internal turn with no `collaboration_mode` field — must not
            # close the plan opened above.
            _task_started("turn_b", "2026-05-01T17:36:00.000Z"),
            _turn_context_without_mode("turn_b", "2026-05-01T17:36:00.001Z"),
            _task_complete("turn_b", "2026-05-01T17:36:30.000Z"),
            # Plan-mode session continues and finalizes a real plan.
            _task_started("turn_c", "2026-05-01T17:37:50.000Z"),
            _turn_context("turn_c", "2026-05-01T17:37:50.001Z", mode="plan"),
            _assistant_message(
                ship_ts,
                "<proposed_plan>\n# Real Plan\n</proposed_plan>",
            ),
            _task_complete("turn_c", "2026-05-01T17:38:01.000Z"),
            _task_complete("turn_a", "2026-05-01T17:35:30.000Z"),
        ],
    )
    plans = iter_plans(rollout, SESSION)
    # One plan, opened at turn_a's plan-mode turn_context, shipped at
    # turn_c's proposed_plan emission. The mode-less turn_b in between
    # was a no-op for plan state.
    assert len(plans) == 1
    plan = plans[0]
    assert plan.completed_at == ship_ts
    assert plan.abandoned_at is None
    assert plan.plan_id == plan_id_for(SESSION, open_ts)


def test_plan_mode_inline_backtick_proposed_plan_does_not_false_ship(
    tmp_path: Path,
):
    """An assistant message that mentions `<proposed_plan>` inline (e.g.
    in backticks while explaining the format) does NOT have a newline
    immediately before/after the tag, so it must not trigger ship."""
    rollout = tmp_path / "rollout.jsonl"
    open_ts = "2026-05-01T17:35:04.125Z"
    _write_jsonl(
        rollout,
        [
            _session_meta(),
            _task_started("turn_plan", "2026-05-01T17:35:04.122Z"),
            _turn_context("turn_plan", open_ts, mode="plan"),
            _assistant_message(
                "2026-05-01T17:36:00.000Z",
                "I'll wrap the final plan in `<proposed_plan>` once intent is clear.",
            ),
            _task_complete("turn_plan", "2026-05-01T17:36:30.000Z"),
        ],
    )
    plans = iter_plans(rollout, SESSION)
    assert len(plans) == 1
    assert plans[0].completed_at is None


# ---- Determinism + privacy ----


def test_plan_id_is_deterministic_across_reruns(tmp_path: Path):
    rollout = tmp_path / "rollout.jsonl"
    ts = "2026-05-01T10:00:00.000Z"
    _write_jsonl(
        rollout,
        [
            _session_meta(),
            _task_started("turn_a", "2026-05-01T09:59:50.000Z"),
            _turn_context("turn_a", "2026-05-01T09:59:50.001Z"),
            _update_plan_call(ts, ["completed"]),
            _task_complete("turn_a", "2026-05-01T10:00:01.000Z"),
        ],
    )
    p1 = iter_plans(rollout, SESSION)
    p2 = iter_plans(rollout, SESSION)
    assert p1[0].plan_id == p2[0].plan_id
    assert p1[0].plan_id == plan_id_for(SESSION, ts)


def test_step_text_and_proposed_plan_body_never_appear_in_attribution(
    tmp_path: Path,
):
    """Privacy regression — neither the step text nor the proposed_plan body
    is bound. The detector returns only the deterministic plan_id and the
    ship/abandon timestamps."""
    rollout = tmp_path / "rollout.jsonl"
    secret_step = "SECRET_REFACTOR_AUTH_LAYER"
    secret_plan_body = "SECRET_PROPOSED_PLAN_BODY_DO_NOT_LEAK"
    _write_jsonl(
        rollout,
        [
            _session_meta(),
            _task_started("turn_a", "2026-05-01T09:59:50.000Z"),
            _turn_context("turn_a", "2026-05-01T09:59:50.001Z"),
            {
                "timestamp": "2026-05-01T10:00:00.000Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "update_plan",
                    "arguments": json.dumps(
                        {
                            "plan": [{"step": secret_step, "status": "completed"}],
                            "explanation": secret_step + "_EXPLAIN",
                        }
                    ),
                    "call_id": "call_x",
                },
            },
            _assistant_message(
                "2026-05-01T10:00:30.000Z",
                f"<proposed_plan>\n{secret_plan_body}\n</proposed_plan>",
            ),
            _task_complete("turn_a", "2026-05-01T10:01:00.000Z"),
        ],
    )
    plans = iter_plans(rollout, SESSION)
    assert plans  # detection succeeded
    # The dataclass fields exposed by the public attribution API are
    # plan_id (UUID) and completed_at (ISO ts) — no text-bearing field.
    attr = detect_plan_for_turn(rollout, SESSION, "turn_a")
    assert attr is not None
    serialised = (str(attr.plan_id), str(attr.completed_at or ""))
    for s in serialised:
        assert secret_step not in s
        assert secret_plan_body not in s
