"""Cursor plan-detector tests — FR-215c (Cursor parity), NFR-318/319.

Covers the CreatePlan + TodoWrite state machine, deterministic plan_id,
the merge:true/false branches, abandonment via fresh CreatePlan and via
all-terminal-non-completed, generation_index pairing, and the
non-binding posture for free-text fields (`name`, `overview`, `plan`,
`todos[].content`).
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from thrum_client.parsers.cursor_plan_detector import (
    CursorPlanAttribution,
    detect_all_plans,
    detect_plan_for_generation,
    iter_plan_events,
    iter_plans,
)
from thrum_client.parsers.plan_detector import plan_id_for


_CONV = "0d883a92-5084-4c09-a2ef-1e9a28c3a6ce"


def _user(text: str = "do something") -> dict:
    return {
        "role": "user",
        "message": {"content": [{"type": "text", "text": text}]},
    }


def _create_plan(seed: list[dict], **extra) -> dict:
    """Build a CreatePlan tool_use line. `seed` is the todos list with
    {id, content, dependencies?} — `content` is free text and tested for
    non-binding."""
    inp = {
        "name": "Free-text plan title",
        "overview": "Free-text overview",
        "plan": "Free-text markdown plan body",
        "todos": seed,
    }
    inp.update(extra)
    return {
        "role": "assistant",
        "message": {
            "content": [{"type": "tool_use", "name": "CreatePlan", "input": inp}]
        },
    }


def _todo_write(updates: list[dict], merge: bool = True) -> dict:
    return {
        "role": "assistant",
        "message": {
            "content": [
                {
                    "type": "tool_use",
                    "name": "TodoWrite",
                    "input": {"merge": merge, "todos": updates},
                }
            ]
        },
    }


def _other_assistant_tool(name: str = "ReadFile") -> dict:
    return {
        "role": "assistant",
        "message": {
            "content": [{"type": "tool_use", "name": name, "input": {"x": 1}}]
        },
    }


def _write(path: Path, lines: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")


# ---------- Happy path ----------


def test_create_plan_then_all_completed_via_todo_writes_ships(tmp_path):
    """4-step plan executed sequentially via merge:true TodoWrites
    ending in all completed → one shipped plan, closing on the
    generation that contains the last completion."""
    transcript = tmp_path / "t.jsonl"
    _write(
        transcript,
        [
            _user("execute the plan"),
            _create_plan(
                [
                    {"id": "audit", "content": "free text"},
                    {"id": "implement", "content": "free text"},
                    {"id": "verify", "content": "free text"},
                ]
            ),
            _todo_write([{"id": "audit", "status": "in_progress"}]),
            _todo_write(
                [
                    {"id": "audit", "status": "completed"},
                    {"id": "implement", "status": "in_progress"},
                ]
            ),
            _todo_write([{"id": "implement", "status": "completed"}]),
            _todo_write([{"id": "verify", "status": "in_progress"}]),
            _todo_write([{"id": "verify", "status": "completed"}]),
        ],
    )

    plans = iter_plans(transcript, _CONV)
    assert len(plans) == 1
    plan = plans[0]
    assert plan.all_completed()
    assert plan.closing_generation_index == 0  # all in one user turn
    assert plan.position == 0
    assert plan.plan_id == plan_id_for(_CONV, "cursor-plan-0")


def test_attribution_for_closing_generation_carries_completed_at(tmp_path):
    transcript = tmp_path / "t.jsonl"
    _write(
        transcript,
        [
            _user("ship"),
            _create_plan([{"id": "a", "content": "x"}]),
            _todo_write([{"id": "a", "status": "completed"}]),
        ],
    )
    attr = detect_plan_for_generation(
        transcript, _CONV, generation_index=0, generation_ts="2026-05-01T19:23:00Z"
    )
    assert attr is not None
    assert attr.plan_id == plan_id_for(_CONV, "cursor-plan-0")
    assert attr.completed_at == "2026-05-01T19:23:00Z"


def test_attribution_for_non_closing_generation_has_null_completed_at(tmp_path):
    """Three-generation conversation where the plan opens in gen 0 but
    completes in gen 2. Gen 0 and gen 1 are participating-but-not-closing
    (completed_at NULL); gen 2 is closing (completed_at set)."""
    transcript = tmp_path / "t.jsonl"
    _write(
        transcript,
        [
            _user("open plan"),
            _create_plan(
                [{"id": "a", "content": "x"}, {"id": "b", "content": "y"}]
            ),
            _todo_write([{"id": "a", "status": "in_progress"}]),
            _user("continue"),
            _todo_write([{"id": "a", "status": "completed"}]),
            _todo_write([{"id": "b", "status": "in_progress"}]),
            _user("finish"),
            _todo_write([{"id": "b", "status": "completed"}]),
        ],
    )

    g0 = detect_plan_for_generation(transcript, _CONV, 0, generation_ts="t0")
    g1 = detect_plan_for_generation(transcript, _CONV, 1, generation_ts="t1")
    g2 = detect_plan_for_generation(transcript, _CONV, 2, generation_ts="t2")

    assert g0 and g1 and g2
    assert g0.plan_id == g1.plan_id == g2.plan_id
    assert g0.completed_at is None
    assert g1.completed_at is None
    assert g2.completed_at == "t2"


def test_partial_completion_returns_attribution_without_completed_at(tmp_path):
    """Plan partially completed (some todos still in_progress) → all
    participating generations get plan_id, completed_at always NULL."""
    transcript = tmp_path / "t.jsonl"
    _write(
        transcript,
        [
            _user("start"),
            _create_plan(
                [{"id": "a", "content": "x"}, {"id": "b", "content": "y"}]
            ),
            _todo_write([{"id": "a", "status": "completed"}]),
            # b never completes
        ],
    )
    attr = detect_plan_for_generation(transcript, _CONV, 0, generation_ts="t")
    assert attr is not None
    assert attr.completed_at is None  # partial, not shipped


# ---------- plan_id determinism ----------


def test_plan_id_deterministic_across_repeated_calls(tmp_path):
    """Same conversation_id + same plan position → same UUID. Critical
    for re-emit-after-restart and the FR-218 backfill dedupe."""
    transcript = tmp_path / "t.jsonl"
    _write(
        transcript,
        [_user("u"), _create_plan([{"id": "a", "content": "x"}])],
    )
    plans1 = iter_plans(transcript, _CONV)
    plans2 = iter_plans(transcript, _CONV)
    assert plans1[0].plan_id == plans2[0].plan_id


def test_two_plans_in_same_conversation_get_distinct_position_seeds(tmp_path):
    """Sequential CreatePlan calls (after the previous shipped) get
    cursor-plan-0 and cursor-plan-1 seeds."""
    transcript = tmp_path / "t.jsonl"
    _write(
        transcript,
        [
            _user("first"),
            _create_plan([{"id": "p1", "content": "x"}]),
            _todo_write([{"id": "p1", "status": "completed"}]),
            _user("second"),
            _create_plan([{"id": "p2", "content": "y"}]),
            _todo_write([{"id": "p2", "status": "completed"}]),
        ],
    )
    plans = iter_plans(transcript, _CONV)
    assert len(plans) == 2
    assert plans[0].plan_id == plan_id_for(_CONV, "cursor-plan-0")
    assert plans[1].plan_id == plan_id_for(_CONV, "cursor-plan-1")
    assert plans[0].plan_id != plans[1].plan_id


# ---------- Abandonment ----------


def test_fresh_create_plan_abandons_previous_incomplete(tmp_path):
    """FR-215c case (b): fresh CreatePlan opens new plan while previous
    is still incomplete → previous marked abandoned, no completed_at,
    no shipped credit. New plan opens fresh."""
    transcript = tmp_path / "t.jsonl"
    _write(
        transcript,
        [
            _user("first"),
            _create_plan(
                [{"id": "a", "content": "x"}, {"id": "b", "content": "y"}]
            ),
            _todo_write([{"id": "a", "status": "completed"}]),
            # b never completes — instead a fresh plan opens
            _user("pivot"),
            _create_plan([{"id": "newA", "content": "z"}]),
            _todo_write([{"id": "newA", "status": "completed"}]),
        ],
    )
    plans = iter_plans(transcript, _CONV)
    assert len(plans) == 2
    # First plan: abandoned, never closed
    assert plans[0].abandoned is True
    assert plans[0].closing_generation_index is None
    # Second plan: shipped
    assert plans[1].closing_generation_index == 1
    assert plans[1].abandoned is False


def test_merge_false_disjoint_ids_treated_as_fresh_plan(tmp_path):
    """Inferred semantics for `merge: false` — when the new id set is
    disjoint from the current plan's seed, treat as plan replacement
    (close old, open new)."""
    transcript = tmp_path / "t.jsonl"
    _write(
        transcript,
        [
            _user("u"),
            _create_plan([{"id": "old1", "content": "x"}]),
            # merge:false with new id "new1" — disjoint from "old1"
            _todo_write([{"id": "new1", "status": "completed"}], merge=False),
        ],
    )
    plans = iter_plans(transcript, _CONV)
    assert len(plans) == 2
    assert plans[0].abandoned is True
    assert plans[1].seed_ids == ["new1"]
    assert plans[1].all_completed()


def test_merge_false_full_overlap_ships_in_place(tmp_path):
    """merge:false with the SAME id set as the current seed → in-place
    status update. Same plan_id, no abandonment."""
    transcript = tmp_path / "t.jsonl"
    _write(
        transcript,
        [
            _user("u"),
            _create_plan(
                [{"id": "a", "content": "x"}, {"id": "b", "content": "y"}]
            ),
            _todo_write(
                [
                    {"id": "a", "status": "completed"},
                    {"id": "b", "status": "completed"},
                ],
                merge=False,
            ),
        ],
    )
    plans = iter_plans(transcript, _CONV)
    assert len(plans) == 1  # in-place, no new plan
    assert plans[0].all_completed()


def test_merge_false_subset_does_NOT_shrink_seed(tmp_path):
    """Review-caught bug (medium): merge:false with a SUBSET of the
    seed previously REPLACED seed_ids with the smaller list, which
    would mark the plan as shipped when the subset hit `completed`
    even though the original other ids never completed. Fix: extend
    seed (additive), never shrink. 5-id seed + merge:false[2 of 5
    completed] → plan still incomplete (3 ids still pending)."""
    transcript = tmp_path / "t.jsonl"
    _write(
        transcript,
        [
            _user("u"),
            _create_plan(
                [
                    {"id": "a", "content": "x"},
                    {"id": "b", "content": "x"},
                    {"id": "c", "content": "x"},
                    {"id": "d", "content": "x"},
                    {"id": "e", "content": "x"},
                ]
            ),
            _todo_write(
                [
                    {"id": "a", "status": "completed"},
                    {"id": "b", "status": "completed"},
                ],
                merge=False,
            ),
        ],
    )
    plans = iter_plans(transcript, _CONV)
    assert len(plans) == 1
    plan = plans[0]
    assert set(plan.seed_ids) == {"a", "b", "c", "d", "e"}  # not shrunk
    assert not plan.all_completed()
    assert plan.closing_generation_index is None  # not shipped


def test_merge_false_partial_overlap_extends_seed(tmp_path):
    """merge:false with SOME existing ids + SOME new ids. New ids get
    added to the seed (status = pending until set), existing ids get
    updated. No abandonment, same plan_id."""
    transcript = tmp_path / "t.jsonl"
    _write(
        transcript,
        [
            _user("u"),
            _create_plan(
                [{"id": "a", "content": "x"}, {"id": "b", "content": "y"}]
            ),
            _todo_write(
                [
                    {"id": "b", "status": "in_progress"},
                    {"id": "c", "status": "pending"},
                ],
                merge=False,
            ),
        ],
    )
    plans = iter_plans(transcript, _CONV)
    assert len(plans) == 1
    plan = plans[0]
    assert set(plan.seed_ids) == {"a", "b", "c"}  # extended, not replaced
    assert plan.statuses["a"] == "pending"  # untouched
    assert plan.statuses["b"] == "in_progress"  # updated
    assert plan.statuses["c"] == "pending"


def test_abandoned_reason_persisted_on_plan(tmp_path):
    """Review-flagged minor: _close_active(reason=...) used to discard
    its argument. Now stored on the plan for future observability."""
    transcript = tmp_path / "t.jsonl"
    _write(
        transcript,
        [
            _user("u1"),
            _create_plan([{"id": "a", "content": "x"}]),
            _user("u2"),
            _create_plan([{"id": "b", "content": "y"}]),
            _todo_write([{"id": "b", "status": "completed"}]),
        ],
    )
    plans = iter_plans(transcript, _CONV)
    assert len(plans) == 2
    assert plans[0].abandoned is True
    assert plans[0].abandoned_reason == "fresh_create_plan"
    assert plans[1].abandoned is False
    assert plans[1].abandoned_reason is None


def test_unknown_status_treated_as_terminal_abandonment(tmp_path):
    """Fail-closed: a status not in {pending, in_progress, completed} is
    treated as terminal-non-completed for case (a). Better than letting
    an unknown enum silently ship the plan."""
    transcript = tmp_path / "t.jsonl"
    _write(
        transcript,
        [
            _user("u"),
            _create_plan(
                [{"id": "a", "content": "x"}, {"id": "b", "content": "y"}]
            ),
            _todo_write([{"id": "a", "status": "deleted"}]),
            _todo_write([{"id": "b", "status": "cancelled"}]),
        ],
    )
    plans = iter_plans(transcript, _CONV)
    assert len(plans) == 1
    assert plans[0].abandoned is True
    assert plans[0].closing_generation_index is None


# ---------- Privacy (NFR-318/319) ----------


def test_free_text_fields_never_bound(tmp_path):
    """Sentinel: name, overview, plan, and todos[].content carry free
    text — must NEVER appear on any captured event or plan dataclass."""
    transcript = tmp_path / "t.jsonl"
    _write(
        transcript,
        [
            _user("SECRET_USER_PROMPT"),
            _create_plan(
                [
                    {
                        "id": "step-1",
                        "content": "SECRET_TODO_CONTENT_BODY",
                    }
                ]
            ),
            _todo_write([{"id": "step-1", "status": "completed"}]),
        ],
    )

    # iter_plan_events
    events = list(iter_plan_events(transcript))
    for ev in events:
        rep = repr(ev)
        assert "SECRET_TODO_CONTENT_BODY" not in rep
        assert "SECRET_USER_PROMPT" not in rep
        assert "Free-text plan title" not in rep
        assert "Free-text overview" not in rep
        assert "Free-text markdown plan body" not in rep

    # iter_plans
    plans = iter_plans(transcript, _CONV)
    for plan in plans:
        rep = repr(plan)
        assert "SECRET_TODO_CONTENT_BODY" not in rep
        assert "Free-text plan title" not in rep


# ---------- Resilience ----------


def test_no_plan_returns_none(tmp_path):
    """Conversation with normal tool calls but no plan tools."""
    transcript = tmp_path / "t.jsonl"
    _write(transcript, [_user("u"), _other_assistant_tool("ReadFile")])
    assert detect_plan_for_generation(transcript, _CONV, 0, "t") is None
    assert iter_plans(transcript, _CONV) == []


def test_todo_write_without_open_plan_ignored(tmp_path):
    """Malformed transcript: TodoWrite arrives without a preceding
    CreatePlan. Don't crash; just skip."""
    transcript = tmp_path / "t.jsonl"
    _write(
        transcript,
        [
            _user("u"),
            _todo_write([{"id": "a", "status": "completed"}]),
        ],
    )
    plans = iter_plans(transcript, _CONV)
    assert plans == []


def test_corrupt_line_tolerated(tmp_path):
    """A line that fails JSON parse must not crash the detector."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        "\n".join(
            [
                json.dumps(_user("u")),
                "{ this is not valid json",
                json.dumps(_create_plan([{"id": "a", "content": "x"}])),
                json.dumps(_todo_write([{"id": "a", "status": "completed"}])),
            ]
        )
        + "\n"
    )
    plans = iter_plans(transcript, _CONV)
    assert len(plans) == 1
    assert plans[0].all_completed()


def test_negative_generation_index_returns_none(tmp_path):
    transcript = tmp_path / "t.jsonl"
    _write(
        transcript,
        [
            _user("u"),
            _create_plan([{"id": "a", "content": "x"}]),
        ],
    )
    assert detect_plan_for_generation(transcript, _CONV, -1, "t") is None


def test_detect_all_plans_yields_every_plan(tmp_path):
    """Backfill helper — yields every plan including abandoned ones."""
    transcript = tmp_path / "t.jsonl"
    _write(
        transcript,
        [
            _user("u1"),
            _create_plan([{"id": "p1", "content": "x"}]),
            # abandoned
            _user("u2"),
            _create_plan([{"id": "p2", "content": "y"}]),
            _todo_write([{"id": "p2", "status": "completed"}]),
        ],
    )
    plans = list(detect_all_plans(transcript, _CONV))
    assert len(plans) == 2
    assert plans[0].abandoned is True
    assert plans[1].all_completed()


def test_attribution_returns_none_for_generation_outside_plan(tmp_path):
    """Generation 1 has no plan tools — should not inherit gen 0's plan."""
    transcript = tmp_path / "t.jsonl"
    _write(
        transcript,
        [
            _user("first"),
            _create_plan([{"id": "a", "content": "x"}]),
            _todo_write([{"id": "a", "status": "completed"}]),
            _user("second — no plan tools"),
            _other_assistant_tool("ReadFile"),
        ],
    )
    g0 = detect_plan_for_generation(transcript, _CONV, 0, "t0")
    g1 = detect_plan_for_generation(transcript, _CONV, 1, "t1")
    assert g0 is not None
    assert g1 is None
