"""Plan-boundary detector tests — FR-215c.

Covers:
  * single-task and multi-task plan completion → completed_at on the
    completing turn only.
  * mid-batch deletion → abandonment (no completed_at).
  * wipe-then-replan → two independent plans, each with its own plan_id.
  * supersession → fresh TaskCreate while previous plan incomplete →
    abandoned at new batch's first timestamp.
  * cross-turn attribution → plan_id present on every turn that touches
    the plan; completed_at only on the completing turn.
  * deterministic plan_id (UUIDv5) survives a re-run.
  * privacy: subject/description/activeForm never bound; sentinel
    planted in fixture content does not appear in detector output.
"""

from __future__ import annotations

from pathlib import Path

from tests.fixtures.transcripts import (
    task_create_record,
    task_update_record,
    write_jsonl,
)
from thrum_client.parsers.plan_detector import (
    PlanAttribution,
    detect_all_plans,
    detect_plan_for_turn,
    iter_tool_use_events,
    plan_id_for,
)


SESSION = "sess-xyz"
SENTINEL = "SENTINEL_PLAN_SUBJECT_xyz"


def _planned_id(first_ts: str) -> str:
    return str(plan_id_for(SESSION, first_ts))


def test_single_task_plan_ships(tmp_path: Path) -> None:
    path = write_jsonl(
        tmp_path / "s.jsonl",
        [
            task_create_record("2026-04-29T10:00:00Z"),
            task_update_record("2026-04-29T10:00:01Z", task_id="1", status="in_progress"),
            task_update_record("2026-04-29T10:05:00Z", task_id="1", status="completed"),
        ],
    )
    plans = detect_all_plans(path, SESSION)
    assert len(plans) == 1
    p = plans[0]
    assert p.task_ids == ["1"]
    assert p.completed_at == "2026-04-29T10:05:00Z"
    assert p.abandoned_at is None
    assert str(p.plan_id) == _planned_id("2026-04-29T10:00:00Z")


def test_multi_task_batch_completed_at_is_last_completed(tmp_path: Path) -> None:
    path = write_jsonl(
        tmp_path / "s.jsonl",
        [
            task_create_record("2026-04-29T10:00:00Z"),
            task_create_record("2026-04-29T10:00:01Z"),
            task_create_record("2026-04-29T10:00:02Z"),
            task_update_record("2026-04-29T10:01:00Z", task_id="1", status="completed"),
            task_update_record("2026-04-29T10:02:00Z", task_id="2", status="completed"),
            task_update_record("2026-04-29T10:03:00Z", task_id="3", status="completed"),
        ],
    )
    plans = detect_all_plans(path, SESSION)
    assert len(plans) == 1
    p = plans[0]
    assert p.task_ids == ["1", "2", "3"]
    assert p.completed_at == "2026-04-29T10:03:00Z"
    assert p.abandoned_at is None


def test_full_wipe_before_completion_marks_abandoned(tmp_path: Path) -> None:
    path = write_jsonl(
        tmp_path / "s.jsonl",
        [
            task_create_record("2026-04-29T10:00:00Z"),
            task_create_record("2026-04-29T10:00:01Z"),
            task_update_record("2026-04-29T10:01:00Z", task_id="1", status="deleted"),
            task_update_record("2026-04-29T10:01:01Z", task_id="2", status="deleted"),
        ],
    )
    plans = detect_all_plans(path, SESSION)
    assert len(plans) == 1
    p = plans[0]
    assert p.completed_at is None
    assert p.abandoned_at == "2026-04-29T10:01:01Z"


def test_wipe_then_replan_yields_two_plans(tmp_path: Path) -> None:
    """6 tasks complete; new TaskCreate batch fires; each batch is its
    own plans_shipped row (FR-215c semantic — two independent plans)."""
    path = write_jsonl(
        tmp_path / "s.jsonl",
        [
            task_create_record("2026-04-29T10:00:00Z"),
            task_update_record("2026-04-29T10:01:00Z", task_id="1", status="completed"),
            task_create_record("2026-04-29T10:02:00Z"),
            task_update_record("2026-04-29T10:03:00Z", task_id="1", status="completed"),
        ],
    )
    plans = detect_all_plans(path, SESSION)
    assert len(plans) == 2
    p1, p2 = plans
    assert p1.completed_at == "2026-04-29T10:01:00Z"
    assert p2.completed_at == "2026-04-29T10:03:00Z"
    assert p1.plan_id != p2.plan_id


def test_supersession_abandons_unfinished_plan(tmp_path: Path) -> None:
    """4-task plan, only 2 completed, then a fresh TaskCreate batch fires
    without `deleted`. Old plan is abandoned at the new batch's first ts."""
    path = write_jsonl(
        tmp_path / "s.jsonl",
        [
            task_create_record("2026-04-29T10:00:00Z"),
            task_create_record("2026-04-29T10:00:01Z"),
            task_create_record("2026-04-29T10:00:02Z"),
            task_create_record("2026-04-29T10:00:03Z"),
            task_update_record("2026-04-29T10:01:00Z", task_id="1", status="completed"),
            task_update_record("2026-04-29T10:02:00Z", task_id="2", status="completed"),
            # New batch — abandons the previous plan at this timestamp.
            task_create_record("2026-04-29T10:05:00Z"),
            task_update_record("2026-04-29T10:06:00Z", task_id="1", status="completed"),
        ],
    )
    plans = detect_all_plans(path, SESSION)
    assert len(plans) == 2
    old, new = plans
    assert old.completed_at is None
    assert old.abandoned_at == "2026-04-29T10:05:00Z"
    assert new.completed_at == "2026-04-29T10:06:00Z"


def test_attribution_per_turn_window(tmp_path: Path) -> None:
    """plan_id is present on every turn that touched the plan;
    completed_at fires only on the turn whose window covers the final
    completed event."""
    path = write_jsonl(
        tmp_path / "s.jsonl",
        [
            task_create_record("2026-04-29T10:00:00Z"),
            task_update_record("2026-04-29T10:01:00Z", task_id="1", status="in_progress"),
            task_update_record("2026-04-29T10:05:00Z", task_id="1", status="completed"),
        ],
    )
    pid = plan_id_for(SESSION, "2026-04-29T10:00:00Z")

    # Turn 1: window covers the TaskCreate. plan_id present, completed_at None.
    turn1 = detect_plan_for_turn(
        path, SESSION,
        turn_start_ts="2026-04-29T09:59:55Z",
        turn_end_ts="2026-04-29T10:00:30Z",
    )
    assert turn1 == PlanAttribution(plan_id=pid, completed_at=None)

    # Turn 2: window covers the in_progress update. plan_id present, completed_at None.
    turn2 = detect_plan_for_turn(
        path, SESSION,
        turn_start_ts="2026-04-29T10:00:31Z",
        turn_end_ts="2026-04-29T10:02:00Z",
    )
    assert turn2 == PlanAttribution(plan_id=pid, completed_at=None)

    # Turn 3: window covers the completed event. completed_at set.
    turn3 = detect_plan_for_turn(
        path, SESSION,
        turn_start_ts="2026-04-29T10:04:00Z",
        turn_end_ts="2026-04-29T10:06:00Z",
    )
    assert turn3 == PlanAttribution(
        plan_id=pid, completed_at="2026-04-29T10:05:00Z"
    )

    # Turn 4: window outside any plan event → None.
    turn4 = detect_plan_for_turn(
        path, SESSION,
        turn_start_ts="2026-04-29T11:00:00Z",
        turn_end_ts="2026-04-29T11:30:00Z",
    )
    assert turn4 is None


def test_plan_id_is_deterministic(tmp_path: Path) -> None:
    path = write_jsonl(
        tmp_path / "s.jsonl",
        [
            task_create_record("2026-04-29T10:00:00Z"),
            task_update_record("2026-04-29T10:05:00Z", task_id="1", status="completed"),
        ],
    )
    p1 = detect_all_plans(path, SESSION)
    p2 = detect_all_plans(path, SESSION)
    assert p1[0].plan_id == p2[0].plan_id


def test_orphan_taskupdate_without_create_is_ignored(tmp_path: Path) -> None:
    """Transcript captured mid-flight: TaskUpdate with no preceding
    TaskCreate. Should not crash, no plan emitted."""
    path = write_jsonl(
        tmp_path / "s.jsonl",
        [
            task_update_record("2026-04-29T10:00:00Z", task_id="7", status="completed"),
        ],
    )
    plans = detect_all_plans(path, SESSION)
    assert plans == []


def test_subject_description_never_appear_in_detector_output(tmp_path: Path) -> None:
    """Privacy posture (NFR-318/319): subject/description/activeForm flow
    through ijson but are never bound, so they cannot leak into the
    detector's PlanAttribution / _Plan output."""
    path = write_jsonl(
        tmp_path / "s.jsonl",
        [
            task_create_record(
                "2026-04-29T10:00:00Z",
                subject=SENTINEL,
                description=SENTINEL,
                active_form=SENTINEL,
            ),
            task_update_record("2026-04-29T10:05:00Z", task_id="1", status="completed"),
        ],
    )
    plans = detect_all_plans(path, SESSION)
    # Walk every field on every plan; SENTINEL must not appear.
    blob = repr(plans)
    assert SENTINEL not in blob
    events = list(iter_tool_use_events(path))
    assert SENTINEL not in repr(events)


def test_non_plan_tool_uses_are_skipped(tmp_path: Path) -> None:
    """Edit / Bash / Read tool_use blocks must not be picked up by the
    plan detector — only TaskCreate / TaskUpdate count."""
    edit_record = {
        "type": "assistant",
        "timestamp": "2026-04-29T10:00:00Z",
        "sessionId": SESSION,
        "message": {
            "id": "msg-edit-1",
            "model": "claude-opus-4-7",
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 20},
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu-edit-1",
                    "name": "Edit",
                    "input": {"file_path": "/tmp/x.py", "old_string": "a", "new_string": "b"},
                }
            ],
        },
    }
    path = write_jsonl(tmp_path / "s.jsonl", [edit_record])
    assert detect_all_plans(path, SESSION) == []
    assert list(iter_tool_use_events(path)) == []
