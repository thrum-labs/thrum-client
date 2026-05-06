"""Transcript parser + FR-221 dedupe tests."""

from __future__ import annotations

from pathlib import Path

from tests.fixtures.transcripts import (
    assistant_record,
    compact_boundary_record,
    user_record,
    write_jsonl,
)
from thrum_client.parsers.transcript import (
    aggregate_subagent,
    aggregate_turn,
    iter_compact_boundaries,
)


SENTINEL = "SENTINEL_TRANSCRIPT_xyz"


def test_fr_221_dedupes_three_sibling_records_into_one_call(tmp_path: Path):
    # Three sibling records, same message.id, identical usage — one API call.
    path = write_jsonl(
        tmp_path / "s.jsonl",
        [
            assistant_record(
                "2026-04-23T10:00:00Z",
                "msg_A",
                stop_reason="end_turn",
                input_tokens=100,
                output_tokens=50,
                sentinel_in_content=SENTINEL,
            ),
            assistant_record(
                "2026-04-23T10:00:01Z",
                "msg_A",
                stop_reason="end_turn",
                input_tokens=100,
                output_tokens=50,
                sentinel_in_content=SENTINEL,
            ),
            assistant_record(
                "2026-04-23T10:00:02Z",
                "msg_A",
                stop_reason="end_turn",
                input_tokens=100,
                output_tokens=50,
                sentinel_in_content=SENTINEL,
            ),
        ],
    )
    agg = aggregate_turn(path, turn_start_ts=None, turn_end_ts=None)
    # Naive sum would be 300/150 — FR-221 dedupe yields 100/50.
    assert agg.tokens_in == 100
    assert agg.tokens_out == 50
    assert agg.message_ids == ["msg_A"]
    # Content must not leak via dataclass repr.
    assert SENTINEL not in repr(agg)


def test_multi_call_turn_sums_distinct_message_ids(tmp_path: Path):
    path = write_jsonl(
        tmp_path / "s.jsonl",
        [
            # First API call — 2 siblings
            assistant_record("2026-04-23T10:00:00Z", "msg_A", stop_reason="tool_use", input_tokens=100, output_tokens=20),
            assistant_record("2026-04-23T10:00:01Z", "msg_A", stop_reason="tool_use", input_tokens=100, output_tokens=20),
            # Second API call — 3 siblings, end_turn
            assistant_record("2026-04-23T10:00:10Z", "msg_B", stop_reason="end_turn", input_tokens=200, output_tokens=40, model="claude-opus-4-7"),
            assistant_record("2026-04-23T10:00:11Z", "msg_B", stop_reason="end_turn", input_tokens=200, output_tokens=40, model="claude-opus-4-7"),
            assistant_record("2026-04-23T10:00:12Z", "msg_B", stop_reason="end_turn", input_tokens=200, output_tokens=40, model="claude-opus-4-7"),
        ],
    )
    agg = aggregate_turn(path, None, None)
    assert agg.tokens_in == 100 + 200
    assert agg.tokens_out == 20 + 40
    assert agg.message_ids == ["msg_A", "msg_B"]
    # Per-turn model from the end_turn record
    assert agg.model == "claude-opus-4-7"
    assert agg.stop_reason == "end_turn"


def test_turn_window_filter_excludes_other_turns(tmp_path: Path):
    path = write_jsonl(
        tmp_path / "s.jsonl",
        [
            # Prior turn — outside window
            assistant_record("2026-04-23T09:00:00Z", "msg_old", input_tokens=999),
            # In-window
            assistant_record("2026-04-23T10:00:00Z", "msg_A", input_tokens=100, stop_reason="end_turn"),
            # Later turn — outside window
            assistant_record("2026-04-23T11:00:00Z", "msg_later", input_tokens=888),
        ],
    )
    agg = aggregate_turn(
        path,
        turn_start_ts="2026-04-23T09:59:00Z",
        turn_end_ts="2026-04-23T10:30:00Z",
    )
    assert agg.message_ids == ["msg_A"]
    assert agg.tokens_in == 100


def test_user_records_are_never_aggregated(tmp_path: Path):
    path = write_jsonl(
        tmp_path / "s.jsonl",
        [
            user_record("2026-04-23T10:00:00Z", sentinel=SENTINEL),
            assistant_record("2026-04-23T10:00:01Z", "msg_A", input_tokens=100, stop_reason="end_turn"),
        ],
    )
    agg = aggregate_turn(path, None, None)
    assert agg.tokens_in == 100
    assert SENTINEL not in repr(agg)


def test_compact_boundary_excluded_from_turn_aggregator(tmp_path: Path):
    path = write_jsonl(
        tmp_path / "s.jsonl",
        [
            compact_boundary_record("2026-04-23T10:00:00Z"),
            assistant_record("2026-04-23T10:00:01Z", "msg_A", input_tokens=100, stop_reason="end_turn"),
        ],
    )
    agg = aggregate_turn(path, None, None)
    # compact_boundary has type=system — never aggregated.
    assert agg.tokens_in == 100
    assert agg.message_ids == ["msg_A"]


def test_iter_compact_boundaries_yields_records(tmp_path: Path):
    path = write_jsonl(
        tmp_path / "s.jsonl",
        [
            compact_boundary_record(
                "2026-04-23T10:00:00Z",
                trigger="manual",
                pre_tokens=72714,
                post_tokens=3086,
                duration_ms=82199,
            ),
            assistant_record("2026-04-23T10:00:01Z", "msg_A"),
            compact_boundary_record(
                "2026-04-23T11:00:00Z",
                trigger="auto",
                pre_tokens=80000,
                post_tokens=5000,
                duration_ms=100000,
            ),
        ],
    )
    boundaries = list(iter_compact_boundaries(path))
    assert len(boundaries) == 2
    assert boundaries[0].trigger == "manual"
    assert boundaries[0].pre_tokens == 72714
    assert boundaries[1].trigger == "auto"


def test_model_fallback_when_no_end_turn(tmp_path: Path):
    path = write_jsonl(
        tmp_path / "s.jsonl",
        [
            assistant_record("2026-04-23T10:00:00Z", "msg_A", stop_reason="tool_use", model="claude-sonnet-4-6"),
            assistant_record("2026-04-23T10:00:01Z", "msg_B", stop_reason="tool_use", model="claude-opus-4-7"),
        ],
    )
    agg = aggregate_turn(path, None, None)
    # No end_turn → fallback to last by timestamp.
    assert agg.model == "claude-opus-4-7"
    assert agg.stop_reason == "tool_use"


def test_subagent_aggregate_sums_whole_file(tmp_path: Path):
    path = write_jsonl(
        tmp_path / "subagent.jsonl",
        [
            assistant_record("2026-04-23T10:00:00Z", "msg_S1", input_tokens=50, output_tokens=10),
            assistant_record("2026-04-23T10:00:01Z", "msg_S1", input_tokens=50, output_tokens=10),  # sibling
            assistant_record("2026-04-23T10:00:02Z", "msg_S2", input_tokens=70, output_tokens=15, stop_reason="end_turn"),
        ],
    )
    agg = aggregate_subagent(path)
    assert agg.tokens_in == 50 + 70
    assert agg.tokens_out == 10 + 15
    assert len(agg.message_ids) == 2


def test_allowlist_never_populates_content_attributes(tmp_path: Path):
    """Structural test — planted content is not in the aggregate."""
    path = write_jsonl(
        tmp_path / "s.jsonl",
        [
            assistant_record(
                "2026-04-23T10:00:00Z",
                "msg_A",
                sentinel_in_content=SENTINEL,
            ),
        ],
    )
    agg = aggregate_turn(path, None, None)
    # repr() would show every field. SENTINEL_* must not appear.
    assert SENTINEL not in repr(agg)
