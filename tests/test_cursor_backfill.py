"""Cursor backfill tests — I1, FR-218f.

Verifies:
* one span per generation (user→assistant boundary)
* token_source='estimated' with char-based approximation
* synthesized external_id projection (cursor|<conv>|backfill-NNNNNN)
* marker prevents re-run unless force=True
* idempotent via the synthesized external_id
* missing projects dir handled gracefully
* plan attribution flows through (participation only — no completed_at
  on backfilled plans by FR-215c semantics)
* canonical tool intents from cursor_transcript projection survive
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from thrum_client.backfill import (
    _CursorBackfillGen,
    _attribution_from_plans_list,
    _cursor_backfill_marker,
    _cursor_conversation_id_from_path,
    _estimate_tokens_from_chars,
    _scan_cursor_transcript_for_backfill,
    emit_cursor_backfill_for_file,
    run_cursor_backfill,
)
from thrum_client.config import SkillSettings


_CONV = "0d883a92-5084-4c09-a2ef-1e9a28c3a6ce"


@pytest.fixture
def settings(tmp_path: Path) -> SkillSettings:
    config_dir = tmp_path / ".config" / "thrum"
    config_dir.mkdir(parents=True)
    return SkillSettings(
        config_dir=config_dir,
        claude_dir=tmp_path / ".claude",
    )


def _user(text: str = "do something") -> dict:
    return {
        "role": "user",
        "message": {"content": [{"type": "text", "text": text}]},
    }


def _assistant_text(text: str) -> dict:
    return {
        "role": "assistant",
        "message": {"content": [{"type": "text", "text": text}]},
    }


def _assistant_tool(name: str, _input: dict | None = None, **kw) -> dict:
    return {
        "role": "assistant",
        "message": {
            "content": [
                {
                    "type": "tool_use",
                    "name": name,
                    "input": _input if _input is not None else dict(kw),
                }
            ]
        },
    }


def _write_transcript(path: Path, lines: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")


def _make_cursor_transcript(
    root: Path, conversation_id: str = _CONV, lines: list[dict] | None = None
) -> Path:
    """Build a transcript at the canonical cursor projects layout
    so run_cursor_backfill's glob matches it."""
    transcript = (
        root
        / "Users-me-proj"
        / "agent-transcripts"
        / conversation_id
        / f"{conversation_id}.jsonl"
    )
    _write_transcript(transcript, lines or [])
    return transcript


def _captured_emits(monkeypatch) -> list[dict]:
    captured: list[dict] = []

    def _fake_emit(view, settings, **kw):
        captured.append({"view": dict(view), **kw})
        from thrum_client.emitter import EmitResult

        return EmitResult(status=200, body_bytes=0)

    monkeypatch.setattr("thrum_client.backfill.emit_turn", _fake_emit)
    return captured


# ---- Helpers ----


def test_estimate_tokens_from_chars_floor_one_when_nonzero():
    assert _estimate_tokens_from_chars(0) == 0
    assert _estimate_tokens_from_chars(1) == 1
    assert _estimate_tokens_from_chars(3) == 1
    assert _estimate_tokens_from_chars(4) == 1
    assert _estimate_tokens_from_chars(8) == 2
    assert _estimate_tokens_from_chars(400) == 100


def test_conversation_id_from_path_uses_stem():
    p = Path(f"/x/agent-transcripts/{_CONV}/{_CONV}.jsonl")
    assert _cursor_conversation_id_from_path(p) == _CONV


def test_scan_cursor_transcript_single_pass(tmp_path):
    """Replaces the prior _cursor_text_chars_for_generation per-generation
    helper with a single-pass scan that returns intents + char counts in
    one walk — fix for the I1 triple-walk perf issue caught in review."""
    transcript = tmp_path / "t.jsonl"
    _write_transcript(
        transcript,
        [
            _user("aaaaa"),  # 5 chars
            _assistant_text("bbbb"),  # 4 chars
            _assistant_tool("ReadFile", targetFile="/x"),
            _user("cc"),  # 2 chars
            _assistant_text("dddddddd"),  # 8 chars
            _assistant_tool("Shell", command="git status"),
        ],
    )
    gens = _scan_cursor_transcript_for_backfill(transcript)
    assert len(gens) == 2
    assert gens[0].user_chars == 5
    assert gens[0].assistant_chars == 4
    assert gens[0].tool_intents == ["read"]
    assert gens[1].user_chars == 2
    assert gens[1].assistant_chars == 8
    assert gens[1].tool_intents == ["git-ops"]


def test_scan_returns_empty_for_corrupt_or_missing_file(tmp_path):
    """Missing file → []."""
    assert _scan_cursor_transcript_for_backfill(tmp_path / "nope.jsonl") == []


def test_scan_text_content_never_bound_in_aggregate(tmp_path):
    """Same NFR-318/319 sentinel as parsers/cursor_transcript: the
    aggregate must not carry user prompt or assistant text body even
    when char counts include them."""
    transcript = tmp_path / "t.jsonl"
    _write_transcript(
        transcript,
        [
            _user("SECRET_USER_BODY_LINE_1"),
            _assistant_text("SECRET_ASSISTANT_BODY_LINE_2"),
        ],
    )
    gens = _scan_cursor_transcript_for_backfill(transcript)
    assert gens
    rep = repr(gens[0])
    assert "SECRET_USER_BODY_LINE_1" not in rep
    assert "SECRET_ASSISTANT_BODY_LINE_2" not in rep


def test_attribution_from_plans_list_finds_match():
    """Pure-Python helper for backfill — no file walk. Mirrors the
    detector's per-generation API so the loop in
    emit_cursor_backfill_for_file doesn't re-walk per generation."""
    import uuid
    from thrum_client.parsers.cursor_plan_detector import _CursorPlan

    plan = _CursorPlan(
        plan_id=uuid.UUID("12345678-1234-5678-1234-567812345678"),
        position=0,
        touched_generations=[0, 1, 2],
        closing_generation_index=2,
    )
    # Generation in touched_generations + closing → completed_at set
    a = _attribution_from_plans_list([plan], 2, generation_ts="t-close")
    assert a is not None
    assert str(a.plan_id) == "12345678-1234-5678-1234-567812345678"
    assert a.completed_at == "t-close"
    # Touched but not closing → plan_id only
    b = _attribution_from_plans_list([plan], 0, generation_ts="t-open")
    assert b is not None and b.completed_at is None
    # Not in touched → None
    assert _attribution_from_plans_list([plan], 5, generation_ts="t") is None
    # Negative index → None
    assert _attribution_from_plans_list([plan], -1, generation_ts="t") is None


# ---- Per-file emit ----


def test_emit_one_span_per_generation(tmp_path, settings, monkeypatch):
    captured = _captured_emits(monkeypatch)
    transcript = tmp_path / f"{_CONV}.jsonl"
    _write_transcript(
        transcript,
        [
            _user("first prompt"),
            _assistant_tool("ReadFile", targetFile="/x"),
            _user("second prompt"),
            _assistant_tool("Shell", command="git status"),
        ],
    )
    emitted = emit_cursor_backfill_for_file(transcript, settings)
    assert emitted == 2
    assert len(captured) == 2
    # Each carries source_tool=cursor and token_source=estimated
    for call in captured:
        assert call["source_tool"] == "cursor"
        assert call["gen_ai_system"] == "cursor"
        assert call["token_source"] == "estimated"
        assert call["backfill"] is True


def test_synthesized_external_id_uses_position(tmp_path, settings, monkeypatch):
    """external_id projection is `cursor|<conv>|backfill-NNNNNN` so
    re-runs dedupe at ingest. The conversation_id comes from the
    cursor_conversation_id metadata kwarg."""
    captured = _captured_emits(monkeypatch)
    transcript = tmp_path / f"{_CONV}.jsonl"
    _write_transcript(
        transcript,
        [
            _user("u1"),
            _assistant_tool("ReadFile", targetFile="/x"),
            _user("u2"),
            _assistant_tool("Shell", command="ls"),
        ],
    )
    emit_cursor_backfill_for_file(transcript, settings)
    assert captured[0]["cursor_conversation_id"] == _CONV
    assert captured[0]["cursor_generation_id"] == "backfill-000000"
    assert captured[1]["cursor_generation_id"] == "backfill-000001"


def test_canonical_tool_intents_flow_through(tmp_path, settings, monkeypatch):
    captured = _captured_emits(monkeypatch)
    transcript = tmp_path / f"{_CONV}.jsonl"
    _write_transcript(
        transcript,
        [
            _user("u"),
            _assistant_tool("ReadFile", targetFile="/x"),
            _assistant_tool("Shell", command="pytest"),
        ],
    )
    emit_cursor_backfill_for_file(transcript, settings)
    assert captured[0]["view"]["tools_used"] == ["read", "run-tests"]


def test_token_estimation_from_user_and_assistant_text(
    tmp_path, settings, monkeypatch
):
    captured = _captured_emits(monkeypatch)
    transcript = tmp_path / f"{_CONV}.jsonl"
    _write_transcript(
        transcript,
        [
            _user("a" * 40),  # 40 chars → 10 tokens
            _assistant_text("b" * 100),  # 100 chars → 25 tokens
        ],
    )
    emit_cursor_backfill_for_file(transcript, settings)
    assert captured[0]["agg"].tokens_in == 10
    assert captured[0]["agg"].tokens_out == 25


def test_plan_attribution_with_completed_at_null_for_backfill(
    tmp_path, settings, monkeypatch
):
    """FR-215c semantics for backfill: plan_id is set on participating
    generations, completed_at is NULL even on the closing generation
    (we don't have a per-line timestamp). Defer real completed_at to
    v2.x via SQLite-derived timestamps."""
    captured = _captured_emits(monkeypatch)
    transcript = tmp_path / f"{_CONV}.jsonl"
    _write_transcript(
        transcript,
        [
            _user("ship"),
            _assistant_tool(
                "CreatePlan", _input={"todos": [{"id": "a", "content": "x"}]}
            ),
            _assistant_tool(
                "TodoWrite",
                _input={
                    "merge": True,
                    "todos": [{"id": "a", "status": "completed"}],
                },
            ),
        ],
    )
    emit_cursor_backfill_for_file(transcript, settings)
    assert captured[0]["plan_id"] is not None
    assert captured[0]["plan_completed_at"] is None  # backfill semantics


# ---- Run-loop ----


def test_run_marker_prevents_rerun(tmp_path, settings, monkeypatch):
    captured = _captured_emits(monkeypatch)
    cursor_root = tmp_path / "cursor-projects"
    _make_cursor_transcript(
        cursor_root,
        lines=[_user("u"), _assistant_tool("ReadFile", targetFile="/x")],
    )
    # First run: emits.
    n1 = run_cursor_backfill(
        settings, cursor_projects_root=cursor_root
    )
    assert n1 == 1
    assert _cursor_backfill_marker(settings).exists()

    # Second run: no emit (marker present).
    captured.clear()
    n2 = run_cursor_backfill(
        settings, cursor_projects_root=cursor_root
    )
    assert n2 == 0
    assert captured == []


def test_run_force_reruns_despite_marker(tmp_path, settings, monkeypatch):
    captured = _captured_emits(monkeypatch)
    cursor_root = tmp_path / "cursor-projects"
    _make_cursor_transcript(
        cursor_root,
        lines=[_user("u"), _assistant_tool("ReadFile", targetFile="/x")],
    )
    run_cursor_backfill(settings, cursor_projects_root=cursor_root)
    captured.clear()

    n = run_cursor_backfill(
        settings, cursor_projects_root=cursor_root, force=True
    )
    assert n == 1
    assert len(captured) == 1


def test_run_missing_projects_dir_creates_marker_and_returns_zero(
    tmp_path, settings, monkeypatch
):
    captured = _captured_emits(monkeypatch)
    # cursor_projects_root does not exist
    n = run_cursor_backfill(
        settings, cursor_projects_root=tmp_path / "nonexistent"
    )
    assert n == 0
    assert captured == []
    assert _cursor_backfill_marker(settings).exists()


def test_run_walks_multiple_workspaces_and_conversations(
    tmp_path, settings, monkeypatch
):
    captured = _captured_emits(monkeypatch)
    cursor_root = tmp_path / "cursor-projects"
    _make_cursor_transcript(
        cursor_root,
        conversation_id="conv-a",
        lines=[_user("u"), _assistant_tool("ReadFile", targetFile="/x")],
    )
    # Different workspace, different conversation
    transcript_b = (
        cursor_root
        / "Users-me-other-proj"
        / "agent-transcripts"
        / "conv-b"
        / "conv-b.jsonl"
    )
    _write_transcript(
        transcript_b,
        [_user("u"), _assistant_tool("Shell", command="pytest")],
    )

    n = run_cursor_backfill(settings, cursor_projects_root=cursor_root)
    assert n == 2
    convs = sorted(c["cursor_conversation_id"] for c in captured)
    assert convs == ["conv-a", "conv-b"]


def test_run_continues_past_individual_file_errors(
    tmp_path, settings, monkeypatch
):
    """A corrupt transcript shouldn't poison the whole run — emit what
    we can and move on."""
    captured = _captured_emits(monkeypatch)
    cursor_root = tmp_path / "cursor-projects"
    # Healthy transcript
    _make_cursor_transcript(
        cursor_root,
        conversation_id="ok",
        lines=[_user("u"), _assistant_tool("ReadFile", targetFile="/x")],
    )
    # Corrupt transcript at the same layout level
    bad = (
        cursor_root
        / "Users-me-bad"
        / "agent-transcripts"
        / "broken"
        / "broken.jsonl"
    )
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("{ this is not json\n")  # not parseable

    # Should not raise; healthy file still emits.
    n = run_cursor_backfill(settings, cursor_projects_root=cursor_root)
    assert n >= 1
