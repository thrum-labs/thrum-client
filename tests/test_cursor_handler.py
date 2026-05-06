"""Cursor handler dispatch tests — D1, FR-218f.

Verifies the source-tool sniff (Claude / Codex / Cursor 3-way), the
13-event Cursor dispatch, generation_index tracking (Fix #2 flush-race
mitigation), and the end-to-end Cursor stop emission against a synthetic
transcript file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from thrum_client.buffer import (
    CLAUDE_SOURCE_TOOL,
    CODEX_SOURCE_TOOL,
    CURSOR_SOURCE_TOOL,
    load_buffer,
)
from thrum_client.config import SkillSettings
from thrum_client.handler import (
    _detect_source_tool,
    _handle_cursor_event,
)


# ---- _detect_source_tool 3-way ----


def test_detect_cursor_payload_via_cursor_version():
    raw = json.dumps(
        {
            "hook_event_name": "stop",
            "conversation_id": "c",
            "session_id": "s",
            "cursor_version": "3.2.16",
        }
    ).encode()
    assert _detect_source_tool(raw) == CURSOR_SOURCE_TOOL


def test_detect_cursor_payload_via_transcript_path_substring():
    """Even without cursor_version (defensive — every captured event
    actually has it), a `/.cursor/projects/` transcript path routes to
    Cursor."""
    raw = json.dumps(
        {
            "hook_event_name": "Stop",
            "session_id": "s",
            "cwd": "/tmp",
            "transcript_path": "/Users/me/.cursor/projects/foo/agent-transcripts/x/x.jsonl",
        }
    ).encode()
    assert _detect_source_tool(raw) == CURSOR_SOURCE_TOOL


def test_detect_codex_still_routes_to_codex():
    raw = json.dumps(
        {
            "hook_event_name": "Stop",
            "session_id": "s",
            "cwd": "/tmp",
            "turn_id": "019dda4a-d28d-73d1",
            "stop_hook_active": False,
        }
    ).encode()
    assert _detect_source_tool(raw) == CODEX_SOURCE_TOOL


def test_detect_cursor_version_wins_over_stray_turn_id():
    """Defensive: if a future Cursor release adds a `turn_id` field
    (matching Codex's payload), `cursor_version` presence still wins
    so the sniff routes to the cursor handler. Locks in the priority
    order documented in `_detect_source_tool`'s docstring."""
    raw = json.dumps(
        {
            "hook_event_name": "stop",
            "conversation_id": "c",
            "session_id": "s",
            "cursor_version": "3.2.16",
            "turn_id": "would-route-to-codex-without-cursor_version-priority",
        }
    ).encode()
    assert _detect_source_tool(raw) == CURSOR_SOURCE_TOOL


def test_detect_claude_still_routes_to_claude():
    raw = json.dumps(
        {
            "hook_event_name": "Stop",
            "session_id": "s",
            "cwd": "/tmp",
            "stop_hook_active": False,
            "transcript_path": "/Users/me/.claude/projects/foo/x.jsonl",
        }
    ).encode()
    assert _detect_source_tool(raw) == CLAUDE_SOURCE_TOOL


# ---- Cursor dispatch ----


@pytest.fixture
def cursor_settings(tmp_path: Path) -> SkillSettings:
    config_dir = tmp_path / ".config" / "thrum"
    config_dir.mkdir(parents=True)
    return SkillSettings(
        config_dir=config_dir,
        claude_dir=tmp_path / ".claude",
    )


_CONV = "0d883a92-5084-4c09-a2ef-1e9a28c3a6ce"
_GEN_1 = "1ed2f56a-2829-4edd-99bf-1acc538eeb4b"
_GEN_2 = "baaea225-aacd-4e84-b4d3-1f932faf9ad6"


def _cursor_payload(transcript_path: str = "", **overrides) -> bytes:
    base = {
        "hook_event_name": "stop",
        "conversation_id": _CONV,
        "generation_id": _GEN_1,
        "session_id": _CONV,  # empirically equal in Cursor captures
        "cursor_version": "3.2.16",
        "model": "default",
        "user_email": "user@example.com",  # PII — must NEVER bind
        "workspace_roots": ["/Users/me/proj"],
        "transcript_path": transcript_path,
    }
    base.update(overrides)
    return json.dumps(base).encode()


def _user_line(text: str = "u") -> dict:
    return {
        "role": "user",
        "message": {"content": [{"type": "text", "text": text}]},
    }


def _assistant_tool(name: str, _input: dict | None = None, **kwargs) -> dict:
    return {
        "role": "assistant",
        "message": {
            "content": [
                {
                    "type": "tool_use",
                    "name": name,
                    "input": _input if _input is not None else dict(kwargs),
                }
            ]
        },
    }


def _write_transcript(path: Path, lines: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")


def _captured_emits(monkeypatch) -> list[dict]:
    """Patch emit_turn at the handler module level and capture every call's
    kwargs — same shim Codex tests use."""
    captured: list[dict] = []

    def _fake_emit(view, settings, **kw):
        captured.append({"view": dict(view), **kw})
        from thrum_client.emitter import EmitResult

        return EmitResult(status=200, body_bytes=0)

    monkeypatch.setattr("thrum_client.handler.emit_turn", _fake_emit)
    return captured


# ---- Lifecycle ----


def test_session_start_creates_buffer_with_cursor_source_tool(
    cursor_settings: SkillSettings,
):
    raw = _cursor_payload(
        hook_event_name="sessionStart",
        generation_id="",  # no turn yet
        composer_mode="agent",
        is_background_agent=False,
    )
    assert _handle_cursor_event(raw, cursor_settings) == 0

    buf = load_buffer(cursor_settings.buffers_dir, _CONV)
    assert buf is not None
    assert buf["source_tool"] == CURSOR_SOURCE_TOOL
    assert buf["cursor_conversation_id"] == _CONV
    assert buf["cursor_generation_index"] == -1  # no beforeSubmitPrompt yet
    assert buf["cursor_version"] == "3.2.16"
    assert buf["cursor_workspace_roots"] == ["/Users/me/proj"]


def test_before_submit_prompt_increments_generation_index(
    cursor_settings: SkillSettings,
):
    """Fix #2 — generation_index increments on EACH beforeSubmitPrompt
    so _emit_for_cursor_turn can pair stop events to the correct
    transcript aggregate."""
    for raw in [
        _cursor_payload(hook_event_name="sessionStart", generation_id=""),
        _cursor_payload(
            hook_event_name="beforeSubmitPrompt", generation_id=_GEN_1
        ),
    ]:
        _handle_cursor_event(raw, cursor_settings)
    buf = load_buffer(cursor_settings.buffers_dir, _CONV)
    assert buf["cursor_generation_index"] == 0
    assert buf["cursor_generation_id"] == _GEN_1

    _handle_cursor_event(
        _cursor_payload(hook_event_name="beforeSubmitPrompt", generation_id=_GEN_2),
        cursor_settings,
    )
    buf = load_buffer(cursor_settings.buffers_dir, _CONV)
    assert buf["cursor_generation_index"] == 1
    assert buf["cursor_generation_id"] == _GEN_2


def test_post_tool_use_does_NOT_record_hook_tool_name_on_buffer(
    cursor_settings: SkillSettings,
):
    """Codex L3 lesson — Rec 1.4 in fbedda4: never carry hook-side raw
    tool_names through to the backend. tools_used is filled at emit
    from the transcript, not appended from the hook."""
    for raw in [
        _cursor_payload(hook_event_name="sessionStart", generation_id=""),
        _cursor_payload(
            hook_event_name="beforeSubmitPrompt", generation_id=_GEN_1
        ),
        _cursor_payload(
            hook_event_name="postToolUse",
            generation_id=_GEN_1,
            tool_name="Read",
            tool_use_id="tu-1",
        ),
        _cursor_payload(
            hook_event_name="postToolUse",
            generation_id=_GEN_1,
            tool_name="Shell",
            tool_use_id="tu-2",
        ),
    ]:
        _handle_cursor_event(raw, cursor_settings)
    buf = load_buffer(cursor_settings.buffers_dir, _CONV)
    # tools_used is empty — neither "Read" nor "Shell" leaked through.
    assert buf["turn"]["tools_used"] == []


# ---- End-to-end stop emit ----


def test_stop_emits_with_cursor_system_and_measured_tokens(
    cursor_settings: SkillSettings, tmp_path: Path, monkeypatch
):
    captured = _captured_emits(monkeypatch)

    transcript = tmp_path / "transcript.jsonl"
    _write_transcript(
        transcript,
        [
            _user_line("list files"),
            _assistant_tool("ReadFile", targetFile="/x"),
            _assistant_tool("Glob", pattern="*.py"),
        ],
    )

    for raw in [
        _cursor_payload(
            hook_event_name="sessionStart",
            generation_id="",
            transcript_path=str(transcript),
        ),
        _cursor_payload(
            hook_event_name="beforeSubmitPrompt",
            generation_id=_GEN_1,
            transcript_path=str(transcript),
        ),
        _cursor_payload(
            hook_event_name="stop",
            generation_id=_GEN_1,
            transcript_path=str(transcript),
            input_tokens=31417,
            output_tokens=233,
            cache_read_tokens=29440,
            cache_write_tokens=0,
            loop_count=0,
            status="completed",
        ),
    ]:
        _handle_cursor_event(raw, cursor_settings)

    assert len(captured) == 1
    call = captured[0]
    assert call["source_tool"] == CURSOR_SOURCE_TOOL
    assert call["gen_ai_system"] == "cursor"
    assert call["token_source"] == "measured"
    assert call["agg"].tokens_in == 31417
    assert call["agg"].tokens_out == 233
    assert call["agg"].cache_read_input_tokens == 29440
    assert call["cursor_conversation_id"] == _CONV
    assert call["cursor_generation_id"] == _GEN_1
    assert call["cursor_loop_count"] == 0
    assert call["cursor_version"] == "3.2.16"
    assert call["cursor_workspace_roots"] == ["/Users/me/proj"]
    # Tools projected from transcript, not from hook.
    assert call["view"]["tools_used"] == ["read"]


def test_stop_without_prior_before_submit_prompt_still_attributes(
    cursor_settings: SkillSettings, tmp_path: Path, monkeypatch
):
    """Fix #3 — even when the FIRST event Thrum sees is `stop` (mid-session
    install or Cursor skipping beforeSubmitPrompt for any reason), the
    per-event refresh in _ensure_cursor_buffer captures the
    generation_id from the stop payload and bumps cursor_generation_index
    to 0. So a single transcript generation pairs correctly and
    tools_used flows through.

    Discovered live (2026-05-03) when an end-to-end Cursor turn
    landed with metadata={} because the stop dispatch never refreshed
    cursor_conversation_id / cursor_generation_id from the event."""
    captured = _captured_emits(monkeypatch)

    transcript = tmp_path / "transcript.jsonl"
    _write_transcript(
        transcript,
        [_user_line("u"), _assistant_tool("ReadFile", targetFile="/x")],
    )

    raw = _cursor_payload(
        hook_event_name="stop",
        generation_id=_GEN_1,
        transcript_path=str(transcript),
        input_tokens=100,
        output_tokens=50,
    )
    _handle_cursor_event(raw, cursor_settings)
    assert len(captured) == 1
    call = captured[0]
    # tools_used comes from the transcript via expected_index=0
    assert call["view"]["tools_used"] == ["read"]
    # cursor_conversation_id / cursor_generation_id were captured from
    # the stop event payload despite no prior beforeSubmitPrompt.
    assert call["cursor_conversation_id"] == _CONV
    assert call["cursor_generation_id"] == _GEN_1


def test_stop_when_transcript_lags_emits_empty_intents_not_stale(
    cursor_settings: SkillSettings, tmp_path: Path, monkeypatch
):
    """Fix #2 race: handler has seen 2 beforeSubmitPrompt events
    (generation_index=1) but transcript only has 1 generation flushed.
    Emit must NOT attribute the first generation's tools to the second
    turn — tools_used stays empty."""
    captured = _captured_emits(monkeypatch)

    transcript = tmp_path / "transcript.jsonl"
    # Only ONE generation flushed, with a Shell command intent.
    _write_transcript(
        transcript,
        [_user_line("first"), _assistant_tool("Shell", command="git status")],
    )

    for raw in [
        _cursor_payload(
            hook_event_name="sessionStart",
            generation_id="",
            transcript_path=str(transcript),
        ),
        _cursor_payload(
            hook_event_name="beforeSubmitPrompt",
            generation_id=_GEN_1,
            transcript_path=str(transcript),
        ),
        _cursor_payload(
            hook_event_name="beforeSubmitPrompt",
            generation_id=_GEN_2,
            transcript_path=str(transcript),
        ),
        # Stop fires for the SECOND generation but transcript only has 1.
        _cursor_payload(
            hook_event_name="stop",
            generation_id=_GEN_2,
            transcript_path=str(transcript),
            input_tokens=200,
            output_tokens=80,
        ),
    ]:
        _handle_cursor_event(raw, cursor_settings)

    assert len(captured) == 1
    # Critical: NOT ["git-ops"] (would be the first generation's intent).
    assert captured[0]["view"]["tools_used"] == []


def test_stop_with_create_plan_attributes_plan_id_and_completed_at(
    cursor_settings: SkillSettings, tmp_path: Path, monkeypatch
):
    """End-to-end FR-215c integration: CreatePlan + completed TodoWrite
    in the transcript → stop emit carries plan_id + completed_at."""
    captured = _captured_emits(monkeypatch)

    transcript = tmp_path / "transcript.jsonl"
    _write_transcript(
        transcript,
        [
            _user_line("ship"),
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

    for raw in [
        _cursor_payload(
            hook_event_name="sessionStart",
            generation_id="",
            transcript_path=str(transcript),
        ),
        _cursor_payload(
            hook_event_name="beforeSubmitPrompt",
            generation_id=_GEN_1,
            transcript_path=str(transcript),
        ),
        _cursor_payload(
            hook_event_name="stop",
            generation_id=_GEN_1,
            transcript_path=str(transcript),
            input_tokens=500,
            output_tokens=100,
        ),
    ]:
        _handle_cursor_event(raw, cursor_settings)

    assert len(captured) == 1
    call = captured[0]
    assert call["plan_id"] is not None
    assert call["plan_completed_at"] is not None  # closes on this generation


def test_post_tool_use_failure_records_tools_failed(
    cursor_settings: SkillSettings,
):
    for raw in [
        _cursor_payload(hook_event_name="sessionStart", generation_id=""),
        _cursor_payload(
            hook_event_name="beforeSubmitPrompt", generation_id=_GEN_1
        ),
        _cursor_payload(
            hook_event_name="postToolUseFailure",
            generation_id=_GEN_1,
            tool_name="Shell",
            tool_use_id="tu-bad",
        ),
    ]:
        _handle_cursor_event(raw, cursor_settings)
    buf = load_buffer(cursor_settings.buffers_dir, _CONV)
    assert buf["turn"]["tools_failed"] == ["Shell"]


def test_session_end_forces_flush(
    cursor_settings: SkillSettings, tmp_path: Path, monkeypatch
):
    """sessionEnd with an open turn forces an emit and deletes the buffer."""
    captured = _captured_emits(monkeypatch)

    transcript = tmp_path / "transcript.jsonl"
    _write_transcript(
        transcript,
        [_user_line("u"), _assistant_tool("ReadFile", targetFile="/x")],
    )

    for raw in [
        _cursor_payload(
            hook_event_name="sessionStart",
            generation_id="",
            transcript_path=str(transcript),
        ),
        _cursor_payload(
            hook_event_name="beforeSubmitPrompt",
            generation_id=_GEN_1,
            transcript_path=str(transcript),
        ),
        # No stop — sessionEnd directly closes.
        _cursor_payload(
            hook_event_name="sessionEnd",
            generation_id=_GEN_1,
            transcript_path=str(transcript),
            reason="user_close",
            duration_ms=1234,
            final_status="completed",
        ),
    ]:
        _handle_cursor_event(raw, cursor_settings)

    assert len(captured) == 1
    assert captured[0]["cursor_session_end_reason"] == "user_close"
    # Buffer deleted post-flush.
    buf = load_buffer(cursor_settings.buffers_dir, _CONV)
    assert buf is None


def test_unknown_hook_event_is_noop():
    """A future Cursor event we haven't registered must not crash; it
    short-circuits out of the parser allowlist as ValueError, the
    handler logs and exits 0."""
    # extract_cursor_hook_event raises ValueError for unknown event names;
    # _handle_cursor_event catches via _PARSE_ERRORS and exits 0.
    raw = _cursor_payload(hook_event_name="someFutureEvent")
    # Not asserting buffer state — we only need to confirm no exception.
    settings = SkillSettings(
        config_dir=Path("/tmp/thrum-test-noop"), claude_dir=Path("/tmp/.claude")
    )
    settings.config_dir.mkdir(parents=True, exist_ok=True)
    assert _handle_cursor_event(raw, settings) == 0


def test_buffer_refresh_picks_up_cursor_metadata_from_any_event(
    cursor_settings: SkillSettings, tmp_path: Path, monkeypatch
):
    """Fix #3 regression: every event payload's conversation_id / version
    / workspace_roots get refreshed onto the buffer. So even if the FIRST
    event was preToolUse (no beforeSubmitPrompt observed), a later stop
    emit carries all the cursor_* metadata."""
    captured = _captured_emits(monkeypatch)

    transcript = tmp_path / "transcript.jsonl"
    _write_transcript(
        transcript,
        [_user_line("u"), _assistant_tool("Shell", command="git status")],
    )

    # First event: preToolUse (no beforeSubmitPrompt, no sessionStart).
    # The OLD buffer-init flow would have left cursor_generation_id empty.
    for raw in [
        _cursor_payload(
            hook_event_name="preToolUse",
            generation_id=_GEN_1,
            transcript_path=str(transcript),
            tool_name="Shell",
            tool_use_id="tu-1",
        ),
        _cursor_payload(
            hook_event_name="stop",
            generation_id=_GEN_1,
            transcript_path=str(transcript),
            input_tokens=200,
            output_tokens=80,
            loop_count=0,
        ),
    ]:
        _handle_cursor_event(raw, cursor_settings)

    assert len(captured) == 1
    call = captured[0]
    # All cursor_* fields populated from the per-event refresh.
    assert call["cursor_conversation_id"] == _CONV
    assert call["cursor_generation_id"] == _GEN_1
    assert call["cursor_version"] == "3.2.16"
    assert call["cursor_workspace_roots"] == ["/Users/me/proj"]
    assert call["view"]["tools_used"] == ["git-ops"]


def test_default_model_substituted_to_cursor_default_on_emit(
    cursor_settings: SkillSettings, tmp_path: Path, monkeypatch
):
    """Cursor's hook payload reports `model: "default"` (the user's
    auto-route preference). _emit_for_cursor_turn substitutes this to
    `"cursor default"` so the activity feed / leaderboard
    distinct_models bucket shows something meaningful instead of the
    opaque token. v2.x model resolution overrides this whole string
    when the real model name lands."""
    captured = _captured_emits(monkeypatch)
    transcript = tmp_path / "transcript.jsonl"
    _write_transcript(
        transcript,
        [_user_line("u"), _assistant_tool("ReadFile", targetFile="/x")],
    )
    raw = _cursor_payload(
        hook_event_name="stop",
        generation_id=_GEN_1,
        transcript_path=str(transcript),
        input_tokens=100,
        output_tokens=50,
    )
    _handle_cursor_event(raw, cursor_settings)
    assert len(captured) == 1
    # Hook payload had `"model": "default"`; emit substitutes.
    assert captured[0]["agg"].model == "cursor default"


def test_real_model_name_in_payload_passes_through_unchanged(
    cursor_settings: SkillSettings, tmp_path: Path, monkeypatch
):
    """Forward-compat: a future Cursor release that ships a real
    model name (e.g. "claude-sonnet-4-7") in the hook payload flows
    through unchanged — substitution only fires for empty / "default"
    inputs."""
    captured = _captured_emits(monkeypatch)
    transcript = tmp_path / "transcript.jsonl"
    _write_transcript(
        transcript,
        [_user_line("u"), _assistant_tool("ReadFile", targetFile="/x")],
    )
    raw = _cursor_payload(
        hook_event_name="stop",
        generation_id=_GEN_1,
        transcript_path=str(transcript),
        input_tokens=100,
        output_tokens=50,
        model="claude-sonnet-4-7",
    )
    _handle_cursor_event(raw, cursor_settings)
    assert len(captured) == 1
    assert captured[0]["agg"].model == "claude-sonnet-4-7"


def test_user_email_from_payload_never_appears_on_emit(
    cursor_settings: SkillSettings, tmp_path: Path, monkeypatch
):
    """End-to-end privacy sentinel: even though every Cursor payload
    carries user_email, no emit kwarg or buffer field captures it."""
    captured = _captured_emits(monkeypatch)

    transcript = tmp_path / "t.jsonl"
    _write_transcript(transcript, [_user_line("u"), _assistant_tool("ReadFile", targetFile="/x")])

    for raw in [
        _cursor_payload(
            hook_event_name="sessionStart",
            generation_id="",
            transcript_path=str(transcript),
            user_email="leaked@example.com",
        ),
        _cursor_payload(
            hook_event_name="beforeSubmitPrompt",
            generation_id=_GEN_1,
            transcript_path=str(transcript),
            user_email="leaked@example.com",
        ),
        _cursor_payload(
            hook_event_name="stop",
            generation_id=_GEN_1,
            transcript_path=str(transcript),
            input_tokens=100,
            output_tokens=50,
            user_email="leaked@example.com",
        ),
    ]:
        _handle_cursor_event(raw, cursor_settings)

    assert len(captured) == 1
    assert "leaked@example.com" not in repr(captured[0])
