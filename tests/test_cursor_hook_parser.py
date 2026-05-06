"""Cursor hook-payload parser tests — FR-218f, NFR-318/319.

Covers the 13 registered Cursor events (sessionStart, sessionEnd,
beforeSubmitPrompt, preToolUse, postToolUse, postToolUseFailure,
subagentStart, subagentStop, beforeShellExecution, afterShellExecution,
preCompact, stop, afterAgentResponse), required-field validation,
unknown-event rejection, content-field non-leakage (esp. user_email,
prompt, text, tool_input, tool_output, command, output, content,
attachments).

Payloads modelled on empirically-captured shapes from ~/.cursor/hook-capture/.
"""

from __future__ import annotations

import json

import pytest

from thrum_client.parsers.cursor_hook import (
    CURSOR_HOOK_EVENT_NAMES,
    CursorHookEvent,
    extract_cursor_hook_event,
)


_BASE_ENVELOPE = {
    "hook_event_name": "stop",  # overridden per test
    "conversation_id": "0d883a92-5084-4c09-a2ef-1e9a28c3a6ce",
    "generation_id": "1ed2f56a-2829-4edd-99bf-1acc538eeb4b",
    "session_id": "0d883a92-5084-4c09-a2ef-1e9a28c3a6ce",
    "cursor_version": "3.2.16",
    "model": "default",
    "user_email": "user@example.com",  # PII — must NEVER bind
    "workspace_roots": ["/Users/me/proj"],
    "transcript_path": (
        "/Users/me/.cursor/projects/Users-me-proj/"
        "agent-transcripts/0d883a92-5084-4c09-a2ef-1e9a28c3a6ce/"
        "0d883a92-5084-4c09-a2ef-1e9a28c3a6ce.jsonl"
    ),
}


def _payload(**overrides) -> bytes:
    p = dict(_BASE_ENVELOPE)
    p.update(overrides)
    return json.dumps(p).encode()


def test_extract_stop_event_with_measured_tokens():
    """FR-218f — stop carries the per-turn token counts."""
    raw = _payload(
        hook_event_name="stop",
        status="completed",
        loop_count=0,
        input_tokens=31417,
        output_tokens=233,
        cache_read_tokens=29440,
        cache_write_tokens=0,
    )
    ev = extract_cursor_hook_event(raw)
    assert ev.hook_event_name == "stop"
    assert ev.conversation_id == "0d883a92-5084-4c09-a2ef-1e9a28c3a6ce"
    assert ev.generation_id == "1ed2f56a-2829-4edd-99bf-1acc538eeb4b"
    assert ev.cursor_version == "3.2.16"
    assert ev.model == "default"
    assert ev.input_tokens == 31417
    assert ev.output_tokens == 233
    assert ev.cache_read_tokens == 29440
    assert ev.cache_write_tokens == 0
    assert ev.status == "completed"
    assert ev.loop_count == 0
    assert ev.workspace_roots == ("/Users/me/proj",)


def test_extract_after_agent_response_does_not_bind_text():
    """afterAgentResponse carries assistant message body in `text`. The
    parser must observe the JSON token (ijson scalar event) but never
    bind the value to any field on the dataclass."""
    raw = _payload(
        hook_event_name="afterAgentResponse",
        text="SECRET_ASSISTANT_REPLY_TEXT_BODY",
        input_tokens=84290,
        output_tokens=864,
    )
    ev = extract_cursor_hook_event(raw)
    assert ev.input_tokens == 84290
    assert ev.output_tokens == 864
    # Sentinel: the secret text must not appear anywhere on the event.
    assert "SECRET_ASSISTANT_REPLY_TEXT_BODY" not in repr(ev)


def test_extract_before_submit_prompt_does_not_bind_prompt():
    """beforeSubmitPrompt carries the user prompt text in `prompt`. Never bind."""
    raw = _payload(
        hook_event_name="beforeSubmitPrompt",
        composer_mode="agent",
        prompt="SECRET_USER_PROMPT_TEXT",
        attachments=["SECRET_ATTACHMENT_1", "SECRET_ATTACHMENT_2"],
    )
    ev = extract_cursor_hook_event(raw)
    assert ev.composer_mode == "agent"
    assert "SECRET_USER_PROMPT_TEXT" not in repr(ev)
    assert "SECRET_ATTACHMENT_1" not in repr(ev)


def test_extract_session_start_minimum_envelope_with_empty_generation():
    """sessionStart fires with no active turn; generation_id is empty
    string in the captured payload, not absent."""
    raw = _payload(
        hook_event_name="sessionStart",
        generation_id="",
        composer_mode="chat",
        is_background_agent=False,
        transcript_path=None,
    )
    ev = extract_cursor_hook_event(raw)
    assert ev.hook_event_name == "sessionStart"
    assert ev.generation_id == ""
    assert ev.composer_mode == "chat"
    assert ev.is_background_agent is False


def test_extract_session_end_with_reason_and_duration():
    raw = _payload(
        hook_event_name="sessionEnd",
        reason="user_close",
        duration_ms=2503883,
        final_status="completed",
        is_background_agent=False,
    )
    ev = extract_cursor_hook_event(raw)
    assert ev.reason == "user_close"
    assert ev.duration_ms == 2503883
    assert ev.final_status == "completed"


def test_extract_pre_tool_use_does_not_bind_tool_input():
    """preToolUse carries the full tool args in `tool_input`. Never bind."""
    raw = _payload(
        hook_event_name="preToolUse",
        tool_name="Shell",
        tool_use_id="7e11a00d-0b65-4dd2-aef0-cadcb7b8ce80",
        tool_input={
            "command": "rm -rf SECRET_PATH",
            "cwd": "/SECRET_CWD",
            "timeout": 30000,
        },
    )
    ev = extract_cursor_hook_event(raw)
    assert ev.tool_name == "Shell"
    assert ev.tool_use_id == "7e11a00d-0b65-4dd2-aef0-cadcb7b8ce80"
    assert "SECRET_PATH" not in repr(ev)
    assert "SECRET_CWD" not in repr(ev)


def test_extract_post_tool_use_does_not_bind_tool_output():
    raw = _payload(
        hook_event_name="postToolUse",
        tool_name="Shell",
        tool_use_id="tu-1",
        duration=642.053,
        tool_input={"command": "ls", "cwd": "/SECRET_CWD"},
        tool_output="SECRET_TOOL_OUTPUT_BODY_12345",
    )
    ev = extract_cursor_hook_event(raw)
    assert ev.tool_name == "Shell"
    assert ev.duration == 642.053
    assert "SECRET_TOOL_OUTPUT_BODY_12345" not in repr(ev)


def test_extract_before_read_file_does_not_bind_content():
    """beforeReadFile carries the FULL file body in `content` — the most
    sensitive field in the Cursor hook surface. We don't register
    beforeReadFile as a Cursor event in v2 (not in CURSOR_HOOK_EVENT_NAMES),
    so this should reject — but if it ever gets added to the registered set,
    the allowlist must NOT include `content`."""
    raw = _payload(
        hook_event_name="beforeReadFile",
        file_path="/SECRET_PATH/secrets.txt",
        content="SECRET_FILE_BODY_LINE_1\nSECRET_FILE_BODY_LINE_2",
    )
    with pytest.raises(ValueError, match="unknown cursor hook_event_name"):
        extract_cursor_hook_event(raw)


def test_user_email_is_never_bound_anywhere():
    """user_email is in every Cursor payload (PII). It must never appear
    on any captured field, regardless of event type."""
    for name in CURSOR_HOOK_EVENT_NAMES:
        raw = _payload(
            hook_event_name=name,
            user_email="leaked@example.com",
        )
        ev = extract_cursor_hook_event(raw)
        # Sentinel: scan the entire repr for the email substring
        assert "leaked@example.com" not in repr(ev), f"event={name}"


def test_missing_required_field_raises_value_error():
    """conversation_id, session_id, cursor_version, hook_event_name are required."""
    bad = json.dumps(
        {
            "hook_event_name": "stop",
            "conversation_id": "c",
            # session_id missing
            "cursor_version": "3.2.16",
        }
    ).encode()
    with pytest.raises(ValueError, match="session_id"):
        extract_cursor_hook_event(bad)


def test_unknown_hook_event_name_rejected():
    """A Claude or Codex event accidentally routed here must raise so the
    caller can fall back. Cursor uses lowercase names; Claude uses
    PascalCase; Codex matches Claude. Ambiguity safe by enum check."""
    raw = _payload(hook_event_name="Stop")  # PascalCase — Claude/Codex shape
    with pytest.raises(ValueError, match="unknown cursor hook_event_name"):
        extract_cursor_hook_event(raw)


def test_workspace_roots_list_captured_correctly():
    raw = _payload(
        hook_event_name="stop",
        workspace_roots=["/Users/me/proj-a", "/Users/me/proj-b"],
    )
    ev = extract_cursor_hook_event(raw)
    assert ev.workspace_roots == ("/Users/me/proj-a", "/Users/me/proj-b")


def test_thirteen_event_names_all_accepted():
    """Sanity check: every registered Cursor event name parses end-to-end."""
    for name in CURSOR_HOOK_EVENT_NAMES:
        raw = _payload(hook_event_name=name)
        ev = extract_cursor_hook_event(raw)
        assert ev.hook_event_name == name


def test_cursor_hook_event_is_frozen_dataclass():
    """Defensive: events must be immutable so a downstream caller can't
    mutate captured fields (NFR-318 spirit)."""
    raw = _payload(hook_event_name="stop")
    ev = extract_cursor_hook_event(raw)
    with pytest.raises(Exception):  # FrozenInstanceError
        ev.session_id = "other"  # type: ignore[misc]


def test_unknown_keys_in_payload_are_ignored():
    """Cursor will grow the payload over time; unknown fields must not
    raise and must not be bound."""
    raw = _payload(
        hook_event_name="stop",
        future_field={"a": 1},
        another_unknown="SECRET_FUTURE_VALUE",
    )
    ev = extract_cursor_hook_event(raw)
    assert "SECRET_FUTURE_VALUE" not in repr(ev)
