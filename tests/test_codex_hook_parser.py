"""Codex hook-payload parser tests — FR-218b, NFR-319.

Covers the six Codex events (`SessionStart`, `UserPromptSubmit`,
`PreToolUse`, `PostToolUse`, `PermissionRequest`, `Stop`), required-field
validation, unknown-event rejection, and content-field non-leakage.
"""

from __future__ import annotations

import json

import pytest

from thrum_client.parsers.codex_hook import (
    CODEX_HOOK_EVENT_NAMES,
    CodexHookEvent,
    extract_codex_hook_event,
)


def _payload(**fields) -> bytes:
    return json.dumps(fields).encode()


def test_session_start_extracts_required_fields_and_model():
    raw = _payload(
        hook_event_name="SessionStart",
        session_id="019dda49-bb2c-7de2-9a1a-1e5ca5fb0465",
        cwd="/Users/me/code",
        model="gpt-5.5",
        turn_id="019dda49-bb2c-7de2-9a1a-1e5ca5fb0466",
    )
    ev = extract_codex_hook_event(raw)
    assert ev.hook_event_name == "SessionStart"
    assert ev.session_id == "019dda49-bb2c-7de2-9a1a-1e5ca5fb0465"
    assert ev.cwd == "/Users/me/code"
    assert ev.model == "gpt-5.5"
    assert ev.turn_id == "019dda49-bb2c-7de2-9a1a-1e5ca5fb0466"


def test_stop_extracts_stop_hook_active_and_turn_id():
    raw = _payload(
        hook_event_name="Stop",
        session_id="s1",
        cwd="/tmp",
        turn_id="t1",
        stop_hook_active=False,
        transcript_path="/Users/me/.codex/sessions/2026/04/29/rollout-x.jsonl",
    )
    ev = extract_codex_hook_event(raw)
    assert ev.hook_event_name == "Stop"
    assert ev.stop_hook_active is False
    assert ev.turn_id == "t1"
    assert ev.transcript_path.endswith("rollout-x.jsonl")


def test_pre_tool_use_extracts_tool_name():
    raw = _payload(
        hook_event_name="PreToolUse",
        session_id="s1",
        cwd="/tmp",
        tool_name="exec_command",
        tool_use_id="tu_42",
    )
    ev = extract_codex_hook_event(raw)
    assert ev.tool_name == "exec_command"
    assert ev.tool_use_id == "tu_42"


def test_missing_session_id_raises_value_error():
    raw = _payload(hook_event_name="Stop", cwd="/tmp")
    with pytest.raises(ValueError, match="session_id"):
        extract_codex_hook_event(raw)


def test_unknown_hook_event_name_rejected():
    """A Claude-only event name (e.g. `SubagentStop`) routed here must
    raise so the caller can fall back to the Claude parser cleanly."""
    raw = _payload(hook_event_name="SubagentStop", session_id="s1", cwd="/tmp")
    with pytest.raises(ValueError, match="unknown codex hook_event_name"):
        extract_codex_hook_event(raw)


def test_six_event_names_all_accepted():
    """Sanity check: every documented Codex event name parses."""
    for name in CODEX_HOOK_EVENT_NAMES:
        raw = _payload(hook_event_name=name, session_id="s", cwd="/tmp")
        ev = extract_codex_hook_event(raw)
        assert ev.hook_event_name == name


def test_unknown_keys_in_payload_are_ignored():
    """Codex may grow the payload over time; unknown fields must not
    raise and must not be bound to the dataclass."""
    raw = _payload(
        hook_event_name="UserPromptSubmit",
        session_id="s",
        cwd="/tmp",
        prompt="SECRET_USER_PROMPT_TEXT",
        last_agent_message="SECRET_RESPONSE",
        future_field={"a": 1, "b": 2},
    )
    ev = extract_codex_hook_event(raw)
    # Sentinel: no string field on the event carries the secret content.
    assert "SECRET_USER_PROMPT_TEXT" not in repr(ev)
    assert "SECRET_RESPONSE" not in repr(ev)


def test_codex_hook_event_is_frozen_dataclass():
    """Defensive: event records must be immutable so a downstream caller
    can't mutate captured fields (review NFR-318 spirit)."""
    raw = _payload(hook_event_name="Stop", session_id="s", cwd="/tmp")
    ev = extract_codex_hook_event(raw)
    with pytest.raises(Exception):  # FrozenInstanceError, dataclasses
        ev.session_id = "other"  # type: ignore[misc]
