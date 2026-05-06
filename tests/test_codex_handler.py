"""Codex handler dispatch tests — D1, FR-218b.

Verifies the source-tool sniff, the six-event Codex dispatch, and the
end-to-end Codex Stop emission against a synthetic rollout file. The
existing `test_handler.py` covers the Claude path; we test the Codex
path here in isolation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from thrum_client.buffer import CODEX_SOURCE_TOOL, load_buffer, new_buffer
from thrum_client.config import SkillSettings
from thrum_client.handler import (
    _detect_source_tool,
    _emit_for_codex_turn,
    _handle_codex_event,
)


# ---- _detect_source_tool ----


def test_detect_codex_payload_with_turn_id():
    raw = json.dumps(
        {
            "hook_event_name": "Stop",
            "session_id": "s1",
            "cwd": "/tmp",
            "turn_id": "019dda4a-d28d-73d1",
            "stop_hook_active": False,
        }
    ).encode()
    assert _detect_source_tool(raw) == "codex-cli"


def test_detect_claude_payload_without_turn_id():
    raw = json.dumps(
        {
            "hook_event_name": "Stop",
            "session_id": "s1",
            "cwd": "/tmp",
            "stop_hook_active": False,
            "transcript_path": "/path/to/transcript.jsonl",
        }
    ).encode()
    assert _detect_source_tool(raw) == "claude-code"


def test_detect_falls_back_to_claude_on_malformed_json():
    """Fail-closed: junk stdin → Claude path → Claude parser raises and we
    log a parse_error. We never want the sniff itself to raise."""
    assert _detect_source_tool(b"{not valid json") == "claude-code"


# ---- Codex dispatch ----


@pytest.fixture
def codex_settings(tmp_path: Path) -> SettingsT:
    """Pre-built settings rooted at tmp_path."""
    config_dir = tmp_path / ".config" / "thrum"
    config_dir.mkdir(parents=True)
    return SkillSettings(
        config_dir=config_dir,
        claude_dir=tmp_path / ".claude",
    )


# Type-helper alias to keep mypy happy without importing TYPE_CHECKING magic.
SettingsT = SkillSettings


def _payload(**fields) -> bytes:
    return json.dumps(fields).encode()


def _rollout_record(record_type: str, **payload) -> dict:
    return {"timestamp": "2026-04-29T17:30:00Z", "type": record_type, "payload": payload}


def _write_rollout(
    path: Path,
    turn_id: str,
    *,
    tokens_in: int,
    tokens_out: int,
    model: str = "gpt-5.5",
    extra_records: list[dict] | None = None,
) -> None:
    records = [
        _rollout_record(
            "session_meta",
            id="019dda49-bb2c-7de2-9a1a-1e5ca5fb0465",
            model_provider="openai",
            originator="codex-tui",
        ),
        _rollout_record("event_msg", type="task_started", turn_id=turn_id),
        _rollout_record("turn_context", turn_id=turn_id, model=model),
    ]
    if extra_records:
        records.extend(extra_records)
    records.extend(
        [
            _rollout_record(
                "event_msg",
                type="token_count",
                info={
                    "last_token_usage": {
                        "input_tokens": tokens_in,
                        "cached_input_tokens": 0,
                        "output_tokens": tokens_out,
                        "reasoning_output_tokens": 5,
                        "total_tokens": tokens_in + tokens_out,
                    }
                },
            ),
            _rollout_record("event_msg", type="task_complete", turn_id=turn_id),
        ]
    )
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def test_codex_session_start_creates_buffer_with_codex_source_tool(
    codex_settings: SkillSettings,
):
    raw = _payload(
        hook_event_name="SessionStart",
        session_id="s1",
        cwd="/tmp",
        turn_id="t1",
        model="gpt-5.5",
        transcript_path="/tmp/rollout.jsonl",
    )
    _handle_codex_event(raw, codex_settings)

    buf = load_buffer(codex_settings.buffers_dir, "s1")
    assert buf is not None
    assert buf["source_tool"] == "codex-cli"
    assert buf["turn_id"] == "t1"
    assert buf["model"] == "gpt-5.5"


def test_codex_user_prompt_submit_opens_turn(
    codex_settings: SkillSettings,
):
    raw = _payload(
        hook_event_name="UserPromptSubmit",
        session_id="s2",
        cwd="/tmp",
        turn_id="t2",
        transcript_path="/tmp/rollout.jsonl",
    )
    _handle_codex_event(raw, codex_settings)

    buf = load_buffer(codex_settings.buffers_dir, "s2")
    assert buf["turn"] is not None
    assert buf["turn"]["turn_start_ts"] is not None


def test_codex_pre_tool_use_records_tool_use_id(
    codex_settings: SkillSettings,
):
    _handle_codex_event(
        _payload(
            hook_event_name="UserPromptSubmit",
            session_id="s3",
            cwd="/tmp",
            turn_id="t3",
        ),
        codex_settings,
    )
    _handle_codex_event(
        _payload(
            hook_event_name="PreToolUse",
            session_id="s3",
            cwd="/tmp",
            tool_name="exec_command",
            tool_use_id="call_42",
        ),
        codex_settings,
    )
    buf = load_buffer(codex_settings.buffers_dir, "s3")
    assert buf["turn"]["tool_use_id_map"]["call_42"] == "exec_command"


def test_codex_permission_request_is_noop_no_emit(
    codex_settings: SkillSettings, monkeypatch
):
    """PermissionRequest must not emit a span — only set a flag."""
    captured: list[Any] = []
    monkeypatch.setattr(
        "thrum_client.handler.emit_turn",
        lambda *a, **k: captured.append(k) or None,
    )
    _handle_codex_event(
        _payload(
            hook_event_name="UserPromptSubmit",
            session_id="s4",
            cwd="/tmp",
            turn_id="t4",
        ),
        codex_settings,
    )
    _handle_codex_event(
        _payload(
            hook_event_name="PermissionRequest",
            session_id="s4",
            cwd="/tmp",
            tool_name="exec_command",
        ),
        codex_settings,
    )
    assert captured == []
    buf = load_buffer(codex_settings.buffers_dir, "s4")
    assert buf["turn"]["tool_flags"].get("permission_requested") is True


def test_codex_stop_with_active_keeps_turn_open(
    codex_settings: SkillSettings, monkeypatch
):
    captured: list[Any] = []
    monkeypatch.setattr(
        "thrum_client.handler.emit_turn",
        lambda *a, **k: captured.append(k) or None,
    )
    _handle_codex_event(
        _payload(
            hook_event_name="UserPromptSubmit",
            session_id="s5",
            cwd="/tmp",
            turn_id="t5",
        ),
        codex_settings,
    )
    _handle_codex_event(
        _payload(
            hook_event_name="Stop",
            session_id="s5",
            cwd="/tmp",
            turn_id="t5",
            stop_hook_active=True,
        ),
        codex_settings,
    )
    assert captured == []
    # Buffer survived (not deleted).
    assert load_buffer(codex_settings.buffers_dir, "s5") is not None


def test_codex_stop_emits_with_openai_system_and_codex_source(
    codex_settings: SkillSettings, tmp_path: Path, monkeypatch
):
    """End-to-end: build a rollout, fire UserPromptSubmit + Stop, assert
    emit_turn is called with `gen_ai_system="openai"` and
    `source_tool="codex-cli"` plus the right tokens / metadata."""
    rollout = tmp_path / "rollout.jsonl"
    _write_rollout(rollout, turn_id="t-final", tokens_in=1000, tokens_out=200)

    captured: list[dict] = []

    def fake_emit_turn(view, settings, **kwargs):
        captured.append(
            {
                "view": view,
                "agg": kwargs.get("agg"),
                "gen_ai_system": kwargs.get("gen_ai_system"),
                "source_tool": kwargs.get("source_tool"),
                "codex_turn_id": kwargs.get("codex_turn_id"),
                "codex_originator": kwargs.get("codex_originator"),
                "reasoning_output_tokens": kwargs.get("reasoning_output_tokens"),
            }
        )

    monkeypatch.setattr("thrum_client.handler.emit_turn", fake_emit_turn)

    _handle_codex_event(
        _payload(
            hook_event_name="UserPromptSubmit",
            session_id="s6",
            cwd="/tmp",
            turn_id="t-final",
            transcript_path=str(rollout),
        ),
        codex_settings,
    )
    _handle_codex_event(
        _payload(
            hook_event_name="Stop",
            session_id="s6",
            cwd="/tmp",
            turn_id="t-final",
            stop_hook_active=False,
            transcript_path=str(rollout),
        ),
        codex_settings,
    )

    assert len(captured) == 1
    call = captured[0]
    assert call["gen_ai_system"] == "openai"
    assert call["source_tool"] == "codex-cli"
    assert call["agg"].tokens_in == 1000
    assert call["agg"].tokens_out == 200
    assert call["agg"].model == "gpt-5.5"
    assert call["codex_turn_id"] == "t-final"
    assert call["codex_originator"] == "codex-tui"
    assert call["reasoning_output_tokens"] == 5


def test_codex_stop_replaces_tools_used_with_canonical_intents(
    codex_settings: SkillSettings, tmp_path: Path, monkeypatch
):
    """At Stop, raw `exec_command` PostToolUse names are replaced by the
    canonical intents projected by the rollout parser."""
    rollout = tmp_path / "rollout.jsonl"
    _write_rollout(
        rollout,
        turn_id="t-int",
        tokens_in=10,
        tokens_out=5,
        extra_records=[
            _rollout_record(
                "event_msg",
                type="exec_command_end",
                turn_id="t-int",
                parsed_cmd=[{"type": "unknown", "cmd": "git status --short"}],
            )
        ],
    )

    captured: list[dict] = []
    monkeypatch.setattr(
        "thrum_client.handler.emit_turn",
        lambda view, settings, **k: captured.append({"view": view, **k}),
    )

    for payload in (
        {
            "hook_event_name": "UserPromptSubmit",
            "session_id": "s7",
            "cwd": "/tmp",
            "turn_id": "t-int",
            "transcript_path": str(rollout),
        },
        {
            "hook_event_name": "PostToolUse",
            "session_id": "s7",
            "cwd": "/tmp",
            "tool_name": "exec_command",
        },
        {
            "hook_event_name": "Stop",
            "session_id": "s7",
            "cwd": "/tmp",
            "turn_id": "t-int",
            "stop_hook_active": False,
            "transcript_path": str(rollout),
        },
    ):
        _handle_codex_event(_payload(**payload), codex_settings)

    assert len(captured) == 1
    # `view["tools_used"]` mirrors what gets emitted on the span.
    tools = captured[0]["view"]["tools_used"]
    assert "git-ops" in tools
    # The raw `exec_command` name is gone — replaced by canonical intents.
    assert "exec_command" not in tools


def test_codex_stop_without_prior_user_prompt_submit_still_emits_intents(
    codex_settings: SkillSettings, tmp_path: Path, monkeypatch
):
    """Edge case fixed in Rec 1: a bare Stop without a prior UserPromptSubmit
    used to alias `buffer.get('turn') or {}` to a fresh empty dict, then
    write canonical intents into the orphan — the `tools_used` list never
    landed on the buffer that emit_turn read from. The fix materialises the
    turn on the buffer first."""
    rollout = tmp_path / "rollout.jsonl"
    _write_rollout(
        rollout,
        turn_id="t-bare",
        tokens_in=50,
        tokens_out=10,
        extra_records=[
            _rollout_record(
                "event_msg",
                type="exec_command_end",
                turn_id="t-bare",
                parsed_cmd=[{"type": "unknown", "cmd": "git diff"}],
            )
        ],
    )

    captured: list[dict] = []
    monkeypatch.setattr(
        "thrum_client.handler.emit_turn",
        lambda view, settings, **k: captured.append({"view": view, **k}),
    )

    # Skip UserPromptSubmit entirely — straight to Stop.
    _handle_codex_event(
        _payload(
            hook_event_name="Stop",
            session_id="s-bare",
            cwd="/tmp",
            turn_id="t-bare",
            stop_hook_active=False,
            transcript_path=str(rollout),
        ),
        codex_settings,
    )

    assert len(captured) == 1
    tools = captured[0]["view"]["tools_used"]
    assert "git-ops" in tools, f"canonical intents lost on bare Stop: {tools}"


def test_codex_stop_clears_raw_tool_names_when_rollout_has_no_match(
    codex_settings: SkillSettings, tmp_path: Path, monkeypatch
):
    """Rec 1: if the rollout exists but the matching turn isn't there yet
    (task_complete pending), raw `exec_command` names captured during
    PostToolUse must be cleared — they would project to `other` in the
    backend classifier and pollute the canonical signal. The missing
    rollout entry will be picked up by the FR-218b backfill later."""
    # Empty-ish rollout — no task_complete for our turn.
    rollout = tmp_path / "rollout.jsonl"
    rollout.write_text(
        '{"type":"session_meta","payload":{"id":"sid","model_provider":"openai","originator":"codex-tui"}}\n'
    )

    captured: list[dict] = []
    monkeypatch.setattr(
        "thrum_client.handler.emit_turn",
        lambda view, settings, **k: captured.append({"view": view, **k}),
    )

    for payload in (
        {
            "hook_event_name": "UserPromptSubmit",
            "session_id": "s-nomatch",
            "cwd": "/tmp",
            "turn_id": "t-nomatch",
            "transcript_path": str(rollout),
        },
        {
            "hook_event_name": "PostToolUse",
            "session_id": "s-nomatch",
            "cwd": "/tmp",
            "tool_name": "exec_command",
        },
        {
            "hook_event_name": "Stop",
            "session_id": "s-nomatch",
            "cwd": "/tmp",
            "turn_id": "t-nomatch",
            "stop_hook_active": False,
            "transcript_path": str(rollout),
        },
    ):
        _handle_codex_event(_payload(**payload), codex_settings)

    assert len(captured) == 1
    # Raw `exec_command` would project to `other` in the classifier; it
    # must be cleared rather than leak through.
    assert captured[0]["view"]["tools_used"] == []


def test_main_dispatches_codex_payload(monkeypatch, tmp_path: Path):
    """The top-level main() entry point must route a Codex payload to the
    Codex handler — not the Claude handler."""
    import thrum_client.handler as h

    config_dir = tmp_path / ".config" / "thrum"
    config_dir.mkdir(parents=True)
    monkeypatch.setenv("THRUM_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("THRUM_CLAUDE_DIR", str(tmp_path / ".claude"))

    raw = _payload(
        hook_event_name="SessionStart",
        session_id="s-main",
        cwd="/tmp",
        turn_id="t-main",
        model="gpt-5.5",
    )

    class _Stdin:
        class buffer:
            @staticmethod
            def read():
                return raw

    monkeypatch.setattr(h.sys, "stdin", _Stdin())

    claude_called = []
    codex_called = []
    real_codex = h._handle_codex_event

    def claude_spy(*a, **k):
        claude_called.append(1)
        return 0

    def codex_spy(*a, **k):
        codex_called.append(1)
        return real_codex(*a, **k)

    monkeypatch.setattr(h, "handle_event", claude_spy)
    monkeypatch.setattr(h, "_handle_codex_event", codex_spy)

    rc = h.main()
    assert rc == 0
    assert codex_called == [1]
    assert claude_called == []


# ---- Plan attribution wiring (FR-215c, Codex parity) ----


def test_codex_stop_threads_plan_id_through_to_emit_turn(
    codex_settings: SkillSettings, tmp_path: Path, monkeypatch
):
    """Wiring regression — `_emit_for_codex_turn` must call `emit_turn`
    with `plan_id` and `plan_completed_at` populated from the Codex plan
    detector when the rollout contains a Plan Mode plan that ships in
    the closing turn.

    This guards against a refactor that drops the kwargs (the unit-level
    detector tests would still pass, but live emits would silently lose
    plan attribution).
    """
    from thrum_client.parsers.plan_detector import plan_id_for

    session_id = "019dda49-bb2c-7de2-9a1a-1e5ca5fb0465"
    plan_open_ts = "2026-04-29T17:30:01.000Z"
    plan_ship_ts = "2026-04-29T17:30:30.000Z"
    rollout = tmp_path / "rollout.jsonl"
    records = [
        {
            "timestamp": "2026-04-29T17:30:00.000Z",
            "type": "session_meta",
            "payload": {
                "id": session_id,
                "model_provider": "openai",
                "originator": "codex-tui",
            },
        },
        {
            "timestamp": "2026-04-29T17:30:00.500Z",
            "type": "event_msg",
            "payload": {"type": "task_started", "turn_id": "turn-plan"},
        },
        {
            "timestamp": plan_open_ts,
            "type": "turn_context",
            "payload": {
                "turn_id": "turn-plan",
                "model": "gpt-5.5",
                "collaboration_mode": {"mode": "plan"},
            },
        },
        {
            "timestamp": plan_ship_ts,
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "<proposed_plan>\n# Title\n</proposed_plan>",
                    }
                ],
            },
        },
        {
            "timestamp": "2026-04-29T17:30:31.000Z",
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "last_token_usage": {
                        "input_tokens": 50,
                        "cached_input_tokens": 0,
                        "output_tokens": 20,
                        "reasoning_output_tokens": 3,
                        "total_tokens": 70,
                    }
                },
            },
        },
        {
            "timestamp": "2026-04-29T17:30:32.000Z",
            "type": "event_msg",
            "payload": {
                "type": "task_complete",
                "turn_id": "turn-plan",
                "completed_at": 1777485032,
            },
        },
    ]
    rollout.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    captured: list[dict] = []
    monkeypatch.setattr(
        "thrum_client.handler.emit_turn",
        lambda view, settings, **k: captured.append({"view": view, **k}),
    )

    _handle_codex_event(
        _payload(
            hook_event_name="UserPromptSubmit",
            session_id="s-plan",
            cwd="/tmp",
            turn_id="turn-plan",
            transcript_path=str(rollout),
        ),
        codex_settings,
    )
    _handle_codex_event(
        _payload(
            hook_event_name="Stop",
            session_id="s-plan",
            cwd="/tmp",
            turn_id="turn-plan",
            stop_hook_active=False,
            transcript_path=str(rollout),
        ),
        codex_settings,
    )

    assert len(captured) == 1
    call = captured[0]
    expected_plan_id = str(plan_id_for(session_id, plan_open_ts))
    assert call["plan_id"] == expected_plan_id
    assert call["plan_completed_at"] == plan_ship_ts


def test_codex_stop_emits_no_plan_id_when_rollout_has_no_plan(
    codex_settings: SkillSettings, tmp_path: Path, monkeypatch
):
    """Negative wiring case — a rollout that contains neither a plan-mode
    turn_context nor an `update_plan` call must NOT set `plan_id`. Guards
    against a regression where the detector returns spurious attribution
    on plain Codex turns."""
    rollout = tmp_path / "rollout.jsonl"
    _write_rollout(rollout, turn_id="t-noplan", tokens_in=10, tokens_out=5)

    captured: list[dict] = []
    monkeypatch.setattr(
        "thrum_client.handler.emit_turn",
        lambda view, settings, **k: captured.append({"view": view, **k}),
    )

    _handle_codex_event(
        _payload(
            hook_event_name="UserPromptSubmit",
            session_id="s-noplan",
            cwd="/tmp",
            turn_id="t-noplan",
            transcript_path=str(rollout),
        ),
        codex_settings,
    )
    _handle_codex_event(
        _payload(
            hook_event_name="Stop",
            session_id="s-noplan",
            cwd="/tmp",
            turn_id="t-noplan",
            stop_hook_active=False,
            transcript_path=str(rollout),
        ),
        codex_settings,
    )

    assert len(captured) == 1
    assert captured[0]["plan_id"] is None
    assert captured[0]["plan_completed_at"] is None
