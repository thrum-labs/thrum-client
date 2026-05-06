"""Handler dispatch + buffer interaction tests.

Privacy fuzz is in test_sentinel_fuzz.py.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tests.fixtures.hook_payloads import (
    SESSION_ID,
    SENTINEL,
    post_tool_use,
    post_tool_use_failure,
    pre_compact,
    pre_tool_use,
    session_end,
    session_start,
    stop,
    subagent_start,
    subagent_stop,
    user_prompt_submit,
)
from thrum_client.buffer import BUFFER_TOP_KEYS, buffer_path, save_buffer
from thrum_client.config import SkillSettings
from thrum_client.handler import handle_event


def _make_settings(tmp_home: Path) -> SkillSettings:
    # tmp_home conftest has set THRUM_CONFIG_DIR / THRUM_CLAUDE_DIR env vars.
    return SkillSettings()


def _read_buffer(tmp_home: Path) -> dict:
    path = buffer_path(
        tmp_home / ".config" / "thrum" / "buffers", SESSION_ID
    )
    return json.loads(path.read_text())


def _buffer_exists(tmp_home: Path) -> bool:
    return buffer_path(
        tmp_home / ".config" / "thrum" / "buffers", SESSION_ID
    ).exists()


def test_opt_out_short_circuits_before_any_io(tmp_home: Path, monkeypatch):
    # Seed a project dir with a .thrum-disable marker
    proj = tmp_home / "project"
    proj.mkdir()
    (proj / ".thrum-disable").write_text("")

    # Tell the payload to use that cwd
    payload = json.dumps(
        {
            "hook_event_name": "UserPromptSubmit",
            "session_id": SESSION_ID,
            "transcript_path": "/tmp/fake.jsonl",
            "cwd": str(proj),
            "prompt": SENTINEL,
        }
    ).encode()

    settings = _make_settings(tmp_home)
    assert handle_event(payload, settings) == 0

    # No buffer file created, no buffers dir touched
    assert not _buffer_exists(tmp_home)


def test_userpromptsubmit_opens_new_turn(tmp_home: Path):
    settings = _make_settings(tmp_home)
    assert handle_event(user_prompt_submit(), settings) == 0

    data = _read_buffer(tmp_home)
    assert set(data.keys()) <= BUFFER_TOP_KEYS
    assert data["session_id"] == SESSION_ID
    assert data["turn"] is not None
    assert data["turn"]["tools_used"] == []
    assert data["turn"]["tools_failed"] == []
    assert "turn_start_ts" in data["turn"]


def test_nested_userpromptsubmit_preserves_active_turn(tmp_home: Path):
    """Review #11: a second UserPromptSubmit while a turn is still active
    (no turn_end_ts, tools already accumulated) must not wipe the turn.
    """
    settings = _make_settings(tmp_home)
    handle_event(user_prompt_submit(), settings)
    handle_event(
        post_tool_use(tool_name="Read", tool_use_id="toolu_1"), settings
    )

    # A second UserPromptSubmit arrives before the Stop. This used to reset
    # buffer["turn"], dropping tools_used.
    handle_event(user_prompt_submit(), settings)

    data = _read_buffer(tmp_home)
    assert data["turn"]["tools_used"] == ["Read"]


def test_posttooluse_appends_tool_name(tmp_home: Path):
    settings = _make_settings(tmp_home)
    assert handle_event(user_prompt_submit(), settings) == 0
    assert handle_event(post_tool_use(tool_name="Read", tool_use_id="toolu_1"), settings) == 0

    data = _read_buffer(tmp_home)
    assert data["turn"]["tools_used"] == ["Read"]
    assert data["turn"]["tools_failed"] == []


def test_posttooluse_failure_appends_to_both(tmp_home: Path):
    settings = _make_settings(tmp_home)
    assert handle_event(user_prompt_submit(), settings) == 0
    assert handle_event(post_tool_use_failure(tool_name="Bash"), settings) == 0

    data = _read_buffer(tmp_home)
    assert data["turn"]["tools_used"] == ["Bash"]
    assert data["turn"]["tools_failed"] == ["Bash"]


def test_bash_category_recorded_on_turn_buffer(tmp_home: Path):
    settings = _make_settings(tmp_home)
    handle_event(user_prompt_submit(), settings)
    handle_event(
        pre_tool_use(tool_name="Bash", command="pytest -k foo"), settings
    )
    handle_event(
        post_tool_use(tool_name="Bash", tool_use_id="toolu_01"), settings
    )

    data = _read_buffer(tmp_home)
    assert data["turn"]["bash_categories"] == ["testing"]
    # The raw command must never be persisted, even transiently.
    raw = json.dumps(data)
    assert "pytest" not in raw


def test_bash_category_deduplicated_across_multiple_runs(tmp_home: Path):
    settings = _make_settings(tmp_home)
    handle_event(user_prompt_submit(), settings)
    handle_event(
        pre_tool_use(tool_name="Bash", tool_use_id="t1", command="pytest"),
        settings,
    )
    handle_event(
        pre_tool_use(tool_name="Bash", tool_use_id="t2", command="pytest -x"),
        settings,
    )
    handle_event(
        pre_tool_use(
            tool_name="Bash", tool_use_id="t3", command="git push origin main"
        ),
        settings,
    )

    data = _read_buffer(tmp_home)
    assert sorted(data["turn"]["bash_categories"]) == ["git_ops", "testing"]


def test_subagent_bash_category_scoped_to_subagent(tmp_home: Path):
    settings = _make_settings(tmp_home)
    handle_event(user_prompt_submit(), settings)
    handle_event(subagent_start(agent_id="agent_1", agent_type="Explore"), settings)
    handle_event(
        pre_tool_use(
            tool_name="Bash",
            tool_use_id="toolu_9",
            agent_id="agent_1",
            command="pytest",
        ),
        settings,
    )

    data = _read_buffer(tmp_home)
    assert data["subagents"]["agent_1"]["bash_categories"] == ["testing"]
    assert data["turn"]["bash_categories"] == []


def test_posttooluse_records_interrupted_flag(tmp_home: Path):
    settings = _make_settings(tmp_home)
    handle_event(user_prompt_submit(), settings)
    handle_event(post_tool_use(tool_name="Bash", interrupted=True), settings)

    data = _read_buffer(tmp_home)
    assert data["turn"]["tool_flags"]["interrupted"] is True


def test_stop_with_stop_hook_active_true_keeps_buffer_open(tmp_home: Path, monkeypatch):
    settings = _make_settings(tmp_home)
    handle_event(user_prompt_submit(), settings)

    emit_calls = []
    monkeypatch.setattr(
        "thrum_client.handler.emit_turn",
        lambda view, settings, **_: emit_calls.append(view.get("session_id")),
    )
    handle_event(stop(stop_hook_active=True), settings)

    # Buffer still present, emit NOT called
    assert _buffer_exists(tmp_home)
    assert emit_calls == []


def test_stop_hook_active_false_emits_and_deletes_buffer(tmp_home: Path, monkeypatch):
    settings = _make_settings(tmp_home)
    handle_event(user_prompt_submit(), settings)
    handle_event(post_tool_use(tool_name="Read", tool_use_id="toolu_1"), settings)

    emit_calls = []
    monkeypatch.setattr(
        "thrum_client.handler.emit_turn",
        lambda view, settings, **_: emit_calls.append(
            (view["session_id"], list(view["tools_used"]))
        ),
    )
    handle_event(stop(stop_hook_active=False), settings)

    assert not _buffer_exists(tmp_home)
    assert emit_calls == [(SESSION_ID, ["Read"])]


def test_subagent_start_opens_independent_buffer(tmp_home: Path):
    settings = _make_settings(tmp_home)
    handle_event(user_prompt_submit(), settings)
    handle_event(subagent_start(agent_id="agent_1", agent_type="Explore"), settings)

    data = _read_buffer(tmp_home)
    assert "agent_1" in data["subagents"]
    assert data["subagents"]["agent_1"]["agent_type"] == "Explore"
    # Parent turn still exists and untouched
    assert data["turn"]["tools_used"] == []


def test_subagent_tool_events_scope_to_subagent_buffer(tmp_home: Path):
    settings = _make_settings(tmp_home)
    handle_event(user_prompt_submit(), settings)
    handle_event(subagent_start(agent_id="agent_1", agent_type="Explore"), settings)
    handle_event(
        post_tool_use(tool_name="Grep", tool_use_id="toolu_9", agent_id="agent_1"),
        settings,
    )

    data = _read_buffer(tmp_home)
    assert data["subagents"]["agent_1"]["tools_used"] == ["Grep"]
    # Parent turn unaffected
    assert data["turn"]["tools_used"] == []


def test_subagent_stop_emits_and_drops_subagent(tmp_home: Path, monkeypatch):
    settings = _make_settings(tmp_home)
    handle_event(user_prompt_submit(), settings)
    handle_event(subagent_start(agent_id="agent_1", agent_type="Explore"), settings)

    emit_calls = []
    monkeypatch.setattr(
        "thrum_client.handler.emit_subagent",
        lambda view, parent_session_id, agent_id, settings, **_: emit_calls.append(
            (agent_id, parent_session_id)
        ),
    )
    handle_event(subagent_stop(agent_id="agent_1"), settings)

    data = _read_buffer(tmp_home)
    assert "agent_1" not in data["subagents"]
    assert emit_calls == [("agent_1", SESSION_ID)]


def test_pre_compact_records_pending_and_emits(tmp_home: Path, monkeypatch):
    settings = _make_settings(tmp_home)
    handle_event(user_prompt_submit(), settings)

    emit_calls = []
    monkeypatch.setattr(
        "thrum_client.handler.emit_compact",
        lambda session_id, trigger, has_custom_instructions, settings, **_: emit_calls.append(
            trigger
        ),
    )
    handle_event(pre_compact(trigger="manual"), settings)

    data = _read_buffer(tmp_home)
    assert data["compact_pending"]["trigger"] == "manual"
    assert data["compact_pending"]["has_custom_instructions"] is False
    # Turn buffer still open (compaction does not close a turn)
    assert data["turn"] is not None
    assert emit_calls == ["manual"]


def test_pre_compact_pending_ttl_expires_and_clears(
    tmp_home: Path, monkeypatch
):
    """If the compact_boundary row never arrives within the TTL, the handler
    must give up — otherwise the session buffer accumulates forever."""
    from thrum_client.handler import _COMPACT_PENDING_TTL

    settings = _make_settings(tmp_home)
    handle_event(user_prompt_submit(), settings)

    # Pretend PreCompact happened 11 minutes ago, with no transcript_path so
    # the scan yields nothing.
    monkeypatch.setattr("thrum_client.handler.emit_compact", lambda *a, **k: None)
    handle_event(pre_compact(trigger="manual"), settings)

    data = _read_buffer(tmp_home)
    old_observed = (
        datetime.now(UTC) - _COMPACT_PENDING_TTL - timedelta(seconds=30)
    ).isoformat()
    data["compact_pending"]["observed_at"] = old_observed
    # Clear transcript_path so the rescan finds nothing.
    data["transcript_path"] = None
    save_buffer(
        tmp_home / ".config" / "thrum" / "buffers", data
    )

    monkeypatch.setattr("thrum_client.handler.emit_turn", lambda *a, **k: None)
    handle_event(stop(stop_hook_active=False), settings)

    # Stale pending → cleared → buffer deleted as usual.
    assert not _buffer_exists(tmp_home)


def test_session_end_flushes_open_turn(tmp_home: Path, monkeypatch):
    settings = _make_settings(tmp_home)
    handle_event(user_prompt_submit(), settings)

    emit_calls = []
    monkeypatch.setattr(
        "thrum_client.handler.emit_session_end_flush",
        lambda session_buffer, settings, **kwargs: emit_calls.append(
            (
                session_buffer["turn"]["forced_flush"],
                kwargs.get("session_end_reason"),
            )
        ),
    )
    handle_event(session_end(reason="prompt_input_exit"), settings)

    assert not _buffer_exists(tmp_home)
    assert emit_calls == [(True, "prompt_input_exit")]


def test_buffer_schema_rejects_unknown_keys_on_read(tmp_home: Path):
    # Pre-seed a buffer file with an extra top-level key
    buffers_dir = tmp_home / ".config" / "thrum" / "buffers"
    buffers_dir.mkdir(parents=True)
    evil = buffers_dir / f"{SESSION_ID}.json"
    evil.write_text(json.dumps({"evil_key": SENTINEL, "session_id": SESSION_ID}))

    settings = _make_settings(tmp_home)
    # Next event should trigger a schema-rejection → buffer deleted, new one created.
    handle_event(user_prompt_submit(), settings)

    # The evil file was replaced by a valid one (no extra key, no sentinel)
    data = _read_buffer(tmp_home)
    assert "evil_key" not in data
    assert SENTINEL not in json.dumps(data)


def test_handler_latency_under_100ms(tmp_home: Path):
    settings = _make_settings(tmp_home)

    # Warm up the path
    handle_event(session_start(), settings)
    handle_event(user_prompt_submit(), settings)

    # Measure the PostToolUse round-trip (the most common hot path).
    payloads = [
        post_tool_use(tool_name=f"T{i:02d}", tool_use_id=f"toolu_{i:02d}")
        for i in range(10)
    ]
    latencies = []
    for p in payloads:
        t0 = time.perf_counter()
        handle_event(p, settings)
        latencies.append((time.perf_counter() - t0) * 1000)
    # p95 well under 100ms
    p95 = sorted(latencies)[int(0.95 * len(latencies))]
    assert p95 < 100, f"p95 {p95:.1f}ms exceeded 100ms budget"
