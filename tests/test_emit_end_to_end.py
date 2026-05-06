"""End-to-end tests — handler → emitter → OTLP wire → in-process receiver.

Backfill is covered in test_backfill.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from freezegun import freeze_time

from tests.fixtures.hook_payloads import (
    SESSION_ID,
    post_tool_use,
    post_tool_use_failure,
    pre_compact,
    session_end,
    session_start,
    stop,
    subagent_start,
    subagent_stop,
    user_prompt_submit,
)
from tests.fixtures.otlp_receiver import Capture, capture_http_post
from tests.fixtures.transcripts import (
    assistant_record,
    compact_boundary_record,
    write_jsonl,
)
from thrum_client.config import SkillSettings
from thrum_client.handler import handle_event


@pytest.fixture
def otlp(monkeypatch):
    cap = Capture()
    post, close = capture_http_post(cap)
    monkeypatch.setattr(httpx, "post", post)
    yield cap
    close()


def _write_token(tmp_home: Path, key: str = "tk_test") -> None:
    cfg = tmp_home / ".config" / "thrum"
    cfg.mkdir(parents=True, exist_ok=True)
    (cfg / "token").write_text(key)


def _payload_for_turn(
    tmp_home: Path, transcript_path: Path
) -> tuple[bytes, bytes]:
    """Return (user_prompt_submit, stop) payloads with transcript_path wired in."""
    up = json.dumps(
        {
            "hook_event_name": "UserPromptSubmit",
            "session_id": SESSION_ID,
            "transcript_path": str(transcript_path),
            "cwd": str(tmp_home),
            "prompt": "harmless",
        }
    ).encode()
    st = json.dumps(
        {
            "hook_event_name": "Stop",
            "session_id": SESSION_ID,
            "transcript_path": str(transcript_path),
            "cwd": str(tmp_home),
            "stop_hook_active": False,
            "last_assistant_message": "goodbye",
        }
    ).encode()
    return up, st


def test_turn_span_dedupes_by_message_id(tmp_home: Path, otlp: Capture):
    _write_token(tmp_home)
    transcript = write_jsonl(
        tmp_home / "t.jsonl",
        [
            # Three siblings, same message.id — FR-221.
            assistant_record("2026-04-23T10:00:30Z", "msg_A", stop_reason="end_turn", input_tokens=100, output_tokens=50),
            assistant_record("2026-04-23T10:00:31Z", "msg_A", stop_reason="end_turn", input_tokens=100, output_tokens=50),
            assistant_record("2026-04-23T10:00:32Z", "msg_A", stop_reason="end_turn", input_tokens=100, output_tokens=50),
        ],
    )
    settings = SkillSettings()
    up, st = _payload_for_turn(tmp_home, transcript)
    with freeze_time("2026-04-23T10:00:00Z"):
        handle_event(up, settings)
    with freeze_time("2026-04-23T10:01:00Z"):
        handle_event(st, settings)

    attrs = otlp.all_attrs()
    assert len(attrs) == 1
    a = attrs[0]
    assert a["_name"] == "thrum.turn"
    # Naive sum would be 300/150; FR-221 dedupe gives 100/50.
    assert a["gen_ai.usage.input_tokens"] == 100
    assert a["gen_ai.usage.output_tokens"] == 50
    assert a["thrum.content_stripped"] is True
    assert a["thrum.source_tool"] == "claude-code"
    assert a["gen_ai.request.model"] == "claude-opus-4-7"


def test_stop_with_stop_hook_active_true_does_not_emit(tmp_home: Path, otlp: Capture):
    _write_token(tmp_home)
    settings = SkillSettings()
    handle_event(user_prompt_submit(), settings)
    handle_event(stop(stop_hook_active=True, cwd=str(tmp_home)), settings)
    assert otlp.all_attrs() == []

    # Next terminating Stop emits exactly one
    handle_event(stop(stop_hook_active=False, cwd=str(tmp_home)), settings)
    assert len(otlp.all_attrs()) == 1


def test_subagent_invocation_emits_separate_activity(tmp_home: Path, otlp: Capture):
    _write_token(tmp_home)
    # Parent transcript — 1 assistant record
    parent_jsonl = write_jsonl(
        tmp_home / "parent.jsonl",
        [assistant_record("2026-04-23T10:00:00Z", "msg_P", stop_reason="end_turn", input_tokens=20, output_tokens=5)],
    )
    # Subagent transcript — tokens disjoint from parent
    sub_jsonl = write_jsonl(
        tmp_home / "sub.jsonl",
        [assistant_record("2026-04-23T10:00:10Z", "msg_S", stop_reason="end_turn", input_tokens=500, output_tokens=200)],
    )

    settings = SkillSettings()
    up, stop_p = _payload_for_turn(tmp_home, parent_jsonl)
    with freeze_time("2026-04-23T09:59:00Z"):
        handle_event(up, settings)
    handle_event(subagent_start(agent_id="agent_X", agent_type="Explore"), settings)
    handle_event(
        post_tool_use(tool_name="Grep", tool_use_id="toolu_s", agent_id="agent_X"),
        settings,
    )

    # Build a SubagentStop referencing our fake subagent transcript
    sub_stop = json.dumps(
        {
            "hook_event_name": "SubagentStop",
            "session_id": SESSION_ID,
            "transcript_path": str(parent_jsonl),
            "cwd": str(tmp_home),
            "agent_id": "agent_X",
            "agent_type": "Explore",
            "agent_transcript_path": str(sub_jsonl),
            "stop_hook_active": False,
            "last_assistant_message": "subagent done",
        }
    ).encode()
    handle_event(sub_stop, settings)
    with freeze_time("2026-04-23T10:01:00Z"):
        handle_event(stop_p, settings)

    attrs = otlp.all_attrs()
    # Expect 2 spans: one subagent (FR-228) + one parent turn.
    assert len(attrs) == 2

    sub = next(a for a in attrs if a.get("thrum.agent_id") == "agent_X")
    parent = next(a for a in attrs if a.get("thrum.agent_id") != "agent_X")

    assert sub["session.id"] == SESSION_ID  # parent session attribution
    assert sub["gen_ai.agent.name"] == "Explore"
    assert sub["gen_ai.usage.input_tokens"] == 500
    assert sub["thrum.tools_used"] == ["Grep"]

    # Parent turn's tokens — only its own record, subagent tokens disjoint.
    assert parent["gen_ai.usage.input_tokens"] == 20
    assert "thrum.agent_id" not in parent
    # Task delegation marker would come from a parent PostToolUse(Task) — not fired here,
    # so we don't assert on it. The disjoint-tokens contract is what matters.


def test_post_tool_use_failure_populates_tools_failed(tmp_home: Path, otlp: Capture):
    _write_token(tmp_home)
    settings = SkillSettings()
    # No transcript → agg is zero but span still emits
    handle_event(user_prompt_submit(), settings)
    handle_event(
        post_tool_use_failure(tool_name="Bash", tool_use_id="toolu_1"),
        settings,
    )
    handle_event(stop(stop_hook_active=False), settings)

    attrs = otlp.all_attrs()
    assert len(attrs) == 1
    assert attrs[0]["thrum.tools_used"] == ["Bash"]
    assert attrs[0]["thrum.tools_failed"] == ["Bash"]


def test_pre_compact_emits_compact_span_and_enriches_from_transcript(
    tmp_home: Path, otlp: Capture
):
    _write_token(tmp_home)
    # Transcript with a compact_boundary record
    tr = write_jsonl(
        tmp_home / "t.jsonl",
        [
            compact_boundary_record(
                "2026-04-23T10:00:00Z",
                trigger="manual",
                pre_tokens=72714,
                post_tokens=3086,
                duration_ms=82199,
            ),
        ],
    )
    up = json.dumps(
        {
            "hook_event_name": "UserPromptSubmit",
            "session_id": SESSION_ID,
            "transcript_path": str(tr),
            "cwd": str(tmp_home),
            "prompt": "harmless",
        }
    ).encode()
    pc = json.dumps(
        {
            "hook_event_name": "PreCompact",
            "session_id": SESSION_ID,
            "transcript_path": str(tr),
            "cwd": str(tmp_home),
            "trigger": "manual",
            "custom_instructions": "",
        }
    ).encode()
    settings = SkillSettings()
    handle_event(up, settings)
    handle_event(pc, settings)

    attrs = otlp.all_attrs()
    # Expect 2 compact spans: the partial (PreCompact) + enrichment (compact_boundary).
    compact = [a for a in attrs if a["_name"] == "thrum.compact"]
    assert len(compact) == 2

    partial = next(c for c in compact if "thrum.compact.pre_tokens" not in c)
    enriched = next(c for c in compact if "thrum.compact.pre_tokens" in c)

    assert partial["thrum.compact.trigger"] == "manual"
    assert partial["thrum.compact.has_custom_instructions"] is False
    assert enriched["thrum.compact.pre_tokens"] == 72714
    assert enriched["thrum.compact.post_tokens"] == 3086
    assert enriched["thrum.compact.duration_ms"] == 82199
    assert enriched["thrum.compact.enrichment"] is True


def test_pre_compact_retries_enrichment_at_next_stop(
    tmp_home: Path, otlp: Capture
):
    """Realistic timing: compact_boundary row is written AFTER PreCompact fires
    (Claude Code writes it when the compaction finishes). The enrichment must
    emit at the next terminating Stop.
    """
    _write_token(tmp_home)
    # Transcript initially empty.
    tr = tmp_home / "t.jsonl"
    tr.write_text("")

    def _p(name: str, **extra) -> bytes:
        base = {
            "hook_event_name": name,
            "session_id": SESSION_ID,
            "transcript_path": str(tr),
            "cwd": str(tmp_home),
        }
        base.update(extra)
        return json.dumps(base).encode()

    settings = SkillSettings()
    handle_event(_p("UserPromptSubmit", prompt="hi"), settings)
    handle_event(
        _p("PreCompact", trigger="manual", custom_instructions=""), settings
    )

    # At this point only the partial (PreCompact) span is out — no enrichment
    # yet because the compact_boundary row hasn't been written.
    compact = [a for a in otlp.all_attrs() if a["_name"] == "thrum.compact"]
    assert len(compact) == 1
    assert "thrum.compact.pre_tokens" not in compact[0]

    # Claude Code finishes compaction and writes the row.
    write_jsonl(
        tr,
        [
            compact_boundary_record(
                "2026-04-23T10:00:00Z",
                trigger="manual",
                pre_tokens=72714,
                post_tokens=3086,
                duration_ms=82199,
            ),
        ],
    )
    handle_event(
        _p("Stop", stop_hook_active=False, last_assistant_message="done"),
        settings,
    )

    compact = [a for a in otlp.all_attrs() if a["_name"] == "thrum.compact"]
    assert len(compact) == 2
    enriched = next(c for c in compact if "thrum.compact.pre_tokens" in c)
    assert enriched["thrum.compact.pre_tokens"] == 72714
    assert enriched["thrum.compact.enrichment"] is True


def test_pre_compact_keeps_buffer_alive_until_enrichment(
    tmp_home: Path, otlp: Capture
):
    """If the compact_boundary row is still absent at Stop time, the buffer
    must not be deleted — otherwise the enrichment can never be emitted."""
    from thrum_client.buffer import buffer_path

    _write_token(tmp_home)
    tr = tmp_home / "t.jsonl"
    tr.write_text("")

    def _p(name: str, **extra) -> bytes:
        base = {
            "hook_event_name": name,
            "session_id": SESSION_ID,
            "transcript_path": str(tr),
            "cwd": str(tmp_home),
        }
        base.update(extra)
        return json.dumps(base).encode()

    settings = SkillSettings()
    handle_event(_p("UserPromptSubmit", prompt="hi"), settings)
    handle_event(
        _p("PreCompact", trigger="manual", custom_instructions=""), settings
    )
    # First Stop fires but the compact_boundary row is still missing.
    handle_event(
        _p("Stop", stop_hook_active=False, last_assistant_message="done"),
        settings,
    )

    buf_file = buffer_path(
        tmp_home / ".config" / "thrum" / "buffers", SESSION_ID
    )
    assert buf_file.exists(), "buffer must survive Stop while compact_pending is unresolved"
    buf = json.loads(buf_file.read_text())
    assert buf["compact_pending"] is not None
    # Emitted turn was flushed; the carried-over buffer has no open turn so
    # a fresh UserPromptSubmit can reopen cleanly.
    assert buf["turn"] is None


def test_session_end_flushes_open_turn(tmp_home: Path, otlp: Capture):
    _write_token(tmp_home)
    settings = SkillSettings()
    handle_event(user_prompt_submit(), settings)
    handle_event(session_end(reason="prompt_input_exit"), settings)

    attrs = otlp.all_attrs()
    assert len(attrs) == 1
    a = attrs[0]
    assert a["thrum.metadata.forced_flush"] is True
    assert a["thrum.metadata.session_end_reason"] == "prompt_input_exit"


def test_sentinel_fuzz_no_content_in_otlp_bytes(tmp_home: Path, otlp: Capture):
    """End-to-end extension of the NFR-318/319/320 fuzz: plant SENTINEL_* in
    every content slot across a full turn + subagent + compaction, then scan:
    - OTLP wire bytes
    - buffer files
    - log files
    No sentinel should appear in any of them.
    """
    _write_token(tmp_home)
    SENTINEL = "SENTINEL_OTLP_xyz"

    tr = write_jsonl(
        tmp_home / "t.jsonl",
        [
            assistant_record(
                "2026-04-23T10:00:00Z",
                "msg_A",
                stop_reason="end_turn",
                input_tokens=100,
                output_tokens=50,
                sentinel_in_content=SENTINEL,
            ),
            compact_boundary_record("2026-04-23T10:00:05Z"),
        ],
    )

    # Payloads with SENTINEL in every content field
    def _p(name: str, **extra) -> bytes:
        base = {
            "hook_event_name": name,
            "session_id": SESSION_ID,
            "transcript_path": str(tr),
            "cwd": str(tmp_home),
        }
        base.update(extra)
        return json.dumps(base).encode()

    settings = SkillSettings()
    handle_event(_p("UserPromptSubmit", prompt=SENTINEL), settings)
    handle_event(
        _p(
            "PreToolUse",
            tool_name="Bash",
            tool_use_id="toolu_1",
            tool_input={"command": f"echo {SENTINEL}", "description": SENTINEL},
        ),
        settings,
    )
    handle_event(
        _p(
            "PostToolUse",
            tool_name="Bash",
            tool_use_id="toolu_1",
            tool_response={
                "stdout": SENTINEL,
                "stderr": SENTINEL,
                "interrupted": False,
                "isImage": False,
                "noOutputExpected": False,
            },
        ),
        settings,
    )
    handle_event(
        _p(
            "PostToolUseFailure",
            tool_name="Bash",
            tool_use_id="toolu_2",
            error=f"EISDIR: {SENTINEL}",
        ),
        settings,
    )
    handle_event(
        _p("PreCompact", trigger="manual", custom_instructions=SENTINEL),
        settings,
    )
    handle_event(
        _p(
            "Stop",
            stop_hook_active=False,
            last_assistant_message=SENTINEL,
        ),
        settings,
    )

    # 1. OTLP wire bytes — assert no sentinel in any captured body
    for body in otlp.bodies:
        assert SENTINEL.encode() not in body, (
            f"sentinel leaked into OTLP bytes: {SENTINEL!r} found in payload"
        )

    # 2. Any file under ~/.config/thrum/ must not contain the sentinel
    config_root = tmp_home / ".config" / "thrum"
    if config_root.exists():
        for p in config_root.rglob("*"):
            if p.is_file():
                assert SENTINEL.encode() not in p.read_bytes(), (
                    f"sentinel leaked to file: {p}"
                )
