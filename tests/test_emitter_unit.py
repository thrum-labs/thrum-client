"""Span-builder unit tests — NFR-318 allowlist enforcement."""

from __future__ import annotations

import pytest

from thrum_client.emitter import (
    ALLOWED_ATTRIBUTE_KEYS,
    AllowlistError,
    _add_attr,
    build_compact_span,
    build_turn_span,
)
from thrum_client.parsers.transcript import CompactBoundaryRecord, TranscriptAggregate


def _get_attr(span, key: str):
    for a in span.attributes:
        if a.key == key:
            v = a.value
            if v.HasField("string_value"):
                return v.string_value
            if v.HasField("int_value"):
                return int(v.int_value)
            if v.HasField("bool_value"):
                return bool(v.bool_value)
            if v.HasField("array_value"):
                return [
                    x.string_value if x.HasField("string_value") else x.int_value
                    for x in v.array_value.values
                ]
    return None


def test_turn_span_sets_content_stripped_and_source_tool():
    buffer = {
        "session_id": "s1",
        "turn_start_ts": "2026-04-23T10:00:00Z",
        "tools_used": ["Read", "Bash"],
        "tools_failed": [],
        "tool_flags": {"interrupted": False, "is_image": False},
    }
    agg = TranscriptAggregate(
        tokens_in=100, tokens_out=50, model="claude-opus-4-7"
    )
    span = build_turn_span(buffer, agg)
    assert _get_attr(span, "thrum.content_stripped") is True
    assert _get_attr(span, "thrum.source_tool") == "claude-code"
    assert _get_attr(span, "gen_ai.system") == "anthropic"
    assert _get_attr(span, "thrum.token_source") == "measured"
    assert _get_attr(span, "session.id") == "s1"
    assert _get_attr(span, "gen_ai.usage.input_tokens") == 100
    assert _get_attr(span, "gen_ai.usage.output_tokens") == 50
    assert _get_attr(span, "gen_ai.request.model") == "claude-opus-4-7"


def test_turn_span_codex_system_and_source_tool():
    """FR-218b — Codex callers pass openai/codex-cli; the builder must
    honour the kwargs without leaking the anthropic/claude-code defaults."""
    buffer = {
        "session_id": "codex_s1",
        "turn_start_ts": "2026-04-23T10:00:00Z",
        "tools_used": ["exec_command"],
        "tools_failed": [],
        "tool_flags": {},
    }
    agg = TranscriptAggregate(tokens_in=800, tokens_out=1200, model="gpt-5.5")
    span = build_turn_span(
        buffer,
        agg,
        gen_ai_system="openai",
        source_tool="codex-cli",
    )
    assert _get_attr(span, "gen_ai.system") == "openai"
    assert _get_attr(span, "thrum.source_tool") == "codex-cli"
    assert _get_attr(span, "thrum.token_source") == "measured"


def test_turn_span_token_source_estimated_pass_through():
    """FR-218e — desktop watchers (Section 6) pass token_source='estimated'."""
    buffer = {
        "session_id": "s1",
        "turn_start_ts": "2026-04-23T10:00:00Z",
        "tools_used": [],
        "tools_failed": [],
        "tool_flags": {},
    }
    span = build_turn_span(buffer, TranscriptAggregate(), token_source="estimated")
    assert _get_attr(span, "thrum.token_source") == "estimated"


def test_compact_span_codex_system_and_source_tool():
    """build_compact_span must accept the same parameterisation."""
    span = build_compact_span(
        "codex_s1",
        "manual",
        False,
        gen_ai_system="openai",
        source_tool="codex-cli",
    )
    assert _get_attr(span, "gen_ai.system") == "openai"
    assert _get_attr(span, "thrum.source_tool") == "codex-cli"


def test_no_anthropic_leakage_when_codex_kwargs_passed():
    """Sentinel: with openai/codex-cli kwargs, the literal 'anthropic' must
    not appear on any string attribute on the span. Catches any forgotten
    hardcoded constant."""
    buffer = {
        "session_id": "s",
        "turn_start_ts": "2026-04-23T10:00:00Z",
        "tools_used": [],
        "tools_failed": [],
        "tool_flags": {},
    }
    span = build_turn_span(
        buffer,
        TranscriptAggregate(model="gpt-5.5"),
        gen_ai_system="openai",
        source_tool="codex-cli",
    )
    for a in span.attributes:
        v = a.value
        if v.HasField("string_value"):
            assert "anthropic" not in v.string_value, a.key
            assert "claude-code" not in v.string_value, a.key


def test_turn_span_tools_arrays_preserve_order():
    buffer = {
        "session_id": "s1",
        "turn_start_ts": "2026-04-23T10:00:00Z",
        "tools_used": ["Read", "Edit", "Bash"],
        "tools_failed": ["Bash"],
        "tool_flags": {},
    }
    span = build_turn_span(buffer, TranscriptAggregate())
    assert _get_attr(span, "thrum.tools_used") == ["Read", "Edit", "Bash"]
    assert _get_attr(span, "thrum.tools_failed") == ["Bash"]


def test_turn_span_backfill_and_flush_flags():
    buffer = {"session_id": "s", "tools_used": [], "tools_failed": [], "tool_flags": {}}
    span = build_turn_span(
        buffer,
        TranscriptAggregate(),
        backfill=True,
        forced_flush=True,
        session_end_reason="prompt_input_exit",
    )
    assert _get_attr(span, "thrum.metadata.backfill") is True
    assert _get_attr(span, "thrum.metadata.forced_flush") is True
    assert _get_attr(span, "thrum.metadata.session_end_reason") == "prompt_input_exit"


def test_allowlist_rejects_unknown_attr_key():
    from opentelemetry.proto.trace.v1.trace_pb2 import Span

    span = Span()
    with pytest.raises(AllowlistError):
        _add_attr(span, "gen_ai.content.prompt", "secret")


def test_allowlist_rejects_oversize_string():
    from opentelemetry.proto.trace.v1.trace_pb2 import Span

    span = Span()
    with pytest.raises(AllowlistError):
        _add_attr(span, "gen_ai.request.model", "x" * 1024)


def test_compact_span_partial_vs_enrichment():
    partial = build_compact_span("s1", "manual", False)
    assert _get_attr(partial, "thrum.event") == "compact"
    assert _get_attr(partial, "thrum.compact.trigger") == "manual"
    assert _get_attr(partial, "thrum.compact.has_custom_instructions") is False

    boundary = CompactBoundaryRecord(
        timestamp="2026-04-23T10:00:00Z",
        trigger="manual",
        pre_tokens=72714,
        post_tokens=3086,
        duration_ms=82199,
    )
    enriched = build_compact_span("s1", "manual", False, boundary=boundary)
    assert _get_attr(enriched, "thrum.compact.enrichment") is True
    assert _get_attr(enriched, "thrum.compact.pre_tokens") == 72714
    assert _get_attr(enriched, "thrum.compact.post_tokens") == 3086
    assert _get_attr(enriched, "thrum.compact.duration_ms") == 82199


def test_subagent_attribution_uses_parent_session_and_agent_name():
    buffer = {
        "session_id": "parent_s",
        "turn_start_ts": "2026-04-23T10:00:00Z",
        "tools_used": ["Grep"],
        "tools_failed": [],
        "tool_flags": {},
    }
    span = build_turn_span(
        buffer,
        TranscriptAggregate(),
        agent_name="Explore",
        agent_id="agent_1",
    )
    assert _get_attr(span, "session.id") == "parent_s"
    assert _get_attr(span, "gen_ai.agent.name") == "Explore"
    assert _get_attr(span, "thrum.agent_id") == "agent_1"


def test_end_ts_override_prevents_now_latency_on_backfill():
    """Backfill spans must not compute end = now(): the turn ended weeks ago.
    Passing `end_ts` pins end_time_unix_nano to the transcript's final record.
    """
    buffer = {
        "session_id": "s",
        "turn_start_ts": "2026-03-01T10:00:00Z",
        "tools_used": [],
        "tools_failed": [],
        "tool_flags": {},
    }
    span = build_turn_span(
        buffer, TranscriptAggregate(), end_ts="2026-03-01T10:00:05Z"
    )
    # End must be ~5 seconds after start, not time.time() - 52 days.
    delta_ns = span.end_time_unix_nano - span.start_time_unix_nano
    assert 0 <= delta_ns < 10 * 1_000_000_000  # under 10 seconds


def test_end_before_start_is_clamped_to_zero_latency():
    buffer = {
        "session_id": "s",
        "turn_start_ts": "2026-03-01T10:00:05Z",
        "tools_used": [],
        "tools_failed": [],
        "tool_flags": {},
    }
    span = build_turn_span(
        buffer, TranscriptAggregate(), end_ts="2026-03-01T10:00:00Z"
    )
    # end < start would produce negative latency; we pin to start instead.
    assert span.end_time_unix_nano == span.start_time_unix_nano


def test_post_span_retries_once_on_transient_http_error(tmp_path, monkeypatch):
    """Review #27: emit_* used to swallow httpx.HTTPError with a single
    POST attempt. A transient blip (RST, timeout) silently dropped the
    activity. _post_span now retries once with a short backoff.
    """
    import httpx

    from thrum_client.config import SkillSettings
    from thrum_client.emitter import emit_turn
    from thrum_client.parsers.transcript import TranscriptAggregate

    # A fake token file so _post_span has headers to send.
    cfg = tmp_path / ".config" / "thrum"
    cfg.mkdir(parents=True)
    (cfg / "token").write_text("tk_test")
    monkeypatch.setenv("THRUM_CONFIG_DIR", str(cfg))

    calls: list[int] = []

    class _FakeResp:
        status_code = 200

    def flaky_post(url, *, content, headers, timeout):
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ConnectError("transient", request=httpx.Request("POST", url))
        return _FakeResp()

    # Make the retry sleep a no-op so the test is instant.
    monkeypatch.setattr("thrum_client.emitter.time.sleep", lambda _s: None)

    buffer = {
        "session_id": "s1",
        "turn_start_ts": "2026-04-23T10:00:00Z",
        "tools_used": ["Read"],
        "tools_failed": [],
        "tool_flags": {},
    }
    settings = SkillSettings()
    result = emit_turn(
        buffer, settings, agg=TranscriptAggregate(), http_post=flaky_post
    )
    assert len(calls) == 2
    assert result.status == 200


def test_allowlist_matches_findings_set():
    # Guardrail: if someone adds an attribute to the builder without
    # updating this expected set, the test fails — forcing the diff to
    # be reviewed deliberately.
    expected = {
        "session.id",
        "thrum.message_ids",
        "thrum.agent_id",
        "gen_ai.system",
        "gen_ai.request.model",
        "thrum.source_tool",
        "thrum.token_source",
        "gen_ai.usage.input_tokens",
        "gen_ai.usage.output_tokens",
        "thrum.usage.cache_read_input_tokens",
        "thrum.usage.cache_creation_input_tokens",
        "thrum.usage.cache_creation_1h",
        "thrum.usage.cache_creation_5m",
        "thrum.usage.server_tool.web_search_requests",
        "thrum.usage.server_tool.web_fetch_requests",
        "thrum.metadata.stop_reason",
        "thrum.metadata.service_tier",
        "thrum.metadata.speed",
        "thrum.metadata.inference_geo",
        "thrum.tools_used",
        "thrum.tools_failed",
        "thrum.bash.categories",
        "thrum.metadata.interrupted",
        "thrum.metadata.is_image",
        "thrum.metadata.backfill",
        "thrum.metadata.forced_flush",
        "thrum.metadata.session_end_reason",
        "gen_ai.agent.name",
        "thrum.event",
        "thrum.compact.trigger",
        "thrum.compact.has_custom_instructions",
        "thrum.compact.pre_tokens",
        "thrum.compact.post_tokens",
        "thrum.compact.duration_ms",
        "thrum.compact.enrichment",
        "thrum.content_stripped",
        "thrum.token_source",
        # FR-215c — plan/task auto-capture
        "thrum.plan.id",
        "thrum.plan.completed_at",
        "thrum.metadata.codex_turn_id",
        "thrum.metadata.codex_originator",
        "thrum.metadata.reasoning_output_tokens",
        "thrum.metadata.cursor_conversation_id",
        "thrum.metadata.cursor_generation_id",
        "thrum.metadata.cursor_loop_count",
        "thrum.metadata.cursor_workspace_roots",
        "thrum.metadata.cursor_version",
        "thrum.metadata.cursor_session_end_reason",
        # FR-700 / FR-705 + FR-707 — group attribution inputs.
        "thrum.metadata.cwd",
        "thrum.personal",
    }
    assert ALLOWED_ATTRIBUTE_KEYS == expected


def test_turn_span_cursor_system_and_source_tool():
    """FR-218f — Cursor callers pass cursor/cursor; the builder must honour
    the kwargs without leaking the anthropic/claude-code or openai/codex-cli
    defaults."""
    buffer = {
        "session_id": "cursor_s1",
        "turn_start_ts": "2026-05-01T19:23:00Z",
        "tools_used": ["Read", "Grep"],
        "tools_failed": [],
        "tool_flags": {},
    }
    agg = TranscriptAggregate(tokens_in=31417, tokens_out=233, model="default")
    span = build_turn_span(
        buffer,
        agg,
        gen_ai_system="cursor",
        source_tool="cursor",
    )
    assert _get_attr(span, "gen_ai.system") == "cursor"
    assert _get_attr(span, "thrum.source_tool") == "cursor"
    assert _get_attr(span, "gen_ai.request.model") == "default"


def test_no_anthropic_or_openai_leakage_when_cursor_kwargs_passed():
    """Sentinel: with cursor/cursor kwargs, neither anthropic/claude-code
    nor openai/codex-cli must appear on any string attribute."""
    buffer = {
        "session_id": "s",
        "turn_start_ts": "2026-05-01T19:23:00Z",
        "tools_used": [],
        "tools_failed": [],
        "tool_flags": {},
    }
    span = build_turn_span(
        buffer,
        TranscriptAggregate(model="default"),
        gen_ai_system="cursor",
        source_tool="cursor",
    )
    forbidden = ("anthropic", "claude-code", "openai", "codex-cli")
    for a in span.attributes:
        v = a.value
        if v.HasField("string_value"):
            for f in forbidden:
                assert f not in v.string_value, f"{a.key}: {v.string_value!r}"


def test_cursor_metadata_keys_emit_when_set():
    """FR-218f — every new Cursor metadata key allowlisted in B1 emits
    successfully via _add_attr (no AllowlistError)."""
    from opentelemetry.proto.trace.v1.trace_pb2 import Span

    span = Span()
    _add_attr(span, "thrum.metadata.cursor_conversation_id", "0d883a92-uuid")
    _add_attr(span, "thrum.metadata.cursor_generation_id", "1ed2f56a-uuid")
    _add_attr(span, "thrum.metadata.cursor_loop_count", 0)
    _add_attr(span, "thrum.metadata.cursor_workspace_roots", ["/Users/me/proj"])
    _add_attr(span, "thrum.metadata.cursor_version", "3.2.16")
    _add_attr(span, "thrum.metadata.cursor_session_end_reason", "user_close")
    assert len(span.attributes) == 6
