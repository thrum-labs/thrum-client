"""OTLP span emission.

The `build_*_span` functions are the single chokepoint for attribute
assembly (NFR-318). They accept only sanitized inputs (tokens, tool names,
enums, opaque IDs) — never a raw hook payload or transcript line. Any
attribute whose key is not in `ALLOWED_ATTRIBUTE_KEYS` raises `AllowlistError`;
the test suite plants sentinel strings in the buffer to assert nothing leaks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import httpx
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
)
from opentelemetry.proto.common.v1.common_pb2 import AnyValue, ArrayValue, KeyValue
from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans, ScopeSpans, Span

from thrum_client.config import SkillSettings
from thrum_client.parsers.transcript import CompactBoundaryRecord, TranscriptAggregate
from thrum_client.safe_log import safe_log


# Full allowlist of attribute keys for any span the skill ever emits.
# Anything else raises in build.
ALLOWED_ATTRIBUTE_KEYS: frozenset[str] = frozenset(
    {
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
        # FR-215c — plan/task auto-capture
        "thrum.plan.id",
        "thrum.plan.completed_at",
        # Codex-specific metadata (FR-218b). Wired in by the Codex handler path
        # (D1) but allowlisted here so the chokepoint stays a single source of
        # truth.
        "thrum.metadata.codex_turn_id",
        "thrum.metadata.codex_originator",
        "thrum.metadata.reasoning_output_tokens",
        # Cursor-specific metadata (FR-218f). Wired in by the Cursor handler
        # path. cursor_generation_id also feeds the backend's external_id
        # projection (`cursor|<conversation_id>|<generation_id>`).
        "thrum.metadata.cursor_conversation_id",
        "thrum.metadata.cursor_generation_id",
        "thrum.metadata.cursor_loop_count",
        "thrum.metadata.cursor_workspace_roots",
        "thrum.metadata.cursor_version",
        "thrum.metadata.cursor_session_end_reason",
        # FR-700 / FR-705 — cwd feeds the backend project_key resolver
        # (FR-706). Sent as a string. Cursor activities also carry it
        # via `cursor_workspace_roots[0]`; this attribute generalises
        # to Claude Code + Codex.
        "thrum.metadata.cwd",
        # FR-707 — `.thrum-personal` short-circuit. When True, the
        # backend forces `activities.group_ids = NULL`. The marker
        # text itself is never sent — only the boolean.
        "thrum.personal",
    }
)

_MAX_STRING_ATTR_LEN = 256  # defense-in-depth cap on any string attribute


class AllowlistError(ValueError):
    """Raised when a disallowed attribute key is passed to a span builder."""


def _to_any_value(v: Any) -> AnyValue:
    if isinstance(v, bool):
        return AnyValue(bool_value=v)
    if isinstance(v, int):
        return AnyValue(int_value=int(v))
    if isinstance(v, float):
        return AnyValue(double_value=float(v))
    if isinstance(v, str):
        return AnyValue(string_value=v)
    if isinstance(v, list):
        return AnyValue(
            array_value=ArrayValue(values=[_to_any_value(x) for x in v])
        )
    if v is None:
        return AnyValue()
    return AnyValue(string_value=str(v))


def _add_attr(span: Span, key: str, value: Any) -> None:
    if key not in ALLOWED_ATTRIBUTE_KEYS:
        raise AllowlistError(f"attribute {key!r} not in allowlist")
    if isinstance(value, str) and len(value) > _MAX_STRING_ATTR_LEN:
        raise AllowlistError(
            f"attribute {key!r} string longer than {_MAX_STRING_ATTR_LEN}"
        )
    span.attributes.append(KeyValue(key=key, value=_to_any_value(value)))


def _to_unix_nano(ts_iso: str | None) -> int:
    if not ts_iso:
        return int(time.time() * 1e9)
    from datetime import datetime
    return int(datetime.fromisoformat(ts_iso.replace("Z", "+00:00")).timestamp() * 1e9)


def build_turn_span(
    buffer: dict[str, Any],
    agg: TranscriptAggregate,
    *,
    backfill: bool = False,
    forced_flush: bool = False,
    session_end_reason: str | None = None,
    agent_name: str | None = None,
    agent_id: str | None = None,
    end_ts: str | None = None,
    plan_id: str | None = None,
    plan_completed_at: str | None = None,
    gen_ai_system: str = "anthropic",
    source_tool: str = "claude-code",
    token_source: str = "measured",
    codex_turn_id: str | None = None,
    codex_originator: str | None = None,
    reasoning_output_tokens: int = 0,
    cursor_conversation_id: str | None = None,
    cursor_generation_id: str | None = None,
    cursor_loop_count: int | None = None,
    cursor_workspace_roots: list[str] | None = None,
    cursor_version: str | None = None,
    cursor_session_end_reason: str | None = None,
    cwd: str | None = None,
    personal: bool = False,
) -> Span:
    """Build one span for a terminating Stop / SubagentStop / SessionEnd flush.

    `buffer` must be a turn or subagent sub-buffer. Every attribute passed here
    comes from the allowlist-sanitized parser or constants — never raw content.

    `gen_ai_system` / `source_tool` default to the Claude Code values for
    backwards compatibility; the Codex handler path (D1) passes
    `gen_ai_system="openai"` and `source_tool="codex-cli"` instead.

    FR-700 attribution attrs:
      - `cwd`: hook payload's cwd, feeds the backend project_key resolver
        (FR-706). String.
      - `personal`: True when `.thrum-personal` was found in the cwd
        ancestry. Backend forces `activities.group_ids = NULL` (FR-707).
        Defaults to False (the no-marker case is the common path).
    """
    span = Span()
    span.name = "thrum.turn"

    turn_start = (
        buffer.get("turn_start_ts")
        or buffer.get("start_ts")
        or buffer.get("created_at")
    )
    span.start_time_unix_nano = _to_unix_nano(turn_start)
    if end_ts:
        span.end_time_unix_nano = _to_unix_nano(end_ts)
    else:
        span.end_time_unix_nano = int(time.time() * 1e9)
    # Guard against end < start (clock skew, bad fixtures). Pin to start
    # so latency_ms is 0 rather than negative / overflowing on the backend.
    if span.end_time_unix_nano < span.start_time_unix_nano:
        span.end_time_unix_nano = span.start_time_unix_nano

    _add_attr(span, "gen_ai.system", gen_ai_system)
    _add_attr(span, "thrum.source_tool", source_tool)
    _add_attr(span, "thrum.token_source", token_source)
    _add_attr(span, "thrum.content_stripped", True)
    _add_attr(span, "session.id", str(buffer.get("session_id") or ""))

    if agg.model:
        _add_attr(span, "gen_ai.request.model", agg.model)
    _add_attr(span, "gen_ai.usage.input_tokens", int(agg.tokens_in))
    _add_attr(span, "gen_ai.usage.output_tokens", int(agg.tokens_out))
    _add_attr(
        span,
        "thrum.usage.cache_read_input_tokens",
        int(agg.cache_read_input_tokens),
    )
    _add_attr(
        span,
        "thrum.usage.cache_creation_input_tokens",
        int(agg.cache_creation_input_tokens),
    )
    if agg.cache_creation_1h:
        _add_attr(span, "thrum.usage.cache_creation_1h", int(agg.cache_creation_1h))
    if agg.cache_creation_5m:
        _add_attr(span, "thrum.usage.cache_creation_5m", int(agg.cache_creation_5m))
    if agg.server_tool_web_search_requests:
        _add_attr(
            span,
            "thrum.usage.server_tool.web_search_requests",
            int(agg.server_tool_web_search_requests),
        )
    if agg.server_tool_web_fetch_requests:
        _add_attr(
            span,
            "thrum.usage.server_tool.web_fetch_requests",
            int(agg.server_tool_web_fetch_requests),
        )
    if agg.stop_reason:
        _add_attr(span, "thrum.metadata.stop_reason", agg.stop_reason)
    if agg.service_tier:
        _add_attr(span, "thrum.metadata.service_tier", agg.service_tier)
    if agg.speed:
        _add_attr(span, "thrum.metadata.speed", agg.speed)
    if agg.inference_geo:
        _add_attr(span, "thrum.metadata.inference_geo", agg.inference_geo)
    if agg.message_ids:
        _add_attr(span, "thrum.message_ids", list(agg.message_ids))

    tools_used = list(buffer.get("tools_used", []))
    tools_failed = list(buffer.get("tools_failed", []))
    _add_attr(span, "thrum.tools_used", tools_used)
    _add_attr(span, "thrum.tools_failed", tools_failed)

    bash_categories = list(buffer.get("bash_categories", []))
    if bash_categories:
        _add_attr(span, "thrum.bash.categories", bash_categories)

    flags = buffer.get("tool_flags") or {}
    if flags.get("interrupted"):
        _add_attr(span, "thrum.metadata.interrupted", True)
    if flags.get("is_image"):
        _add_attr(span, "thrum.metadata.is_image", True)

    if backfill:
        _add_attr(span, "thrum.metadata.backfill", True)
    if forced_flush:
        _add_attr(span, "thrum.metadata.forced_flush", True)
    if session_end_reason:
        _add_attr(span, "thrum.metadata.session_end_reason", session_end_reason)
    if agent_name:
        _add_attr(span, "gen_ai.agent.name", agent_name)
    if agent_id:
        _add_attr(span, "thrum.agent_id", agent_id)

    if plan_id:
        _add_attr(span, "thrum.plan.id", plan_id)
    if plan_completed_at:
        _add_attr(span, "thrum.plan.completed_at", plan_completed_at)

    # Codex-specific metadata (FR-218b). Emitted only when the call site
    # supplies them — Claude turns leave these unset.
    if codex_turn_id:
        _add_attr(span, "thrum.metadata.codex_turn_id", codex_turn_id)
    if codex_originator:
        _add_attr(span, "thrum.metadata.codex_originator", codex_originator)
    if reasoning_output_tokens:
        _add_attr(
            span,
            "thrum.metadata.reasoning_output_tokens",
            int(reasoning_output_tokens),
        )

    # Cursor-specific metadata (FR-218f). Emitted only when the call site
    # supplies them — Claude / Codex turns leave these unset.
    if cursor_conversation_id:
        _add_attr(
            span, "thrum.metadata.cursor_conversation_id", cursor_conversation_id
        )
    if cursor_generation_id:
        _add_attr(
            span, "thrum.metadata.cursor_generation_id", cursor_generation_id
        )
    if cursor_loop_count is not None:
        _add_attr(span, "thrum.metadata.cursor_loop_count", int(cursor_loop_count))
    if cursor_workspace_roots:
        _add_attr(
            span, "thrum.metadata.cursor_workspace_roots", list(cursor_workspace_roots)
        )
    if cursor_version:
        _add_attr(span, "thrum.metadata.cursor_version", cursor_version)
    if cursor_session_end_reason:
        _add_attr(
            span,
            "thrum.metadata.cursor_session_end_reason",
            cursor_session_end_reason,
        )
    # FR-700 / FR-705 — cwd feeds the backend group resolver. Empty
    # cwds (rare; only when the host doesn't supply one) are skipped
    # so the backend's "no project_key → home_group fallback" path
    # naturally fires.
    if cwd:
        _add_attr(span, "thrum.metadata.cwd", cwd)
    # FR-707 — only emit when True; the False default is the common
    # case and would clutter every span.
    if personal:
        _add_attr(span, "thrum.personal", True)

    return span


def build_compact_span(
    session_id: str,
    trigger: str | None,
    has_custom_instructions: bool,
    *,
    boundary: CompactBoundaryRecord | None = None,
    gen_ai_system: str = "anthropic",
    source_tool: str = "claude-code",
    token_source: str = "measured",
) -> Span:
    span = Span()
    span.name = "thrum.compact"
    span.start_time_unix_nano = int(time.time() * 1e9)
    span.end_time_unix_nano = span.start_time_unix_nano

    _add_attr(span, "gen_ai.system", gen_ai_system)
    _add_attr(span, "thrum.source_tool", source_tool)
    _add_attr(span, "thrum.token_source", token_source)
    _add_attr(span, "thrum.content_stripped", True)
    _add_attr(span, "session.id", str(session_id))
    _add_attr(span, "thrum.event", "compact")
    if trigger:
        _add_attr(span, "thrum.compact.trigger", trigger)
    _add_attr(
        span, "thrum.compact.has_custom_instructions", bool(has_custom_instructions)
    )
    if boundary is not None:
        _add_attr(span, "thrum.compact.enrichment", True)
        if boundary.pre_tokens is not None:
            _add_attr(span, "thrum.compact.pre_tokens", int(boundary.pre_tokens))
        if boundary.post_tokens is not None:
            _add_attr(span, "thrum.compact.post_tokens", int(boundary.post_tokens))
        if boundary.duration_ms is not None:
            _add_attr(span, "thrum.compact.duration_ms", int(boundary.duration_ms))
    return span


@dataclass
class EmitResult:
    status: int
    body_bytes: int


def _envelope(span: Span) -> bytes:
    req = ExportTraceServiceRequest()
    rs = req.resource_spans.add()
    ss = rs.scope_spans.add()
    ss.spans.append(span)
    return req.SerializeToString()


def _read_token(settings: SkillSettings) -> str:
    if not settings.token_path.exists():
        return ""
    return settings.token_path.read_text().strip()


_EMIT_RETRY_BACKOFF_S = 0.1


def _post_span(
    body: bytes,
    settings: SkillSettings,
    http_post: Callable[..., httpx.Response] | None = None,
) -> EmitResult:
    """POST with one bounded retry on transient HTTP errors.

    Hooks block Claude Code's UI, so we cap at a single 100 ms-delayed
    retry — enough to ride out a collector hiccup or a stray connection
    reset without making the user wait. After that we drop and let the
    caller log the failure.
    """
    headers = {
        "content-type": "application/x-protobuf",
        "x-api-key": _read_token(settings),
    }
    post = http_post if http_post is not None else httpx.post

    for attempt in range(2):
        try:
            resp = post(
                settings.collector_url,
                content=body,
                headers=headers,
                timeout=5.0,
            )
            return EmitResult(status=resp.status_code, body_bytes=len(body))
        except httpx.HTTPError:
            if attempt == 0:
                time.sleep(_EMIT_RETRY_BACKOFF_S)
                continue
            raise
    # Unreachable; the loop either returns or raises.
    return EmitResult(status=0, body_bytes=0)


def emit_turn(
    buffer: dict[str, Any],
    settings: SkillSettings,
    *,
    agg: TranscriptAggregate | None = None,
    backfill: bool = False,
    forced_flush: bool = False,
    session_end_reason: str | None = None,
    end_ts: str | None = None,
    plan_id: str | None = None,
    plan_completed_at: str | None = None,
    http_post: Callable[..., httpx.Response] | None = None,
    gen_ai_system: str = "anthropic",
    source_tool: str = "claude-code",
    token_source: str = "measured",
    codex_turn_id: str | None = None,
    codex_originator: str | None = None,
    reasoning_output_tokens: int = 0,
    cursor_conversation_id: str | None = None,
    cursor_generation_id: str | None = None,
    cursor_loop_count: int | None = None,
    cursor_workspace_roots: list[str] | None = None,
    cursor_version: str | None = None,
    cursor_session_end_reason: str | None = None,
    cwd: str | None = None,
    personal: bool = False,
) -> EmitResult:
    try:
        effective_agg = agg or TranscriptAggregate()
        span = build_turn_span(
            buffer,
            effective_agg,
            backfill=backfill,
            forced_flush=forced_flush,
            session_end_reason=session_end_reason,
            end_ts=end_ts,
            plan_id=plan_id,
            plan_completed_at=plan_completed_at,
            gen_ai_system=gen_ai_system,
            source_tool=source_tool,
            token_source=token_source,
            codex_turn_id=codex_turn_id,
            codex_originator=codex_originator,
            reasoning_output_tokens=reasoning_output_tokens,
            cursor_conversation_id=cursor_conversation_id,
            cursor_generation_id=cursor_generation_id,
            cursor_loop_count=cursor_loop_count,
            cursor_workspace_roots=cursor_workspace_roots,
            cursor_version=cursor_version,
            cursor_session_end_reason=cursor_session_end_reason,
            cwd=cwd,
            personal=personal,
        )
        body = _envelope(span)
        return _post_span(body, settings, http_post=http_post)
    except (AllowlistError, httpx.HTTPError) as exc:
        safe_log(
            "emit_turn_error",
            log_path=settings.log_path,
            session_id=buffer.get("session_id"),
            error_category=type(exc).__name__,
        )
        return EmitResult(status=0, body_bytes=0)


def emit_subagent(
    sub_buffer: dict[str, Any],
    parent_session_id: str,
    agent_id: str,
    settings: SkillSettings,
    *,
    agg: TranscriptAggregate | None = None,
    end_ts: str | None = None,
    http_post: Callable[..., httpx.Response] | None = None,
    gen_ai_system: str = "anthropic",
    source_tool: str = "claude-code",
    token_source: str = "measured",
) -> EmitResult:
    # For the builder, we want session.id = parent; reuse the same builder.
    effective_agg = agg or TranscriptAggregate()
    # Clone the sub-buffer with the parent session id so the span gets the
    # right session attribution (one of FR-228's requirements).
    buffer_view = dict(sub_buffer)
    buffer_view["session_id"] = parent_session_id
    try:
        span = build_turn_span(
            buffer_view,
            effective_agg,
            agent_name=sub_buffer.get("agent_type") or "subagent",
            agent_id=agent_id,
            end_ts=end_ts,
            gen_ai_system=gen_ai_system,
            source_tool=source_tool,
            token_source=token_source,
        )
        body = _envelope(span)
        return _post_span(body, settings, http_post=http_post)
    except (AllowlistError, httpx.HTTPError) as exc:
        safe_log(
            "emit_subagent_error",
            log_path=settings.log_path,
            session_id=parent_session_id,
            agent_id=agent_id,
            error_category=type(exc).__name__,
        )
        return EmitResult(status=0, body_bytes=0)


def emit_compact(
    session_id: str,
    trigger: str | None,
    has_custom_instructions: bool,
    settings: SkillSettings,
    *,
    boundary: CompactBoundaryRecord | None = None,
    http_post: Callable[..., httpx.Response] | None = None,
    gen_ai_system: str = "anthropic",
    source_tool: str = "claude-code",
    token_source: str = "measured",
) -> EmitResult:
    try:
        span = build_compact_span(
            session_id=session_id,
            trigger=trigger,
            has_custom_instructions=has_custom_instructions,
            boundary=boundary,
            gen_ai_system=gen_ai_system,
            source_tool=source_tool,
            token_source=token_source,
        )
        body = _envelope(span)
        return _post_span(body, settings, http_post=http_post)
    except (AllowlistError, httpx.HTTPError) as exc:
        safe_log(
            "emit_compact_error",
            log_path=settings.log_path,
            session_id=session_id,
            error_category=type(exc).__name__,
        )
        return EmitResult(status=0, body_bytes=0)


def emit_session_end_flush(
    session_buffer: dict[str, Any],
    settings: SkillSettings,
    *,
    agg: TranscriptAggregate | None = None,
    session_end_reason: str | None = None,
    end_ts: str | None = None,
    plan_id: str | None = None,
    plan_completed_at: str | None = None,
    http_post: Callable[..., httpx.Response] | None = None,
    gen_ai_system: str = "anthropic",
    source_tool: str = "claude-code",
    token_source: str = "measured",
) -> EmitResult:
    turn = session_buffer.get("turn") or {}
    view = {
        "session_id": session_buffer.get("session_id"),
        "turn_start_ts": turn.get("turn_start_ts"),
        "tools_used": turn.get("tools_used", []),
        "tools_failed": turn.get("tools_failed", []),
        "tool_flags": turn.get("tool_flags", {}),
        "bash_categories": turn.get("bash_categories", []),
    }
    return emit_turn(
        view,
        settings,
        agg=agg,
        forced_flush=True,
        session_end_reason=session_end_reason,
        end_ts=end_ts,
        plan_id=plan_id,
        plan_completed_at=plan_completed_at,
        http_post=http_post,
        gen_ai_system=gen_ai_system,
        source_tool=source_tool,
        token_source=token_source,
    )


def turn_view(session_buffer: dict[str, Any]) -> dict[str, Any]:
    """Flatten a session buffer + its open turn into the dict shape emit_turn expects."""
    turn = session_buffer.get("turn") or {}
    return {
        "session_id": session_buffer.get("session_id"),
        "turn_start_ts": turn.get("turn_start_ts"),
        "tools_used": turn.get("tools_used", []),
        "tools_failed": turn.get("tools_failed", []),
        "tool_flags": turn.get("tool_flags", {}),
        "bash_categories": turn.get("bash_categories", []),
    }


def subagent_view(sub_buffer: dict[str, Any]) -> dict[str, Any]:
    return {
        "turn_start_ts": sub_buffer.get("start_ts"),
        "tools_used": sub_buffer.get("tools_used", []),
        "tools_failed": sub_buffer.get("tools_failed", []),
        "tool_flags": sub_buffer.get("tool_flags", {}),
        "agent_type": sub_buffer.get("agent_type"),
        "bash_categories": sub_buffer.get("bash_categories", []),
    }
