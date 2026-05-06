"""Transcript JSONL streaming parser — NFR-319 + FR-221 dedupe.

Claude Code persists each session as `~/.claude/projects/<hash>/<session>.jsonl`.
A single API response appears as multiple `assistant` records (one per content
block — thinking / text / tool_use), each carrying the identical `usage` block.
Per FR-221 we dedupe by `message.id` before summing.

Content keys (`message.content[].text`, `.thinking`, `.input`, `.result`, the
top-level `content` on user records) are NEVER bound to a Python variable —
`ijson.parse` yields tokens we ignore.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import ijson


TRANSCRIPT_ALLOWED_PATHS: frozenset[str] = frozenset(
    {
        "type",
        "uuid",
        "parentUuid",
        "timestamp",
        "sessionId",
        "isSidechain",
        "requestId",
        "message.id",
        "message.model",
        "message.stop_reason",
        "message.usage.input_tokens",
        "message.usage.output_tokens",
        "message.usage.cache_creation_input_tokens",
        "message.usage.cache_read_input_tokens",
        "message.usage.cache_creation.ephemeral_1h_input_tokens",
        "message.usage.cache_creation.ephemeral_5m_input_tokens",
        "message.usage.server_tool_use.web_search_requests",
        "message.usage.server_tool_use.web_fetch_requests",
        "message.usage.service_tier",
        "message.usage.speed",
        "message.usage.inference_geo",
        # compact_boundary enrichment — applicable only when type=="system"
        "subtype",
        "compactMetadata.trigger",
        "compactMetadata.preTokens",
        "compactMetadata.postTokens",
        "compactMetadata.durationMs",
    }
)

_SCALAR_EVENTS: frozenset[str] = frozenset({"string", "number", "boolean", "null"})


@dataclass(frozen=True)
class AssistantRecord:
    timestamp: str
    message_id: str
    model: str | None
    stop_reason: str | None
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    cache_creation_1h: int
    cache_creation_5m: int
    server_tool_web_search_requests: int
    server_tool_web_fetch_requests: int
    service_tier: str | None
    speed: str | None
    inference_geo: str | None


@dataclass(frozen=True)
class CompactBoundaryRecord:
    timestamp: str
    trigger: str | None
    pre_tokens: int | None
    post_tokens: int | None
    duration_ms: int | None


@dataclass
class TranscriptAggregate:
    tokens_in: int = 0
    tokens_out: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_creation_1h: int = 0
    cache_creation_5m: int = 0
    server_tool_web_search_requests: int = 0
    server_tool_web_fetch_requests: int = 0
    model: str | None = None
    stop_reason: str | None = None
    service_tier: str | None = None
    speed: str | None = None
    inference_geo: str | None = None
    message_ids: list[str] = field(default_factory=list)


def _extract_line(raw: bytes) -> dict[str, Any]:
    captured: dict[str, Any] = {}
    for prefix, event, value in ijson.parse(io.BytesIO(raw)):
        if event in _SCALAR_EVENTS and prefix in TRANSCRIPT_ALLOWED_PATHS:
            captured[prefix] = value
    return captured


def _to_int(v: Any) -> int:
    if v is None:
        return 0
    if isinstance(v, bool):
        return int(v)
    return int(v)


def _coerce_assistant(row: dict[str, Any]) -> AssistantRecord | None:
    if row.get("type") != "assistant":
        return None
    mid = row.get("message.id")
    ts = row.get("timestamp")
    if not mid or not ts:
        return None
    return AssistantRecord(
        timestamp=ts,
        message_id=mid,
        model=row.get("message.model"),
        stop_reason=row.get("message.stop_reason"),
        input_tokens=_to_int(row.get("message.usage.input_tokens")),
        output_tokens=_to_int(row.get("message.usage.output_tokens")),
        cache_creation_input_tokens=_to_int(
            row.get("message.usage.cache_creation_input_tokens")
        ),
        cache_read_input_tokens=_to_int(
            row.get("message.usage.cache_read_input_tokens")
        ),
        cache_creation_1h=_to_int(
            row.get("message.usage.cache_creation.ephemeral_1h_input_tokens")
        ),
        cache_creation_5m=_to_int(
            row.get("message.usage.cache_creation.ephemeral_5m_input_tokens")
        ),
        server_tool_web_search_requests=_to_int(
            row.get("message.usage.server_tool_use.web_search_requests")
        ),
        server_tool_web_fetch_requests=_to_int(
            row.get("message.usage.server_tool_use.web_fetch_requests")
        ),
        service_tier=row.get("message.usage.service_tier"),
        speed=row.get("message.usage.speed"),
        inference_geo=row.get("message.usage.inference_geo"),
    )


def _coerce_compact(row: dict[str, Any]) -> CompactBoundaryRecord | None:
    if row.get("type") != "system" or row.get("subtype") != "compact_boundary":
        return None
    return CompactBoundaryRecord(
        timestamp=row.get("timestamp") or "",
        trigger=row.get("compactMetadata.trigger"),
        pre_tokens=(
            int(row["compactMetadata.preTokens"])
            if row.get("compactMetadata.preTokens") is not None
            else None
        ),
        post_tokens=(
            int(row["compactMetadata.postTokens"])
            if row.get("compactMetadata.postTokens") is not None
            else None
        ),
        duration_ms=(
            int(row["compactMetadata.durationMs"])
            if row.get("compactMetadata.durationMs") is not None
            else None
        ),
    )


def iter_records(path: Path) -> Iterable[dict[str, Any]]:
    """Yield one extracted row per JSONL line. Blank lines and parse errors skipped."""
    with path.open("rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield _extract_line(line)
            except (ijson.JSONError, ijson.IncompleteJSONError):
                continue


def iter_assistant_records(path: Path) -> Iterable[AssistantRecord]:
    for row in iter_records(path):
        rec = _coerce_assistant(row)
        if rec is not None:
            yield rec


def iter_compact_boundaries(path: Path) -> Iterable[CompactBoundaryRecord]:
    for row in iter_records(path):
        rec = _coerce_compact(row)
        if rec is not None:
            yield rec


def aggregate_turn(
    path: Path,
    turn_start_ts: str | None,
    turn_end_ts: str | None,
) -> TranscriptAggregate:
    """Dedupe assistant records by message.id, sum usage, pick per-turn model.

    Records with `type != "assistant"` are ignored. `compact_boundary` rows
    have `type == "system"` so they're implicitly excluded.
    """
    seen: dict[str, AssistantRecord] = {}
    # Remember insertion order so we can pick the "last by timestamp" fallback.
    order: list[str] = []
    for rec in iter_assistant_records(path):
        if turn_start_ts is not None and rec.timestamp < turn_start_ts:
            continue
        if turn_end_ts is not None and rec.timestamp > turn_end_ts:
            continue
        if rec.message_id in seen:
            continue
        seen[rec.message_id] = rec
        order.append(rec.message_id)

    agg = TranscriptAggregate()
    for mid in order:
        r = seen[mid]
        agg.tokens_in += r.input_tokens
        agg.tokens_out += r.output_tokens
        agg.cache_read_input_tokens += r.cache_read_input_tokens
        agg.cache_creation_input_tokens += r.cache_creation_input_tokens
        agg.cache_creation_1h += r.cache_creation_1h
        agg.cache_creation_5m += r.cache_creation_5m
        agg.server_tool_web_search_requests += r.server_tool_web_search_requests
        agg.server_tool_web_fetch_requests += r.server_tool_web_fetch_requests
        agg.message_ids.append(mid)

    # Per-turn model: end_turn record if present, else last by timestamp.
    end_turn = next(
        (seen[m] for m in order if seen[m].stop_reason == "end_turn"), None
    )
    picked = end_turn
    if picked is None and order:
        picked = max(seen.values(), key=lambda r: r.timestamp)
    if picked is not None:
        agg.model = picked.model
        agg.stop_reason = picked.stop_reason
        agg.service_tier = picked.service_tier
        agg.speed = picked.speed
        agg.inference_geo = picked.inference_geo
    return agg


def aggregate_subagent(path: Path) -> TranscriptAggregate:
    """FR-228: a subagent transcript is scoped to one invocation, no time filter."""
    return aggregate_turn(path, turn_start_ts=None, turn_end_ts=None)
