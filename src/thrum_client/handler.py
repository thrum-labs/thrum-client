"""`thrum-hook` — per-event hook handler.

Called by Claude Code once per registered event with the JSON payload on
stdin. Order of operations:

1. Parse stdin with the NFR-319 allowlist.
2. Opt-out check (FR-227) — walk up from payload.cwd looking for
   `.thrum-disable`. Short-circuit before any buffer write, transcript
   read, or network call.
3. Load (or create) the per-session turn buffer. Schema-reject on
   unknown keys.
4. Dispatch by `hook_event_name` — updates the buffer, and for certain
   events (terminating Stop, SubagentStop, PreCompact, SessionEnd) calls
   into the emitter.
5. Persist the buffer (or delete it) and exit.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import ijson

from thrum_client.buffer import (
    BufferError,
    CLAUDE_SOURCE_TOOL,
    CODEX_SOURCE_TOOL,
    CURSOR_SOURCE_TOOL,
    buffer_lock,
    delete_buffer,
    load_buffer,
    new_buffer,
    save_buffer,
)
from thrum_client.config import SkillSettings, load_settings
from thrum_client.emitter import (
    emit_compact,
    emit_session_end_flush,
    emit_subagent,
    emit_turn,
    subagent_view,
    turn_view,
)
from thrum_client.opt_out import has_disable_marker, has_personal_marker
from thrum_client.parsers.codex_hook import (
    CodexHookEvent,
    extract_codex_hook_event,
)
from thrum_client.parsers.codex_plan_detector import (
    CodexPlanAttribution,
    detect_plan_for_turn as detect_codex_plan_for_turn,
)
from thrum_client.parsers.codex_rollout import (
    CodexTurnAggregate,
    iter_codex_turns,
    read_session_meta,
    read_turn,
)
from thrum_client.parsers.cursor_hook import (
    CursorHookEvent,
    extract_cursor_hook_event,
)
from thrum_client.parsers.cursor_plan_detector import (
    CursorPlanAttribution,
    detect_plan_for_generation as detect_cursor_plan_for_generation,
)
from thrum_client.parsers.cursor_transcript import (
    count_generations as count_cursor_generations,
    read_turn as read_cursor_turn,
)
from thrum_client.parsers.hook import HookEvent, extract_hook_event
from thrum_client.parsers.plan_detector import (
    PlanAttribution,
    detect_plan_for_turn,
)
from thrum_client.parsers.transcript import (
    CompactBoundaryRecord,
    TranscriptAggregate,
    aggregate_subagent,
    aggregate_turn,
    iter_compact_boundaries,
)
from thrum_client.safe_log import safe_log


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _new_turn() -> dict[str, Any]:
    return {
        "turn_start_ts": _now_iso(),
        "tools_used": [],
        "tools_failed": [],
        "tool_use_id_map": {},
        "tool_flags": {"interrupted": False, "is_image": False},
        "bash_categories": [],
    }


def _new_subagent(event: HookEvent) -> dict[str, Any]:
    return {
        "agent_type": event.agent_type,
        "cwd": event.cwd,
        "agent_transcript_path": event.agent_transcript_path,
        "start_ts": _now_iso(),
        "tools_used": [],
        "tools_failed": [],
        "tool_use_id_map": {},
        "tool_flags": {"interrupted": False, "is_image": False},
        "bash_categories": [],
    }


def _record_bash_category(scope: dict[str, Any], event: HookEvent) -> None:
    if not event.bash_category:
        return
    bucket = scope.setdefault("bash_categories", [])
    if event.bash_category not in bucket:
        bucket.append(event.bash_category)


def _ensure_buffer(settings: SkillSettings, event: HookEvent) -> dict[str, Any]:
    try:
        existing = load_buffer(settings.buffers_dir, event.session_id)
    except BufferError:
        safe_log(
            "buffer_schema_rejected",
            log_path=settings.log_path,
            session_id=event.session_id,
            error_category="BufferError",
        )
        delete_buffer(settings.buffers_dir, event.session_id)
        existing = None
    if existing is not None:
        return existing
    return new_buffer(
        event.session_id,
        event.cwd,
        transcript_path=event.transcript_path,
    )


def _scope_for_tool(
    buffer: dict[str, Any], event: HookEvent
) -> dict[str, Any]:
    if event.agent_id and event.agent_id in buffer["subagents"]:
        return buffer["subagents"][event.agent_id]
    if buffer["turn"] is None:
        buffer["turn"] = _new_turn()
    return buffer["turn"]


def _record_tool_flags(scope: dict[str, Any], event: HookEvent) -> None:
    if event.tool_response_interrupted:
        scope["tool_flags"]["interrupted"] = True
    if event.tool_response_is_image:
        scope["tool_flags"]["is_image"] = True


Persistence = Literal["save", "delete", "noop"]
Emit = Literal[
    "none", "turn", "subagent", "compact", "flush"
]


@dataclass
class Action:
    persist: Persistence
    emit: Emit = "none"
    agent_id: str | None = None


def dispatch(buffer: dict[str, Any], event: HookEvent) -> Action:
    name = event.hook_event_name

    if name == "SessionStart":
        buffer["model"] = event.model or buffer.get("model")
        return Action(persist="save")

    if name == "UserPromptSubmit":
        # Only open a fresh turn when the previous one has closed (Stop
        # wrote turn_end_ts or the buffer slot is empty). A nested
        # UserPromptSubmit on a live turn would otherwise wipe the
        # accumulated tools_used / tools_failed.
        current = buffer.get("turn")
        if current is None or current.get("turn_end_ts"):
            buffer["turn"] = _new_turn()
        if event.transcript_path:
            buffer["transcript_path"] = event.transcript_path
        return Action(persist="save")

    if name == "PreToolUse":
        scope = _scope_for_tool(buffer, event)
        if event.tool_use_id and event.tool_name:
            scope["tool_use_id_map"][event.tool_use_id] = event.tool_name
        _record_bash_category(scope, event)
        return Action(persist="save")

    if name == "PostToolUse":
        if not event.tool_name:
            return Action(persist="save")
        scope = _scope_for_tool(buffer, event)
        scope["tools_used"].append(event.tool_name)
        _record_tool_flags(scope, event)
        _record_bash_category(scope, event)
        return Action(persist="save")

    if name == "PostToolUseFailure":
        if not event.tool_name:
            return Action(persist="save")
        scope = _scope_for_tool(buffer, event)
        scope["tools_used"].append(event.tool_name)
        scope["tools_failed"].append(event.tool_name)
        _record_tool_flags(scope, event)
        _record_bash_category(scope, event)
        return Action(persist="save")

    if name == "SubagentStart":
        if event.agent_id:
            buffer["subagents"][event.agent_id] = _new_subagent(event)
        return Action(persist="save")

    if name == "SubagentStop":
        if event.agent_id and event.agent_id in buffer["subagents"]:
            sub = buffer["subagents"][event.agent_id]
            sub["stop_ts"] = _now_iso()
            if event.agent_transcript_path:
                sub["agent_transcript_path"] = event.agent_transcript_path
            return Action(persist="save", emit="subagent", agent_id=event.agent_id)
        return Action(persist="save")

    if name == "PreCompact":
        buffer["compact_pending"] = {
            "trigger": event.trigger,
            "has_custom_instructions": event.has_custom_instructions,
            "observed_at": _now_iso(),
            "last_scanned_at": None,
        }
        return Action(persist="save", emit="compact")

    if name in ("Stop", "StopFailure"):
        if event.stop_hook_active is True:
            return Action(persist="save")
        if buffer["turn"] is not None:
            buffer["turn"]["turn_end_ts"] = _now_iso()
        return Action(persist="delete", emit="turn")

    if name == "SessionEnd":
        if buffer["turn"] is not None:
            buffer["turn"]["session_end_reason"] = event.reason
            buffer["turn"]["forced_flush"] = True
            return Action(persist="delete", emit="flush")
        return Action(persist="delete")

    safe_log(
        "unknown_hook_event",
        log_path=None,
        hook_event_name=name,
    )
    return Action(persist="noop")


def _emit_for_turn(
    buffer: dict[str, Any],
    settings: SkillSettings,
) -> None:
    turn = buffer.get("turn") or {}
    end_ts = turn.get("turn_end_ts") or _now_iso()
    transcript_path = buffer.get("transcript_path")
    agg = TranscriptAggregate()
    plan: PlanAttribution | None = None
    if transcript_path and Path(transcript_path).exists():
        path = Path(transcript_path)
        agg = aggregate_turn(path, turn.get("turn_start_ts"), end_ts)
        # FR-215c — re-scan the transcript for plan boundaries. Cheap relative
        # to the turn aggregator (we ignore content, only TaskCreate/TaskUpdate
        # tool_use blocks). Each Stop replays the full session state machine
        # so plan_id stays deterministic across retries / backfill.
        session_id = str(buffer.get("session_id") or "")
        if session_id:
            plan = detect_plan_for_turn(
                path, session_id, turn.get("turn_start_ts"), end_ts
            )
    view = turn_view(buffer)
    cwd, personal = _attribution_attrs(buffer.get("cwd"))
    emit_turn(
        view,
        settings,
        agg=agg,
        end_ts=end_ts,
        plan_id=str(plan.plan_id) if plan else None,
        plan_completed_at=plan.completed_at if plan else None,
        cwd=cwd,
        personal=personal,
    )


def _attribution_attrs(cwd_str: str | None) -> tuple[str | None, bool]:
    """FR-700 / FR-707 — derive the (cwd, personal) span attrs from a
    raw cwd string. Returns (None, False) when cwd is missing /
    unresolvable so the backend resolver falls through to home_group.
    """
    if not cwd_str:
        return None, False
    try:
        path = Path(cwd_str)
    except (TypeError, ValueError):
        return None, False
    return cwd_str, has_personal_marker(path)


def _emit_for_subagent(
    buffer: dict[str, Any],
    settings: SkillSettings,
    agent_id: str,
) -> None:
    sub = buffer["subagents"].get(agent_id)
    if sub is None:
        return
    agent_transcript_path = sub.get("agent_transcript_path")
    agg = TranscriptAggregate()
    if agent_transcript_path and Path(agent_transcript_path).exists():
        agg = aggregate_subagent(Path(agent_transcript_path))
    view = subagent_view(sub)
    view["session_id"] = buffer.get("session_id")  # not used by builder, kept for symmetry
    emit_subagent(
        view,
        parent_session_id=str(buffer.get("session_id") or ""),
        agent_id=agent_id,
        settings=settings,
        agg=agg,
        end_ts=sub.get("stop_ts") or _now_iso(),
    )
    buffer["subagents"].pop(agent_id, None)


def _emit_for_compact(
    buffer: dict[str, Any],
    settings: SkillSettings,
) -> None:
    pending = buffer.get("compact_pending") or {}
    emit_compact(
        session_id=str(buffer.get("session_id") or ""),
        trigger=pending.get("trigger"),
        has_custom_instructions=bool(pending.get("has_custom_instructions")),
        settings=settings,
    )
    _scan_for_compact_enrichment(buffer, settings)


_COMPACT_PENDING_TTL = timedelta(minutes=10)


def _scan_for_compact_enrichment(
    buffer: dict[str, Any],
    settings: SkillSettings,
) -> None:
    """Look for new `compact_boundary` transcript rows and emit one enrichment
    span per hit. Records scanned once (we track `last_scanned_at`)."""
    pending = buffer.get("compact_pending") or {}
    transcript_path = buffer.get("transcript_path")
    if not transcript_path or not Path(transcript_path).exists():
        return
    last_scanned = pending.get("last_scanned_at")
    latest_ts = last_scanned
    for boundary in iter_compact_boundaries(Path(transcript_path)):
        if last_scanned is not None and boundary.timestamp <= last_scanned:
            continue
        emit_compact(
            session_id=str(buffer.get("session_id") or ""),
            trigger=boundary.trigger,
            has_custom_instructions=False,
            settings=settings,
            boundary=boundary,
        )
        if latest_ts is None or boundary.timestamp > latest_ts:
            latest_ts = boundary.timestamp
    if latest_ts is not None and latest_ts != last_scanned:
        pending["last_scanned_at"] = latest_ts
        buffer["compact_pending"] = pending


def _compact_pending_is_stale(pending: dict[str, Any]) -> bool:
    observed = pending.get("observed_at")
    if not isinstance(observed, str) or not observed:
        return True
    try:
        observed_dt = datetime.fromisoformat(observed.replace("Z", "+00:00"))
    except ValueError:
        return True
    return datetime.now(UTC) - observed_dt > _COMPACT_PENDING_TTL


def _resolve_compact_pending(
    buffer: dict[str, Any],
    settings: SkillSettings,
) -> None:
    """Try to enrich a pending PreCompact span from the live transcript.

    Called at every terminating Stop / SessionEnd. The
    compact_boundary row may not yet be in the transcript when PreCompact
    fires, so we retry on each subsequent transcript read. Pending state is
    cleared once enrichment emits or when the TTL expires.
    """
    pending = buffer.get("compact_pending")
    if not pending:
        return
    _scan_for_compact_enrichment(buffer, settings)
    pending = buffer.get("compact_pending") or {}
    if pending.get("last_scanned_at") is not None:
        buffer["compact_pending"] = None
        return
    if _compact_pending_is_stale(pending):
        safe_log(
            "compact_enrichment_missing",
            log_path=settings.log_path,
            session_id=buffer.get("session_id"),
        )
        buffer["compact_pending"] = None


def _emit_for_flush(
    buffer: dict[str, Any],
    settings: SkillSettings,
) -> None:
    turn = buffer.get("turn") or {}
    reason = turn.get("session_end_reason")
    end_ts = _now_iso()
    transcript_path = buffer.get("transcript_path")
    agg = TranscriptAggregate()
    plan: PlanAttribution | None = None
    if transcript_path and Path(transcript_path).exists():
        path = Path(transcript_path)
        agg = aggregate_turn(path, turn.get("turn_start_ts"), end_ts)
        session_id = str(buffer.get("session_id") or "")
        if session_id:
            plan = detect_plan_for_turn(
                path, session_id, turn.get("turn_start_ts"), end_ts
            )
    emit_session_end_flush(
        buffer,
        settings,
        agg=agg,
        session_end_reason=reason,
        end_ts=end_ts,
        plan_id=str(plan.plan_id) if plan else None,
        plan_completed_at=plan.completed_at if plan else None,
    )


_PARSE_ERRORS: tuple[type[Exception], ...] = (
    ValueError,
    ijson.JSONError,
    ijson.IncompleteJSONError,
)


def handle_event(raw: bytes, settings: SkillSettings) -> int:
    try:
        event = extract_hook_event(raw)
    except _PARSE_ERRORS as exc:
        safe_log(
            "parse_error",
            log_path=settings.log_path,
            error_category=type(exc).__name__,
        )
        return 0

    if has_disable_marker(Path(event.cwd)):
        return 0

    with buffer_lock(settings.buffers_dir, event.session_id):
        buffer = _ensure_buffer(settings, event)
        action = dispatch(buffer, event)

        if action.emit == "turn":
            _emit_for_turn(buffer, settings)
            _resolve_compact_pending(buffer, settings)
            # If the compact_boundary row hasn't landed in the transcript yet,
            # keep the buffer alive so the next Stop can retry.
            # Clear the emitted turn so a fresh UserPromptSubmit opens a new one.
            if action.persist == "delete" and buffer.get("compact_pending"):
                buffer["turn"] = None
                action = Action(persist="save", emit=action.emit)
        elif action.emit == "subagent" and action.agent_id:
            save_buffer(settings.buffers_dir, buffer)
            _emit_for_subagent(buffer, settings, action.agent_id)
        elif action.emit == "compact":
            save_buffer(settings.buffers_dir, buffer)
            _emit_for_compact(buffer, settings)
        elif action.emit == "flush":
            _emit_for_flush(buffer, settings)
            _resolve_compact_pending(buffer, settings)

        if action.persist == "save":
            save_buffer(settings.buffers_dir, buffer)
        elif action.persist == "delete":
            delete_buffer(settings.buffers_dir, event.session_id)
    return 0


# ---- Codex CLI dispatch path (FR-218b) ----


def _detect_source_tool(raw: bytes) -> str:
    """Identify whether the stdin payload is from Claude Code, Codex CLI,
    or Cursor.

    Three signals, scanned in one ijson pass (early exit on the strongest):

    1. `cursor_version` — present on every Cursor hook payload (verified
       empirically against Cursor 3.2.16 captures). The strongest signal
       because Claude / Codex never emit it.
    2. `turn_id` — present on every Codex hook payload EXCEPT SessionStart.
       Claude payloads never carry it.
    3. `transcript_path` — substring fallback for events that lack the
       above (Codex SessionStart). `/.codex/sessions/` → Codex,
       `/.cursor/projects/` → Cursor, anything else → Claude default.

    Defaults to Claude on parse error or when no signal is present.
    """
    import io as _io

    transcript_path: str | None = None
    try:
        for prefix, event, value in ijson.parse(_io.BytesIO(raw)):
            if prefix == "cursor_version" and event == "string" and value:
                return CURSOR_SOURCE_TOOL
            if prefix == "turn_id" and event in {"string", "number"} and value:
                return CODEX_SOURCE_TOOL
            if prefix == "transcript_path" and event == "string":
                transcript_path = value
    except (ijson.JSONError, ijson.IncompleteJSONError):
        return CLAUDE_SOURCE_TOOL
    if transcript_path:
        if "/.codex/sessions/" in transcript_path:
            return CODEX_SOURCE_TOOL
        if "/.cursor/projects/" in transcript_path:
            return CURSOR_SOURCE_TOOL
    return CLAUDE_SOURCE_TOOL


def _ensure_codex_buffer(
    settings: SkillSettings, event: CodexHookEvent
) -> dict[str, Any]:
    try:
        existing = load_buffer(settings.buffers_dir, event.session_id)
    except BufferError:
        safe_log(
            "buffer_schema_rejected",
            log_path=settings.log_path,
            session_id=event.session_id,
            error_category="BufferError",
        )
        delete_buffer(settings.buffers_dir, event.session_id)
        existing = None
    if existing is not None:
        # The buffer might have been created by Claude on a previous run with
        # the same session_id — unusual but defensible. Trust the source_tool
        # already on disk, fall through.
        return existing
    return new_buffer(
        event.session_id,
        event.cwd,
        transcript_path=event.transcript_path,
        source_tool=CODEX_SOURCE_TOOL,
        turn_id=event.turn_id,
    )


def _new_codex_turn() -> dict[str, Any]:
    return {
        "turn_start_ts": _now_iso(),
        # Populated from CodexTurnAggregate.tool_intents on Stop emission;
        # cleared here so a stale list from the previous turn doesn't leak.
        "tools_used": [],
        "tools_failed": [],
        "tool_use_id_map": {},
        "tool_flags": {"interrupted": False, "is_image": False},
        "bash_categories": [],
    }


def _dispatch_codex(
    buffer: dict[str, Any], event: CodexHookEvent
) -> Action:
    """Six events: SessionStart, UserPromptSubmit, PreToolUse, PostToolUse,
    PermissionRequest, Stop. No SubagentStop/PreCompact/SessionEnd."""
    name = event.hook_event_name

    if name == "SessionStart":
        if event.model:
            buffer["model"] = event.model
        if event.turn_id:
            buffer["turn_id"] = event.turn_id
        if event.transcript_path:
            buffer["transcript_path"] = event.transcript_path
        return Action(persist="save")

    if name == "UserPromptSubmit":
        current = buffer.get("turn")
        if current is None or current.get("turn_end_ts"):
            buffer["turn"] = _new_codex_turn()
        if event.turn_id:
            buffer["turn_id"] = event.turn_id
        if event.transcript_path:
            buffer["transcript_path"] = event.transcript_path
        return Action(persist="save")

    if name == "PreToolUse":
        if buffer.get("turn") is None:
            buffer["turn"] = _new_codex_turn()
        if event.tool_use_id and event.tool_name:
            buffer["turn"]["tool_use_id_map"][event.tool_use_id] = event.tool_name
        return Action(persist="save")

    if name == "PostToolUse":
        # On Codex, the per-call tool_name is generic (`exec_command`,
        # `write_stdin`). The canonical-intent vocabulary is filled at Stop
        # from the rollout's parsed_cmd[].type — see _emit_for_codex_turn.
        # We still record the raw name here so a partial picture is available
        # if the rollout read fails.
        if buffer.get("turn") is None:
            buffer["turn"] = _new_codex_turn()
        if event.tool_name:
            buffer["turn"]["tools_used"].append(event.tool_name)
        return Action(persist="save")

    if name == "PermissionRequest":
        # No span emitted; recorded as a flag for the next emission only.
        if buffer.get("turn") is not None:
            buffer["turn"]["tool_flags"]["permission_requested"] = True
        return Action(persist="save")

    if name == "Stop":
        if event.stop_hook_active is True:
            return Action(persist="save")
        if buffer.get("turn") is not None:
            buffer["turn"]["turn_end_ts"] = _now_iso()
        # Carry the live turn_id through to the emit step in case the buffer
        # was created from a SessionStart with a stale id.
        if event.turn_id:
            buffer["turn_id"] = event.turn_id
        return Action(persist="delete", emit="turn")

    safe_log(
        "unknown_codex_hook_event",
        log_path=None,
        hook_event_name=name,
    )
    return Action(persist="noop")


def _emit_for_codex_turn(
    buffer: dict[str, Any], settings: SkillSettings
) -> None:
    """Read the rollout JSONL, find the matching turn, emit one OTLP span.

    Tokens come directly from `info.last_token_usage` (per FR-218b — no
    cumulative-delta math). `tools_used` is the canonical intent list
    from the rollout parser (`edit`, `git-ops`, ...) so the backend
    classifier sees the same axis for Codex as for Claude. The raw
    `exec_command` / `write_stdin` names captured during PostToolUse are
    DROPPED here: they would project to `other` in the classifier and
    pollute the canonical signal — the rollout is the authoritative
    source for what happened in the turn.
    """
    # Materialise a turn dict on the buffer if absent (a bare Stop without
    # prior UserPromptSubmit) so later mutations actually persist via the
    # buffer reference rather than being lost on an orphan {} alias.
    if buffer.get("turn") is None:
        buffer["turn"] = _new_codex_turn()
    turn = buffer["turn"]
    end_ts = turn.get("turn_end_ts") or _now_iso()
    transcript_path = buffer.get("transcript_path")
    target_turn_id = buffer.get("turn_id") or ""

    agg = TranscriptAggregate()
    originator: str | None = None
    reasoning_tokens = 0
    matching: CodexTurnAggregate | None = None
    plan: CodexPlanAttribution | None = None

    if transcript_path and Path(transcript_path).exists():
        path = Path(transcript_path)
        meta = read_session_meta(path)
        if meta is not None:
            originator = meta.originator
        # Try the completed-turn iterator first. On a live Stop hook, Codex
        # often hasn't flushed `task_complete` to the rollout yet (the hook
        # fires synchronously, the JSONL append is async) — so completed
        # iteration commonly returns nothing for the closing turn. Fall back
        # to `read_turn`, which also returns in-progress state populated from
        # `task_started` + `turn_context` + the latest `token_count`. Without
        # this fallback every live Stop emits zeros.
        if target_turn_id:
            matching = read_turn(path, target_turn_id)
        else:
            for codex_agg in iter_codex_turns(path):
                matching = codex_agg
                break
        # FR-215c — Codex plan attribution. Mirrors the Claude path
        # (`_emit_for_turn`): re-scan the rollout for `update_plan` checklist
        # and Plan Mode `<proposed_plan>` boundaries. Cheap relative to the
        # turn aggregator. plan_id is deterministic across this live emit and
        # the FR-218b backfill so the backend dedupes idempotently.
        plan_session_id = (
            (meta.session_id if meta and meta.session_id else None)
            or str(buffer.get("session_id") or "")
        )
        if plan_session_id and target_turn_id:
            plan = detect_codex_plan_for_turn(
                path, plan_session_id, target_turn_id
            )

    if matching is not None:
        agg = TranscriptAggregate(
            tokens_in=matching.tokens_in,
            tokens_out=matching.tokens_out,
            cache_read_input_tokens=matching.cached_input_tokens,
            model=matching.model,
        )
        reasoning_tokens = matching.reasoning_output_tokens
        turn["tools_used"] = list(matching.tool_intents)
    else:
        # Rollout missing or turn not yet flushed (task_complete pending).
        # Clear any raw `exec_command` / `write_stdin` names so the backend
        # doesn't see a misleading partial signal — the missing rollout will
        # be picked up by the FR-218b backfill when it eventually lands.
        turn["tools_used"] = []

    view = turn_view(buffer)
    cwd, personal = _attribution_attrs(buffer.get("cwd"))
    emit_turn(
        view,
        settings,
        agg=agg,
        end_ts=end_ts,
        gen_ai_system="openai",
        source_tool=CODEX_SOURCE_TOOL,
        codex_turn_id=target_turn_id or None,
        codex_originator=originator,
        reasoning_output_tokens=reasoning_tokens,
        plan_id=str(plan.plan_id) if plan else None,
        plan_completed_at=plan.completed_at if plan else None,
        cwd=cwd,
        personal=personal,
    )


def _handle_codex_event(raw: bytes, settings: SkillSettings) -> int:
    try:
        event = extract_codex_hook_event(raw)
    except _PARSE_ERRORS as exc:
        safe_log(
            "parse_error",
            log_path=settings.log_path,
            error_category=type(exc).__name__,
            source_tool=CODEX_SOURCE_TOOL,
        )
        return 0

    if has_disable_marker(Path(event.cwd)):
        return 0

    with buffer_lock(settings.buffers_dir, event.session_id):
        buffer = _ensure_codex_buffer(settings, event)
        action = _dispatch_codex(buffer, event)

        if action.emit == "turn":
            _emit_for_codex_turn(buffer, settings)

        if action.persist == "save":
            save_buffer(settings.buffers_dir, buffer)
        elif action.persist == "delete":
            delete_buffer(settings.buffers_dir, event.session_id)
    return 0


def _ensure_cursor_buffer(
    settings: SkillSettings, event: CursorHookEvent
) -> dict[str, Any]:
    try:
        existing = load_buffer(settings.buffers_dir, event.session_id)
    except BufferError:
        safe_log(
            "buffer_schema_rejected",
            log_path=settings.log_path,
            session_id=event.session_id,
            error_category="BufferError",
        )
        delete_buffer(settings.buffers_dir, event.session_id)
        existing = None

    if existing is None:
        # cwd is not on Cursor's payload; use the first workspace_root as
        # the opt-out anchor (`.thrum-disable` is checked relative to cwd).
        cwd = event.workspace_roots[0] if event.workspace_roots else ""
        buf = new_buffer(
            event.session_id,
            cwd,
            model=event.model or None,
            transcript_path=event.transcript_path,
            source_tool=CURSOR_SOURCE_TOOL,
        )
        buf["cursor_generation_index"] = -1
    else:
        buf = existing

    # Refresh buffer-level fields from EVERY event payload — defensive
    # against the case where the first event Thrum saw wasn't sessionStart
    # or beforeSubmitPrompt (e.g. install happened mid-session, or Cursor
    # skipped beforeSubmitPrompt for any reason). Without this, an emit
    # triggered by a bare `stop` would carry empty cursor_conversation_id
    # / cursor_generation_id and the backend's external_id projection
    # would yield NULL.
    if event.conversation_id:
        buf["cursor_conversation_id"] = event.conversation_id
    if event.cursor_version:
        buf["cursor_version"] = event.cursor_version
    if event.workspace_roots:
        # Latest observed wins. Single-workspace is the typical case;
        # if Cursor ever supports multi-workspace and the user adds
        # one mid-conversation, the closing emit attributes the most
        # recent root set rather than the one the conversation opened
        # with. Acceptable trade-off without a per-event metadata
        # snapshot mechanism.
        buf["cursor_workspace_roots"] = list(event.workspace_roots)
    if event.transcript_path and not buf.get("transcript_path"):
        buf["transcript_path"] = event.transcript_path
    if event.model and not buf.get("model"):
        buf["model"] = event.model

    # Generation tracking: bump generation_index whenever a new
    # generation_id is observed, not just on beforeSubmitPrompt. Each
    # generation_id is stable across all events of one Cursor turn
    # (verified empirically — common envelope), so detecting a NEW
    # generation_id by != comparison is sufficient.
    if event.generation_id:
        last_seen = buf.get("cursor_generation_id")
        if last_seen != event.generation_id:
            buf["cursor_generation_index"] = int(
                buf.get("cursor_generation_index", -1)
            ) + 1
            buf["cursor_generation_id"] = event.generation_id

    return buf


def _new_cursor_turn() -> dict[str, Any]:
    return {
        "turn_start_ts": _now_iso(),
        # Populated from cursor_transcript.read_turn() at emit time;
        # cleared here so a stale list from the previous turn doesn't leak.
        "tools_used": [],
        "tools_failed": [],
        "tool_use_id_map": {},
        "tool_flags": {"interrupted": False, "is_image": False},
        "bash_categories": [],
    }


def _dispatch_cursor(
    buffer: dict[str, Any], event: CursorHookEvent
) -> Action:
    """13 registered Cursor events. The lifecycle subset that matters for
    per-turn ingest: beforeSubmitPrompt opens, stop closes; the rest fold
    into buffer state for the closing emit.

    Tab events (beforeTabFileRead, afterTabFileEdit) are not registered
    in `~/.cursor/hooks.json` so they never reach this dispatch. MCP
    events (before/afterMCPExecution) likewise — deferred until real-world
    MCP usage justifies the surface.
    """
    name = event.hook_event_name

    if name == "sessionStart":
        if event.cursor_version:
            buffer["cursor_version"] = event.cursor_version
        if event.workspace_roots:
            buffer["cursor_workspace_roots"] = list(event.workspace_roots)
        return Action(persist="save")

    if name == "sessionEnd":
        # Forced flush — close any open turn and emit. `reason` is recorded
        # in the closing span's metadata so downstream observability can
        # distinguish user-close from timeout.
        if buffer.get("turn") is not None:
            buffer["turn"]["turn_end_ts"] = _now_iso()
            if event.reason:
                buffer["turn"]["session_end_reason"] = event.reason
            return Action(persist="delete", emit="turn")
        return Action(persist="delete")

    if name == "beforeSubmitPrompt":
        # Opens a new turn slot. Generation tracking
        # (cursor_generation_id + cursor_generation_index) is centralised
        # in _ensure_cursor_buffer so it works even when beforeSubmitPrompt
        # is missed (mid-session install, Cursor skipping the event, etc.).
        current = buffer.get("turn")
        if current is None or current.get("turn_end_ts"):
            buffer["turn"] = _new_cursor_turn()
        if event.transcript_path:
            buffer["transcript_path"] = event.transcript_path
        return Action(persist="save")

    if name in ("preToolUse", "postToolUse"):
        # Cursor's hook tool_name is normalised (Read/Grep/Shell/Write).
        # Model-facing tool names (ReadFile/Glob/rg/etc) don't surface
        # here. The canonical-intent vocabulary is filled at stop time
        # from the transcript JSONL — see _emit_for_cursor_turn.
        # We don't even record the hook tool_name on the buffer; the
        # transcript is authoritative (Codex L3 lesson — Rec 1.4 in fbedda4).
        if buffer.get("turn") is None:
            buffer["turn"] = _new_cursor_turn()
        return Action(persist="save")

    if name == "postToolUseFailure":
        if buffer.get("turn") is None:
            buffer["turn"] = _new_cursor_turn()
        if event.tool_name:
            buffer["turn"]["tools_failed"].append(event.tool_name)
        return Action(persist="save")

    if name in ("subagentStart", "subagentStop"):
        # Mark the turn as having delegated work; the canonical
        # task-delegation intent is added at emit time from the transcript.
        if buffer.get("turn") is None:
            buffer["turn"] = _new_cursor_turn()
        return Action(persist="save")

    if name in ("beforeShellExecution", "afterShellExecution"):
        # Shell-class events feed into the canonical shell-class intent
        # projection (run-tests / build / git-ops). The transcript also
        # carries Shell tool_use blocks with the command string, so we
        # rely on cursor_transcript._project_tool_use to do classification.
        # Hook event recorded as a no-op for buffer continuity.
        if buffer.get("turn") is None:
            buffer["turn"] = _new_cursor_turn()
        return Action(persist="save")

    if name == "preCompact":
        if buffer.get("turn") is None:
            buffer["turn"] = _new_cursor_turn()
        # Mirror Codex / Claude — record compact intent on the turn so
        # the classifier sees it. Token snapshot deferred until preCompact's
        # full payload schema is empirically captured.
        if "compact" not in buffer["turn"]["tools_used"]:
            buffer["turn"]["tools_used"].append("compact")
        return Action(persist="save")

    if name == "afterAgentResponse":
        # Pre-stop signal carrying the same measured tokens as `stop`.
        # Stash the snapshot on the turn so the closing stop emit doesn't
        # have to wait for hook re-entry. (Some Cursor sessions only emit
        # afterAgentResponse without a paired `stop` — defensive parity.)
        if buffer.get("turn") is None:
            buffer["turn"] = _new_cursor_turn()
        if event.input_tokens is not None:
            buffer["turn"]["input_tokens"] = event.input_tokens
        if event.output_tokens is not None:
            buffer["turn"]["output_tokens"] = event.output_tokens
        if event.cache_read_tokens is not None:
            buffer["turn"]["cache_read_tokens"] = event.cache_read_tokens
        if event.cache_write_tokens is not None:
            buffer["turn"]["cache_write_tokens"] = event.cache_write_tokens
        return Action(persist="save")

    if name == "stop":
        # Per-turn boundary. Tokens come from the hook payload directly
        # (FR-218f — `token_source='measured'`); transcript supplies
        # tools_used + plan attribution. See _emit_for_cursor_turn.
        if buffer.get("turn") is None:
            buffer["turn"] = _new_cursor_turn()
        if event.input_tokens is not None:
            buffer["turn"]["input_tokens"] = event.input_tokens
        if event.output_tokens is not None:
            buffer["turn"]["output_tokens"] = event.output_tokens
        if event.cache_read_tokens is not None:
            buffer["turn"]["cache_read_tokens"] = event.cache_read_tokens
        if event.cache_write_tokens is not None:
            buffer["turn"]["cache_write_tokens"] = event.cache_write_tokens
        if event.loop_count is not None:
            buffer["turn"]["loop_count"] = event.loop_count
        buffer["turn"]["turn_end_ts"] = _now_iso()
        return Action(persist="delete", emit="turn")

    safe_log(
        "unknown_cursor_hook_event",
        log_path=None,
        hook_event_name=name,
    )
    return Action(persist="noop")


def _emit_for_cursor_turn(
    buffer: dict[str, Any], settings: SkillSettings
) -> None:
    """Read the Cursor transcript JSONL, find the matching generation,
    emit one OTLP span.

    Tokens come directly from the `stop` (or `afterAgentResponse`) hook
    payload (FR-218f — `token_source='measured'`). `tools_used` is the
    canonical intent list from the transcript parser so the backend
    classifier sees the same axis as Claude / Codex.

    Hook-vs-transcript flush race (Fix #2): the handler tracks
    `cursor_generation_index` per buffer (incremented on each
    `beforeSubmitPrompt`). When the transcript hasn't caught up yet,
    `read_cursor_turn(path, expected_index=...)` returns None and the
    activity emits with empty `tool_intents` rather than mis-attributing
    the previous generation's tools to this turn.
    """
    if buffer.get("turn") is None:
        buffer["turn"] = _new_cursor_turn()
    turn = buffer["turn"]
    end_ts = turn.get("turn_end_ts") or _now_iso()
    transcript_path = buffer.get("transcript_path")
    generation_index = int(buffer.get("cursor_generation_index", -1))
    conversation_id = buffer.get("cursor_conversation_id") or ""
    generation_id = buffer.get("cursor_generation_id") or ""

    plan: CursorPlanAttribution | None = None
    canonical_intents: list[str] = []

    if transcript_path and Path(transcript_path).exists() and generation_index >= 0:
        path = Path(transcript_path)
        cursor_agg = read_cursor_turn(
            path, expected_index=generation_index
        )
        if cursor_agg is not None:
            canonical_intents = list(cursor_agg.tool_intents)
        # FR-215c — Cursor plan attribution. Cheap re-walk over the same
        # transcript. `generation_ts` = `end_ts` (the hook's stop time)
        # so completed_at lands on the closing turn only.
        if conversation_id:
            plan = detect_cursor_plan_for_generation(
                path, conversation_id, generation_index, generation_ts=end_ts
            )

    # Replace tools_used at emit (Codex L3 lesson — Rec 1.4 in fbedda4):
    # never carry hook-side raw tool_names through to the backend. If the
    # transcript hasn't caught up (Fix #2 race), tools_used stays empty.
    # Preserve the special "compact" intent if a preCompact arrived earlier.
    has_compact = "compact" in turn.get("tools_used", [])
    turn["tools_used"] = canonical_intents
    if has_compact and "compact" not in turn["tools_used"]:
        turn["tools_used"].append("compact")

    # Build the aggregate that emit_turn / build_turn_span consumes.
    raw_model = buffer.get("model") or ""
    # Cursor's hook payload reports `model: "default"` — the user's
    # auto-route preference, not a resolved model id (FR-218f). Substitute
    # to "cursor default" so the activity feed / distinct_models aggregation
    # shows something meaningful (routing layer = cursor, selection =
    # default/auto) rather than the opaque "default" token. Forward-
    # compatible: a future Cursor release that ships a real model name
    # in the hook payload (e.g. "claude-sonnet-4-7") flows through
    # unchanged, and v2.x's ai-tracking.db join overrides this whole
    # string with the resolved name.
    model = (
        "cursor default"
        if (not raw_model or raw_model == "default")
        else raw_model
    )
    agg = TranscriptAggregate(
        tokens_in=int(turn.get("input_tokens", 0) or 0),
        tokens_out=int(turn.get("output_tokens", 0) or 0),
        cache_read_input_tokens=int(turn.get("cache_read_tokens", 0) or 0),
        cache_creation_input_tokens=int(turn.get("cache_write_tokens", 0) or 0),
        model=model,
    )

    view = turn_view(buffer)
    # Cursor's cwd-equivalent is `workspace_roots[0]`. The buffer carries
    # both fields; prefer cwd when present so cross-source attribution
    # stays uniform.
    cursor_cwd = buffer.get("cwd") or (
        (buffer.get("cursor_workspace_roots") or [None])[0]
    )
    cwd, personal = _attribution_attrs(cursor_cwd)
    emit_turn(
        view,
        settings,
        agg=agg,
        end_ts=end_ts,
        gen_ai_system="cursor",
        source_tool=CURSOR_SOURCE_TOOL,
        token_source="measured",
        plan_id=str(plan.plan_id) if plan else None,
        plan_completed_at=plan.completed_at if plan else None,
        cursor_conversation_id=conversation_id or None,
        cursor_generation_id=generation_id or None,
        cursor_loop_count=turn.get("loop_count"),
        cursor_workspace_roots=buffer.get("cursor_workspace_roots") or None,
        cursor_version=buffer.get("cursor_version") or None,
        cursor_session_end_reason=turn.get("session_end_reason"),
        cwd=cwd,
        personal=personal,
    )


def _handle_cursor_event(raw: bytes, settings: SkillSettings) -> int:
    try:
        event = extract_cursor_hook_event(raw)
    except _PARSE_ERRORS as exc:
        safe_log(
            "parse_error",
            log_path=settings.log_path,
            error_category=type(exc).__name__,
            source_tool=CURSOR_SOURCE_TOOL,
        )
        return 0

    cwd_anchor = (
        event.workspace_roots[0] if event.workspace_roots else ""
    )
    if cwd_anchor and has_disable_marker(Path(cwd_anchor)):
        return 0

    with buffer_lock(settings.buffers_dir, event.session_id):
        buffer = _ensure_cursor_buffer(settings, event)
        action = _dispatch_cursor(buffer, event)

        if action.emit == "turn":
            _emit_for_cursor_turn(buffer, settings)

        if action.persist == "save":
            save_buffer(settings.buffers_dir, buffer)
        elif action.persist == "delete":
            delete_buffer(settings.buffers_dir, event.session_id)
    return 0


def main() -> int:
    raw = sys.stdin.buffer.read()
    settings = load_settings()
    detected = _detect_source_tool(raw)
    if detected == CURSOR_SOURCE_TOOL:
        return _handle_cursor_event(raw, settings)
    if detected == CODEX_SOURCE_TOOL:
        return _handle_codex_event(raw, settings)
    return handle_event(raw, settings)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
