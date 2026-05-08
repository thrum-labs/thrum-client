"""FR-217 — one-time pass over existing transcripts at first install.

Enumerates `~/.claude/projects/*/*.jsonl`. Within each file we walk the
assistant records in order, apply FR-221 dedupe per turn, and emit one
span (`thrum.metadata.backfill=true`) per turn boundary (an assistant
record with `stop_reason == "end_turn"`).

For Codex (FR-218b): a sister loop over
`${CODEX_HOME:-~/.codex}/sessions/YYYY/MM/DD/rollout-*.jsonl`. One span
per `task_complete` record in the rollout, idempotent via a separate
`.codex_backfill_done` marker.

For Cursor (FR-218f): a sister loop over
`~/.cursor/projects/*/agent-transcripts/*/*.jsonl`. One span per
generation (user→assistant boundary), idempotent via a separate
`.cursor_backfill_done` marker. Cursor transcripts are content-only —
no token data — so backfill spans set `token_source='estimated'` with
char-based approximation (~4 chars = 1 token).

All three passes idempotent: on success, touch the relevant marker.
Subsequent runs exit immediately. `external_id` projection per migration
`d2e3f4a5b6c7` provides ingest-side dedupe so `--force` re-runs don't
double-count.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from thrum_client.codex_config import codex_home
from thrum_client.config import SkillSettings
from thrum_client.cursor_config import cursor_home
from thrum_client.emitter import emit_turn
from thrum_client.parsers.codex_plan_detector import (
    detect_plan_for_turn as detect_codex_plan_for_turn,
)
from thrum_client.parsers.codex_rollout import (
    CodexSessionMeta,
    CodexTurnAggregate,
    iter_codex_turns,
    read_session_meta,
)
from thrum_client.parsers.cursor_plan_detector import (
    CursorPlanAttribution,
    detect_plan_for_generation as detect_cursor_plan_for_generation,
    iter_plans as iter_cursor_plans,
)
from thrum_client.parsers.cursor_transcript import (
    _project_tool_use,
    _walk_content_blocks,
    iter_cursor_turns,
)
from thrum_client.parsers.plan_detector import detect_plan_for_turn
from thrum_client.parsers.transcript import (
    AssistantRecord,
    TranscriptAggregate,
    iter_assistant_records,
)
from thrum_client.safe_log import safe_log


def _aggregate(records: list[AssistantRecord]) -> TranscriptAggregate:
    agg = TranscriptAggregate()
    seen: dict[str, AssistantRecord] = {}
    order: list[str] = []
    for r in records:
        if r.message_id in seen:
            continue
        seen[r.message_id] = r
        order.append(r.message_id)
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
    end_turn = next((seen[m] for m in order if seen[m].stop_reason == "end_turn"), None)
    picked = end_turn or (seen[order[-1]] if order else None)
    if picked is not None:
        agg.model = picked.model
        agg.stop_reason = picked.stop_reason
        agg.service_tier = picked.service_tier
        agg.speed = picked.speed
        agg.inference_geo = picked.inference_geo
    return agg


def _turn_view(session_id: str, records: list[AssistantRecord]) -> dict[str, Any]:
    start = records[0].timestamp if records else None
    return {
        "session_id": session_id,
        "turn_start_ts": start,
        "tools_used": [],
        "tools_failed": [],
        "tool_flags": {},
    }


def _session_id_from_path(jsonl: Path) -> str:
    return jsonl.stem


def emit_backfill_for_file(
    jsonl: Path,
    settings: SkillSettings,
    *,
    http_post: Callable | None = None,
) -> int:
    """Walk assistant records in order; close a turn at the first end_turn
    record for a new `message.id` (FR-221 dedupe — sibling records of the
    same API response share one message.id and must not each trigger a
    span).
    """
    session_id = _session_id_from_path(jsonl)
    group: list[AssistantRecord] = []
    # FR-221 dedupe is per-turn in spec, but we keep `seen` file-wide here
    # on purpose: a malformed transcript that has multiple `end_turn`
    # siblings sharing one message.id would otherwise emit duplicate
    # backfill spans, one per sibling. Since message.id is globally unique
    # per real API call, file-wide dedup is indistinguishable from per-turn
    # dedup in well-formed input — and stricter on bad input. Review #35
    # flagged the drift; leaving it as-is with this comment is the fix.
    seen: set[str] = set()
    spans_emitted = 0
    for rec in iter_assistant_records(jsonl):
        if rec.message_id in seen:
            continue
        seen.add(rec.message_id)
        group.append(rec)
        if rec.stop_reason == "end_turn":
            agg = _aggregate(group)
            view = _turn_view(session_id, group)
            turn_start_ts = group[0].timestamp
            end_ts = group[-1].timestamp
            # FR-215c — historical plans get the same plan_id treatment so
            # backfill rows attribute to plans the same way live Stop hooks do.
            plan = detect_plan_for_turn(jsonl, session_id, turn_start_ts, end_ts)
            emit_turn(
                view,
                settings,
                agg=agg,
                backfill=True,
                end_ts=end_ts,
                plan_id=str(plan.plan_id) if plan else None,
                plan_completed_at=plan.completed_at if plan else None,
                http_post=http_post,
            )
            spans_emitted += 1
            group = []
    return spans_emitted


def run_backfill(
    settings: SkillSettings,
    *,
    projects_root: Path | None = None,
    http_post: Callable | None = None,
    force: bool = False,
) -> int:
    """Walk ~/.claude/projects/*/*.jsonl, emit one backfill span per turn.

    Idempotent via `.backfill_done` marker in the config dir. Pass `force=True`
    to re-run.
    """
    marker = settings.backfill_marker
    if marker.exists() and not force:
        safe_log(
            "backfill_skipped_marker_present",
            log_path=settings.log_path,
        )
        return 0

    root = projects_root or (settings.claude_dir / "projects")
    if not root.exists():
        safe_log("backfill_no_projects_dir", log_path=settings.log_path)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(datetime.now(UTC).isoformat())
        return 0

    total = 0
    for jsonl in sorted(root.glob("*/*.jsonl")):
        try:
            total += emit_backfill_for_file(jsonl, settings, http_post=http_post)
        except Exception as exc:
            safe_log(
                "backfill_file_error",
                log_path=settings.log_path,
                error_category=type(exc).__name__,
            )
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(datetime.now(UTC).isoformat())
    safe_log(
        "backfill_complete",
        log_path=settings.log_path,
        span_count=total,
    )
    return total


# ---- Codex CLI backfill (FR-218b sister loop to FR-217) ----


def _codex_backfill_marker(settings: SkillSettings) -> Path:
    return settings.config_dir / ".codex_backfill_done"


def _codex_session_id(meta: CodexSessionMeta | None, jsonl: Path) -> str:
    """Authoritative source is `session_meta.payload.id` (UUIDv7 in the
    rollout itself). Fall back to the trailing UUID parsed from the file
    name only when session_meta is absent or its `id` is empty — the
    filename pattern can drift across Codex versions, the in-file id can't.
    """
    if meta is not None and meta.session_id:
        return meta.session_id
    # Fallback: rollouts are named `rollout-<iso-ts>-<uuid>.jsonl`; the
    # last 5 dash-segments are the UUIDv7 (8-4-4-4-12).
    stem = jsonl.stem
    if stem.startswith("rollout-"):
        parts = stem.split("-")
        if len(parts) >= 5:
            return "-".join(parts[-5:])
    return stem


def _codex_turn_view(
    session_id: str, turn: CodexTurnAggregate
) -> dict[str, Any]:
    """Buffer-shaped view for emit_turn. `tools_used` is set to the canonical
    tool_intents projected by C1 — same axis the live D1 path emits."""
    return {
        "session_id": session_id,
        "turn_start_ts": None,  # Codex backfill doesn't track per-turn start
        "tools_used": list(turn.tool_intents),
        "tools_failed": [],
        "tool_flags": {},
    }


def _codex_agg_to_transcript(turn: CodexTurnAggregate) -> TranscriptAggregate:
    return TranscriptAggregate(
        tokens_in=turn.tokens_in,
        tokens_out=turn.tokens_out,
        cache_read_input_tokens=turn.cached_input_tokens,
        model=turn.model,
    )


def emit_codex_backfill_for_file(
    rollout: Path,
    settings: SkillSettings,
    *,
    http_post: Callable | None = None,
) -> int:
    """Emit one backfill span per `task_complete` record in the rollout."""
    meta = read_session_meta(rollout)
    session_id = _codex_session_id(meta, rollout)
    originator = meta.originator if meta else None
    spans_emitted = 0
    for turn in iter_codex_turns(rollout):
        agg = _codex_agg_to_transcript(turn)
        view = _codex_turn_view(session_id, turn)
        # FR-215c — historical Codex plans get the same plan_id treatment
        # as live `_emit_for_codex_turn` so backfill rows attribute to the
        # same plan_id (UUIDv5(session, first_event_ts)) the live hook would
        # have produced. Idempotent across re-runs.
        plan = detect_codex_plan_for_turn(rollout, session_id, turn.turn_id)
        emit_turn(
            view,
            settings,
            agg=agg,
            backfill=True,
            gen_ai_system="openai",
            source_tool="codex-cli",
            codex_turn_id=turn.turn_id,
            codex_originator=originator,
            reasoning_output_tokens=turn.reasoning_output_tokens,
            plan_id=str(plan.plan_id) if plan else None,
            plan_completed_at=plan.completed_at if plan else None,
            http_post=http_post,
        )
        spans_emitted += 1
    return spans_emitted


def run_codex_backfill(
    settings: SkillSettings,
    *,
    sessions_root: Path | None = None,
    http_post: Callable | None = None,
    force: bool = False,
) -> int:
    """Walk `${CODEX_HOME:-~/.codex}/sessions/**/rollout-*.jsonl` and emit
    one backfill span per `task_complete` record.

    Separate marker file (`.codex_backfill_done`) so the Claude and Codex
    loops are independently idempotent — installing Thrum after Codex was
    already in use, or vice versa, doesn't re-emit either history.
    """
    marker = _codex_backfill_marker(settings)
    if marker.exists() and not force:
        safe_log(
            "codex_backfill_skipped_marker_present",
            log_path=settings.log_path,
        )
        return 0

    root = sessions_root or (codex_home() / "sessions")
    if not root.exists():
        safe_log("codex_backfill_no_sessions_dir", log_path=settings.log_path)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(datetime.now(UTC).isoformat())
        return 0

    total = 0
    for rollout in sorted(root.rglob("rollout-*.jsonl")):
        try:
            total += emit_codex_backfill_for_file(
                rollout, settings, http_post=http_post
            )
        except Exception as exc:
            safe_log(
                "codex_backfill_file_error",
                log_path=settings.log_path,
                error_category=type(exc).__name__,
            )
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(datetime.now(UTC).isoformat())
    safe_log(
        "codex_backfill_complete",
        log_path=settings.log_path,
        span_count=total,
    )
    return total


# ---- Cursor backfill (FR-218f sister loop to FR-217 / FR-218b) ----


def _cursor_backfill_marker(settings: SkillSettings) -> Path:
    return settings.config_dir / ".cursor_backfill_done"


def _attribution_from_plans_list(
    plans: list,
    generation_index: int,
    generation_ts: str | None = None,
) -> CursorPlanAttribution | None:
    """Pure-Python lookup against a pre-built plans list. Mirrors
    `parsers/cursor_plan_detector.detect_plan_for_generation` but
    without the file walk — backfill builds the list once and reuses it
    across every generation.
    """
    if generation_index < 0:
        return None
    for plan in plans:
        if generation_index not in plan.touched_generations:
            continue
        completed_at: str | None = None
        if (
            plan.closing_generation_index == generation_index
            and generation_ts is not None
        ):
            completed_at = generation_ts
        return CursorPlanAttribution(
            plan_id=plan.plan_id, completed_at=completed_at
        )
    return None


def _cursor_conversation_id_from_path(transcript: Path) -> str:
    """Cursor's transcript path is
    `<projects>/<sanitised-cwd>/agent-transcripts/<conversation_id>/<conversation_id>.jsonl`
    — the basename (sans `.jsonl`) is the conversation_id, and it
    matches the parent dir name. Use the stem; fall back to parent name
    if the stem is empty (defensive)."""
    stem = transcript.stem
    if stem:
        return stem
    return transcript.parent.name


def _estimate_tokens_from_chars(char_count: int) -> int:
    """Coarse char-based token estimator. ~4 chars per token is a
    long-standing rule of thumb across the major models. Used only for
    Cursor backfill spans, marked `token_source='estimated'` so the
    backend (and downstream leaderboards) can filter them out of
    measured-only views."""
    if char_count <= 0:
        return 0
    return max(1, char_count // 4)


@dataclass
class _CursorBackfillGen:
    """Per-generation aggregate built in a single transcript pass.

    Combines what `iter_cursor_turns` (tool_intents) and the prior
    `_cursor_text_chars_for_generation` (char counts) used to compute in
    separate walks. Held by `_scan_cursor_transcript_for_backfill`.
    """

    tool_intents: list[str] = field(default_factory=list)
    user_chars: int = 0
    assistant_chars: int = 0


def _scan_cursor_transcript_for_backfill(
    transcript: Path,
) -> list[_CursorBackfillGen]:
    """Single-pass walk of a Cursor transcript JSONL.

    Reuses `parsers/cursor_transcript._project_tool_use` and
    `_walk_content_blocks` so the intent projection stays consistent
    with the live emit path. Sums `text` content lengths per generation
    for the char-based token estimator (see
    `_estimate_tokens_from_chars`). The text values are only used for
    `len()`; they live in the local `text` variable across one block
    iteration and are not stored on the returned aggregate. Same
    privacy posture as `iter_cursor_turns`.

    Replaces the prior pattern of looping `iter_cursor_turns(...)` AND
    calling `_cursor_text_chars_for_generation(...)` per generation,
    which scaled O(N²) in file reads.
    """
    import json as _json

    out: list[_CursorBackfillGen] = []
    current: _CursorBackfillGen | None = None
    open_gen = False

    try:
        with transcript.open("r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    line = _json.loads(raw)
                except _json.JSONDecodeError:
                    continue
                if not isinstance(line, dict):
                    continue
                role = line.get("role")
                if role == "user":
                    if open_gen and current is not None:
                        out.append(current)
                    current = _CursorBackfillGen()
                    open_gen = True
                    for block in _walk_content_blocks(line.get("message")):
                        if block.get("type") == "text":
                            text = block.get("text")
                            if isinstance(text, str):
                                current.user_chars += len(text)
                    continue
                if role != "assistant" or current is None:
                    continue
                for block in _walk_content_blocks(line.get("message")):
                    btype = block.get("type")
                    if btype == "tool_use":
                        intent = _project_tool_use(
                            block.get("name"), block.get("input")
                        )
                        if intent and intent not in current.tool_intents:
                            current.tool_intents.append(intent)
                    elif btype == "text":
                        text = block.get("text")
                        if isinstance(text, str):
                            current.assistant_chars += len(text)
    except OSError:
        return []

    if open_gen and current is not None:
        out.append(current)
    return out


def emit_cursor_backfill_for_file(
    transcript: Path,
    settings: SkillSettings,
    *,
    http_post: Callable | None = None,
) -> int:
    """Emit one backfill span per generation in a Cursor transcript JSONL.

    Each user→assistant boundary becomes one span with:
      * source_tool='cursor', gen_ai_system='cursor', model='default'
      * token_source='estimated' (transcripts don't carry tokens; we
        approximate from char counts of user prompt + assistant text
        via the ~4-chars-per-token rule of thumb. Bias direction:
        UNDER-counts for tool-heavy turns because we only count
        `message.content[].text` blocks — tool inputs / tool results
        live in other fields and contribute zero tokens to the estimate.
        `token_source='estimated'` is the escape valve so consumers can
        filter these out of measured-only views.)
      * external_id='cursor|<conversation_id>|backfill-<6-digit-index>'
        — synthesized so re-runs dedupe at ingest. Distinct from the
        live `cursor|<conv>|<live-generation-uuid>` projection so
        backfill rows don't collide with subsequent live emits for
        the same generation (the live row wins on UI/leaderboard
        because it carries `metadata.backfill=false` implicitly).
      * plan attribution via detect_cursor_plan_for_generation; backfill
        passes `generation_ts=None` so completed_at stays NULL even on
        closing generations (FR-215c semantics for backfilled plans —
        participation only, no shipped credit, until v2.x adds
        SQLite-derived timestamps).

    Performance: previously did 3N file walks (outer iter_cursor_turns
    + per-iteration char count + per-iteration plan detect). Now does
    2 walks total — one for intent + char aggregation, one for plan
    detection (kept separate because the plan detector uses its own
    ijson-based privacy allowlist).
    """
    conversation_id = _cursor_conversation_id_from_path(transcript)

    # Pre-walk 1: aggregate per-generation tool intents + char counts.
    per_generation = _scan_cursor_transcript_for_backfill(transcript)
    if not per_generation:
        return 0

    # Pre-walk 2: build the full plans list once. The plan detector
    # uses ijson allowlist (privacy isolation from the json.loads-based
    # walk above), so it stays a separate pass.
    plans = list(iter_cursor_plans(transcript, conversation_id))

    spans_emitted = 0
    for generation_index, gen_data in enumerate(per_generation):
        agg = TranscriptAggregate(
            tokens_in=_estimate_tokens_from_chars(gen_data.user_chars),
            tokens_out=_estimate_tokens_from_chars(gen_data.assistant_chars),
            # Match the live emit's substitution (handler._emit_for_cursor_turn)
            # so backfill rows and live rows share the same model placeholder
            # string in the leaderboard's distinct_models bucket.
            model="cursor default",
        )
        synthesized_generation_id = f"backfill-{generation_index:06d}"
        view: dict[str, Any] = {
            "session_id": conversation_id,
            "turn_start_ts": None,  # Cursor backfill has no per-turn ts
            "tools_used": list(gen_data.tool_intents),
            "tools_failed": [],
            "tool_flags": {},
        }
        plan = _attribution_from_plans_list(
            plans, generation_index, generation_ts=None
        )
        emit_turn(
            view,
            settings,
            agg=agg,
            backfill=True,
            gen_ai_system="cursor",
            source_tool="cursor",
            token_source="estimated",
            cursor_conversation_id=conversation_id,
            cursor_generation_id=synthesized_generation_id,
            plan_id=str(plan.plan_id) if plan else None,
            plan_completed_at=plan.completed_at if plan else None,
            http_post=http_post,
        )
        spans_emitted += 1
    return spans_emitted


def run_cursor_backfill(
    settings: SkillSettings,
    *,
    cursor_projects_root: Path | None = None,
    http_post: Callable | None = None,
    force: bool = False,
) -> int:
    """Walk `~/.cursor/projects/*/agent-transcripts/*/*.jsonl` and emit
    one backfill span per generation.

    Separate marker file (`.cursor_backfill_done`) so Claude / Codex /
    Cursor backfill loops are independently idempotent — a user
    installing Thrum after one was already in use doesn't re-emit any
    pre-existing histories.
    """
    marker = _cursor_backfill_marker(settings)
    if marker.exists() and not force:
        safe_log(
            "cursor_backfill_skipped_marker_present",
            log_path=settings.log_path,
        )
        return 0

    root = cursor_projects_root or (cursor_home() / "projects")
    if not root.exists():
        safe_log("cursor_backfill_no_projects_dir", log_path=settings.log_path)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(datetime.now(UTC).isoformat())
        return 0

    total = 0
    for transcript in sorted(root.glob("*/agent-transcripts/*/*.jsonl")):
        try:
            total += emit_cursor_backfill_for_file(
                transcript, settings, http_post=http_post
            )
        except Exception as exc:
            safe_log(
                "cursor_backfill_file_error",
                log_path=settings.log_path,
                error_category=type(exc).__name__,
            )
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(datetime.now(UTC).isoformat())
    safe_log(
        "cursor_backfill_complete",
        log_path=settings.log_path,
        span_count=total,
    )
    return total
