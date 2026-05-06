"""Cursor IDE hook-payload streaming extractor — NFR-319 (FR-218f).

Mirrors `parsers/codex_hook.py` for the Cursor hook surface. Cursor 3.2.x
exposes 18 documented events in `~/.cursor/skills-cursor/create-hook/SKILL.md`;
Thrum registers the lifecycle subset that maps to the per-turn ingest model:

  sessionStart, sessionEnd,
  beforeSubmitPrompt,
  preToolUse, postToolUse, postToolUseFailure,
  subagentStart, subagentStop,
  beforeShellExecution, afterShellExecution,
  preCompact,
  stop, afterAgentResponse

Tab events (`beforeTabFileRead`, `afterTabFileEdit`) are deliberately NOT
in the subset — they fire per keystroke when inline-completion is on and
would generate one event per character. MCP events (`beforeMCPExecution`,
`afterMCPExecution`) are a future extension once Cursor MCP usage warrants
the surface.

Content fields (`prompt`, `text`, `tool_input`, `tool_output`, `command`,
`output`, `content`, `attachments`, `user_email`) are NEVER bound — `ijson.parse`
yields tokens that aren't in the allowlist; we ignore them. The `user_email`
exclusion is critical: every Cursor hook payload carries the account email,
and binding it would forward PII across the wire (NFR-318).

The hook payload schemas in this module are the empirically-verified shapes
captured from real Cursor sessions. Any
new field a future Cursor release adds will be silently ignored unless
explicitly added to `CURSOR_ALLOWED_PATHS` — content-stripping fail-closed.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any

import ijson


# Cursor hook payload allowlist. Distinct from Claude's and Codex's:
# - common envelope: hook_event_name, conversation_id, generation_id,
#   session_id, model, transcript_path, cursor_version, is_background_agent
# - workspace_roots is a list — bound via the `.item` ijson list-element path
# - per-event extras are scalars on top-level keys
# - explicitly excluded: user_email (PII), prompt, text, tool_input,
#   tool_output, command, output, content, attachments (all content fields)
CURSOR_ALLOWED_PATHS: frozenset[str] = frozenset(
    {
        # Common envelope
        "hook_event_name",
        "conversation_id",
        "generation_id",
        "session_id",
        "model",
        "transcript_path",
        "cursor_version",
        "is_background_agent",
        # workspace_roots[]: ijson scalar event for each list element
        "workspace_roots.item",
        # stop, afterAgentResponse — measured token fields
        "status",
        "loop_count",
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        # sessionStart, beforeSubmitPrompt
        "composer_mode",
        # sessionEnd
        "reason",
        "duration_ms",
        "final_status",
        # preToolUse, postToolUse
        "tool_name",
        "tool_use_id",
        "duration",
        # before/afterShellExecution
        "sandbox",
    }
)

_SCALAR_EVENTS: frozenset[str] = frozenset({"string", "number", "boolean", "null"})

CURSOR_HOOK_EVENT_NAMES: frozenset[str] = frozenset(
    {
        "sessionStart",
        "sessionEnd",
        "beforeSubmitPrompt",
        "preToolUse",
        "postToolUse",
        "postToolUseFailure",
        "subagentStart",
        "subagentStop",
        "beforeShellExecution",
        "afterShellExecution",
        "preCompact",
        "stop",
        "afterAgentResponse",
    }
)


@dataclass(frozen=True)
class CursorHookEvent:
    """One captured Cursor hook payload, with content fields stripped.

    Only ids, enums, counts, and timestamps cross this boundary. The
    `user_email`, `prompt`, `text`, `tool_input`, `tool_output`,
    `command`, `output`, `content`, and `attachments` fields are never
    bound (NFR-318/319).
    """

    hook_event_name: str
    conversation_id: str
    session_id: str
    cursor_version: str
    model: str
    # `generation_id` may be empty on sessionStart / sessionEnd when no
    # turn is active — treat "" as "no generation in scope".
    generation_id: str = ""
    transcript_path: str | None = None
    is_background_agent: bool | None = None
    workspace_roots: tuple[str, ...] = field(default_factory=tuple)
    # stop / afterAgentResponse — measured tokens
    status: str | None = None
    loop_count: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    # sessionStart / beforeSubmitPrompt
    composer_mode: str | None = None
    # sessionEnd
    reason: str | None = None
    duration_ms: int | None = None
    final_status: str | None = None
    # preToolUse / postToolUse
    tool_name: str | None = None
    tool_use_id: str | None = None
    duration: float | None = None
    # before/afterShellExecution
    sandbox: bool | None = None


def extract_cursor_hook_event(raw: bytes) -> CursorHookEvent:
    """Parse a Cursor hook stdin payload. Raises ValueError on bad shape.

    Validates that `hook_event_name` is one of the registered Cursor
    events; an unknown name (e.g. a Claude-only `SubagentStop` accidentally
    routed here) is rejected so the caller can fall back to the
    appropriate parser.
    """
    captured: dict[str, Any] = {}
    workspace_roots: list[str] = []

    for prefix, event, value in ijson.parse(io.BytesIO(raw)):
        if event not in _SCALAR_EVENTS:
            continue
        if prefix == "workspace_roots.item" and isinstance(value, str):
            workspace_roots.append(value)
            continue
        if prefix in CURSOR_ALLOWED_PATHS:
            captured[prefix] = value

    required = ("hook_event_name", "conversation_id", "session_id", "cursor_version")
    for k in required:
        if k not in captured:
            raise ValueError(f"missing required key: {k}")

    name = captured["hook_event_name"]
    if name not in CURSOR_HOOK_EVENT_NAMES:
        raise ValueError(f"unknown cursor hook_event_name: {name!r}")

    # ijson returns Decimal for fractional numbers and int for whole numbers.
    # Normalise to native int/float so downstream callers (handler, emitter)
    # don't need to special-case Decimal arithmetic.
    def _as_int(v: Any) -> int | None:
        if v is None:
            return None
        return int(v)

    def _as_float(v: Any) -> float | None:
        if v is None:
            return None
        return float(v)

    return CursorHookEvent(
        hook_event_name=name,
        conversation_id=captured["conversation_id"],
        session_id=captured["session_id"],
        cursor_version=captured["cursor_version"],
        model=captured.get("model", ""),
        generation_id=captured.get("generation_id", "") or "",
        transcript_path=captured.get("transcript_path"),
        is_background_agent=captured.get("is_background_agent"),
        workspace_roots=tuple(workspace_roots),
        status=captured.get("status"),
        loop_count=_as_int(captured.get("loop_count")),
        input_tokens=_as_int(captured.get("input_tokens")),
        output_tokens=_as_int(captured.get("output_tokens")),
        cache_read_tokens=_as_int(captured.get("cache_read_tokens")),
        cache_write_tokens=_as_int(captured.get("cache_write_tokens")),
        composer_mode=captured.get("composer_mode"),
        reason=captured.get("reason"),
        duration_ms=_as_int(captured.get("duration_ms")),
        final_status=captured.get("final_status"),
        tool_name=captured.get("tool_name"),
        tool_use_id=captured.get("tool_use_id"),
        duration=_as_float(captured.get("duration")),
        sandbox=captured.get("sandbox"),
    )
