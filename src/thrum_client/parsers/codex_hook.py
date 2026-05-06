"""Codex CLI hook-payload streaming extractor — NFR-319 (FR-218b).

Mirrors `parsers/hook.py` for the Codex CLI hook surface. Six events:
`SessionStart`, `UserPromptSubmit`, `PreToolUse`, `PostToolUse`,
`PermissionRequest`, `Stop`. No `SubagentStop`, no `PreCompact`, no
`SessionEnd` — the Codex hook surface is narrower than Claude Code's.

Content fields (`prompt`, `last_agent_message`, `tool_response.*` content,
`exec_command.arguments.cmd`, `parsed_cmd[].cmd`, etc.) are NEVER bound to
a program variable — `ijson.parse` yields tokens that aren't in the
allowlist; we ignore them.

Bash-style command classification is **not** done from the hook payload
for Codex (unlike Claude). Codex pre-classifies commands inside the
rollout's `exec_command_end.parsed_cmd[].type` field, so the rollout
parser (`parsers/codex_rollout.py`, C1) is the canonical source.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

import ijson


# Codex hook payload allowlist. Distinct from Claude's:
# - adds `turn_id` (UUIDv7 paired to rollout records)
# - drops `agent_id`/`agent_type`/`agent_transcript_path` (no subagents)
# - drops `tool_input.*` paths (Codex doesn't expose tool_input on hooks)
# - keeps `tool_response.interrupted` for parity if Codex grows it
CODEX_ALLOWED_PATHS: frozenset[str] = frozenset(
    {
        "hook_event_name",
        "session_id",
        "transcript_path",
        "cwd",
        "tool_name",
        "tool_use_id",
        "turn_id",
        "stop_hook_active",
        "model",
        "permission_mode",
        # SessionStart only — `startup` / `resume` / `clear`. Content-free enum.
        "source",
    }
)

_SCALAR_EVENTS: frozenset[str] = frozenset({"string", "number", "boolean", "null"})

CODEX_HOOK_EVENT_NAMES: frozenset[str] = frozenset(
    {
        "SessionStart",
        "UserPromptSubmit",
        "PreToolUse",
        "PostToolUse",
        "PermissionRequest",
        "Stop",
    }
)


@dataclass(frozen=True)
class CodexHookEvent:
    hook_event_name: str
    session_id: str
    cwd: str
    transcript_path: str | None = None
    tool_name: str | None = None
    tool_use_id: str | None = None
    turn_id: str | None = None
    stop_hook_active: bool | None = None
    model: str | None = None
    permission_mode: str | None = None
    source: str | None = None  # SessionStart: startup / resume / clear


def extract_codex_hook_event(raw: bytes) -> CodexHookEvent:
    """Parse a Codex hook stdin payload. Raises ValueError on bad shape.

    Validates that `hook_event_name` is one of the six Codex events; an
    unknown name (e.g. a Claude-only event accidentally routed here) is
    rejected so the caller can fall back to the Claude path.
    """
    captured: dict[str, Any] = {}
    for prefix, event, value in ijson.parse(io.BytesIO(raw)):
        if event in _SCALAR_EVENTS and prefix in CODEX_ALLOWED_PATHS:
            captured[prefix] = value

    required = ("hook_event_name", "session_id", "cwd")
    for k in required:
        if k not in captured:
            raise ValueError(f"missing required key: {k}")

    name = captured["hook_event_name"]
    if name not in CODEX_HOOK_EVENT_NAMES:
        raise ValueError(f"unknown codex hook_event_name: {name!r}")

    return CodexHookEvent(
        hook_event_name=name,
        session_id=captured["session_id"],
        cwd=captured["cwd"],
        transcript_path=captured.get("transcript_path"),
        tool_name=captured.get("tool_name"),
        tool_use_id=captured.get("tool_use_id"),
        turn_id=captured.get("turn_id"),
        stop_hook_active=captured.get("stop_hook_active"),
        model=captured.get("model"),
        permission_mode=captured.get("permission_mode"),
        source=captured.get("source"),
    )
