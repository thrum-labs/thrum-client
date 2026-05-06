"""Hook-payload streaming extractor — NFR-319.

Content keys (`prompt`, `last_assistant_message`, `tool_input.*`, `tool_response.*`
where not explicitly allowed, `error`, `task`) are NEVER bound to a program
variable. `ijson.parse` yields low-level events; we only capture scalars whose
path is in `ALLOWED_PATHS`.

`custom_instructions` is a presence-only path: we observe the event to set
`has_custom_instructions` but never store the string.

`tool_input.command` is a classify-and-drop path (Bash only): in the same
ijson iteration where the string appears, we run a fixed set of patterns
(git / test / build) and retain only the resulting enum label. The raw
command string is never stored, logged, or emitted.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Any

import ijson


# Mirrored verbatim from backend/app/services/classifier.py. The backend
# side no longer reads command strings (privacy: NFR-318), so the client
# is the single place these patterns apply.
_GIT_COMMAND_RE = re.compile(r"\bgit\s+\w")
_TEST_COMMAND_RE = re.compile(
    r"\b(pytest|npm\s+test|cargo\s+test|go\s+test|jest|vitest|rspec)\b"
)
_BUILD_COMMAND_RE = re.compile(
    r"\b(docker\s+build|make\b|npm\s+run\s+build|cargo\s+build|go\s+build|"
    r"tsc\b|webpack|vite\s+build)\b"
)


def _classify_bash_command(cmd: str) -> str | None:
    """Return one of {git_ops, testing, build_deploy} or None.

    First match wins in the same order the backend classifier used to.
    Called inside the ijson loop so the caller never binds `cmd` past the
    next iteration.
    """
    if _GIT_COMMAND_RE.search(cmd):
        return "git_ops"
    if _TEST_COMMAND_RE.search(cmd):
        return "testing"
    if _BUILD_COMMAND_RE.search(cmd):
        return "build_deploy"
    return None


# Paths are ijson-prefix literals — `tool_response.interrupted` matches only
# when `tool_response` is an object with a scalar `interrupted` field. If a
# future Claude Code version ever returns `tool_response` as an array, the
# prefixes become `tool_response.item.interrupted` and these allow-list
# entries stop matching. That is the fail-closed outcome we want (the flag
# silently drops rather than emitting a guessed value).
ALLOWED_PATHS: frozenset[str] = frozenset(
    {
        "hook_event_name",
        "session_id",
        "transcript_path",
        "cwd",
        "permission_mode",
        "tool_name",
        "tool_use_id",
        "agent_id",
        "agent_type",
        "agent_transcript_path",
        "stop_hook_active",
        "is_interrupt",
        "source",
        "model",
        "trigger",
        "reason",
        # Non-content flags from PostToolUse.tool_response (object shape).
        "tool_response.interrupted",
        "tool_response.isImage",
        "tool_response.noOutputExpected",
    }
)

PRESENCE_PATHS: frozenset[str] = frozenset({"custom_instructions"})

_SCALAR_EVENTS: frozenset[str] = frozenset({"string", "number", "boolean", "null"})


@dataclass(frozen=True)
class HookEvent:
    hook_event_name: str
    session_id: str
    cwd: str
    transcript_path: str | None = None
    permission_mode: str | None = None
    tool_name: str | None = None
    tool_use_id: str | None = None
    agent_id: str | None = None
    agent_type: str | None = None
    agent_transcript_path: str | None = None
    stop_hook_active: bool | None = None
    is_interrupt: bool | None = None
    source: str | None = None
    model: str | None = None
    trigger: str | None = None
    reason: str | None = None
    tool_response_interrupted: bool | None = None
    tool_response_is_image: bool | None = None
    tool_response_no_output_expected: bool | None = None
    has_custom_instructions: bool = False
    bash_category: str | None = None


def extract_hook_event(raw: bytes) -> HookEvent:
    """Parse a hook-stdin payload. Raises ValueError on bad shape."""
    captured: dict[str, Any] = {}
    has_custom_instructions = False
    bash_category: str | None = None

    for prefix, event, value in ijson.parse(io.BytesIO(raw)):
        if event not in _SCALAR_EVENTS:
            continue
        if prefix in ALLOWED_PATHS:
            captured[prefix] = value
        elif prefix in PRESENCE_PATHS:
            # Presence-only: observe that a non-empty string appeared,
            # but never retain the value. After this iteration `value`
            # is rebound by the loop and the string is unreferenced.
            if event == "string" and isinstance(value, str) and value:
                has_custom_instructions = True
        elif prefix == "tool_input.command":
            # Classify-and-drop: same scoping guarantee as the presence
            # branch above — `value` is rebound by the next iteration.
            if event == "string" and isinstance(value, str) and value:
                hit = _classify_bash_command(value)
                if hit is not None and bash_category is None:
                    bash_category = hit

    required = ("hook_event_name", "session_id", "cwd")
    for k in required:
        if k not in captured:
            raise ValueError(f"missing required key: {k}")

    return HookEvent(
        hook_event_name=captured["hook_event_name"],
        session_id=captured["session_id"],
        cwd=captured["cwd"],
        transcript_path=captured.get("transcript_path"),
        permission_mode=captured.get("permission_mode"),
        tool_name=captured.get("tool_name"),
        tool_use_id=captured.get("tool_use_id"),
        agent_id=captured.get("agent_id"),
        agent_type=captured.get("agent_type"),
        agent_transcript_path=captured.get("agent_transcript_path"),
        stop_hook_active=captured.get("stop_hook_active"),
        is_interrupt=captured.get("is_interrupt"),
        source=captured.get("source"),
        model=captured.get("model"),
        trigger=captured.get("trigger"),
        reason=captured.get("reason"),
        tool_response_interrupted=captured.get("tool_response.interrupted"),
        tool_response_is_image=captured.get("tool_response.isImage"),
        tool_response_no_output_expected=captured.get("tool_response.noOutputExpected"),
        has_custom_instructions=has_custom_instructions,
        bash_category=bash_category,
    )
