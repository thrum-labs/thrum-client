"""Hand-crafted hook payload fixtures.

Every payload mixes allowlisted keys with content keys (prompt, stdout, etc.)
so tests can verify the NFR-319 extractor does not bind the content.
"""

from __future__ import annotations

import json


SESSION_ID = "2aa32619-b387-4c07-8d50-d6c44d97a272"
CWD = "/Users/example/proj"
TRANSCRIPT = f"/Users/example/.claude/projects/-proj/{SESSION_ID}.jsonl"

SENTINEL = "SENTINEL_PRIVATE_abc"


def session_start() -> bytes:
    return json.dumps(
        {
            "hook_event_name": "SessionStart",
            "session_id": SESSION_ID,
            "transcript_path": TRANSCRIPT,
            "cwd": CWD,
            "source": "startup",
            "model": "claude-opus-4-7",
        }
    ).encode()


def user_prompt_submit(prompt: str = SENTINEL) -> bytes:
    return json.dumps(
        {
            "hook_event_name": "UserPromptSubmit",
            "session_id": SESSION_ID,
            "transcript_path": TRANSCRIPT,
            "cwd": CWD,
            "prompt": prompt,  # content — MUST NOT be bound
        }
    ).encode()


def pre_tool_use(
    tool_name: str = "Bash",
    tool_use_id: str = "toolu_01",
    agent_id: str | None = None,
    command: str | None = None,
) -> bytes:
    payload = {
        "hook_event_name": "PreToolUse",
        "session_id": SESSION_ID,
        "transcript_path": TRANSCRIPT,
        "cwd": CWD,
        "tool_name": tool_name,
        "tool_use_id": tool_use_id,
        "tool_input": {
            # Default command still carries the sentinel so other tests can
            # assert it never leaks; override lets the classify-and-drop path
            # be exercised with realistic verbs (pytest, git, docker build…).
            "command": command if command is not None else f"echo {SENTINEL}",
            "description": SENTINEL,
        },
    }
    if agent_id is not None:
        payload["agent_id"] = agent_id
    return json.dumps(payload).encode()


def post_tool_use(
    tool_name: str = "Bash",
    tool_use_id: str = "toolu_01",
    interrupted: bool = False,
    is_image: bool = False,
    agent_id: str | None = None,
) -> bytes:
    payload = {
        "hook_event_name": "PostToolUse",
        "session_id": SESSION_ID,
        "transcript_path": TRANSCRIPT,
        "cwd": CWD,
        "tool_name": tool_name,
        "tool_use_id": tool_use_id,
        "tool_response": {
            "stdout": f"{SENTINEL} output",  # content
            "stderr": f"{SENTINEL} error",  # content
            "interrupted": interrupted,
            "isImage": is_image,
            "noOutputExpected": False,
        },
    }
    if agent_id is not None:
        payload["agent_id"] = agent_id
    return json.dumps(payload).encode()


def post_tool_use_failure(
    tool_name: str = "Bash",
    tool_use_id: str = "toolu_02",
    agent_id: str | None = None,
) -> bytes:
    payload = {
        "hook_event_name": "PostToolUseFailure",
        "session_id": SESSION_ID,
        "transcript_path": TRANSCRIPT,
        "cwd": CWD,
        "tool_name": tool_name,
        "tool_use_id": tool_use_id,
        "error": f"EISDIR: {SENTINEL}",  # content
    }
    if agent_id is not None:
        payload["agent_id"] = agent_id
    return json.dumps(payload).encode()


def subagent_start(
    agent_id: str = "agent_01",
    agent_type: str = "Explore",
) -> bytes:
    return json.dumps(
        {
            "hook_event_name": "SubagentStart",
            "session_id": SESSION_ID,
            "transcript_path": TRANSCRIPT,
            "cwd": CWD,
            "agent_id": agent_id,
            "agent_type": agent_type,
        }
    ).encode()


def subagent_stop(
    agent_id: str = "agent_01",
    agent_type: str = "Explore",
    agent_transcript_path: str = "/tmp/fake-subagent.jsonl",
    stop_hook_active: bool = False,
) -> bytes:
    return json.dumps(
        {
            "hook_event_name": "SubagentStop",
            "session_id": SESSION_ID,
            "transcript_path": TRANSCRIPT,
            "cwd": CWD,
            "agent_id": agent_id,
            "agent_type": agent_type,
            "agent_transcript_path": agent_transcript_path,
            "permission_mode": "default",
            "stop_hook_active": stop_hook_active,
            "last_assistant_message": SENTINEL,  # content
        }
    ).encode()


def pre_compact(
    trigger: str = "manual",
    custom_instructions: str = "",
) -> bytes:
    return json.dumps(
        {
            "hook_event_name": "PreCompact",
            "session_id": SESSION_ID,
            "transcript_path": TRANSCRIPT,
            "cwd": CWD,
            "trigger": trigger,
            "custom_instructions": custom_instructions,  # presence-only
        }
    ).encode()


def stop(stop_hook_active: bool = False, cwd: str = CWD) -> bytes:
    return json.dumps(
        {
            "hook_event_name": "Stop",
            "session_id": SESSION_ID,
            "transcript_path": TRANSCRIPT,
            "cwd": cwd,
            "stop_hook_active": stop_hook_active,
            "last_assistant_message": SENTINEL,  # content
        }
    ).encode()


def session_end(reason: str = "prompt_input_exit") -> bytes:
    return json.dumps(
        {
            "hook_event_name": "SessionEnd",
            "session_id": SESSION_ID,
            "transcript_path": TRANSCRIPT,
            "cwd": CWD,
            "reason": reason,
        }
    ).encode()
