"""JSONL transcript fixture builders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def assistant_record(
    timestamp: str,
    message_id: str,
    *,
    model: str = "claude-opus-4-7",
    stop_reason: str = "tool_use",
    input_tokens: int = 100,
    output_tokens: int = 50,
    cache_read: int = 0,
    cache_creation: int = 0,
    cache_1h: int = 0,
    cache_5m: int = 0,
    web_search: int = 0,
    web_fetch: int = 0,
    service_tier: str = "standard",
    speed: str = "standard",
    inference_geo: str = "",
    sentinel_in_content: str | None = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {
        "id": message_id,
        "model": model,
        "stop_reason": stop_reason,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_creation_input_tokens": cache_creation,
            "cache_read_input_tokens": cache_read,
            "cache_creation": {
                "ephemeral_1h_input_tokens": cache_1h,
                "ephemeral_5m_input_tokens": cache_5m,
            },
            "server_tool_use": {
                "web_search_requests": web_search,
                "web_fetch_requests": web_fetch,
            },
            "service_tier": service_tier,
            "speed": speed,
            "inference_geo": inference_geo,
        },
        "content": [
            # Content block — must never leak into the aggregator.
            {"type": "text", "text": sentinel_in_content or "harmless text"},
        ],
    }
    return {
        "type": "assistant",
        "uuid": f"uuid-{message_id}",
        "timestamp": timestamp,
        "requestId": f"req-{message_id}",
        "sessionId": "sess-xyz",
        "isSidechain": False,
        "message": message,
    }


def task_create_record(
    timestamp: str,
    *,
    subject: str = "Some plan task",
    description: str = "Detailed description that must never leak to the backend.",
    active_form: str = "Doing the task",
    sessionId: str = "sess-xyz",
) -> dict[str, Any]:
    """One assistant turn that fires a single TaskCreate tool_use.

    `subject`, `description`, `activeForm` are present so privacy tests can
    plant sentinels and assert nothing leaks into emitted attributes.
    """
    return {
        "type": "assistant",
        "uuid": f"uuid-tc-{timestamp}",
        "timestamp": timestamp,
        "sessionId": sessionId,
        "isSidechain": False,
        "message": {
            "id": f"msg-tc-{timestamp}",
            "model": "claude-opus-4-7",
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "content": [
                {
                    "type": "tool_use",
                    "id": f"toolu-tc-{timestamp}",
                    "name": "TaskCreate",
                    "input": {
                        "subject": subject,
                        "description": description,
                        "activeForm": active_form,
                    },
                }
            ],
        },
    }


def task_update_record(
    timestamp: str,
    *,
    task_id: str,
    status: str,
    sessionId: str = "sess-xyz",
) -> dict[str, Any]:
    return {
        "type": "assistant",
        "uuid": f"uuid-tu-{timestamp}-{task_id}",
        "timestamp": timestamp,
        "sessionId": sessionId,
        "isSidechain": False,
        "message": {
            "id": f"msg-tu-{timestamp}-{task_id}",
            "model": "claude-opus-4-7",
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "content": [
                {
                    "type": "tool_use",
                    "id": f"toolu-tu-{timestamp}-{task_id}",
                    "name": "TaskUpdate",
                    "input": {"taskId": task_id, "status": status},
                }
            ],
        },
    }


def user_record(timestamp: str, sentinel: str | None = None) -> dict[str, Any]:
    return {
        "type": "user",
        "timestamp": timestamp,
        "content": sentinel or "what is 2+2?",
    }


def compact_boundary_record(
    timestamp: str,
    *,
    trigger: str = "manual",
    pre_tokens: int = 72714,
    post_tokens: int = 3086,
    duration_ms: int = 82199,
) -> dict[str, Any]:
    return {
        "type": "system",
        "subtype": "compact_boundary",
        "timestamp": timestamp,
        "compactMetadata": {
            "trigger": trigger,
            "preTokens": pre_tokens,
            "postTokens": post_tokens,
            "durationMs": duration_ms,
            "preCompactDiscoveredTools": ["TaskCreate"],
        },
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path
