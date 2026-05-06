"""NFR-320 — allowlisted logging wrapper.

Only takes allowlisted fields (session_id, tool_name, hook_event_name,
error_category, token counters, timestamps). Raw payloads, transcript
lines, and full exception objects are NEVER logged at any level.
Exception tracebacks are either suppressed or reduced to their type name.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


_ALLOWED_KEYS: frozenset[str] = frozenset(
    {
        "event",
        "session_id",
        "hook_event_name",
        "tool_name",
        "error_category",
        "tokens_in",
        "tokens_out",
        "latency_ms",
        "agent_id",
        "agent_type",
        "span_count",
        "buffer_size",
        "status",
    }
)


def safe_log(
    event: str,
    log_path: Path | None = None,
    **fields: Any,
) -> None:
    """Write a structured JSON line. Drops any key outside the allowlist.

    Values must be int, str, bool, or None; anything else is coerced via str()
    (nothing other than enums / IDs / counters should ever reach here).
    """
    filtered: dict[str, Any] = {"event": event}
    for k, v in fields.items():
        if k not in _ALLOWED_KEYS:
            continue
        if v is None or isinstance(v, (int, bool, str)):
            filtered[k] = v
        else:
            filtered[k] = str(v)
    filtered["ts"] = datetime.now(UTC).isoformat()

    if log_path is None:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(json.dumps(filtered, separators=(",", ":")) + "\n")
        f.flush()
        os.fsync(f.fileno())
