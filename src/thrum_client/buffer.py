"""On-disk turn buffer — per-session state for the hook handler.

One file per session at `<buffers_dir>/<session_id>.json`. The schema is
a fixed set of top-level keys; unknown keys trigger a reject-and-delete
on read (NFR-319 defence-in-depth: prevents a buggy prior write or a
planted file from leaking content back into our process).
"""

from __future__ import annotations

import json
import re
import sys
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

if sys.platform != "win32":
    import fcntl


_SESSION_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")

BUFFER_TOP_KEYS: frozenset[str] = frozenset(
    {
        "version",
        "session_id",
        "cwd",
        "model",
        "transcript_path",
        "turn",
        "subagents",
        "compact_pending",
        "created_at",
        "updated_at",
        # FR-218b — populated for codex-cli buffers, omitted/None for claude-code.
        "source_tool",
        "turn_id",
        # FR-218f — populated for cursor buffers only. `cursor_generation_index`
        # is the per-conversation generation counter (incremented on each
        # `beforeSubmitPrompt` event) used to pair stop-hook emits to the
        # correct transcript aggregate (Fix #2 flush-race mitigation).
        "cursor_conversation_id",
        "cursor_generation_id",
        "cursor_generation_index",
        "cursor_workspace_roots",
        "cursor_version",
    }
)

BUFFER_VERSION = 1

CLAUDE_SOURCE_TOOL = "claude-code"
CODEX_SOURCE_TOOL = "codex-cli"
CURSOR_SOURCE_TOOL = "cursor"


class BufferError(Exception):
    """Raised when a buffer file fails schema validation."""


def _validate_session_id(session_id: str) -> None:
    if not _SESSION_ID_RE.match(session_id):
        raise BufferError(f"invalid session_id shape: {session_id!r}")


def buffer_path(buffers_dir: Path, session_id: str) -> Path:
    _validate_session_id(session_id)
    return buffers_dir / f"{session_id}.json"


def new_buffer(
    session_id: str,
    cwd: str,
    *,
    model: str | None = None,
    transcript_path: str | None = None,
    source_tool: str = CLAUDE_SOURCE_TOOL,
    turn_id: str | None = None,
) -> dict[str, Any]:
    now = datetime.now(UTC).isoformat()
    buf: dict[str, Any] = {
        "version": BUFFER_VERSION,
        "session_id": session_id,
        "cwd": cwd,
        "model": model,
        "transcript_path": transcript_path,
        "turn": None,
        "subagents": {},
        "compact_pending": None,
        "created_at": now,
        "updated_at": now,
    }
    # Only set the Codex fields when relevant — keeps Claude-only buffers
    # byte-identical to the pre-D1 shape.
    if source_tool != CLAUDE_SOURCE_TOOL:
        buf["source_tool"] = source_tool
    if turn_id is not None:
        buf["turn_id"] = turn_id
    return buf


def load_buffer(buffers_dir: Path, session_id: str) -> dict[str, Any] | None:
    path = buffer_path(buffers_dir, session_id)
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise BufferError(f"unreadable buffer: {type(exc).__name__}")
    if not isinstance(data, dict):
        raise BufferError("buffer root is not a JSON object")
    extra = set(data.keys()) - BUFFER_TOP_KEYS
    if extra:
        raise BufferError(f"unknown buffer keys: {sorted(extra)}")
    return data


def save_buffer(buffers_dir: Path, data: dict[str, Any]) -> None:
    data["updated_at"] = datetime.now(UTC).isoformat()
    path = buffer_path(buffers_dir, data["session_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
    tmp.replace(path)


def delete_buffer(buffers_dir: Path, session_id: str) -> bool:
    path = buffer_path(buffers_dir, session_id)
    if path.exists():
        path.unlink()
        return True
    return False


@contextmanager
def buffer_lock(buffers_dir: Path, session_id: str) -> Iterator[None]:
    """Serialise load→mutate→save on a single session's buffer.

    Claude Code currently runs hooks serially per session, so the hot race
    is theoretical. The atomic-rename in save_buffer already prevents
    partial writes, but two interleaved load→save cycles could still clobber
    each other's tool-name lists. POSIX `fcntl.flock` on a companion
    `.lock` file closes that window with zero cross-process state.

    Windows has no equivalent that matches `LOCK_EX`'s indefinite-block
    semantics (`msvcrt.locking(LK_LOCK)` raises after 10s under contention,
    which is worse than no lock for our serial-per-session use case), so
    the lock is a no-op there. Acceptable because the race is theoretical.
    """
    _validate_session_id(session_id)
    buffers_dir.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        yield
        return
    lock_path = buffers_dir / f"{session_id}.lock"
    with lock_path.open("w") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
