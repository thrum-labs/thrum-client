"""Cursor IDE hook registration — FR-218f.

Round-trips `~/.cursor/hooks.json` with the stdlib `json` module. Cursor's
`hooks.json` schema (per `~/.cursor/skills-cursor/create-hook/SKILL.md`)
is **strict JSON** with an explicit `version: 1` framing — not JSONC,
not VS Code-flavoured `settings.json`. Comments and trailing commas would
break the file.

Cursor watches `hooks.json` and hot-reloads on save (no `cursor` CLI restart
needed). The atomic temp-then-rename write pattern in `_write_atomic`
guarantees Cursor never sees a partial file mid-write.

Ownership rule (mirrors Codex / Claude): a hook entry is "ours" iff its
`command` field's basename equals `thrum-hook`. `unmerge_cursor_hooks`
removes only owned entries so a third-party tool registering its own hook
in the same JSON survives our uninstall. Cursor's spec also tolerates
arbitrary additional keys per entry, so we mark each owned entry with
`"_thrumManaged": true` as a defense-in-depth signal — useful when a user
has aliased the binary to a non-`thrum-hook` name.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


# Lifecycle subset Thrum registers.
# Tab events (beforeTabFileRead, afterTabFileEdit) excluded — keystroke flood.
# MCP events excluded — defer until usage data justifies inclusion.
CURSOR_HOOK_EVENTS: tuple[str, ...] = (
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
)

CURSOR_HOOKS_SCHEMA_VERSION: int = 1

_THRUM_HOOK_BASENAME = "thrum-hook"
_THRUM_MANAGED_KEY = "_thrumManaged"


def cursor_home() -> Path:
    """`~/.cursor` — Cursor's user-scope config directory.

    Honours `CURSOR_HOME` env var for test injection. The IDE itself does
    not currently read this env var (verified empirically via `~/.cursor/`
    inspection during the spike), but the override is useful for tmp_path
    test isolation and for users with non-default install layouts.
    """
    raw = os.environ.get("CURSOR_HOME")
    return Path(raw) if raw else Path.home() / ".cursor"


def cursor_hooks_path() -> Path:
    return cursor_home() / "hooks.json"


def _is_thrum_command(cmd: str) -> bool:
    return Path(cmd).name == _THRUM_HOOK_BASENAME


def _is_thrum_managed(entry: object) -> bool:
    """An entry is Thrum-owned iff EITHER its `command` basename matches
    OR it carries our explicit `_thrumManaged: true` sentinel. Sentinel-
    based ownership survives the user aliasing the binary to a non-default
    name."""
    if not isinstance(entry, dict):
        return False
    if entry.get(_THRUM_MANAGED_KEY) is True:
        return True
    cmd = entry.get("command")
    return isinstance(cmd, str) and _is_thrum_command(cmd)


def _entries_for_event(hooks_block: dict, event: str) -> list:
    raw = hooks_block.get(event)
    return raw if isinstance(raw, list) else []


def _make_entry(command: str) -> dict[str, Any]:
    """Per-event entry shape: `{command, _thrumManaged}`. Cursor accepts
    additional keys per entry (see SKILL.md § Hooks File Format)."""
    return {"command": command, _THRUM_MANAGED_KEY: True}


def _write_atomic(path: Path, content: str) -> None:
    """Write via temp-then-rename so Cursor's hot-reload watcher never
    sees a partial JSON file. macOS `os.rename` is atomic within the same
    filesystem; a failed mid-write leaves the original intact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def merge_cursor_hooks(hooks_path: Path, command: str) -> bool:
    """Insert Thrum hook entries for the registered Cursor events. Idempotent.

    Creates `hooks.json` with `{version: 1, hooks: {}}` if it doesn't exist;
    otherwise loads the existing file and patches in our entries. Preserves
    every unrelated key in the doc (other top-level keys, other event
    arrays, other entries within registered event arrays).

    Returns True iff the file was changed.
    """
    if hooks_path.exists():
        try:
            doc = json.loads(hooks_path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"hooks.json at {hooks_path} is not valid JSON: {exc}"
            ) from exc
        if not isinstance(doc, dict):
            raise ValueError(
                f"hooks.json at {hooks_path} top-level must be an object"
            )
    else:
        doc = {}

    changed = False
    if doc.get("version") != CURSOR_HOOKS_SCHEMA_VERSION:
        doc["version"] = CURSOR_HOOKS_SCHEMA_VERSION
        changed = True

    hooks_block = doc.get("hooks")
    if not isinstance(hooks_block, dict):
        hooks_block = {}
        doc["hooks"] = hooks_block
        changed = True

    for event in CURSOR_HOOK_EVENTS:
        existing = _entries_for_event(hooks_block, event)
        # Already-managed-with-this-command? Skip.
        if any(
            isinstance(e, dict) and e.get("command") == command
            for e in existing
        ):
            continue
        new_list = list(existing) + [_make_entry(command)]
        hooks_block[event] = new_list
        changed = True

    if not changed:
        return False
    _write_atomic(hooks_path, json.dumps(doc, indent=2) + "\n")
    return True


def unmerge_cursor_hooks(hooks_path: Path) -> bool:
    """Remove every Thrum-owned hook entry. Returns True iff a change was made.

    Owned = `_thrumManaged: true` OR `command` basename equals `thrum-hook`.
    Third-party entries in the same arrays survive; arrays that become empty
    are deleted along with their event key. The `hooks` block is removed if
    it ends up empty, but `version: 1` stays in the file (it's a top-level
    schema declaration, not Thrum-owned). If after cleanup the doc has only
    `version`, the whole file is deleted so we don't leave a stray
    Thrum-shaped artifact behind.
    """
    if not hooks_path.exists():
        return False

    try:
        doc = json.loads(hooks_path.read_text())
    except json.JSONDecodeError:
        # Corrupt file — leave alone; user must repair manually.
        return False
    if not isinstance(doc, dict):
        return False

    hooks_block = doc.get("hooks")
    if not isinstance(hooks_block, dict):
        return False

    changed = False
    for event in list(hooks_block.keys()):
        entries = hooks_block[event]
        if not isinstance(entries, list):
            continue
        kept = [e for e in entries if not _is_thrum_managed(e)]
        if len(kept) == len(entries):
            continue
        changed = True
        if kept:
            hooks_block[event] = kept
        else:
            del hooks_block[event]

    if not changed:
        return False

    if not hooks_block:
        del doc["hooks"]

    # If only `version` is left, remove the whole file (was Thrum-only).
    if set(doc.keys()) <= {"version"}:
        try:
            hooks_path.unlink()
        except OSError:
            pass
        return True

    _write_atomic(hooks_path, json.dumps(doc, indent=2) + "\n")
    return True
