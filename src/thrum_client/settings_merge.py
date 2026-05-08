"""Additive merge of Thrum hook entries into `~/.claude/settings.json`.

Contract:
- NEVER overwrite an existing `hooks` block.
- Inserts entries for the Claude Code settings-schema-compatible FR-216 events.
- Idempotent: running merge twice with the same `command` is a no-op.
- Preserves every non-hook key (`theme`, `env`, `permissions`, ...).

Claude Code hook JSON shape (observed from docs):
{
  "hooks": {
    "UserPromptSubmit": [
      {"hooks": [{"type": "command", "command": "/path/to/thrum-hook"}]}
    ],
    ...
  }
}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# Thrum's handler can process the full FR-216 runtime set below, but Claude Code
# 2.1.119 rejects three of these names during settings.json schema validation
# and then skips the whole settings file. Keep the default registration list
# limited to keys accepted by the settings schema until upstream validation is
# fixed.
RUNTIME_HOOK_EVENTS: tuple[str, ...] = (
    "UserPromptSubmit",
    "PreToolUse",
    "PostToolUse",
    "PostToolUseFailure",
    "SubagentStart",
    "SubagentStop",
    "PreCompact",
    "Stop",
    "StopFailure",
    "SessionEnd",
)

SETTINGS_SCHEMA_REJECTED_HOOK_EVENTS: tuple[str, ...] = (
    "PostToolUseFailure",
    "SubagentStart",
    "StopFailure",
)

HOOK_EVENTS: tuple[str, ...] = tuple(
    event
    for event in RUNTIME_HOOK_EVENTS
    if event not in SETTINGS_SCHEMA_REJECTED_HOOK_EVENTS
)


_OUR_BASENAME = "thrum-hook"


def _is_our_hook(command: str) -> bool:
    """Match only hooks we installed. The basename check is strict enough to
    distinguish `/usr/local/bin/thrum-hook` (ours) from a user-named
    `/opt/my-thrum-hook-wrapper.sh` (theirs).
    """
    return Path(command).name == _OUR_BASENAME


def _load(settings_path: Path) -> dict[str, Any]:
    if not settings_path.exists():
        return {}
    with settings_path.open(encoding="utf-8") as f:
        raw = f.read().strip() or "{}"
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"{settings_path} is not a JSON object")
    return parsed


def _already_registered(entries: list[dict[str, Any]], command: str) -> bool:
    for entry in entries:
        for h in entry.get("hooks", []) or []:
            if isinstance(h, dict) and h.get("command") == command:
                return True
    return False


def _remove_schema_rejected_own_hooks(
    hooks: dict[str, Any],
    command: str,
) -> bool:
    """Remove stale Thrum registrations from events Claude settings rejects."""
    changed = False
    for event in SETTINGS_SCHEMA_REJECTED_HOOK_EVENTS:
        entries = hooks.get(event)
        if not isinstance(entries, list):
            continue

        new_entries: list[dict[str, Any]] = []
        for entry in entries:
            inner = entry.get("hooks") if isinstance(entry, dict) else None
            if not isinstance(inner, list):
                new_entries.append(entry)
                continue

            kept = []
            for h in inner:
                if (
                    isinstance(h, dict)
                    and isinstance(h.get("command"), str)
                    and (h["command"] == command or _is_our_hook(h["command"]))
                ):
                    changed = True
                    continue
                kept.append(h)

            if kept:
                new_entry = dict(entry)
                new_entry["hooks"] = kept
                new_entries.append(new_entry)

        if new_entries:
            hooks[event] = new_entries
        else:
            del hooks[event]
            changed = True
    return changed


def merge_hooks(
    settings_path: Path,
    command: str,
    events: tuple[str, ...] = HOOK_EVENTS,
) -> bool:
    """Ensure `command` is registered for each event in `events`.

    Returns True if any change was made, False if the file already had
    every entry and no stale rejected registrations needed removal.
    """
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    data = _load(settings_path)
    hooks = data.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        raise ValueError("settings.json hooks is not an object — refusing to modify")
    changed = _remove_schema_rejected_own_hooks(hooks, command)

    for event in events:
        entries = hooks.setdefault(event, [])
        if not isinstance(entries, list):
            raise ValueError(
                f"settings.json hooks.{event!r} is not a list — refusing to modify"
            )
        if _already_registered(entries, command):
            continue
        entries.append({"hooks": [{"type": "command", "command": command}]})
        changed = True

    if not changed:
        return False

    tmp = settings_path.with_suffix(settings_path.suffix + ".thrum-tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    tmp.replace(settings_path)
    return True


def unmerge_hooks(
    settings_path: Path,
) -> bool:
    """Remove every hook entry we installed from `settings.json`.

    Uses the basename (not a substring) to decide ownership so user-named
    hooks that happen to contain "thrum-hook" are preserved. Other tools'
    entries are always preserved.

    Returns True if anything was removed.
    """
    if not settings_path.exists():
        return False
    data = _load(settings_path)
    hooks = data.get("hooks")
    if not isinstance(hooks, dict):
        return False

    changed = False
    for event in list(hooks.keys()):
        entries = hooks[event]
        if not isinstance(entries, list):
            continue
        new_entries: list[dict[str, Any]] = []
        for entry in entries:
            inner = entry.get("hooks") if isinstance(entry, dict) else None
            if not isinstance(inner, list):
                new_entries.append(entry)
                continue
            kept = [
                h
                for h in inner
                if not (
                    isinstance(h, dict)
                    and isinstance(h.get("command"), str)
                    and _is_our_hook(h["command"])
                )
            ]
            if len(kept) != len(inner):
                changed = True
            if kept:
                new_entry = dict(entry)
                new_entry["hooks"] = kept
                new_entries.append(new_entry)
        if new_entries:
            hooks[event] = new_entries
        else:
            del hooks[event]
            changed = True

    if not hooks:
        data.pop("hooks", None)

    if not changed:
        return False

    tmp = settings_path.with_suffix(settings_path.suffix + ".thrum-tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    tmp.replace(settings_path)
    return True
