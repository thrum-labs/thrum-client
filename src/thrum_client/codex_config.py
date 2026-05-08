"""Codex CLI hook registration — FR-218b/c/d.

Round-trips `${CODEX_HOME:-~/.codex}/config.toml` with `tomlkit` so unrelated
tables (`[projects.<path>]` trust state, `[tui.*]` UI preferences, anything
the user has manually configured) survive install→uninstall byte-for-byte
except for the entries we own.

Ownership is decided by the same basename rule as the Claude path
(`settings_merge.py`): a hook is "ours" iff its `command` field's basename
equals `thrum-hook`. `unmerge_codex_hooks` removes only owned entries so a
third-party tool registering its own hook in the same TOML survives our
uninstall.

Version gate: Codex hooks landed under a stable `codex_hooks` flag in
0.123.0 but inline `[hooks]` table support in `config.toml` only arrived
in 0.124.0 — so the gate is `>= 0.124.0` (FR-218c). Later releases
renamed the flag from `codex_hooks` to `hooks` and started emitting a
deprecation warning for the old name; we write the new key and migrate
any pre-existing `codex_hooks` entry.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import tomlkit


CODEX_HOOK_EVENTS: tuple[str, ...] = (
    "SessionStart",
    "UserPromptSubmit",
    "PreToolUse",
    "PostToolUse",
    "PermissionRequest",
    "Stop",
)

MIN_CODEX_VERSION: tuple[int, int, int] = (0, 124, 0)

_THRUM_HOOK_BASENAME = "thrum-hook"


def codex_home() -> Path:
    """`~/.codex` by default; honour `CODEX_HOME` env per Codex docs."""
    raw = os.environ.get("CODEX_HOME")
    return Path(raw) if raw else Path.home() / ".codex"


def codex_config_path() -> Path:
    return codex_home() / "config.toml"


def _is_thrum_command(cmd: str) -> bool:
    return Path(cmd).name == _THRUM_HOOK_BASENAME


_VERSION_RE = re.compile(r"(\d+)\.(\d+)\.(\d+)")


def check_codex_version(
    min_version: tuple[int, int, int] = MIN_CODEX_VERSION,
    *,
    binary: str | None = None,
    runner=subprocess.run,
) -> bool:
    """True iff `codex --version` reports `>= min_version`.

    Fails closed — missing binary, subprocess error, or unparseable output
    all return False. `binary` and `runner` are injection points for tests;
    in production both default to `shutil.which("codex")` and
    `subprocess.run`.
    """
    bin_path = binary or shutil.which("codex")
    if not bin_path:
        return False
    try:
        result = runner(
            [bin_path, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (subprocess.SubprocessError, OSError):
        return False
    match = _VERSION_RE.search((result.stdout or "") + (result.stderr or ""))
    if not match:
        return False
    found = tuple(int(g) for g in match.groups())
    return found >= min_version


def _make_matcher_group(command: str) -> tomlkit.items.InlineTable:
    """`{ matcher = "*", hooks = [{ type = "command", command = "..." }] }`
    as a single TOML inline table — Codex's `MatcherGroup` schema (PR #18893).
    """
    inner = tomlkit.inline_table()
    inner["type"] = "command"
    inner["command"] = command
    inner_arr = tomlkit.array()
    inner_arr.append(inner)
    group = tomlkit.inline_table()
    group["matcher"] = "*"
    group["hooks"] = inner_arr
    return group


def _group_contains_command(group: object, command: str) -> bool:
    """True iff a MatcherGroup carries a hook with the given command. Tolerant
    to either the matcher-group shape `{matcher, hooks: [...]}` or the legacy
    flat shape `{type, command}` so re-running merge over an old config (or a
    third-party tool's pre-existing entry) stays idempotent."""
    if not isinstance(group, dict):
        return False
    inner_hooks = group.get("hooks")
    if isinstance(inner_hooks, list):
        return any(
            isinstance(h, dict) and h.get("command") == command for h in inner_hooks
        )
    return group.get("command") == command


def _ensure_codex_hooks_feature_flag(doc) -> bool:
    """Set `[features] hooks = true`. The flag is `stable` (per
    `codex features list`), but that's a maturity tag — the runtime state
    still requires opt-in via `[features]`. Without this, hooks compile in
    but never load. Verified empirically against 0.125.

    Older Codex releases used the name `codex_hooks`; newer releases
    renamed it to `hooks` and emit a deprecation warning for the old key.
    Migrate any existing `codex_hooks` entry to `hooks`.

    Returns True iff the doc was changed.
    """
    if "features" not in doc:
        doc["features"] = tomlkit.table()
    features = doc["features"]
    changed = False
    if "codex_hooks" in features:
        del features["codex_hooks"]
        changed = True
    if features.get("hooks") is not True:
        features["hooks"] = True
        changed = True
    return changed


def merge_codex_hooks(config_path: Path, command: str) -> bool:
    """Insert Thrum hook entries for the six Codex events plus enable the
    `hooks` feature flag. Idempotent.

    Schema is `{event: [{matcher, hooks: [{type, command}]}]}` — the same
    matcher-group form Claude Code uses, deserialized by Codex into
    `Vec<MatcherGroup>`. Anything flatter (e.g. `event: [{type, command}]`)
    parses as TOML but doesn't fire. Empirically verified against 0.125.

    Returns True iff the file was changed. Preserves every unrelated
    table (e.g. `[projects.<path>]`, `[tui.*]`) via tomlkit round-trip.
    """
    if config_path.exists():
        doc = tomlkit.parse(config_path.read_text(encoding="utf-8"))
    else:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        doc = tomlkit.document()

    changed = _ensure_codex_hooks_feature_flag(doc)

    if "hooks" not in doc:
        doc["hooks"] = tomlkit.table()
    hooks_table = doc["hooks"]

    for event in CODEX_HOOK_EVENTS:
        entries = hooks_table.get(event)
        if entries is None:
            arr = tomlkit.array()
            arr.append(_make_matcher_group(command))
            hooks_table[event] = arr
            changed = True
            continue

        # Existing array — append only if our command isn't already in any
        # existing matcher group (either schema shape).
        already = any(_group_contains_command(g, command) for g in entries)
        if already:
            continue
        entries.append(_make_matcher_group(command))
        changed = True

    if not changed:
        return False
    config_path.write_text(tomlkit.dumps(doc), encoding="utf-8")
    return True


def _matcher_group_owned_by_thrum(group: object) -> bool:
    """True iff a MatcherGroup contains only Thrum-owned hook commands.
    Handles both schemas (matcher-wrapped + legacy-flat) for forward and
    backward compat."""
    if not isinstance(group, dict):
        return False
    inner_hooks = group.get("hooks")
    if isinstance(inner_hooks, list):
        if not inner_hooks:
            return False
        return all(
            isinstance(h, dict)
            and isinstance(h.get("command"), str)
            and _is_thrum_command(h["command"])
            for h in inner_hooks
        )
    cmd = group.get("command")
    return isinstance(cmd, str) and _is_thrum_command(cmd)


def unmerge_codex_hooks(config_path: Path) -> bool:
    """Remove every Thrum-owned hook entry. Returns True iff a change was made.

    Owned = basename of `command` equals `thrum-hook`. Third-party entries
    in the same array survive; arrays that become empty are deleted along
    with their event key. The `[hooks]` table itself is removed if it ends
    up empty so we don't leave a stray header. The `[features] hooks =
    true` we set in merge is left in place — flipping it to false would
    surprise the user if they have non-Thrum hooks registered.
    """
    if not config_path.exists():
        return False
    doc = tomlkit.parse(config_path.read_text(encoding="utf-8"))
    if "hooks" not in doc:
        return False
    hooks_table = doc["hooks"]

    changed = False
    for event in list(hooks_table.keys()):
        entries = hooks_table[event]
        # tomlkit arrays are list-like
        try:
            iter(entries)
        except TypeError:
            continue

        # Collect indices to remove (iterate descending so removal is safe).
        to_remove = [
            i for i, g in enumerate(entries)
            if _matcher_group_owned_by_thrum(g)
        ]

        if not to_remove:
            continue

        for i in reversed(to_remove):
            del entries[i]
        changed = True

        if len(entries) == 0:
            del hooks_table[event]

    # If [hooks] is now empty, drop it entirely.
    if changed and not list(hooks_table.keys()):
        del doc["hooks"]

    if not changed:
        return False
    config_path.write_text(tomlkit.dumps(doc), encoding="utf-8")
    return True
