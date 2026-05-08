"""Codex CLI hook registration tests — FR-218b/c/d.

Covers:
- Version gate (FR-218c): blocks `< 0.124.0`, passes `>= 0.124.0`.
- TOML round-trip preservation (FR-218d): `[projects.<path>]` and `[tui.*]`
  blocks survive install→uninstall byte-for-byte.
- Idempotent merge: running merge twice produces a single entry per event.
- Surgical unmerge: third-party entries in the same hook array survive.
"""

from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path
from types import SimpleNamespace

import tomlkit

from thrum_client.codex_config import (
    CODEX_HOOK_EVENTS,
    check_codex_version,
    merge_codex_hooks,
    unmerge_codex_hooks,
)


# ---- Version gate (FR-218c) ----


def _fake_runner(stdout: str, stderr: str = "", returncode: int = 0):
    def runner(*args, **kwargs):
        return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)
    return runner


def test_version_check_passes_at_minimum():
    runner = _fake_runner("codex-cli 0.124.0\n")
    assert check_codex_version(binary="/usr/bin/codex", runner=runner) is True


def test_version_check_passes_above_minimum():
    runner = _fake_runner("codex-cli 0.125.3\n")
    assert check_codex_version(binary="/usr/bin/codex", runner=runner) is True


def test_version_check_blocks_below_minimum():
    """0.123.0 has the codex_hooks flag but no inline [hooks] support."""
    runner = _fake_runner("codex-cli 0.123.0\n")
    assert check_codex_version(binary="/usr/bin/codex", runner=runner) is False


def test_version_check_returns_false_when_binary_missing(monkeypatch):
    """Fail-closed: no codex on PATH → no registration attempt.
    `binary=None` (the default) triggers shutil.which lookup; mock it to None."""
    monkeypatch.setattr("thrum_client.codex_config.shutil.which", lambda _: None)
    assert check_codex_version() is False


def test_version_check_returns_false_on_unparseable_output():
    runner = _fake_runner("garbage\n")
    assert check_codex_version(binary="/usr/bin/codex", runner=runner) is False


def test_version_check_returns_false_on_subprocess_error():
    def boom(*a, **k):
        raise subprocess.SubprocessError("oops")
    assert check_codex_version(binary="/usr/bin/codex", runner=boom) is False


# ---- merge_codex_hooks ----


_USER_BLOCKS = textwrap.dedent("""\
    [projects."/Users/me/work/project-a"]
    trust_level = "trusted"

    [projects."/Users/me/work/project-b"]
    trust_level = "ask"

    [tui.model_availability_nux]
    "gpt-5.5" = 1
    """)


def _write_user_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_USER_BLOCKS)


def test_merge_creates_six_hook_events_in_fresh_file(tmp_path: Path):
    cfg = tmp_path / "config.toml"
    changed = merge_codex_hooks(cfg, "/usr/local/bin/thrum-hook")
    assert changed is True

    doc = tomlkit.parse(cfg.read_text())
    assert "hooks" in doc
    for event in CODEX_HOOK_EVENTS:
        assert event in doc["hooks"], f"{event} missing"
        entries = doc["hooks"][event]
        assert len(entries) == 1
        # Matcher-group schema: {matcher, hooks: [{type, command}]}
        group = entries[0]
        assert group["matcher"] == "*"
        assert len(group["hooks"]) == 1
        assert group["hooks"][0]["type"] == "command"
        assert group["hooks"][0]["command"] == "/usr/local/bin/thrum-hook"


def test_merge_enables_hooks_feature_flag(tmp_path: Path):
    """`codex features list` shows the flag as stable but that's a maturity
    label — the runtime requires `[features] hooks = true` in config.toml.
    Without it hooks compile in but never load (verified empirically
    against codex-cli 0.125.0). Older Codex releases used `codex_hooks`;
    newer releases deprecated that name in favour of `hooks`."""
    cfg = tmp_path / "config.toml"
    merge_codex_hooks(cfg, "/usr/local/bin/thrum-hook")
    doc = tomlkit.parse(cfg.read_text())
    assert doc["features"]["hooks"] is True
    assert "codex_hooks" not in doc["features"]


def test_merge_migrates_legacy_codex_hooks_flag(tmp_path: Path):
    """If a previous install wrote `[features] codex_hooks = true`, replace
    it with `hooks = true` so newer Codex stops emitting the deprecation
    warning."""
    cfg = tmp_path / "config.toml"
    cfg.write_text(textwrap.dedent("""\
        [features]
        codex_hooks = true
        """))

    merge_codex_hooks(cfg, "/usr/local/bin/thrum-hook")

    doc = tomlkit.parse(cfg.read_text())
    assert doc["features"]["hooks"] is True
    assert "codex_hooks" not in doc["features"]


def test_merge_preserves_user_projects_and_tui_blocks(tmp_path: Path):
    cfg = tmp_path / "config.toml"
    _write_user_config(cfg)

    merge_codex_hooks(cfg, "/usr/local/bin/thrum-hook")

    doc = tomlkit.parse(cfg.read_text())
    assert doc["projects"]["/Users/me/work/project-a"]["trust_level"] == "trusted"
    assert doc["projects"]["/Users/me/work/project-b"]["trust_level"] == "ask"
    assert doc["tui"]["model_availability_nux"]["gpt-5.5"] == 1


def test_merge_is_idempotent(tmp_path: Path):
    cfg = tmp_path / "config.toml"
    first = merge_codex_hooks(cfg, "/usr/local/bin/thrum-hook")
    second = merge_codex_hooks(cfg, "/usr/local/bin/thrum-hook")

    assert first is True
    assert second is False  # no-op

    doc = tomlkit.parse(cfg.read_text())
    for event in CODEX_HOOK_EVENTS:
        assert len(doc["hooks"][event]) == 1


def test_merge_appends_alongside_third_party_matcher_group(tmp_path: Path):
    """Third-party tool with the matcher-group schema must coexist."""
    cfg = tmp_path / "config.toml"
    cfg.write_text(textwrap.dedent("""\
        [hooks]
        Stop = [
            { matcher = "*", hooks = [{ type = "command", command = "/opt/other/hook" }] },
        ]
        """))

    merge_codex_hooks(cfg, "/usr/local/bin/thrum-hook")

    doc = tomlkit.parse(cfg.read_text())
    stop_groups = doc["hooks"]["Stop"]
    commands = [g["hooks"][0]["command"] for g in stop_groups]
    assert "/opt/other/hook" in commands
    assert "/usr/local/bin/thrum-hook" in commands


def test_merge_is_idempotent_against_legacy_flat_schema(tmp_path: Path):
    """If the user's config.toml carries a pre-existing entry in the flat
    legacy shape (`{type, command}` directly), don't double-add ours."""
    cfg = tmp_path / "config.toml"
    cfg.write_text(textwrap.dedent("""\
        [hooks]
        Stop = [
            { type = "command", command = "/usr/local/bin/thrum-hook" },
        ]
        """))

    changed = merge_codex_hooks(cfg, "/usr/local/bin/thrum-hook")
    # We DO change the file (need to add features flag + the other 5 events)
    # but Stop should NOT gain a duplicate thrum-hook entry.
    assert changed is True
    doc = tomlkit.parse(cfg.read_text())
    stop_groups = doc["hooks"]["Stop"]
    commands = []
    for g in stop_groups:
        if "hooks" in g:
            commands.extend(h["command"] for h in g["hooks"])
        else:
            commands.append(g["command"])
    assert commands.count("/usr/local/bin/thrum-hook") == 1


# ---- unmerge_codex_hooks ----


def test_unmerge_removes_only_thrum_matcher_groups(tmp_path: Path):
    cfg = tmp_path / "config.toml"
    cfg.write_text(textwrap.dedent("""\
        [hooks]
        Stop = [
            { matcher = "*", hooks = [{ type = "command", command = "/opt/other/hook" }] },
            { matcher = "*", hooks = [{ type = "command", command = "/usr/local/bin/thrum-hook" }] },
        ]
        """))

    changed = unmerge_codex_hooks(cfg)
    assert changed is True

    doc = tomlkit.parse(cfg.read_text())
    stop_groups = doc["hooks"]["Stop"]
    assert len(stop_groups) == 1
    assert stop_groups[0]["hooks"][0]["command"] == "/opt/other/hook"


def test_unmerge_tolerates_legacy_flat_schema(tmp_path: Path):
    """Backward compat: an old config produced by the pre-fix merge would
    have flat-schema entries. unmerge must still remove those too."""
    cfg = tmp_path / "config.toml"
    cfg.write_text(textwrap.dedent("""\
        [hooks]
        Stop = [
            { type = "command", command = "/opt/other/hook" },
            { type = "command", command = "/usr/local/bin/thrum-hook" },
        ]
        """))

    unmerge_codex_hooks(cfg)
    doc = tomlkit.parse(cfg.read_text())
    stop_entries = doc["hooks"]["Stop"]
    commands = [e["command"] for e in stop_entries]
    assert commands == ["/opt/other/hook"]


def test_unmerge_drops_event_key_when_array_becomes_empty(tmp_path: Path):
    cfg = tmp_path / "config.toml"
    merge_codex_hooks(cfg, "/usr/local/bin/thrum-hook")
    unmerge_codex_hooks(cfg)

    doc = tomlkit.parse(cfg.read_text())
    # [hooks] table should be gone since every event was Thrum-only.
    assert "hooks" not in doc


def test_install_then_uninstall_preserves_user_blocks(tmp_path: Path):
    """T14 threat-model test: install + uninstall must leave
    `[projects.<path>]` and `[tui.*]` exactly as the user wrote them.
    Note: `[features] hooks = true` (set on install) is intentionally
    left behind on uninstall — flipping it would surprise the user if they
    have non-Thrum hooks registered."""
    cfg = tmp_path / "config.toml"
    _write_user_config(cfg)

    merge_codex_hooks(cfg, "/usr/local/bin/thrum-hook")
    unmerge_codex_hooks(cfg)

    doc = tomlkit.parse(cfg.read_text())
    assert doc["projects"]["/Users/me/work/project-a"]["trust_level"] == "trusted"
    assert doc["projects"]["/Users/me/work/project-b"]["trust_level"] == "ask"
    assert doc["tui"]["model_availability_nux"]["gpt-5.5"] == 1
    assert "hooks" not in doc
    # hooks feature flag is sticky on uninstall — see merge docstring.
    assert doc["features"]["hooks"] is True


def test_unmerge_returns_false_when_file_missing(tmp_path: Path):
    cfg = tmp_path / "nonexistent.toml"
    assert unmerge_codex_hooks(cfg) is False


def test_unmerge_returns_false_when_no_hooks_table(tmp_path: Path):
    cfg = tmp_path / "config.toml"
    _write_user_config(cfg)
    assert unmerge_codex_hooks(cfg) is False


def test_unmerge_does_not_remove_third_party_named_thrum_hook(tmp_path: Path):
    """A user-wrapped hook at `/opt/my-thrum-hook-wrapper.sh` has a
    different basename ('my-thrum-hook-wrapper.sh') from `thrum-hook`,
    so the basename-equality check leaves it alone."""
    cfg = tmp_path / "config.toml"
    cfg.write_text(textwrap.dedent("""\
        [hooks]
        Stop = [
            { matcher = "*", hooks = [{ type = "command", command = "/opt/my-thrum-hook-wrapper.sh" }] },
        ]
        """))

    changed = unmerge_codex_hooks(cfg)
    assert changed is False
    doc = tomlkit.parse(cfg.read_text())
    group = doc["hooks"]["Stop"][0]
    assert group["hooks"][0]["command"] == "/opt/my-thrum-hook-wrapper.sh"
