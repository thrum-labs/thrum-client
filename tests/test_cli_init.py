from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import httpx
import pytest
from click.testing import CliRunner

from thrum_client.cli import main
from thrum_client.settings_merge import HOOK_EVENTS


@pytest.fixture
def mock_guest_register(monkeypatch):
    """Intercept httpx.post, return a canned guest-register payload."""
    captured = {"calls": 0, "last_url": None}

    class _Resp:
        status_code = 201

        def json(self) -> dict:
            return {
                "data": {
                    "user_id": "00000000-0000-0000-0000-000000000001",
                    "username": "guest_00000000",
                    "api_key": "tk_" + "a" * 32,
                }
            }

    def fake_post(url, *args, **kwargs):
        captured["calls"] += 1
        captured["last_url"] = url
        return _Resp()

    monkeypatch.setattr(httpx, "post", fake_post)
    return captured


def test_init_writes_token_file_mode_0600(tmp_home: Path, mock_guest_register, monkeypatch):
    monkeypatch.setenv("THRUM_HOOK_CMD", "/fake/bin/thrum-hook")
    runner = CliRunner()
    result = runner.invoke(main, ["init"])
    assert result.exit_code == 0, result.output

    token_path = tmp_home / ".config" / "thrum" / "token"
    assert token_path.exists()
    assert token_path.read_text() == "tk_" + "a" * 32
    mode = stat.S_IMODE(token_path.stat().st_mode)
    assert mode == 0o600, f"expected 0600, got {mode:o}"
    assert mock_guest_register["calls"] == 1


def test_init_merges_hooks_into_claude_settings(tmp_home: Path, mock_guest_register, monkeypatch):
    monkeypatch.setenv("THRUM_HOOK_CMD", "/fake/bin/thrum-hook")
    runner = CliRunner()
    result = runner.invoke(main, ["init"])
    assert result.exit_code == 0, result.output

    settings_path = tmp_home / ".claude" / "settings.json"
    data = json.loads(settings_path.read_text())
    for event in HOOK_EVENTS:
        cmds = [h["command"] for e in data["hooks"][event] for h in e["hooks"]]
        assert cmds == ["/fake/bin/thrum-hook"]


def test_init_skips_when_token_exists(tmp_home: Path, mock_guest_register, monkeypatch):
    monkeypatch.setenv("THRUM_HOOK_CMD", "/fake/bin/thrum-hook")
    # Pre-seed a token
    token_path = tmp_home / ".config" / "thrum" / "token"
    token_path.parent.mkdir(parents=True)
    token_path.write_text("tk_existing")
    os.chmod(token_path, 0o600)

    runner = CliRunner()
    result = runner.invoke(main, ["init"])
    assert result.exit_code == 0, result.output
    # httpx.post was not called
    assert mock_guest_register["calls"] == 0
    # Token unchanged
    assert token_path.read_text() == "tk_existing"
    # Existing installs still reconcile hooks, including compatibility cleanups.
    settings_path = tmp_home / ".claude" / "settings.json"
    data = json.loads(settings_path.read_text())
    assert set(data["hooks"].keys()) == set(HOOK_EVENTS)


def test_init_does_not_run_backfill_on_first_install(tmp_home: Path, monkeypatch):
    from tests.fixtures.otlp_receiver import Capture, capture_http_post
    from tests.fixtures.transcripts import assistant_record, write_jsonl

    # Seed one historical transcript with one end_turn — would be picked up
    # by backfill if init still ran it.
    projects = tmp_home / ".claude" / "projects" / "hash"
    projects.mkdir(parents=True)
    write_jsonl(
        projects / "sess-old.jsonl",
        [
            assistant_record(
                "2026-04-01T10:00:00Z",
                "msg_H",
                stop_reason="end_turn",
                input_tokens=42,
                output_tokens=7,
            ),
        ],
    )

    cap = Capture()
    collector_post, close = capture_http_post(cap)

    class _FakeResp:
        status_code = 201

        def json(self):
            return {
                "data": {
                    "user_id": "00000000-0000-0000-0000-000000000001",
                    "username": "guest_deadbeef",
                    "api_key": "tk_" + "a" * 32,
                }
            }

    def routing_post(url, *args, **kwargs):
        if "/auth/guest-register" in url:
            return _FakeResp()
        return collector_post(url, *args, **kwargs)

    monkeypatch.setattr(httpx, "post", routing_post)
    monkeypatch.setenv("THRUM_HOOK_CMD", "/fake/bin/thrum-hook")

    try:
        runner = CliRunner()
        result = runner.invoke(main, ["init"])
        assert result.exit_code == 0, result.output
        assert "Backfilled" not in result.output
        assert cap.all_attrs() == []
        assert not (tmp_home / ".config" / "thrum" / ".backfill_done").exists()
    finally:
        close()


def test_init_reports_backend_error(tmp_home: Path, monkeypatch):
    class _Resp:
        status_code = 500

        def json(self) -> dict:
            return {}

    def fake_post(url, *args, **kwargs):
        return _Resp()

    monkeypatch.setattr(httpx, "post", fake_post)
    runner = CliRunner()
    result = runner.invoke(main, ["init"])
    assert result.exit_code != 0
    assert "guest-register failed" in result.output


def test_init_registers_codex_hooks_when_version_ok(
    tmp_home: Path, mock_guest_register, monkeypatch
):
    """FR-218b/c — Codex >= 0.124.0 → hooks merge into ~/.codex/config.toml."""
    monkeypatch.setenv("THRUM_HOOK_CMD", "/fake/bin/thrum-hook")
    monkeypatch.setenv("CODEX_HOME", str(tmp_home / ".codex"))
    # Force version check to return True without actually running codex.
    monkeypatch.setattr(
        "thrum_client.cli.check_codex_version", lambda: True
    )

    runner = CliRunner()
    result = runner.invoke(main, ["init"])
    assert result.exit_code == 0, result.output
    assert "Codex hooks:" in result.output

    cfg = tmp_home / ".codex" / "config.toml"
    assert cfg.exists()
    import tomlkit

    doc = tomlkit.parse(cfg.read_text())
    assert "hooks" in doc
    # Feature flag must be set or hooks compile in but never load (see
    # codex_config._ensure_codex_hooks_feature_flag).
    assert doc["features"]["hooks"] is True
    # Six events registered in matcher-group schema.
    for event in (
        "SessionStart",
        "UserPromptSubmit",
        "PreToolUse",
        "PostToolUse",
        "PermissionRequest",
        "Stop",
    ):
        groups = doc["hooks"][event]
        flat_cmds = [
            h["command"] for g in groups for h in g.get("hooks", [])
        ]
        assert "/fake/bin/thrum-hook" in flat_cmds


def test_init_skips_codex_when_version_too_low(
    tmp_home: Path, mock_guest_register, monkeypatch
):
    """FR-218c — Codex missing or `< 0.124.0` → no config.toml written,
    clear skip message, Claude-side install still completes."""
    monkeypatch.setenv("THRUM_HOOK_CMD", "/fake/bin/thrum-hook")
    monkeypatch.setenv("CODEX_HOME", str(tmp_home / ".codex"))
    monkeypatch.setattr(
        "thrum_client.cli.check_codex_version", lambda: False
    )

    runner = CliRunner()
    result = runner.invoke(main, ["init"])
    assert result.exit_code == 0, result.output
    assert "skipping Codex hook registration" in result.output

    cfg = tmp_home / ".codex" / "config.toml"
    assert not cfg.exists()


def test_uninstall_removes_codex_hooks_when_present(
    tmp_home: Path, monkeypatch
):
    """FR-218d — uninstall calls unmerge_codex_hooks for the config.toml
    entries it owns; preserves user `[projects.*]` blocks."""
    import textwrap

    monkeypatch.setenv("CODEX_HOME", str(tmp_home / ".codex"))
    cfg = tmp_home / ".codex" / "config.toml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(textwrap.dedent("""\
        [projects."/Users/me/work"]
        trust_level = "trusted"

        [hooks]
        Stop = [
            { type = "command", command = "/usr/local/bin/thrum-hook" },
            { type = "command", command = "/opt/other/hook" },
        ]
        """))

    runner = CliRunner()
    result = runner.invoke(main, ["uninstall"])
    assert result.exit_code == 0, result.output
    assert "Unregistered Codex hooks" in result.output

    import tomlkit

    doc = tomlkit.parse(cfg.read_text())
    # User block survives.
    assert doc["projects"]["/Users/me/work"]["trust_level"] == "trusted"
    # Third-party hook survives, ours gone.
    commands = [e["command"] for e in doc["hooks"]["Stop"]]
    assert commands == ["/opt/other/hook"]


def test_status_reports_token_and_backfill(tmp_home: Path):
    runner = CliRunner()
    r1 = runner.invoke(main, ["status"])
    assert "Token present: False" in r1.output
    assert "Backfill done: False" in r1.output

    (tmp_home / ".config" / "thrum").mkdir(parents=True)
    (tmp_home / ".config" / "thrum" / "token").write_text("tk_x")
    (tmp_home / ".config" / "thrum" / ".backfill_done").write_text("")

    r2 = runner.invoke(main, ["status"])
    assert "Token present: True" in r2.output
    assert "Backfill done: True" in r2.output


def test_uninstall_unregisters_hooks_and_removes_token(tmp_home: Path):
    # Simulate a prior install
    from thrum_client.settings_merge import merge_hooks

    settings_path = tmp_home / ".claude" / "settings.json"
    merge_hooks(settings_path, "/fake/bin/thrum-hook")
    (tmp_home / ".config" / "thrum").mkdir(parents=True)
    (tmp_home / ".config" / "thrum" / "token").write_text("tk_x")

    runner = CliRunner()
    r = runner.invoke(main, ["uninstall"])
    assert r.exit_code == 0, r.output
    assert "Unregistered hooks" in r.output
    assert "Token removed" in r.output

    # settings.json no longer has hooks
    data = json.loads(settings_path.read_text())
    assert "hooks" not in data
    # Token gone
    assert not (tmp_home / ".config" / "thrum" / "token").exists()
    # Buffers/log dir retained (no --full)
    assert (tmp_home / ".config" / "thrum").exists()


def test_uninstall_full_removes_config_dir(tmp_home: Path):
    from thrum_client.settings_merge import merge_hooks

    settings_path = tmp_home / ".claude" / "settings.json"
    merge_hooks(settings_path, "/fake/bin/thrum-hook")
    cfg = tmp_home / ".config" / "thrum"
    cfg.mkdir(parents=True)
    (cfg / "token").write_text("tk_x")
    (cfg / "skill.log").write_text('{"event":"init_ok"}\n')

    runner = CliRunner()
    r = runner.invoke(main, ["uninstall", "--full"])
    assert r.exit_code == 0, r.output
    assert not cfg.exists()


def test_uninstall_without_prior_install_is_graceful(tmp_home: Path):
    runner = CliRunner()
    r = runner.invoke(main, ["uninstall"])
    assert r.exit_code == 0, r.output
    # No hooks to remove, no token to remove — graceful messaging.
    assert "No thrum-hook entries" in r.output


def test_logout_removes_token_locally(tmp_home: Path):
    (tmp_home / ".config" / "thrum").mkdir(parents=True)
    token_path = tmp_home / ".config" / "thrum" / "token"
    token_path.write_text("tk_x")

    runner = CliRunner()
    r = runner.invoke(main, ["logout"])
    assert r.exit_code == 0
    assert not token_path.exists()
    assert "still valid server-side" in r.output


# ---- backfill subcommand + re-init backfill ----


def _seed_token(tmp_home: Path) -> Path:
    cfg = tmp_home / ".config" / "thrum"
    cfg.mkdir(parents=True, exist_ok=True)
    token = cfg / "token"
    token.write_text("tk_existing")
    os.chmod(token, 0o600)
    return token


def test_backfill_requires_install(tmp_home: Path):
    runner = CliRunner()
    r = runner.invoke(main, ["backfill"])
    assert r.exit_code != 0
    assert "thrum init" in r.output


def test_backfill_runs_all_three_loops_by_default(tmp_home: Path, monkeypatch):
    """Default-no-flags backfill now runs claude + codex + cursor.
    Renamed from `..._both_loops...` after the cursor wire (was
    asserting 2 of 3 sources fired; now explicitly asserts all three)."""
    _seed_token(tmp_home)

    calls: dict[str, dict] = {"claude": {}, "codex": {}, "cursor": {}}

    def fake_run_backfill(settings, *, force=False, **kwargs):
        calls["claude"] = {"force": force}
        return 0

    def fake_run_codex_backfill(settings, *, force=False, **kwargs):
        calls["codex"] = {"force": force}
        return 0

    def fake_run_cursor_backfill(settings, *, force=False, **kwargs):
        calls["cursor"] = {"force": force}
        return 0

    monkeypatch.setattr(
        "thrum_client.backfill.run_backfill", fake_run_backfill
    )
    monkeypatch.setattr(
        "thrum_client.backfill.run_codex_backfill", fake_run_codex_backfill
    )
    monkeypatch.setattr(
        "thrum_client.backfill.run_cursor_backfill", fake_run_cursor_backfill
    )

    runner = CliRunner()
    r = runner.invoke(main, ["backfill"])
    assert r.exit_code == 0, r.output
    assert calls["claude"] == {"force": False}
    assert calls["codex"] == {"force": False}
    assert calls["cursor"] == {"force": False}


def test_backfill_codex_only(tmp_home: Path, monkeypatch):
    _seed_token(tmp_home)

    called: dict[str, bool] = {"claude": False, "codex": False}

    def fake_run_backfill(*args, **kwargs):
        called["claude"] = True
        return 0

    def fake_run_codex_backfill(*args, **kwargs):
        called["codex"] = True
        return 0

    monkeypatch.setattr(
        "thrum_client.backfill.run_backfill", fake_run_backfill
    )
    monkeypatch.setattr(
        "thrum_client.backfill.run_codex_backfill", fake_run_codex_backfill
    )

    runner = CliRunner()
    r = runner.invoke(main, ["backfill", "--no-claude", "--codex"])
    assert r.exit_code == 0, r.output
    assert called["claude"] is False
    assert called["codex"] is True


def test_backfill_force_passes_force_flag(tmp_home: Path, monkeypatch):
    _seed_token(tmp_home)

    received_force: dict[str, bool] = {}

    def fake_run_backfill(settings, *, force=False, **kwargs):
        received_force["claude"] = force
        return 0

    def fake_run_codex_backfill(settings, *, force=False, **kwargs):
        received_force["codex"] = force
        return 0

    monkeypatch.setattr(
        "thrum_client.backfill.run_backfill", fake_run_backfill
    )
    monkeypatch.setattr(
        "thrum_client.backfill.run_codex_backfill", fake_run_codex_backfill
    )

    runner = CliRunner()
    r = runner.invoke(main, ["backfill", "--force"])
    assert r.exit_code == 0, r.output
    assert received_force == {"claude": True, "codex": True}


def test_backfill_rejects_both_sources_disabled(tmp_home: Path, monkeypatch):
    _seed_token(tmp_home)
    runner = CliRunner()
    r = runner.invoke(main, ["backfill", "--no-claude", "--no-codex"])
    assert r.exit_code != 0
    assert "Nothing to do" in r.output


def test_init_does_not_run_backfill_on_re_init_with_existing_token(
    tmp_home: Path, mock_guest_register, monkeypatch
):
    """`thrum init` no longer runs backfill — neither on first install nor
    on re-init. Users invoke `thrum backfill` explicitly."""
    _seed_token(tmp_home)
    monkeypatch.setenv("THRUM_HOOK_CMD", "/fake/bin/thrum-hook")

    called = {"claude": False, "codex": False}

    def fake_run_backfill(*args, **kwargs):
        called["claude"] = True
        return 0

    def fake_run_codex_backfill(*args, **kwargs):
        called["codex"] = True
        return 0

    monkeypatch.setattr(
        "thrum_client.backfill.run_backfill", fake_run_backfill
    )
    monkeypatch.setattr(
        "thrum_client.backfill.run_codex_backfill", fake_run_codex_backfill
    )

    runner = CliRunner()
    r = runner.invoke(main, ["init"])
    assert r.exit_code == 0, r.output
    assert "Already installed" in r.output
    assert mock_guest_register["calls"] == 0
    assert called == {"claude": False, "codex": False}


# ---- Cursor (FR-218f) ----


def test_init_registers_cursor_hooks_when_cursor_dir_exists(
    tmp_home: Path, mock_guest_register, monkeypatch
):
    """FR-218f — `~/.cursor/` exists → hooks merge into hooks.json."""
    monkeypatch.setenv("THRUM_HOOK_CMD", "/fake/bin/thrum-hook")
    # Cursor home is HOME/.cursor under tmp_home
    cursor_home = tmp_home / ".cursor"
    cursor_home.mkdir(parents=True)

    runner = CliRunner()
    result = runner.invoke(main, ["init"])
    assert result.exit_code == 0, result.output
    assert "Cursor hooks:" in result.output

    cfg = cursor_home / "hooks.json"
    assert cfg.exists()
    doc = json.loads(cfg.read_text())
    assert doc["version"] == 1
    # Per-event registration via CURSOR_HOOK_EVENTS in cursor_config.
    assert "stop" in doc["hooks"]
    entries = doc["hooks"]["stop"]
    assert any(
        e.get("command") == "/fake/bin/thrum-hook"
        and e.get("_thrumManaged") is True
        for e in entries
    )


def test_init_skips_cursor_when_cursor_dir_absent(
    tmp_home: Path, mock_guest_register, monkeypatch
):
    """No `~/.cursor/` → skip with a clear user-visible message; no
    hooks.json created, no error. Review-flagged UX fix: previously
    silent (asymmetric with Codex's "skipping Codex" line)."""
    monkeypatch.setenv("THRUM_HOOK_CMD", "/fake/bin/thrum-hook")
    # Deliberately do NOT create ~/.cursor

    runner = CliRunner()
    result = runner.invoke(main, ["init"])
    assert result.exit_code == 0, result.output
    # Skip message present.
    assert "Cursor hooks: not installed" in result.output
    assert "skipping" in result.output
    # Genuine error path (different message) should not have fired.
    assert "Cursor hook registration skipped" not in result.output


def test_uninstall_removes_cursor_hooks_when_present(
    tmp_home: Path, monkeypatch
):
    """uninstall calls unmerge_cursor_hooks for the hooks.json entries
    it owns; preserves any user-managed entries."""
    cursor_home = tmp_home / ".cursor"
    cursor_home.mkdir()
    cfg = cursor_home / "hooks.json"
    # Pre-existing user entry alongside a Thrum-managed one.
    cfg.write_text(
        json.dumps(
            {
                "version": 1,
                "hooks": {
                    "stop": [
                        {"command": "/Users/me/my-hook.sh"},
                        {
                            "command": "/usr/local/bin/thrum-hook",
                            "_thrumManaged": True,
                        },
                    ],
                    "preToolUse": [
                        {
                            "command": "/usr/local/bin/thrum-hook",
                            "_thrumManaged": True,
                        }
                    ],
                },
            }
        )
    )
    # Seed token so uninstall doesn't bail early on auth-state reasons.
    (tmp_home / ".config" / "thrum").mkdir(parents=True, exist_ok=True)
    (tmp_home / ".config" / "thrum" / "token").write_text("tk_test")

    runner = CliRunner()
    result = runner.invoke(main, ["uninstall"])
    assert result.exit_code == 0, result.output
    assert "Unregistered Cursor hooks" in result.output

    after = json.loads(cfg.read_text())
    # User entry on `stop` survives; Thrum entry on `stop` and the
    # entire `preToolUse` array (Thrum-only) gone.
    assert after["hooks"]["stop"] == [
        {"command": "/Users/me/my-hook.sh"}
    ]
    assert "preToolUse" not in after["hooks"]


def test_backfill_default_runs_cursor_too(tmp_home: Path, monkeypatch):
    """`thrum backfill` with no flags now runs claude + codex + cursor."""
    _seed_token(tmp_home)

    called = {"claude": False, "codex": False, "cursor": False}

    def fake_claude(*a, **kw):
        called["claude"] = True
        return 0

    def fake_codex(*a, **kw):
        called["codex"] = True
        return 0

    def fake_cursor(*a, **kw):
        called["cursor"] = True
        return 0

    monkeypatch.setattr("thrum_client.backfill.run_backfill", fake_claude)
    monkeypatch.setattr("thrum_client.backfill.run_codex_backfill", fake_codex)
    monkeypatch.setattr("thrum_client.backfill.run_cursor_backfill", fake_cursor)

    runner = CliRunner()
    r = runner.invoke(main, ["backfill"])
    assert r.exit_code == 0, r.output
    assert called == {"claude": True, "codex": True, "cursor": True}


def test_backfill_cursor_only_via_flag(tmp_home: Path, monkeypatch):
    _seed_token(tmp_home)

    called = {"claude": False, "codex": False, "cursor": False}

    def fake(name):
        def _f(*a, **kw):
            called[name] = True
            return 0

        return _f

    monkeypatch.setattr("thrum_client.backfill.run_backfill", fake("claude"))
    monkeypatch.setattr("thrum_client.backfill.run_codex_backfill", fake("codex"))
    monkeypatch.setattr("thrum_client.backfill.run_cursor_backfill", fake("cursor"))

    runner = CliRunner()
    r = runner.invoke(
        main, ["backfill", "--no-claude", "--no-codex", "--cursor"]
    )
    assert r.exit_code == 0, r.output
    assert called == {"claude": False, "codex": False, "cursor": True}


def test_backfill_all_disabled_errors(tmp_home: Path):
    _seed_token(tmp_home)
    runner = CliRunner()
    r = runner.invoke(
        main, ["backfill", "--no-claude", "--no-codex", "--no-cursor"]
    )
    assert r.exit_code != 0
    assert "all sources disabled" in r.output


def test_init_existing_token_does_not_run_cursor_backfill(
    tmp_home: Path, mock_guest_register, monkeypatch
):
    """Re-init on an installed machine no longer triggers any backfill loop."""
    _seed_token(tmp_home)
    monkeypatch.setenv("THRUM_HOOK_CMD", "/fake/bin/thrum-hook")

    called = {"claude": False, "codex": False, "cursor": False}

    monkeypatch.setattr(
        "thrum_client.backfill.run_backfill",
        lambda *a, **kw: (called.__setitem__("claude", True), 0)[1],
    )
    monkeypatch.setattr(
        "thrum_client.backfill.run_codex_backfill",
        lambda *a, **kw: (called.__setitem__("codex", True), 0)[1],
    )
    monkeypatch.setattr(
        "thrum_client.backfill.run_cursor_backfill",
        lambda *a, **kw: (called.__setitem__("cursor", True), 0)[1],
    )

    runner = CliRunner()
    r = runner.invoke(main, ["init"])
    assert r.exit_code == 0, r.output
    assert called == {"claude": False, "codex": False, "cursor": False}
