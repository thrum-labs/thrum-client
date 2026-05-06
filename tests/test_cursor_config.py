"""Cursor hooks.json round-trip tests — FR-218f.

Strict-JSON merge/unmerge behaviour for `~/.cursor/hooks.json`. Mirrors
the test surface from `test_codex_cli_init.py` but for Cursor's simpler
JSON file format.
"""

from __future__ import annotations

import json

import pytest

from thrum_client.cursor_config import (
    CURSOR_HOOK_EVENTS,
    CURSOR_HOOKS_SCHEMA_VERSION,
    merge_cursor_hooks,
    unmerge_cursor_hooks,
)


def test_merge_creates_hooks_json_when_absent(tmp_path):
    hooks_path = tmp_path / "hooks.json"
    assert not hooks_path.exists()

    changed = merge_cursor_hooks(hooks_path, "/usr/local/bin/thrum-hook")
    assert changed is True
    assert hooks_path.exists()

    doc = json.loads(hooks_path.read_text())
    assert doc["version"] == CURSOR_HOOKS_SCHEMA_VERSION
    assert set(doc["hooks"].keys()) == set(CURSOR_HOOK_EVENTS)
    for event in CURSOR_HOOK_EVENTS:
        assert doc["hooks"][event] == [
            {"command": "/usr/local/bin/thrum-hook", "_thrumManaged": True}
        ]


def test_merge_idempotent_no_duplicates_on_rerun(tmp_path):
    hooks_path = tmp_path / "hooks.json"
    cmd = "/usr/local/bin/thrum-hook"

    assert merge_cursor_hooks(hooks_path, cmd) is True
    # Second call must be a no-op (no changes).
    assert merge_cursor_hooks(hooks_path, cmd) is False

    doc = json.loads(hooks_path.read_text())
    for event in CURSOR_HOOK_EVENTS:
        # Still exactly one entry per event.
        assert len(doc["hooks"][event]) == 1


def test_merge_preserves_user_managed_entries(tmp_path):
    """User has a hand-edited hooks.json with their own preToolUse hook.
    Our merge must not clobber it."""
    hooks_path = tmp_path / "hooks.json"
    hooks_path.write_text(
        json.dumps(
            {
                "version": 1,
                "hooks": {
                    "preToolUse": [
                        {"command": "/Users/me/my-audit-hook.sh"}
                    ],
                    "afterFileEdit": [
                        {"command": "/Users/me/my-formatter.sh"}
                    ],
                },
            }
        )
    )

    merge_cursor_hooks(hooks_path, "/usr/local/bin/thrum-hook")
    doc = json.loads(hooks_path.read_text())

    # Our preToolUse entry sits alongside the user's.
    pre_entries = doc["hooks"]["preToolUse"]
    assert len(pre_entries) == 2
    assert any(e.get("command") == "/Users/me/my-audit-hook.sh" for e in pre_entries)
    assert any(
        e.get("command") == "/usr/local/bin/thrum-hook"
        and e.get("_thrumManaged") is True
        for e in pre_entries
    )

    # afterFileEdit is NOT in our registered set — the user's entry survives
    # untouched, no Thrum entry added.
    assert doc["hooks"]["afterFileEdit"] == [
        {"command": "/Users/me/my-formatter.sh"}
    ]


def test_unmerge_removes_only_thrum_managed(tmp_path):
    hooks_path = tmp_path / "hooks.json"
    cmd = "/usr/local/bin/thrum-hook"
    # Pre-existing user entry on a registered event:
    hooks_path.write_text(
        json.dumps(
            {
                "version": 1,
                "hooks": {
                    "preToolUse": [
                        {"command": "/Users/me/my-hook.sh"}
                    ]
                },
            }
        )
    )

    merge_cursor_hooks(hooks_path, cmd)
    pre_a = json.loads(hooks_path.read_text())["hooks"]["preToolUse"]
    assert len(pre_a) == 2  # user + thrum

    unmerge_cursor_hooks(hooks_path)
    doc = json.loads(hooks_path.read_text())
    pre_b = doc["hooks"]["preToolUse"]
    assert pre_b == [{"command": "/Users/me/my-hook.sh"}]
    # version stays in place (it's not Thrum-owned).
    assert doc["version"] == 1


def test_unmerge_empty_thrum_only_file_removes_file_entirely(tmp_path):
    hooks_path = tmp_path / "hooks.json"
    cmd = "/usr/local/bin/thrum-hook"

    merge_cursor_hooks(hooks_path, cmd)
    # Doc currently has only `version` + Thrum-managed entries.
    unmerge_cursor_hooks(hooks_path)
    # Should be deleted (no Thrum-shaped artifact left behind).
    assert not hooks_path.exists()


def test_unmerge_preserves_non_thrum_top_level_keys(tmp_path):
    """If the user's hooks.json has top-level keys other than `version`/`hooks`
    (Cursor may grow new sections), unmerge must NOT delete the whole file
    even when our entries are the only ones in `hooks`."""
    hooks_path = tmp_path / "hooks.json"
    hooks_path.write_text(
        json.dumps(
            {
                "version": 1,
                "userPreferences": {"theme": "dark"},
                "hooks": {},
            }
        )
    )

    merge_cursor_hooks(hooks_path, "/usr/local/bin/thrum-hook")
    unmerge_cursor_hooks(hooks_path)

    assert hooks_path.exists()
    doc = json.loads(hooks_path.read_text())
    assert doc["userPreferences"] == {"theme": "dark"}
    assert "hooks" not in doc  # block emptied + removed


def test_strict_json_no_jsonc_comments_in_output(tmp_path):
    """Cursor parses hooks.json as strict JSON (per SKILL.md). Our writer
    must never emit JSONC-style comments."""
    hooks_path = tmp_path / "hooks.json"
    merge_cursor_hooks(hooks_path, "/usr/local/bin/thrum-hook")
    text = hooks_path.read_text()
    assert "//" not in text
    assert "/*" not in text
    # Round-trip via stdlib json must succeed (proves valid strict JSON).
    json.loads(text)


def test_merge_rejects_invalid_json_gracefully(tmp_path):
    """A user with a corrupt hooks.json gets a clear error, not a clobber."""
    hooks_path = tmp_path / "hooks.json"
    hooks_path.write_text("{ not valid json")

    with pytest.raises(ValueError, match="not valid JSON"):
        merge_cursor_hooks(hooks_path, "/usr/local/bin/thrum-hook")


def test_merge_rejects_top_level_array(tmp_path):
    """Cursor's spec mandates a top-level object. Reject anything else."""
    hooks_path = tmp_path / "hooks.json"
    hooks_path.write_text("[]")

    with pytest.raises(ValueError, match="top-level must be an object"):
        merge_cursor_hooks(hooks_path, "/usr/local/bin/thrum-hook")


def test_atomic_write_no_corruption_on_partial_failure(tmp_path, monkeypatch):
    """If something fails between writing the temp file and the atomic
    rename, the original file (or absence) must be preserved — never a
    half-written hooks.json that would crash Cursor's loader."""
    hooks_path = tmp_path / "hooks.json"
    # Pre-existing valid file:
    original_doc = {"version": 1, "hooks": {"preToolUse": [{"command": "/x"}]}}
    hooks_path.write_text(json.dumps(original_doc))

    # Force os.replace to fail. The temp file (if any) should be cleaned up,
    # and the original must remain readable + unchanged.
    import os

    real_replace = os.replace

    def boom(*args, **kwargs):  # noqa: ANN001
        raise OSError("simulated rename failure")

    monkeypatch.setattr("thrum_client.cursor_config.os.replace", boom)

    with pytest.raises(OSError, match="simulated rename failure"):
        merge_cursor_hooks(hooks_path, "/usr/local/bin/thrum-hook")

    # Original is intact.
    assert json.loads(hooks_path.read_text()) == original_doc
    # Restore so test teardown isn't surprising.
    monkeypatch.setattr("thrum_client.cursor_config.os.replace", real_replace)
