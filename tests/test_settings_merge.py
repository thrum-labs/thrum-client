import json
from pathlib import Path

from thrum_client.settings_merge import (
    HOOK_EVENTS,
    RUNTIME_HOOK_EVENTS,
    SETTINGS_SCHEMA_REJECTED_HOOK_EVENTS,
    merge_hooks,
    unmerge_hooks,
)


def test_merge_creates_settings_when_missing(tmp_path: Path) -> None:
    settings_path = tmp_path / ".claude" / "settings.json"
    changed = merge_hooks(settings_path, "/usr/local/bin/thrum-hook")
    assert changed is True
    data = json.loads(settings_path.read_text())
    assert set(data["hooks"].keys()) == set(HOOK_EVENTS)
    for event in HOOK_EVENTS:
        entries = data["hooks"][event]
        assert len(entries) == 1
        assert entries[0]["hooks"][0]["command"] == "/usr/local/bin/thrum-hook"


def test_default_merge_excludes_schema_rejected_runtime_events(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.json"
    merge_hooks(settings_path, "/opt/thrum-hook")
    data = json.loads(settings_path.read_text())

    assert set(SETTINGS_SCHEMA_REJECTED_HOOK_EVENTS) < set(RUNTIME_HOOK_EVENTS)
    for event in SETTINGS_SCHEMA_REJECTED_HOOK_EVENTS:
        assert event not in data["hooks"]


def test_merge_removes_stale_own_schema_rejected_hooks(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps(
            {
                "hooks": {
                    "StopFailure": [
                        {
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": "/old/bin/thrum-hook",
                                }
                            ]
                        }
                    ],
                    "PostToolUseFailure": [],
                }
            }
        )
    )

    changed = merge_hooks(settings_path, "/new/bin/thrum-hook")
    data = json.loads(settings_path.read_text())

    assert changed is True
    assert "StopFailure" not in data["hooks"]
    assert "PostToolUseFailure" not in data["hooks"]


def test_merge_preserves_user_hooks_under_schema_rejected_events(
    tmp_path: Path,
) -> None:
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps(
            {
                "hooks": {
                    "StopFailure": [
                        {
                            "hooks": [
                                {"type": "command", "command": "/opt/user-custom"},
                                {
                                    "type": "command",
                                    "command": "/usr/local/bin/thrum-hook",
                                },
                            ]
                        }
                    ]
                }
            }
        )
    )

    merge_hooks(settings_path, "/new/bin/thrum-hook")
    data = json.loads(settings_path.read_text())

    commands = [
        h["command"]
        for entry in data["hooks"]["StopFailure"]
        for h in entry["hooks"]
    ]
    assert commands == ["/opt/user-custom"]


def test_merge_preserves_existing_hooks(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps(
            {
                "theme": "dark",
                "hooks": {
                    "UserPromptSubmit": [
                        {
                            "hooks": [
                                {"type": "command", "command": "/opt/user-custom"}
                            ]
                        }
                    ]
                },
            }
        )
    )
    merge_hooks(settings_path, "/opt/thrum-hook")
    data = json.loads(settings_path.read_text())

    # Theme and the pre-existing hook still there
    assert data["theme"] == "dark"
    entries = data["hooks"]["UserPromptSubmit"]
    commands = [h["command"] for e in entries for h in e["hooks"]]
    assert "/opt/user-custom" in commands
    assert "/opt/thrum-hook" in commands

    # Thrum is registered against all other events too
    for event in HOOK_EVENTS:
        if event == "UserPromptSubmit":
            continue
        cmds = [
            h["command"]
            for e in data["hooks"][event]
            for h in e["hooks"]
        ]
        assert cmds == ["/opt/thrum-hook"]


def test_merge_is_idempotent(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.json"
    first = merge_hooks(settings_path, "/opt/thrum-hook")
    second = merge_hooks(settings_path, "/opt/thrum-hook")
    assert first is True
    assert second is False

    data = json.loads(settings_path.read_text())
    for event in HOOK_EVENTS:
        cmds = [
            h["command"]
            for e in data["hooks"][event]
            for h in e["hooks"]
        ]
        # Exactly one thrum-hook entry per event
        assert cmds.count("/opt/thrum-hook") == 1


def test_merge_rejects_non_list_hooks_block(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps({"hooks": {"UserPromptSubmit": {"bad": "shape"}}})
    )
    try:
        merge_hooks(settings_path, "/opt/thrum-hook")
    except ValueError as e:
        assert "UserPromptSubmit" in str(e)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_unmerge_removes_only_matching_entries(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps(
            {
                "theme": "dark",
                "hooks": {
                    "UserPromptSubmit": [
                        {"hooks": [{"type": "command", "command": "/opt/user-custom"}]},
                        {"hooks": [{"type": "command", "command": "/usr/local/bin/thrum-hook"}]},
                    ],
                    "Stop": [
                        {"hooks": [{"type": "command", "command": "/usr/local/bin/thrum-hook"}]},
                    ],
                },
            }
        )
    )
    removed = unmerge_hooks(settings_path)
    assert removed is True

    data = json.loads(settings_path.read_text())
    assert data["theme"] == "dark"
    # User's own hook preserved
    assert data["hooks"]["UserPromptSubmit"] == [
        {"hooks": [{"type": "command", "command": "/opt/user-custom"}]}
    ]
    # Stop had only thrum-hook, so the event key is gone
    assert "Stop" not in data["hooks"]


def test_unmerge_removes_hooks_block_entirely_when_empty(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps(
            {
                "theme": "dark",
                "hooks": {
                    "UserPromptSubmit": [
                        {"hooks": [{"type": "command", "command": "/usr/local/bin/thrum-hook"}]},
                    ],
                },
            }
        )
    )
    unmerge_hooks(settings_path)
    data = json.loads(settings_path.read_text())
    assert "hooks" not in data
    assert data["theme"] == "dark"


def test_unmerge_is_idempotent(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.json"
    # Start with thrum hooks
    merge_hooks(settings_path, "/opt/thrum-hook")
    first = unmerge_hooks(settings_path)
    second = unmerge_hooks(settings_path)
    assert first is True
    assert second is False


def test_unmerge_matches_any_install_path_with_thrum_hook_basename(
    tmp_path: Path,
) -> None:
    """Install paths vary (venvs, upgrades). Uninstall matches by basename."""
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps(
            {
                "hooks": {
                    "Stop": [
                        {"hooks": [{"type": "command", "command": "/Users/x/.local/share/uv/tools/thrum-client/bin/thrum-hook"}]},
                    ],
                },
            }
        )
    )
    assert unmerge_hooks(settings_path) is True
    assert json.loads(settings_path.read_text()) == {}


def test_unmerge_preserves_user_named_hooks_containing_thrum_hook(
    tmp_path: Path,
) -> None:
    """Review #20: a user-authored hook like `/opt/my-thrum-hook-wrapper.sh`
    used to be silently deleted by substring match. Basename match keeps it.
    """
    settings_path = tmp_path / "settings.json"
    user_hooks = [
        "/opt/my-thrum-hook-wrapper.sh",
        "/home/user/tools/pre-thrum-hook",
        "/srv/thrum-hook-audit.py",
    ]
    settings_path.write_text(
        json.dumps(
            {
                "hooks": {
                    "Stop": (
                        [{"hooks": [{"type": "command", "command": c}]} for c in user_hooks]
                        + [{"hooks": [{"type": "command", "command": "/usr/local/bin/thrum-hook"}]}]
                    ),
                },
            }
        )
    )

    assert unmerge_hooks(settings_path) is True
    data = json.loads(settings_path.read_text())
    remaining = [
        h["command"]
        for entry in data["hooks"]["Stop"]
        for h in entry["hooks"]
    ]
    assert sorted(remaining) == sorted(user_hooks)


def test_unmerge_on_missing_file_is_noop(tmp_path: Path) -> None:
    assert unmerge_hooks(tmp_path / "does-not-exist.json") is False
