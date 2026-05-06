from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture
def tmp_home(tmp_path, monkeypatch) -> Path:
    """Redirect HOME + THRUM_CONFIG_DIR + THRUM_CLAUDE_DIR at a tmp path.

    Any code in thrum_client that resolves `Path.home()` or loads
    `SkillSettings()` inside this fixture reads from the temp dir.
    """
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("THRUM_CONFIG_DIR", str(home / ".config" / "thrum"))
    monkeypatch.setenv("THRUM_CLAUDE_DIR", str(home / ".claude"))
    return home
