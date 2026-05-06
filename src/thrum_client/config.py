"""Client configuration — env + .env loading via pydantic-settings.

All paths are plain `~/.config/thrum/...` and `~/.claude/...` on every
platform; platformdirs is intentionally not used so the documented paths
and the runtime agree literally.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# Anchored to the user config dir rather than cwd — hook handlers run with
# the Claude Code project dir as cwd, so a project-local `.env` could
# silently retarget the skill to an attacker-controlled backend.
_ENV_FILE = Path.home() / ".config" / "thrum" / "env"


class SkillSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_prefix="THRUM_",
        extra="ignore",
    )

    # Backend API base URL used by `thrum init` to call /auth/guest-register
    # and by the emitter to POST OTLP spans to the local collector.
    api_url: str = "http://127.0.0.1:8000"

    # Local OTLP/HTTP collector endpoint — the emitter targets this directly.
    collector_url: str = "http://127.0.0.1:4318/v1/traces"

    # Base dir for skill state (token, buffers, logs, markers). Overridable
    # for tests.
    config_dir: Path = Path.home() / ".config" / "thrum"

    # Claude Code settings dir (for hook registration).
    claude_dir: Path = Path.home() / ".claude"

    @property
    def token_path(self) -> Path:
        return self.config_dir / "token"

    @property
    def state_path(self) -> Path:
        return self.config_dir / "state.json"

    @property
    def buffers_dir(self) -> Path:
        return self.config_dir / "buffers"

    @property
    def log_path(self) -> Path:
        return self.config_dir / "skill.log"

    @property
    def backfill_marker(self) -> Path:
        return self.config_dir / ".backfill_done"

    @property
    def claude_settings_path(self) -> Path:
        return self.claude_dir / "settings.json"


def load_settings() -> SkillSettings:
    return SkillSettings()
