"""NFR-318/319/320 sentinel fuzz — planted strings must never leak.

Covers the handler side: parse → buffer → emitter stub → log. The
end-to-end variant in test_emit_end_to_end.py covers OTLP wire bytes.
"""

from __future__ import annotations

from pathlib import Path

from tests.fixtures.hook_payloads import (
    SENTINEL,
    post_tool_use,
    post_tool_use_failure,
    pre_compact,
    pre_tool_use,
    stop,
    subagent_stop,
    user_prompt_submit,
)
from thrum_client.config import SkillSettings
from thrum_client.handler import handle_event


def _settings(tmp_home: Path) -> SkillSettings:
    return SkillSettings()


def _scan_dir_for(root: Path, needle: str) -> list[Path]:
    hits = []
    if not root.exists():
        return hits
    for p in root.rglob("*"):
        if p.is_file():
            try:
                content = p.read_bytes()
            except OSError:
                continue
            if needle.encode() in content:
                hits.append(p)
    return hits


def test_no_sentinel_in_buffer_or_log_after_full_turn(tmp_home: Path):
    settings = _settings(tmp_home)

    # Fire an event sequence that plants SENTINEL in prompt, tool_input,
    # tool_response.stdout/stderr, error, last_assistant_message, and
    # PreCompact.custom_instructions — a realistic spread of leak sites.
    handle_event(user_prompt_submit(prompt=SENTINEL), settings)
    handle_event(pre_tool_use(tool_name="Bash", tool_use_id="toolu_1"), settings)
    handle_event(
        post_tool_use(tool_name="Bash", tool_use_id="toolu_1"), settings
    )
    handle_event(
        post_tool_use_failure(tool_name="Bash", tool_use_id="toolu_2"),
        settings,
    )
    handle_event(pre_compact(custom_instructions=SENTINEL), settings)
    handle_event(subagent_stop(agent_id="agent_x"), settings)
    handle_event(stop(stop_hook_active=False), settings)

    # Assert no file under ~/.config/thrum/ contains the sentinel.
    hits = _scan_dir_for(tmp_home / ".config" / "thrum", SENTINEL)
    assert hits == [], f"sentinel leaked into: {[str(h) for h in hits]}"


def test_presence_only_for_non_empty_custom_instructions_does_not_leak_value(
    tmp_home: Path,
):
    settings = _settings(tmp_home)
    handle_event(user_prompt_submit(prompt="harmless"), settings)
    handle_event(
        pre_compact(custom_instructions=f"please {SENTINEL} the thing"),
        settings,
    )

    hits = _scan_dir_for(tmp_home / ".config" / "thrum", SENTINEL)
    assert hits == []
