"""Parser-level tests — NFR-319 allowlist extraction."""

from tests.fixtures.hook_payloads import (
    SENTINEL,
    post_tool_use,
    post_tool_use_failure,
    pre_compact,
    pre_tool_use,
    session_start,
    stop,
    subagent_start,
    subagent_stop,
    user_prompt_submit,
)
from thrum_client.parsers.hook import extract_hook_event


def test_session_start_captures_model_and_source():
    ev = extract_hook_event(session_start())
    assert ev.hook_event_name == "SessionStart"
    assert ev.model == "claude-opus-4-7"
    assert ev.source == "startup"


def test_user_prompt_submit_never_binds_prompt():
    ev = extract_hook_event(user_prompt_submit())
    # The extractor cannot expose prompt — the dataclass has no such field.
    assert not hasattr(ev, "prompt")
    # The dataclass itself, stringified, contains nothing from the prompt.
    assert SENTINEL not in repr(ev)


def test_pre_tool_use_drops_tool_input_contents():
    ev = extract_hook_event(pre_tool_use(tool_name="Read", tool_use_id="toolu_42"))
    assert ev.tool_name == "Read"
    assert ev.tool_use_id == "toolu_42"
    assert SENTINEL not in repr(ev)


def test_pre_tool_use_classifies_bash_pytest_without_retaining_command():
    ev = extract_hook_event(
        pre_tool_use(tool_name="Bash", command=f"pytest -k {SENTINEL}")
    )
    assert ev.bash_category == "testing"
    # The raw command (including the sentinel) must never be retained.
    assert SENTINEL not in repr(ev)


def test_pre_tool_use_classifies_git_commit():
    ev = extract_hook_event(
        pre_tool_use(tool_name="Bash", command="git commit -m 'fix'")
    )
    assert ev.bash_category == "git_ops"


def test_pre_tool_use_classifies_docker_build():
    ev = extract_hook_event(
        pre_tool_use(tool_name="Bash", command="docker build -t app .")
    )
    assert ev.bash_category == "build_deploy"


def test_pre_tool_use_unknown_command_has_no_category():
    ev = extract_hook_event(
        pre_tool_use(tool_name="Bash", command="echo hello")
    )
    assert ev.bash_category is None


def test_pre_tool_use_first_match_wins_when_multiple_patterns_hit():
    # A pretend-bad command hitting both git and test patterns should pick
    # git first — mirrors the backend classifier's priority order.
    ev = extract_hook_event(
        pre_tool_use(tool_name="Bash", command="git checkout && pytest")
    )
    assert ev.bash_category == "git_ops"


def test_post_tool_use_keeps_flags_not_stdout():
    ev = extract_hook_event(post_tool_use(tool_name="Bash", interrupted=True))
    assert ev.tool_name == "Bash"
    assert ev.tool_response_interrupted is True
    assert ev.tool_response_is_image is False
    assert ev.tool_response_no_output_expected is False
    assert SENTINEL not in repr(ev)


def test_post_tool_use_failure_drops_error_string():
    ev = extract_hook_event(post_tool_use_failure(tool_name="Bash"))
    assert ev.tool_name == "Bash"
    assert SENTINEL not in repr(ev)


def test_subagent_start_captures_identity():
    ev = extract_hook_event(subagent_start(agent_id="agent_xyz", agent_type="Explore"))
    assert ev.agent_id == "agent_xyz"
    assert ev.agent_type == "Explore"


def test_subagent_stop_captures_transcript_path_not_last_message():
    ev = extract_hook_event(subagent_stop())
    assert ev.agent_transcript_path == "/tmp/fake-subagent.jsonl"
    assert ev.stop_hook_active is False
    assert SENTINEL not in repr(ev)


def test_pre_compact_custom_instructions_is_presence_only():
    ev_empty = extract_hook_event(pre_compact(custom_instructions=""))
    assert ev_empty.trigger == "manual"
    assert ev_empty.has_custom_instructions is False

    ev_with = extract_hook_event(pre_compact(custom_instructions="summarize the design"))
    assert ev_with.has_custom_instructions is True
    # The literal string must not survive.
    assert "summarize" not in repr(ev_with)


def test_stop_captures_stop_hook_active_flag():
    ev_chained = extract_hook_event(stop(stop_hook_active=True))
    assert ev_chained.stop_hook_active is True
    ev_done = extract_hook_event(stop(stop_hook_active=False))
    assert ev_done.stop_hook_active is False
    assert SENTINEL not in repr(ev_done)
