"""Codex rollout JSONL parser tests — FR-218b, NFR-319.

Synthetic fixtures matching the empirical schema captured from
`~/.codex/sessions/2026/04/29/rollout-2026-04-29T20-29-18-...jsonl`
(codex-cli 0.125.0).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from thrum_client.parsers.codex_rollout import (
    CodexSessionMeta,
    CodexTurnAggregate,
    iter_codex_turns,
    read_session_meta,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def _session_meta(**payload) -> dict:
    base = {
        "id": "019dda49-bb2c-7de2-9a1a-1e5ca5fb0465",
        "model_provider": "openai",
        "originator": "codex-tui",
        "cli_version": "0.125.0",
    }
    return {
        "timestamp": "2026-04-29T17:30:29.943Z",
        "type": "session_meta",
        "payload": {**base, **payload},
    }


def _turn_context(turn_id: str, model: str = "gpt-5.5") -> dict:
    return {
        "timestamp": "2026-04-29T17:30:29.945Z",
        "type": "turn_context",
        "payload": {
            "turn_id": turn_id,
            "model": model,
            "cwd": "/tmp",
        },
    }


def _task_started(turn_id: str) -> dict:
    return {
        "type": "event_msg",
        "payload": {"type": "task_started", "turn_id": turn_id},
    }


def _token_count(
    input_tokens: int,
    output_tokens: int,
    cached: int = 0,
    reasoning: int = 0,
    total: int | None = None,
) -> dict:
    return {
        "type": "event_msg",
        "payload": {
            "type": "token_count",
            "info": {
                "last_token_usage": {
                    "input_tokens": input_tokens,
                    "cached_input_tokens": cached,
                    "output_tokens": output_tokens,
                    "reasoning_output_tokens": reasoning,
                    "total_tokens": total or (input_tokens + output_tokens),
                }
            },
        },
    }


def _task_complete(turn_id: str) -> dict:
    return {
        "type": "event_msg",
        "payload": {
            "type": "task_complete",
            "turn_id": turn_id,
            "completed_at": 1777483900,
            "duration_ms": 12345,
            # `last_agent_message` is content — must not leak into intents/aggs
            "last_agent_message": "SECRET_AGENT_RESPONSE",
        },
    }


def _function_call(name: str, args: str = "{}") -> dict:
    return {
        "type": "response_item",
        "payload": {
            "type": "function_call",
            "name": name,
            "arguments": args,  # raw content; must not leak
            "call_id": "call_x",
        },
    }


def _exec_command_end(turn_id: str, *parsed_cmds: dict) -> dict:
    return {
        "type": "event_msg",
        "payload": {
            "type": "exec_command_end",
            "turn_id": turn_id,
            "parsed_cmd": list(parsed_cmds),
            "stdout": "SECRET_STDOUT_CONTENT",
            "stderr": "SECRET_STDERR_CONTENT",
        },
    }


def test_single_turn_aggregates_token_usage(tmp_path: Path):
    rollout = tmp_path / "rollout.jsonl"
    _write_jsonl(
        rollout,
        [
            _session_meta(),
            _task_started("019dda4a-d28d-73d1-b8fc-afdf5c70657f"),
            _turn_context("019dda4a-d28d-73d1-b8fc-afdf5c70657f"),
            _token_count(input_tokens=13076, output_tokens=217, cached=11648, reasoning=14),
            _task_complete("019dda4a-d28d-73d1-b8fc-afdf5c70657f"),
        ],
    )

    turns = list(iter_codex_turns(rollout))
    assert len(turns) == 1
    t = turns[0]
    assert t.turn_id == "019dda4a-d28d-73d1-b8fc-afdf5c70657f"
    assert t.tokens_in == 13076
    assert t.tokens_out == 217
    assert t.cached_input_tokens == 11648
    assert t.reasoning_output_tokens == 14
    assert t.model == "gpt-5.5"


def test_multiple_turns_produce_multiple_aggregates(tmp_path: Path):
    rollout = tmp_path / "rollout.jsonl"
    _write_jsonl(
        rollout,
        [
            _session_meta(),
            # Turn 1
            _task_started("turn_1"),
            _turn_context("turn_1"),
            _token_count(input_tokens=100, output_tokens=50),
            _task_complete("turn_1"),
            # Turn 2
            _task_started("turn_2"),
            _turn_context("turn_2", model="gpt-5.3-codex"),
            _token_count(input_tokens=200, output_tokens=80),
            _task_complete("turn_2"),
        ],
    )

    turns = list(iter_codex_turns(rollout))
    assert len(turns) == 2
    assert [t.turn_id for t in turns] == ["turn_1", "turn_2"]
    assert turns[0].tokens_in == 100 and turns[0].model == "gpt-5.5"
    assert turns[1].tokens_in == 200 and turns[1].model == "gpt-5.3-codex"


def test_apply_patch_function_call_projects_to_edit_intent(tmp_path: Path):
    rollout = tmp_path / "rollout.jsonl"
    _write_jsonl(
        rollout,
        [
            _session_meta(),
            _task_started("t1"),
            _turn_context("t1"),
            _function_call("apply_patch", args='{"input": "*** Begin Patch ..."}'),
            _token_count(input_tokens=10, output_tokens=20),
            _task_complete("t1"),
        ],
    )
    [t] = list(iter_codex_turns(rollout))
    assert "edit" in t.tool_intents


def test_exec_command_end_git_command_classified_as_git_ops(tmp_path: Path):
    """Codex labels git commands `parsed_cmd[].type='unknown'` in v0.125;
    the parser falls back to classify-and-drop on `parsed_cmd[].cmd` and
    yields the `git-ops` canonical intent."""
    rollout = tmp_path / "rollout.jsonl"
    _write_jsonl(
        rollout,
        [
            _session_meta(),
            _task_started("t1"),
            _turn_context("t1"),
            _function_call("exec_command", args='{"cmd": "git status --short"}'),
            _exec_command_end(
                "t1", {"type": "unknown", "cmd": "git status --short"}
            ),
            _token_count(input_tokens=10, output_tokens=20),
            _task_complete("t1"),
        ],
    )
    [t] = list(iter_codex_turns(rollout))
    assert "git-ops" in t.tool_intents


def test_exec_command_end_known_parsed_cmd_type_used_when_present(tmp_path: Path):
    """When Codex labels `parsed_cmd[].type='git'` (e.g. on a future version),
    the parser uses that directly without needing the regex fallback."""
    rollout = tmp_path / "rollout.jsonl"
    _write_jsonl(
        rollout,
        [
            _session_meta(),
            _task_started("t1"),
            _turn_context("t1"),
            _exec_command_end(
                "t1", {"type": "git", "cmd": "git log --oneline"}
            ),
            _token_count(input_tokens=5, output_tokens=10),
            _task_complete("t1"),
        ],
    )
    [t] = list(iter_codex_turns(rollout))
    assert "git-ops" in t.tool_intents


def test_session_meta_returns_provider_and_originator(tmp_path: Path):
    rollout = tmp_path / "rollout.jsonl"
    _write_jsonl(rollout, [_session_meta()])
    meta = read_session_meta(rollout)
    assert isinstance(meta, CodexSessionMeta)
    assert meta.model_provider == "openai"
    assert meta.originator == "codex-tui"
    assert meta.session_id == "019dda49-bb2c-7de2-9a1a-1e5ca5fb0465"


def test_session_meta_returns_none_on_empty_file(tmp_path: Path):
    rollout = tmp_path / "rollout.jsonl"
    rollout.write_text("")
    assert read_session_meta(rollout) is None


def test_blank_lines_and_malformed_records_skipped(tmp_path: Path):
    rollout = tmp_path / "rollout.jsonl"
    rollout.write_text(
        "\n"
        + json.dumps(_session_meta())
        + "\n\n"
        + "{ this is not json\n"
        + json.dumps(_task_started("t1"))
        + "\n"
        + json.dumps(_token_count(input_tokens=1, output_tokens=2))
        + "\n"
        + json.dumps(_task_complete("t1"))
        + "\n"
    )
    turns = list(iter_codex_turns(rollout))
    assert len(turns) == 1
    assert turns[0].turn_id == "t1"


def test_no_content_leaks_in_aggregate_repr(tmp_path: Path):
    """NFR-319 sentinel: secret strings planted in content fields
    (last_agent_message, arguments, stdout, stderr) must not appear in
    the aggregate's repr or any field on it."""
    rollout = tmp_path / "rollout.jsonl"
    _write_jsonl(
        rollout,
        [
            _session_meta(),
            _task_started("t1"),
            _turn_context("t1"),
            _function_call(
                "exec_command",
                args='{"cmd": "echo SECRET_ARG_CONTENT"}',
            ),
            _exec_command_end(
                "t1", {"type": "unknown", "cmd": "echo SECRET_PARSED_CMD"}
            ),
            _token_count(input_tokens=1, output_tokens=2),
            _task_complete("t1"),
        ],
    )
    [t] = list(iter_codex_turns(rollout))
    rep = repr(t)
    for sentinel in (
        "SECRET_AGENT_RESPONSE",
        "SECRET_ARG_CONTENT",
        "SECRET_PARSED_CMD",
        "SECRET_STDOUT_CONTENT",
        "SECRET_STDERR_CONTENT",
    ):
        assert sentinel not in rep, f"{sentinel!r} leaked into {t!r}"


def test_token_count_without_an_open_turn_is_ignored(tmp_path: Path):
    """Pre-turn token_count snapshots (rate-limit pings before the first
    task_started) must not crash and must not attribute tokens to anyone."""
    rollout = tmp_path / "rollout.jsonl"
    _write_jsonl(
        rollout,
        [
            _session_meta(),
            _token_count(input_tokens=999, output_tokens=999),
            _task_started("t1"),
            _turn_context("t1"),
            _token_count(input_tokens=10, output_tokens=20),
            _task_complete("t1"),
        ],
    )
    [t] = list(iter_codex_turns(rollout))
    # The pre-turn token_count must NOT have been folded in.
    assert t.tokens_in == 10
    assert t.tokens_out == 20


def test_apply_patch_heredoc_in_exec_command_projects_to_edit(tmp_path: Path):
    """When Codex shells out to apply_patch via exec_command (not the
    dedicated function_call), parsed_cmd[].type is 'unknown' and only the
    heredoc command string survives. The parser must still detect this
    as an `edit` intent — otherwise real coding edits get classified as
    `general` on the backend (P2 bug)."""
    rollout = tmp_path / "rollout.jsonl"
    _write_jsonl(
        rollout,
        [
            _session_meta(),
            _task_started("t-edit"),
            _turn_context("t-edit"),
            _function_call(
                "exec_command",
                args='{"cmd": "apply_patch <<\'PATCH\'\\n...\\nPATCH"}',
            ),
            _exec_command_end(
                "t-edit",
                {
                    "type": "unknown",
                    "cmd": "apply_patch <<'PATCH'\n*** Begin Patch\nPATCH",
                },
            ),
            _token_count(input_tokens=10, output_tokens=20),
            _task_complete("t-edit"),
        ],
    )
    [t] = list(iter_codex_turns(rollout))
    assert "edit" in t.tool_intents


def test_read_turn_returns_completed_aggregate_with_completed_at(tmp_path: Path):
    """When task_complete is present, read_turn returns the same data
    iter_codex_turns would yield, plus completed_at as a cursor signal."""
    rollout = tmp_path / "rollout.jsonl"
    _write_jsonl(
        rollout,
        [
            _session_meta(),
            _task_started("t-done"),
            _turn_context("t-done"),
            _token_count(input_tokens=100, output_tokens=50),
            _task_complete("t-done"),
        ],
    )
    from thrum_client.parsers.codex_rollout import read_turn

    agg = read_turn(rollout, "t-done")
    assert agg is not None
    assert agg.turn_id == "t-done"
    assert agg.tokens_in == 100
    assert agg.tokens_out == 50
    assert agg.model == "gpt-5.5"


def test_read_turn_returns_pending_aggregate_when_task_complete_missing(
    tmp_path: Path,
):
    """The hook race fix: Codex Stop fires before task_complete is flushed
    to the rollout. read_turn must still return the latest token_count and
    model so the live span carries real numbers — not zeros."""
    rollout = tmp_path / "rollout.jsonl"
    _write_jsonl(
        rollout,
        [
            _session_meta(),
            _task_started("t-pending"),
            _turn_context("t-pending"),
            _token_count(input_tokens=13075, output_tokens=5, cached=11648),
            # NO task_complete — Codex hasn't flushed it yet.
        ],
    )
    from thrum_client.parsers.codex_rollout import read_turn

    agg = read_turn(rollout, "t-pending")
    assert agg is not None
    assert agg.turn_id == "t-pending"
    assert agg.tokens_in == 13075
    assert agg.tokens_out == 5
    assert agg.cached_input_tokens == 11648
    assert agg.model == "gpt-5.5"
    assert agg.completed_at is None


def test_read_turn_returns_none_for_unknown_turn_id(tmp_path: Path):
    """Don't fabricate a turn we never saw start."""
    rollout = tmp_path / "rollout.jsonl"
    _write_jsonl(rollout, [_session_meta()])
    from thrum_client.parsers.codex_rollout import read_turn

    assert read_turn(rollout, "nonexistent") is None


def test_apply_patch_word_boundary_avoids_false_positive(tmp_path: Path):
    """A path containing `apply_patch` as a substring (e.g. inside a tool
    name like `foo_apply_patch_helper`) must NOT trigger the edit intent.
    Only the standalone word at a word boundary counts."""
    rollout = tmp_path / "rollout.jsonl"
    _write_jsonl(
        rollout,
        [
            _session_meta(),
            _task_started("t-fp"),
            _turn_context("t-fp"),
            _exec_command_end(
                "t-fp",
                {"type": "unknown", "cmd": "ls /opt/foo_apply_patch_x/"},
            ),
            _token_count(input_tokens=1, output_tokens=1),
            _task_complete("t-fp"),
        ],
    )
    [t] = list(iter_codex_turns(rollout))
    assert "edit" not in t.tool_intents
