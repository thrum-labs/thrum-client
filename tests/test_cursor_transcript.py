"""Cursor transcript JSONL parser tests — FR-218f, FR-214, NFR-318/319.

Covers tool_use → canonical-intent projection, generation boundary
detection (role:user transitions), latest-generation read, and the
non-binding posture for content fields (message text, tool input/output).

Fixtures modelled on empirically-captured transcripts from
~/.cursor/projects/.../agent-transcripts/.../*.jsonl.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from thrum_client.parsers.cursor_transcript import (
    CursorTurnAggregate,
    count_generations,
    iter_cursor_turns,
    read_turn,
)


def _write_transcript(path: Path, lines: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")


def _user(text: str = "do something") -> dict:
    return {
        "role": "user",
        "message": {"content": [{"type": "text", "text": text}]},
    }


def _assistant_text(text: str) -> dict:
    return {
        "role": "assistant",
        "message": {"content": [{"type": "text", "text": text}]},
    }


def _assistant_tool(tool_name: str, _input: dict | None = None, **input_kwargs) -> dict:
    """Build an assistant tool_use line. Pass `_input` for tool inputs that
    contain a `name` key (e.g. CreatePlan); otherwise use kwargs."""
    return {
        "role": "assistant",
        "message": {
            "content": [
                {
                    "type": "tool_use",
                    "name": tool_name,
                    "input": _input if _input is not None else dict(input_kwargs),
                }
            ]
        },
    }


def test_iter_one_generation_one_user_two_assistants(tmp_path):
    transcript = tmp_path / "t.jsonl"
    _write_transcript(
        transcript,
        [
            _user("list files in this project"),
            _assistant_tool("ReadFile", targetFile="/proj/index.html"),
            _assistant_text("Here are the files: ..."),
        ],
    )
    aggregates = list(iter_cursor_turns(transcript))
    assert len(aggregates) == 1
    assert aggregates[0].tool_intents == ["read"]


def test_iter_two_generations_split_at_user_boundary(tmp_path):
    transcript = tmp_path / "t.jsonl"
    _write_transcript(
        transcript,
        [
            _user("first prompt"),
            _assistant_tool("ReadFile", targetFile="/x.html"),
            _user("second prompt"),
            _assistant_tool("Shell", command="git status"),
        ],
    )
    aggregates = list(iter_cursor_turns(transcript))
    assert len(aggregates) == 2
    assert aggregates[0].tool_intents == ["read"]
    assert aggregates[1].tool_intents == ["git-ops"]


def test_intents_dedup_within_generation(tmp_path):
    """Same intent emitted by multiple tool_use blocks in one generation
    appears once in the aggregate (set semantics, not multiset)."""
    transcript = tmp_path / "t.jsonl"
    _write_transcript(
        transcript,
        [
            _user("read several files"),
            _assistant_tool("ReadFile", targetFile="/a"),
            _assistant_tool("Glob", pattern="*.py"),
            _assistant_tool("rg", query="foo"),
            _assistant_tool("ReadFile", targetFile="/b"),
        ],
    )
    aggregates = list(iter_cursor_turns(transcript))
    assert len(aggregates) == 1
    assert aggregates[0].tool_intents == ["read"]  # not ["read", "read", "read", "read"]


def test_intents_preserve_first_seen_order(tmp_path):
    """List order matters for downstream classifier rules — first-seen
    wins (avoids set non-determinism in user-visible debug output)."""
    transcript = tmp_path / "t.jsonl"
    _write_transcript(
        transcript,
        [
            _user("multi-step turn"),
            _assistant_tool("ReadFile", targetFile="/x"),
            _assistant_tool("Shell", command="pytest"),
            _assistant_tool("Write", path="/y"),
            _assistant_tool("Shell", command="git commit -m foo"),
        ],
    )
    aggregates = list(iter_cursor_turns(transcript))
    assert aggregates[0].tool_intents == ["read", "run-tests", "edit", "git-ops"]


def test_shell_command_classifier_drops_through_to_other(tmp_path):
    """An unrecognised shell command (not git/test/build) projects to `other`."""
    transcript = tmp_path / "t.jsonl"
    _write_transcript(
        transcript,
        [
            _user("run something arbitrary"),
            _assistant_tool("Shell", command="echo hello"),
        ],
    )
    aggregates = list(iter_cursor_turns(transcript))
    assert aggregates[0].tool_intents == ["other"]


def test_create_plan_and_todo_write_not_projected_as_intents(tmp_path):
    """CreatePlan / TodoWrite are plan-detector signals, NOT tool intents.
    The transcript parser must skip them so they don't pollute
    tools_used (which would mislead the classifier)."""
    transcript = tmp_path / "t.jsonl"
    _write_transcript(
        transcript,
        [
            _user("execute a 4-step plan"),
            _assistant_tool(
                "CreatePlan",
                _input={
                    "name": "My Plan",
                    "overview": "Top-level overview",
                    "plan": "markdown body",
                    "todos": [{"id": "step-1", "content": "step text"}],
                },
            ),
            _assistant_tool("ReadFile", targetFile="/x"),
            _assistant_tool(
                "TodoWrite",
                _input={
                    "merge": True,
                    "todos": [{"id": "step-1", "status": "completed"}],
                },
            ),
        ],
    )
    aggregates = list(iter_cursor_turns(transcript))
    # Only ReadFile contributed an intent.
    assert aggregates[0].tool_intents == ["read"]


def test_text_content_blocks_never_bound(tmp_path):
    """Sentinel: assistant `text` content (the message body) and user
    `text` (the prompt) must never appear on any aggregate field."""
    transcript = tmp_path / "t.jsonl"
    _write_transcript(
        transcript,
        [
            _user("SECRET_USER_PROMPT_TEXT"),
            _assistant_text("SECRET_ASSISTANT_REPLY_BODY"),
            _assistant_tool("ReadFile", targetFile="/SECRET_PATH/secret.txt"),
        ],
    )
    aggregates = list(iter_cursor_turns(transcript))
    assert len(aggregates) == 1
    rep = repr(aggregates[0])
    assert "SECRET_USER_PROMPT_TEXT" not in rep
    assert "SECRET_ASSISTANT_REPLY_BODY" not in rep
    assert "SECRET_PATH" not in rep


def test_tool_input_args_for_non_shell_never_bound(tmp_path):
    """`input` on non-Shell tools (the path on ReadFile, the pattern on
    Glob, etc.) is never bound — only the intent label survives."""
    transcript = tmp_path / "t.jsonl"
    _write_transcript(
        transcript,
        [
            _user("read"),
            _assistant_tool(
                "ReadFile",
                targetFile="/SECRET_FILE_PATH",
                charsLimit=999999,
            ),
            _assistant_tool("Glob", pattern="**/SECRET_GLOB_PATTERN/*"),
        ],
    )
    aggregates = list(iter_cursor_turns(transcript))
    rep = repr(aggregates[0])
    assert "SECRET_FILE_PATH" not in rep
    assert "SECRET_GLOB_PATTERN" not in rep


def test_str_replace_variants_project_to_edit(tmp_path):
    """Live-validation catch (2026-05-03): a real Cursor plan-mode session
    used `StrReplace` 5 times for edits; the original intent map had
    Write/WriteFile/ApplyPatch/Edit/MultiEdit but missed StrReplace and
    its variants. Without these, an entire plan-execution turn classified
    as `general` because tools_used = {read, other}."""
    transcript = tmp_path / "t.jsonl"
    _write_transcript(
        transcript,
        [
            _user("apply edits"),
            _assistant_tool("StrReplace", path="/x", old="foo", new="bar"),
            _assistant_tool("StrReplaceEditor", path="/x"),
            _assistant_tool("str_replace", path="/x"),
            _assistant_tool("str_replace_editor", path="/x"),
        ],
    )
    aggregates = list(iter_cursor_turns(transcript))
    assert aggregates[0].tool_intents == ["edit"]


def test_unknown_tool_name_does_not_pollute_intents(tmp_path):
    """A tool name not in our projection map (e.g. a future Cursor tool)
    silently drops — better than a `general` fallback that would mask
    classification gaps."""
    transcript = tmp_path / "t.jsonl"
    _write_transcript(
        transcript,
        [
            _user("use a future tool"),
            _assistant_tool("UnknownFutureTool", whatever="x"),
            _assistant_tool("ReadFile", targetFile="/x"),
        ],
    )
    aggregates = list(iter_cursor_turns(transcript))
    assert aggregates[0].tool_intents == ["read"]


def test_read_turn_returns_latest_aggregate(tmp_path):
    """Live emit path: read_turn(path, target=any) returns the most-recent
    generation's aggregate — that's what the stop-hook just closed."""
    transcript = tmp_path / "t.jsonl"
    _write_transcript(
        transcript,
        [
            _user("first"),
            _assistant_tool("ReadFile", targetFile="/x"),
            _user("second"),
            _assistant_tool("Shell", command="pytest"),
        ],
    )
    agg = read_turn(transcript, target_generation_id="some-uuid")
    assert agg is not None
    assert agg.tool_intents == ["run-tests"]  # the latest


def test_read_turn_returns_none_for_empty_transcript(tmp_path):
    transcript = tmp_path / "t.jsonl"
    transcript.write_text("")
    assert read_turn(transcript) is None


def test_read_turn_handles_corrupt_lines_gracefully(tmp_path):
    """A line that fails JSON parse must not crash the reader — Cursor
    has been observed to write partial lines during shutdown."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        "\n".join(
            [
                json.dumps(_user("ok")),
                "{ this is not valid json",
                json.dumps(_assistant_tool("ReadFile", targetFile="/x")),
            ]
        )
        + "\n"
    )
    agg = read_turn(transcript)
    assert agg is not None
    assert agg.tool_intents == ["read"]


def test_read_turn_with_expected_index_returns_specific_generation(tmp_path):
    """Live handler path: handler tracks `expected_index` by counting
    beforeSubmitPrompt events. read_turn must return THAT generation,
    not the latest, so the index → generation pairing is deterministic
    even when the latest generation is in flight elsewhere."""
    transcript = tmp_path / "t.jsonl"
    _write_transcript(
        transcript,
        [
            _user("first"),
            _assistant_tool("ReadFile", targetFile="/x"),
            _user("second"),
            _assistant_tool("Shell", command="pytest"),
            _user("third"),
            _assistant_tool("Write", path="/y"),
        ],
    )
    assert read_turn(transcript, expected_index=0).tool_intents == ["read"]
    assert read_turn(transcript, expected_index=1).tool_intents == ["run-tests"]
    assert read_turn(transcript, expected_index=2).tool_intents == ["edit"]


def test_read_turn_returns_none_when_transcript_lags_behind_expected_index(tmp_path):
    """Hook-vs-transcript flush race (Codex L2 catch). Handler has seen
    3 beforeSubmitPrompt events (expected_index=2 for the third stop),
    but Cursor has only flushed 2 generations to the transcript.
    read_turn must return None — D1 will then emit with empty
    tool_intents rather than mis-attribute the second generation's
    tools to the third turn."""
    transcript = tmp_path / "t.jsonl"
    _write_transcript(
        transcript,
        [
            _user("first"),
            _assistant_tool("ReadFile", targetFile="/x"),
            _user("second"),
            _assistant_tool("Shell", command="pytest"),
        ],
    )
    # Handler expects the third generation (zero-indexed: 2) but transcript
    # has only 2 generations (indices 0, 1).
    assert read_turn(transcript, expected_index=2) is None
    # Sanity: the second generation IS available.
    assert read_turn(transcript, expected_index=1).tool_intents == ["run-tests"]


def test_read_turn_negative_index_returns_none(tmp_path):
    """Defensive: a malformed handler counter (negative) must not pull
    aggregates from the end of the list (Python's negative-index magic)."""
    transcript = tmp_path / "t.jsonl"
    _write_transcript(
        transcript,
        [_user("u"), _assistant_tool("ReadFile", targetFile="/x")],
    )
    assert read_turn(transcript, expected_index=-1) is None


def test_count_generations_matches_iter_count(tmp_path):
    transcript = tmp_path / "t.jsonl"
    _write_transcript(
        transcript,
        [
            _user("a"),
            _assistant_tool("ReadFile", targetFile="/x"),
            _user("b"),
            _assistant_tool("Shell", command="ls"),
        ],
    )
    assert count_generations(transcript) == 2


def test_count_generations_zero_for_empty_transcript(tmp_path):
    transcript = tmp_path / "t.jsonl"
    transcript.write_text("")
    assert count_generations(transcript) == 0


def test_iter_returns_empty_when_only_user_lines(tmp_path):
    """User prompts without any assistant content yield one empty
    aggregate (the open generation closes at EOF with no intents)."""
    transcript = tmp_path / "t.jsonl"
    _write_transcript(transcript, [_user("just a prompt")])
    aggregates = list(iter_cursor_turns(transcript))
    assert aggregates == [CursorTurnAggregate(tool_intents=[])]
