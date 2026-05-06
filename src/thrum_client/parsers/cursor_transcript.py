"""Cursor IDE transcript JSONL parser — FR-218f + NFR-318/319.

Cursor persists each conversation as
`~/.cursor/projects/<sanitised-cwd>/agent-transcripts/<conversation_id>/<conversation_id>.jsonl`.
Path is exposed by the `stop` / `afterAgentResponse` / `beforeReadFile`
hook payloads as `transcript_path`.

Per-line shape (verified empirically against captured transcripts):

    {"role": "user" | "assistant",
     "message": {"content": [{"type": "text", "text": "..."},
                              {"type": "tool_use", "name": "...",
                               "input": {...}},
                              ...]}}

Unlike Codex's rollout, the transcript is **content-only** — no `model`
field on the message, no `usage` block. Token + provider/model signal lives
in the hook payload (`parsers/cursor_hook.py`); this parser only projects
tool_use blocks into canonical intent strings for FR-214.

Cursor's hook surface uses normalised tool categories (`Read`/`Grep`/
`Shell`/`Write`/etc) at `pre/postToolUse` time, but the transcript JSONL
records the model-facing tool names (`ReadFile`/`Glob`/`rg`/`ApplyPatch`/
`CreatePlan`/`TodoWrite`/etc). The intent projection here works against
the model-facing names — cleaner because each is unambiguous and Cursor
documents them in its slash-commands reference.

Privacy posture (NFR-318/319): every `text` field on a message content
block is the message body — never bound. Every `input.*` field on a
tool_use block carries the call arguments — never bound (the plan detector
in `parsers/cursor_plan_detector.py` uses its own narrow allowlist for
the structural plan/todo fields it needs). This module only reads
`role`, `message.content[].type`, and `message.content[].name`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from thrum_client.parsers.hook import _classify_bash_command


# Tool-name → canonical intent. Drawn from Cursor's slash-command + tool
# inventory, observed empirically in test transcripts plus the SKILL.md
# matcher reference.
_NAME_TO_INTENT: dict[str, str] = {
    # Reading / search
    "Read": "read",
    "ReadFile": "read",
    "ReadLints": "read",
    "Glob": "read",
    "Grep": "read",
    "rg": "read",
    "ripgrep_raw_search": "read",
    "glob_file_search": "read",
    "read_file_v2": "read",
    "ListDir": "read",
    # Editing / writing
    "Write": "edit",
    "WriteFile": "edit",
    "ApplyPatch": "edit",
    "Edit": "edit",
    "MultiEdit": "edit",
    "StrReplace": "edit",
    "StrReplaceEditor": "edit",
    "str_replace": "edit",
    "str_replace_editor": "edit",
    # Web
    "WebSearch": "web-search",
    "WebFetch": "web-search",
    "web_search": "web-search",
    # Subagent / task delegation
    "Task": "task-delegation",
    "Agent": "task-delegation",
    # Plan/todo tools — NOT projected as intents; consumed by the plan
    # detector instead. Listed here as a comment so future readers know
    # they're intentionally absent from this map.
    # "CreatePlan": <plan-detector signal>
    # "TodoWrite":  <plan-detector signal>
}

# Shell-class tool names. The actual category (run-tests / build / git-ops)
# is derived from the command string itself via `_classify_bash_command`,
# same pattern as Claude Code's hook parser.
_SHELL_NAMES: frozenset[str] = frozenset(
    {"Shell", "Bash", "bash", "execute_shell", "exec_command"}
)

_BASH_CATEGORY_TO_INTENT: dict[str, str] = {
    "git_ops": "git-ops",
    "testing": "run-tests",
    "build_deploy": "build",
}


@dataclass(frozen=True)
class CursorTurnAggregate:
    """One Cursor generation's transcript-derived signal.

    Token data does NOT come from the transcript (Cursor's transcript is
    content-only); the handler combines this aggregate with the hook
    payload's measured tokens at emit time. `tool_intents` is the canonical
    intent projection ready to flow through the FR-214 classifier.
    """

    tool_intents: list[str] = field(default_factory=list)


def _project_tool_use(name: str | None, input_value: object) -> str | None:
    """Project a tool_use block to a canonical intent. Returns None for
    plan/todo tools (consumed by the plan detector) and unknown names.

    `input_value` is observed only for shell-class tools — we read its
    `command` field to classify git/test/build, then drop. Other tool
    names ignore `input` entirely so their args never bind.
    """
    if not name:
        return None
    intent = _NAME_TO_INTENT.get(name)
    if intent:
        return intent
    if name in _SHELL_NAMES:
        cmd = None
        if isinstance(input_value, dict):
            raw = input_value.get("command")
            if isinstance(raw, str) and raw:
                cmd = raw
        if cmd:
            cat = _classify_bash_command(cmd)
            if cat:
                return _BASH_CATEGORY_TO_INTENT[cat]
        # Shell with no recognised command pattern — generic shell intent.
        return "other"
    return None


def _walk_content_blocks(message: object) -> Iterable[dict]:
    """Yield each content block dict from a transcript line's `message`.

    Cursor wraps `message.content` as a list of typed dicts. Anything else
    is silently skipped — fail-closed against a future Cursor schema
    change rather than crashing the parser.
    """
    if not isinstance(message, dict):
        return
    content = message.get("content")
    if not isinstance(content, list):
        return
    for block in content:
        if isinstance(block, dict):
            yield block


def iter_cursor_turns(path: Path) -> Iterable[CursorTurnAggregate]:
    """Walk a transcript and yield one aggregate per generation boundary.

    A generation boundary = a `role: "user"` line opens a new generation;
    every subsequent `role: "assistant"` line accumulates into that
    generation. End-of-file closes the last open generation.

    The transcript contains no explicit `generation_id` per line, so
    aggregates are returned in transcript order — the handler's live path
    pairs them with hook-payload `generation_id`s by recency
    (most-recent-first). The backfill path attributes by transcript order.
    """
    current: list[str] = []  # current generation's tool intents
    open_gen = False

    with path.open("r") as f:
        for raw_line in f:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                line = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if not isinstance(line, dict):
                continue

            role = line.get("role")
            if role == "user":
                # Close out previous generation before opening a new one.
                if open_gen:
                    yield CursorTurnAggregate(tool_intents=list(current))
                current = []
                open_gen = True
                continue

            if role != "assistant":
                continue

            for block in _walk_content_blocks(line.get("message")):
                if block.get("type") != "tool_use":
                    continue
                intent = _project_tool_use(
                    block.get("name"), block.get("input")
                )
                if intent and intent not in current:
                    current.append(intent)

    if open_gen:
        yield CursorTurnAggregate(tool_intents=list(current))


def read_turn(
    path: Path,
    target_generation_id: str | None = None,
    *,
    expected_index: int | None = None,
) -> CursorTurnAggregate | None:
    """Return one generation's aggregate from the transcript.

    Two modes:

    * `expected_index` set (live handler path): return `aggregates[index]`
      iff the transcript has at least `index + 1` complete generations.
      Returns None when the transcript hasn't caught up yet (the
      "hook-fired-before-flush" race that Codex hit in 0da407a — Cursor's
      transcript writer is also async). The handler should then emit the
      activity with empty `tool_intents` rather than mis-attributing the
      previous generation's tools to the new turn. The handler tracks
      `expected_index` by counting `beforeSubmitPrompt` events per
      conversation (zero-indexed: first prompt → index 0, second → 1).

    * `expected_index` None (backfill / legacy path): return the latest
      aggregate. Caller takes the trade-off: latest is correct for emit-
      after-stop only when no race is in flight.

    `target_generation_id` is currently advisory — Cursor's transcript
    has no per-line generation_id, so the parameter cannot be honoured
    directly. Kept in the signature for symmetry with
    `parsers/codex_rollout.read_turn` and so a future Cursor schema
    addition can be honoured without changing the call site.

    Returns None if the file is empty, contains no assistant content, or
    (for `expected_index`) has fewer than `expected_index + 1` complete
    generations.
    """
    aggregates = list(iter_cursor_turns(path))
    if not aggregates:
        return None
    if expected_index is not None:
        if expected_index < 0 or expected_index >= len(aggregates):
            return None
        return aggregates[expected_index]
    return aggregates[-1]


def count_generations(path: Path) -> int:
    """Count the complete generations (closed user→assistant blocks)
    visible in the transcript right now. Used by the handler to detect
    the hook-vs-transcript flush race: if the handler has seen K
    `beforeSubmitPrompt` events but the transcript reports fewer than K
    generations, the latest stop is firing ahead of its content.
    """
    return sum(1 for _ in iter_cursor_turns(path))
