"""Codex CLI rollout JSONL parser — FR-218b + NFR-319.

The Codex CLI persists each session as
`${CODEX_HOME:-~/.codex}/sessions/YYYY/MM/DD/rollout-<iso-ts>-<uuid>.jsonl`.

Top-level record types observed empirically (codex-cli 0.125.0):
- `session_meta`     — once per session; carries `model_provider`, `originator`.
- `turn_context`     — once per turn; carries `turn_id`, `model`.
- `event_msg`        — many sub-types via `payload.type`:
    * `task_started`     opens a turn (carries `turn_id`)
    * `token_count`      running token-usage snapshot (no `turn_id`; the
                         per-turn delta is `info.last_token_usage`)
    * `exec_command_end` tool-execution result with `parsed_cmd[].type`
                         (carries `turn_id`)
    * `task_complete`    closes a turn (carries `turn_id`)
- `response_item`    — `function_call`, `message`, `reasoning`, etc.

Per-turn algorithm:
1. `task_started(turn_id=X)` → open buffer keyed by X.
2. `turn_context(turn_id=X)` → fill in `model`.
3. `token_count` (no turn_id) → update the current open turn's tokens
   from `info.last_token_usage`. The "current turn" is the most recently
   opened `task_started` whose `task_complete` has not yet fired.
4. `function_call` (any pending) → project name → intent.
5. `exec_command_end(turn_id=X)` → project parsed_cmd → intents.
6. `task_complete(turn_id=X)` → yield the aggregate, close buffer X.

Content keys (`payload.last_agent_message`, `arguments` raw JSON,
`stdout`/`stderr`, `parsed_cmd[].cmd`, `base_instructions.text`,
`developer_instructions`, `agent_message.message`, `user_message.*`,
`reasoning.text`, `web_search_call.query`) are NEVER bound to a Python
variable: ijson.parse yields tokens we ignore.

For `parsed_cmd[].cmd` we follow the same classify-and-drop pattern
as `parsers/hook.py` for `tool_input.command` — Codex's own classifier
labels git/test/build commands as `"unknown"` in the captured session,
so we run our own regex within the ijson loop and retain only the
canonical intent label.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import ijson

from thrum_client.parsers.hook import _classify_bash_command


# Codex sometimes shells out to apply_patch via exec_command rather than
# calling the dedicated function (`function_call.name="apply_patch"`).
# In that path `parsed_cmd[].type` is `"unknown"` and only the heredoc
# command string survives — we detect the literal `apply_patch` token
# at word-boundary so a path like `/opt/foo-apply_patch-helper` doesn't
# false-positive.
_APPLY_PATCH_RE = re.compile(r"\bapply_patch\b")


# Allowlist for ijson scalar capture. Paths reflect the structure inside a
# single JSON line (the ijson parser is reset per line in `iter_codex_turns`).
ROLLOUT_ALLOWED_PATHS: frozenset[str] = frozenset(
    {
        # Top-level
        "type",
        # session_meta payload
        "payload.id",
        "payload.model_provider",
        "payload.originator",
        # turn_context payload
        "payload.turn_id",
        "payload.model",
        # event_msg payload (task_started / task_complete / token_count /
        # exec_command_end share `payload.type` discriminator)
        "payload.type",
        # task_complete fields used as the per-turn cursor for incremental
        # sync (FR-218b polling path).
        "payload.completed_at",
        # token_count.info.last_token_usage.*
        "payload.info.last_token_usage.input_tokens",
        "payload.info.last_token_usage.cached_input_tokens",
        "payload.info.last_token_usage.output_tokens",
        "payload.info.last_token_usage.reasoning_output_tokens",
        "payload.info.last_token_usage.total_tokens",
        # function_call name (NOT arguments — arguments is content)
        "payload.name",
        # exec_command_end parsed_cmd[].type — pre-classified by Codex
        "payload.parsed_cmd.item.type",
    }
)

# Classify-and-drop paths: the value is observed inside the ijson loop,
# fed to a fixed-set classifier, and the resulting enum (not the raw
# string) is what we keep. Mirrors `parsers/hook.py` discipline.
ROLLOUT_CLASSIFY_PATHS: frozenset[str] = frozenset(
    {
        "payload.parsed_cmd.item.cmd",
    }
)

_SCALAR_EVENTS: frozenset[str] = frozenset({"string", "number", "boolean", "null"})


# Canonical intent vocabulary (FR-214). Maps:
# function_call.name (and a fallback bash-classifier label) → intent string.
_FUNCTION_NAME_TO_INTENT: dict[str, str] = {
    "apply_patch": "edit",
    "edit": "edit",
    "write": "edit",
    "str_replace_editor": "edit",
    "web_search": "web-search",
    "web_search_call": "web-search",
    "task": "task-delegation",
}

# parsed_cmd[].type → canonical intent. Codex labels are not stable across
# versions; only the values empirically observed are mapped, with a fallback
# via _classify_bash_command for `unknown` (the most common label in v0.125).
_PARSED_CMD_TYPE_TO_INTENT: dict[str, str] = {
    "git": "git-ops",
    "test": "run-tests",
    "build": "build",
    "read": "read",
}

# Bash-classifier label (from hook._classify_bash_command) → canonical intent.
_BASH_LABEL_TO_INTENT: dict[str, str] = {
    "git_ops": "git-ops",
    "testing": "run-tests",
    "build_deploy": "build",
}


@dataclass(frozen=True)
class CodexSessionMeta:
    """First `session_meta` record of a rollout."""

    session_id: str | None
    model_provider: str | None
    originator: str | None


@dataclass
class CodexTurnAggregate:
    """One turn = one `task_complete` boundary. Fields are populated by the
    state machine in `iter_codex_turns`. Tokens come directly from
    `info.last_token_usage` — no cumulative-delta math needed (the field
    already holds the per-turn delta on token_count events emitted within
    the turn).

    `completed_at` (Unix epoch seconds from `task_complete.payload.completed_at`)
    serves as the per-turn cursor for incremental sync. Codex turns are
    sequential within a session, so completed_at is monotonic per file —
    `run_codex_sync` skips turns whose completed_at is `<= cursor`.
    """

    turn_id: str
    model: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    cached_input_tokens: int = 0
    reasoning_output_tokens: int = 0
    total_tokens: int = 0
    tool_intents: list[str] = field(default_factory=list)
    completed_at: int | None = None


def _add_intent(intents: list[str], intent: str | None) -> None:
    """Append intent to the list once; preserves first-seen order."""
    if intent and intent not in intents:
        intents.append(intent)


def _extract_record(raw: bytes) -> tuple[dict[str, Any], list[str]]:
    """Return (allowlisted captures, derived intents) for one JSONL line.

    Captures repeat-keys for `payload.parsed_cmd.item.type` only as the
    most-recent value — the array iteration is folded into the intent
    list directly, not into the captures dict. Same for the cmd
    classify-and-drop branch.
    """
    captures: dict[str, Any] = {}
    intents: list[str] = []
    for prefix, event, value in ijson.parse(io.BytesIO(raw)):
        if event not in _SCALAR_EVENTS:
            continue
        if prefix == "payload.parsed_cmd.item.type" and isinstance(value, str):
            mapped = _PARSED_CMD_TYPE_TO_INTENT.get(value)
            if mapped:
                _add_intent(intents, mapped)
            continue
        if prefix in ROLLOUT_CLASSIFY_PATHS:
            if event == "string" and isinstance(value, str) and value:
                # apply_patch heredoc takes priority over the bash classifier:
                # an `apply_patch <<'PATCH' ... PATCH` shell-out is an edit, not
                # an `other` shell command. Without this Codex turns where the
                # agent shells out to apply_patch (instead of calling the
                # dedicated function) silently lose their `edit` intent and the
                # backend mis-classifies the activity as `general`.
                if _APPLY_PATCH_RE.search(value):
                    _add_intent(intents, "edit")
                hit = _classify_bash_command(value)
                if hit:
                    _add_intent(intents, _BASH_LABEL_TO_INTENT.get(hit, hit))
            continue
        if prefix in ROLLOUT_ALLOWED_PATHS:
            captures[prefix] = value
    return captures, intents


def _to_int(v: Any) -> int:
    if v is None:
        return 0
    if isinstance(v, bool):
        return int(v)
    return int(v)


def read_session_meta(path: Path) -> CodexSessionMeta | None:
    """Read the first `session_meta` line of the rollout. Returns None if
    no such record is present (treated as a malformed file)."""
    with path.open("rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                cap, _ = _extract_record(line)
            except (ijson.JSONError, ijson.IncompleteJSONError):
                continue
            if cap.get("type") == "session_meta":
                return CodexSessionMeta(
                    session_id=cap.get("payload.id"),
                    model_provider=cap.get("payload.model_provider"),
                    originator=cap.get("payload.originator"),
                )
    return None


def read_turn(path: Path, turn_id: str) -> CodexTurnAggregate | None:
    """Return the aggregate for a single turn, completed OR in-progress.

    The Stop hook fires before Codex flushes `task_complete` to the
    rollout JSONL — so a live `_emit_for_codex_turn` reading the file
    racing the writer often sees the turn opened (`task_started`,
    `turn_context`, several `token_count` snapshots) but no closing
    `task_complete`. `iter_codex_turns` yields only completed turns
    by design, so this helper is the fallback path: rebuild the same
    aggregate but return the pending state at end-of-stream when no
    `task_complete` was seen for the requested `turn_id`.

    Returns None iff the turn was never opened.
    """
    target = turn_id
    pending: CodexTurnAggregate | None = None

    with path.open("rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                cap, intents = _extract_record(line)
            except (ijson.JSONError, ijson.IncompleteJSONError):
                continue

            top_type = cap.get("type")

            if top_type == "event_msg":
                sub = cap.get("payload.type")
                if sub == "task_started":
                    if cap.get("payload.turn_id") == target:
                        pending = CodexTurnAggregate(turn_id=target)
                elif sub == "task_complete":
                    if cap.get("payload.turn_id") == target and pending is not None:
                        for it in intents:
                            _add_intent(pending.tool_intents, it)
                        pending.completed_at = (
                            int(cap["payload.completed_at"])
                            if cap.get("payload.completed_at") is not None
                            else None
                        )
                        return pending
                elif sub == "token_count" and pending is not None:
                    # The most-recent token_count BEFORE task_complete is the
                    # best proxy for the closing turn's usage. Codex turns are
                    # sequential per session so there's no risk of conflating
                    # parallel turns. We overwrite each snapshot since
                    # `last_token_usage` is the latest cumulative-for-this-turn.
                    pending.tokens_in = _to_int(
                        cap.get("payload.info.last_token_usage.input_tokens")
                    )
                    pending.tokens_out = _to_int(
                        cap.get("payload.info.last_token_usage.output_tokens")
                    )
                    pending.cached_input_tokens = _to_int(
                        cap.get(
                            "payload.info.last_token_usage.cached_input_tokens"
                        )
                    )
                    pending.reasoning_output_tokens = _to_int(
                        cap.get(
                            "payload.info.last_token_usage."
                            "reasoning_output_tokens"
                        )
                    )
                    pending.total_tokens = _to_int(
                        cap.get("payload.info.last_token_usage.total_tokens")
                    )
                elif sub == "exec_command_end":
                    if (
                        cap.get("payload.turn_id") == target
                        and pending is not None
                    ):
                        for it in intents:
                            _add_intent(pending.tool_intents, it)

            elif top_type == "turn_context":
                if cap.get("payload.turn_id") == target and pending is not None:
                    pending.model = cap.get("payload.model")

            elif top_type == "response_item":
                if (
                    cap.get("payload.type") == "function_call"
                    and pending is not None
                ):
                    name = cap.get("payload.name")
                    if name:
                        intent = _FUNCTION_NAME_TO_INTENT.get(name, "other")
                        _add_intent(pending.tool_intents, intent)

    return pending  # in-progress (no task_complete seen) — caller decides


def iter_codex_turns(path: Path) -> Iterable[CodexTurnAggregate]:
    """Yield one `CodexTurnAggregate` per `task_complete` record.

    State machine:
    - `task_started(turn_id=X)`  → open `pending[X]`
    - `turn_context(turn_id=X)`  → record model
    - `token_count` (no turn_id) → update `current_turn` (the most
                                   recently opened turn) with last_token_usage
    - `function_call`            → project name → intent → current_turn
    - `exec_command_end(turn_id=X)` → intents from parsed_cmd[].type → X
    - `task_complete(turn_id=X)` → yield pending[X]; del pending[X]
    """
    pending: dict[str, CodexTurnAggregate] = {}
    current_turn_id: str | None = None  # most-recently-opened turn

    with path.open("rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                cap, intents = _extract_record(line)
            except (ijson.JSONError, ijson.IncompleteJSONError):
                continue

            top_type = cap.get("type")

            if top_type == "event_msg":
                sub = cap.get("payload.type")
                if sub == "task_started":
                    tid = cap.get("payload.turn_id")
                    if tid:
                        pending[tid] = CodexTurnAggregate(turn_id=tid)
                        current_turn_id = tid
                elif sub == "task_complete":
                    tid = cap.get("payload.turn_id")
                    if tid and tid in pending:
                        for it in intents:
                            _add_intent(pending[tid].tool_intents, it)
                        yield pending.pop(tid)
                        if current_turn_id == tid:
                            current_turn_id = (
                                next(reversed(pending), None) if pending else None
                            )
                elif sub == "token_count":
                    target = current_turn_id
                    if target and target in pending:
                        agg = pending[target]
                        agg.tokens_in = _to_int(
                            cap.get("payload.info.last_token_usage.input_tokens")
                        )
                        agg.tokens_out = _to_int(
                            cap.get("payload.info.last_token_usage.output_tokens")
                        )
                        agg.cached_input_tokens = _to_int(
                            cap.get(
                                "payload.info.last_token_usage.cached_input_tokens"
                            )
                        )
                        agg.reasoning_output_tokens = _to_int(
                            cap.get(
                                "payload.info.last_token_usage."
                                "reasoning_output_tokens"
                            )
                        )
                        agg.total_tokens = _to_int(
                            cap.get("payload.info.last_token_usage.total_tokens")
                        )
                elif sub == "exec_command_end":
                    tid = cap.get("payload.turn_id") or current_turn_id
                    if tid and tid in pending:
                        for it in intents:
                            _add_intent(pending[tid].tool_intents, it)

            elif top_type == "turn_context":
                tid = cap.get("payload.turn_id")
                if tid and tid in pending:
                    pending[tid].model = cap.get("payload.model")

            elif top_type == "response_item":
                # Only `function_call` response_items carry a tool name; other
                # payload types (`message`, `reasoning`) populate `payload.name`
                # for unrelated reasons or not at all.
                if cap.get("payload.type") == "function_call":
                    name = cap.get("payload.name")
                    target = current_turn_id
                    if name and target and target in pending:
                        intent = _FUNCTION_NAME_TO_INTENT.get(name, "other")
                        _add_intent(pending[target].tool_intents, intent)
