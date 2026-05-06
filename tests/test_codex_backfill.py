"""Codex backfill tests — I1, FR-218b sister loop to FR-217."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from tests.fixtures.otlp_receiver import Capture, capture_http_post
from thrum_client.backfill import (
    emit_codex_backfill_for_file,
    run_codex_backfill,
)
from thrum_client.config import SkillSettings


@pytest.fixture
def otlp(monkeypatch):
    cap = Capture()
    post, close = capture_http_post(cap)
    monkeypatch.setattr(httpx, "post", post)
    yield cap
    close()


def _seed_token(tmp_home: Path) -> None:
    cfg = tmp_home / ".config" / "thrum"
    cfg.mkdir(parents=True, exist_ok=True)
    (cfg / "token").write_text("tk_test")


def _rollout(record_type: str, **payload) -> dict:
    return {"timestamp": "2026-04-29T17:30:00Z", "type": record_type, "payload": payload}


def _write_rollout(
    path: Path,
    *turns: tuple[str, int, int],  # (turn_id, tokens_in, tokens_out)
) -> None:
    """Write a rollout file with N turns, each opened by task_started and
    closed by task_complete with a token_count snapshot in between."""
    records: list[dict] = [
        _rollout(
            "session_meta",
            id="019dda49-bb2c-7de2-9a1a-1e5ca5fb0465",
            model_provider="openai",
            originator="codex-tui",
        ),
    ]
    for turn_id, tokens_in, tokens_out in turns:
        records.extend(
            [
                _rollout("event_msg", type="task_started", turn_id=turn_id),
                _rollout("turn_context", turn_id=turn_id, model="gpt-5.5"),
                _rollout(
                    "event_msg",
                    type="token_count",
                    info={
                        "last_token_usage": {
                            "input_tokens": tokens_in,
                            "cached_input_tokens": 0,
                            "output_tokens": tokens_out,
                            "reasoning_output_tokens": 5,
                            "total_tokens": tokens_in + tokens_out,
                        }
                    },
                ),
                _rollout("event_msg", type="task_complete", turn_id=turn_id),
            ]
        )
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def _seed_codex_sessions(tmp_home: Path) -> Path:
    root = tmp_home / ".codex" / "sessions" / "2026" / "04" / "29"
    root.mkdir(parents=True)
    _write_rollout(
        root / "rollout-2026-04-29T17-30-18-019dda49-bb2c-7de2-9a1a-1e5ca5fb0465.jsonl",
        ("turn-a", 100, 50),
        ("turn-b", 200, 80),
    )
    return tmp_home / ".codex" / "sessions"


def test_codex_backfill_emits_one_span_per_task_complete(
    tmp_home: Path, otlp: Capture, monkeypatch
):
    _seed_token(tmp_home)
    sessions_root = _seed_codex_sessions(tmp_home)
    monkeypatch.setenv("CODEX_HOME", str(tmp_home / ".codex"))

    settings = SkillSettings(
        config_dir=tmp_home / ".config" / "thrum",
        claude_dir=tmp_home / ".claude",
    )
    count = run_codex_backfill(settings, sessions_root=sessions_root)
    assert count == 2

    attrs_list = otlp.all_attrs()
    assert len(attrs_list) == 2
    for attrs in attrs_list:
        assert attrs["thrum.metadata.backfill"] is True
        assert attrs["gen_ai.system"] == "openai"
        assert attrs["thrum.source_tool"] == "codex-cli"
        assert attrs["gen_ai.request.model"] == "gpt-5.5"


def test_codex_backfill_marker_prevents_rerun(
    tmp_home: Path, otlp: Capture, monkeypatch
):
    _seed_token(tmp_home)
    sessions_root = _seed_codex_sessions(tmp_home)
    monkeypatch.setenv("CODEX_HOME", str(tmp_home / ".codex"))

    settings = SkillSettings(
        config_dir=tmp_home / ".config" / "thrum",
        claude_dir=tmp_home / ".claude",
    )
    count1 = run_codex_backfill(settings, sessions_root=sessions_root)
    count2 = run_codex_backfill(settings, sessions_root=sessions_root)
    assert count1 == 2
    assert count2 == 0
    # Spans were emitted exactly once — second run was a marker no-op.
    assert len(otlp.all_attrs()) == 2


def test_codex_backfill_force_re_emits(
    tmp_home: Path, otlp: Capture, monkeypatch
):
    _seed_token(tmp_home)
    sessions_root = _seed_codex_sessions(tmp_home)
    monkeypatch.setenv("CODEX_HOME", str(tmp_home / ".codex"))

    settings = SkillSettings(
        config_dir=tmp_home / ".config" / "thrum",
        claude_dir=tmp_home / ".claude",
    )
    run_codex_backfill(settings, sessions_root=sessions_root)
    count = run_codex_backfill(
        settings, sessions_root=sessions_root, force=True
    )
    assert count == 2
    assert len(otlp.all_attrs()) == 4


def test_codex_backfill_missing_sessions_dir_is_graceful(
    tmp_home: Path, otlp: Capture
):
    _seed_token(tmp_home)
    settings = SkillSettings(
        config_dir=tmp_home / ".config" / "thrum",
        claude_dir=tmp_home / ".claude",
    )
    # No sessions root created.
    count = run_codex_backfill(
        settings, sessions_root=tmp_home / "missing"
    )
    assert count == 0
    assert otlp.all_attrs() == []
    # Marker still touched so subsequent runs short-circuit.
    assert (tmp_home / ".config" / "thrum" / ".codex_backfill_done").exists()


def test_emit_codex_backfill_for_file_attaches_originator_and_reasoning(
    tmp_home: Path, otlp: Capture
):
    _seed_token(tmp_home)
    rollout = (
        tmp_home
        / "rollout-2026-04-29T17-30-18-019dda49-bb2c-7de2-9a1a-1e5ca5fb0465.jsonl"
    )
    _write_rollout(rollout, ("turn-1", 1000, 200))

    settings = SkillSettings(
        config_dir=tmp_home / ".config" / "thrum",
        claude_dir=tmp_home / ".claude",
    )
    n = emit_codex_backfill_for_file(rollout, settings)
    assert n == 1

    [attrs] = otlp.all_attrs()
    assert attrs["thrum.metadata.codex_originator"] == "codex-tui"
    assert attrs["thrum.metadata.codex_turn_id"] == "turn-1"
    assert attrs["thrum.metadata.reasoning_output_tokens"] == 5
    assert attrs["gen_ai.usage.input_tokens"] == 1000
    assert attrs["gen_ai.usage.output_tokens"] == 200


def test_codex_backfill_uses_session_meta_id_over_filename(
    tmp_home: Path, otlp: Capture
):
    """Rec 1: session_id derivation must prefer `session_meta.payload.id`
    over the filename pattern. If the file is renamed (or Codex changes
    its naming convention), the in-file id is still authoritative."""
    _seed_token(tmp_home)
    rollout = tmp_home / "weirdly-named-file.jsonl"
    _write_rollout(rollout, ("turn-id-x", 100, 50))

    settings = SkillSettings(
        config_dir=tmp_home / ".config" / "thrum",
        claude_dir=tmp_home / ".claude",
    )
    n = emit_codex_backfill_for_file(rollout, settings)
    assert n == 1

    [attrs] = otlp.all_attrs()
    # session_meta.payload.id from _write_rollout's session_meta record.
    assert attrs["session.id"] == "019dda49-bb2c-7de2-9a1a-1e5ca5fb0465"


def test_codex_backfill_independent_of_claude_marker(
    tmp_home: Path, otlp: Capture, monkeypatch
):
    """Claude `.backfill_done` and Codex `.codex_backfill_done` are
    independent — completing one does not gate the other."""
    _seed_token(tmp_home)
    cfg = tmp_home / ".config" / "thrum"
    cfg.mkdir(parents=True, exist_ok=True)
    # Pretend Claude backfill already ran.
    (cfg / ".backfill_done").write_text("done")

    sessions_root = _seed_codex_sessions(tmp_home)
    monkeypatch.setenv("CODEX_HOME", str(tmp_home / ".codex"))
    settings = SkillSettings(
        config_dir=cfg,
        claude_dir=tmp_home / ".claude",
    )

    count = run_codex_backfill(settings, sessions_root=sessions_root)
    assert count == 2  # Codex ran even though Claude marker is present
