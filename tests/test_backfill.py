"""FR-217 backfill tests."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from tests.fixtures.otlp_receiver import Capture, capture_http_post
from tests.fixtures.transcripts import assistant_record, write_jsonl
from thrum_client.backfill import run_backfill
from thrum_client.config import SkillSettings


@pytest.fixture
def otlp(monkeypatch):
    cap = Capture()
    post, close = capture_http_post(cap)
    monkeypatch.setattr(httpx, "post", post)
    yield cap
    close()


def _seed_projects(tmp_home: Path) -> Path:
    root = tmp_home / ".claude" / "projects"
    # Two sessions, each with one turn ending in end_turn
    a = root / "session-a-hash"
    b = root / "session-b-hash"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    write_jsonl(
        a / "sess-a.jsonl",
        [
            assistant_record("2026-04-23T10:00:00Z", "msg_A1", stop_reason="tool_use", input_tokens=50, output_tokens=10),
            assistant_record("2026-04-23T10:00:10Z", "msg_A2", stop_reason="end_turn", input_tokens=80, output_tokens=20),
        ],
    )
    write_jsonl(
        b / "sess-b.jsonl",
        [
            # Three siblings of the same message.id — FR-221 must dedupe.
            assistant_record("2026-04-23T10:05:00Z", "msg_B", stop_reason="end_turn", input_tokens=200, output_tokens=60),
            assistant_record("2026-04-23T10:05:01Z", "msg_B", stop_reason="end_turn", input_tokens=200, output_tokens=60),
            assistant_record("2026-04-23T10:05:02Z", "msg_B", stop_reason="end_turn", input_tokens=200, output_tokens=60),
        ],
    )
    return root


def _seed_token(tmp_home: Path) -> None:
    cfg = tmp_home / ".config" / "thrum"
    cfg.mkdir(parents=True, exist_ok=True)
    (cfg / "token").write_text("tk_test")


def test_backfill_emits_one_span_per_end_turn(tmp_home: Path, otlp: Capture):
    _seed_token(tmp_home)
    _seed_projects(tmp_home)

    settings = SkillSettings()
    emitted = run_backfill(settings)
    assert emitted == 2

    attrs = otlp.all_attrs()
    assert len(attrs) == 2
    for a in attrs:
        assert a["thrum.metadata.backfill"] is True
        assert a["thrum.content_stripped"] is True

    # Session A turn: 50+80 = 130 input_tokens, 10+20 = 30 output
    sess_a = next(
        a for a in attrs if a["session.id"] == "sess-a"
    )
    assert sess_a["gen_ai.usage.input_tokens"] == 130
    assert sess_a["gen_ai.usage.output_tokens"] == 30

    # Session B turn: FR-221 dedupe → 200 / 60 (not 600 / 180)
    sess_b = next(
        a for a in attrs if a["session.id"] == "sess-b"
    )
    assert sess_b["gen_ai.usage.input_tokens"] == 200
    assert sess_b["gen_ai.usage.output_tokens"] == 60


def test_backfill_is_idempotent_via_marker(tmp_home: Path, otlp: Capture):
    _seed_token(tmp_home)
    _seed_projects(tmp_home)

    settings = SkillSettings()
    first = run_backfill(settings)
    assert first == 2
    # Marker written
    assert settings.backfill_marker.exists()

    second = run_backfill(settings)
    assert second == 0
    # Emission count unchanged
    assert len(otlp.all_attrs()) == 2


def test_backfill_force_reemits(tmp_home: Path, otlp: Capture):
    _seed_token(tmp_home)
    _seed_projects(tmp_home)

    settings = SkillSettings()
    run_backfill(settings)
    second = run_backfill(settings, force=True)
    assert second == 2
    assert len(otlp.all_attrs()) == 4


def test_backfill_no_projects_dir_still_writes_marker(tmp_home: Path, otlp: Capture):
    _seed_token(tmp_home)
    settings = SkillSettings()
    assert run_backfill(settings) == 0
    assert settings.backfill_marker.exists()
    assert otlp.all_attrs() == []
