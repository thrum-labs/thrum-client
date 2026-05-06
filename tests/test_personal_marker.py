"""FR-707 — `.thrum-personal` walk + span emission."""

from __future__ import annotations

from pathlib import Path

import pytest

from thrum_skill.emitter import build_turn_span
from thrum_skill.opt_out import (
    DISABLE_MARKER,
    PERSONAL_MARKER,
    has_disable_marker,
    has_personal_marker,
)
from thrum_skill.parsers.transcript import TranscriptAggregate


def _buffer() -> dict:
    return {
        "session_id": "s1",
        "turn_start_ts": "2026-05-05T12:00:00Z",
        "turn_end_ts": "2026-05-05T12:00:30Z",
        "tools_used": [],
    }


# ── walk-up semantics ──────────────────────────────────────────────


def test_personal_marker_present_at_cwd(tmp_path):
    (tmp_path / PERSONAL_MARKER).touch()
    assert has_personal_marker(tmp_path) is True


def test_personal_marker_present_at_ancestor(tmp_path):
    deep = tmp_path / "a" / "b" / "c"
    deep.mkdir(parents=True)
    (tmp_path / PERSONAL_MARKER).touch()
    assert has_personal_marker(deep) is True


def test_personal_marker_absent(tmp_path):
    assert has_personal_marker(tmp_path) is False


def test_personal_independent_of_disable(tmp_path):
    """The two markers don't shadow each other."""
    (tmp_path / DISABLE_MARKER).touch()
    assert has_disable_marker(tmp_path) is True
    assert has_personal_marker(tmp_path) is False

    (tmp_path / PERSONAL_MARKER).touch()
    assert has_personal_marker(tmp_path) is True


def test_marker_walk_fails_open_on_unresolvable(tmp_path):
    """Bogus path — walk should not raise."""
    bogus = tmp_path / "does" / "not" / "exist"
    assert has_personal_marker(bogus) is False


# ── emitter wiring ────────────────────────────────────────────────


def _attr_dict(span) -> dict:
    """Project Span attrs into a {key: AnyValue} dict for assertions."""
    return {kv.key: kv.value for kv in span.attributes}


def test_build_turn_span_omits_attrs_when_default():
    span = build_turn_span(_buffer(), TranscriptAggregate())
    attrs = _attr_dict(span)
    assert "thrum.metadata.cwd" not in attrs
    assert "thrum.personal" not in attrs


def test_build_turn_span_emits_cwd_when_provided():
    span = build_turn_span(
        _buffer(), TranscriptAggregate(), cwd="/Users/me/work/repo"
    )
    attrs = _attr_dict(span)
    assert "thrum.metadata.cwd" in attrs
    assert attrs["thrum.metadata.cwd"].string_value == "/Users/me/work/repo"


def test_build_turn_span_emits_personal_when_true():
    span = build_turn_span(
        _buffer(), TranscriptAggregate(), personal=True
    )
    attrs = _attr_dict(span)
    assert "thrum.personal" in attrs
    assert attrs["thrum.personal"].bool_value is True


def test_build_turn_span_personal_false_is_omitted():
    """The False default is the common case — don't clutter every span."""
    span = build_turn_span(
        _buffer(), TranscriptAggregate(), personal=False
    )
    attrs = _attr_dict(span)
    assert "thrum.personal" not in attrs
