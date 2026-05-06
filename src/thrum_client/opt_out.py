"""FR-227 + FR-707 per-project marker walks.

Two markers, two semantics:

- `.thrum-disable` (FR-227) — silence ingest entirely. The handler
  exits 0 before any transcript read or network call.
- `.thrum-personal` (FR-707) — let the activity flow into the user's
  own profile but **exclude it from any group**. The handler still
  emits the span; the backend resolver sees `thrum.personal=true` on
  the span and forces `activities.group_ids = NULL`.

Both walks share the same shape — walk up from cwd looking for the
marker file. The shared `_has_marker` helper keeps them consistent.
"""

from __future__ import annotations

from pathlib import Path


DISABLE_MARKER = ".thrum-disable"
PERSONAL_MARKER = ".thrum-personal"


def _has_marker(cwd: Path, marker: str) -> bool:
    try:
        current = cwd.resolve()
    except (OSError, RuntimeError):
        return False
    while True:
        if (current / marker).exists():
            return True
        parent = current.parent
        if parent == current:
            return False
        current = parent


def has_disable_marker(cwd: Path) -> bool:
    """FR-227 — walk from `cwd` toward the filesystem root looking for
    `.thrum-disable`. Fail open (False) on unresolvable paths."""
    return _has_marker(cwd, DISABLE_MARKER)


def has_personal_marker(cwd: Path) -> bool:
    """FR-707 — walk from `cwd` toward the filesystem root looking for
    `.thrum-personal`. Fail open (False) on unresolvable paths.

    Distinct from `has_disable_marker`: a `.thrum-personal` tree still
    emits activity (the user sees it on their own profile) but the
    backend resolver short-circuits group attribution.
    """
    return _has_marker(cwd, PERSONAL_MARKER)
