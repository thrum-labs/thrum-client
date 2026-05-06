"""`thrum groups` — interactive picker for FR-700 group attribution.

Implements Group H (H1–H5):

- `thrum groups`              — pick which groups this project counts
                                toward (interactive numeric prompts)
- `thrum groups --all`        — list all known per-project mappings
- `thrum groups --primary X`  — set home_group_id to slug X
- `thrum groups --backfill`   — POST /users/me/projects/{key}/groups/backfill
                                for the current cwd

Talks to the backend with the api_key from `~/.config/thrum/token`. The
new `get_current_user_or_api_key` dep on the picker endpoints accepts
either Bearer JWT (web) or x-api-key (CLI) — same trust model as
ingest, since the api_key already represents the user.

Project-key derivation mirrors `services.project_keys.derive_project_key`
on the backend so the CLI and ingest agree on what `cwd → key` means.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote

import click
import httpx

from thrum_client.config import SkillSettings, load_settings


_REMOTE_URL_RE = re.compile(r"^\s*url\s*=\s*(\S+)\s*$", flags=re.MULTILINE)


# ── project_key derivation (mirrors backend services/project_keys.py) ──


def _normalise_remote(url: str) -> str:
    url = url.strip()
    url = re.sub(r"^(https?://)[^@/]+@", r"\1", url)
    url = url.rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]
    return url


def _read_git_remote(git_dir: Path) -> str | None:
    config = git_dir / "config"
    try:
        content = config.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    in_origin = False
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_origin = stripped == '[remote "origin"]'
            continue
        if in_origin:
            m = _REMOTE_URL_RE.match(line)
            if m:
                return _normalise_remote(m.group(1))
    return None


def _walk_to_git(start: Path) -> Path | None:
    cur = start.resolve(strict=False)
    while True:
        if (cur / ".git").is_dir():
            return cur
        parent = cur.parent
        if parent == cur:
            return None
        cur = parent


def derive_project_key(cwd: str | None) -> str | None:
    """Mirror of `backend/.../project_keys.derive_project_key`. The CLI
    and the backend MUST agree on this function or the picker writes a
    key that the resolver will never match.
    """
    if not cwd:
        return None
    cwd_path = Path(cwd)
    repo_root = _walk_to_git(cwd_path)
    if repo_root is not None:
        remote = _read_git_remote(repo_root / ".git")
        if remote:
            return f"git:{remote}"
        return f"path:{os.fspath(repo_root)}"
    return f"cwd:{os.fspath(cwd_path)}"


# ── HTTP helpers ──────────────────────────────────────────────────


def _api_url(settings: SkillSettings, path: str) -> str:
    return settings.api_url.rstrip("/") + "/api/v1" + path


def _headers(settings: SkillSettings) -> dict[str, str]:
    if not settings.token_path.exists():
        raise click.ClickException(
            f"Not installed — no token at {settings.token_path}. "
            "Run `thrum init` first."
        )
    return {"x-api-key": settings.token_path.read_text().strip()}


def _http(settings: SkillSettings) -> httpx.Client:
    return httpx.Client(timeout=10.0, headers=_headers(settings))


def _raise_for_status(r: httpx.Response, context: str) -> None:
    """Translate httpx HTTPStatusError into a friendly ClickException
    so users see `Error: …` rather than a Python traceback (Fix #14).
    """
    if r.is_success:
        return
    detail = ""
    try:
        body = r.json()
        detail = body.get("error", {}).get("detail") or body.get("detail") or ""
    except Exception:  # body wasn't JSON or schema didn't match
        detail = r.text[:200]
    raise click.ClickException(
        f"{context} failed ({r.status_code}): {detail}".strip()
    )


def _list_my_groups(settings: SkillSettings) -> list[dict[str, Any]]:
    with _http(settings) as c:
        r = c.get(_api_url(settings, "/users/me/groups"))
    _raise_for_status(r, "list groups")
    return r.json()["data"]


def _get_mapping(
    settings: SkillSettings, project_key: str
) -> list[str] | None:
    encoded = quote(project_key, safe="")
    with _http(settings) as c:
        r = c.get(
            _api_url(settings, f"/users/me/projects/{encoded}/groups")
        )
    if r.status_code == 404:
        return None
    _raise_for_status(r, "read project mapping")
    return r.json()["data"]["group_ids"]


def _put_mapping(
    settings: SkillSettings, project_key: str, group_ids: list[str]
) -> None:
    encoded = quote(project_key, safe="")
    with _http(settings) as c:
        r = c.put(
            _api_url(settings, f"/users/me/projects/{encoded}/groups"),
            json={"group_ids": group_ids},
        )
    _raise_for_status(r, "save project mapping")


def _delete_mapping(
    settings: SkillSettings, project_key: str
) -> None:
    encoded = quote(project_key, safe="")
    with _http(settings) as c:
        r = c.delete(
            _api_url(settings, f"/users/me/projects/{encoded}/groups")
        )
    if r.status_code not in (204, 404):
        _raise_for_status(r, "clear project mapping")


def _set_home_group(settings: SkillSettings, slug: str) -> None:
    """Resolve slug → group_id, then PATCH /users/me with the group's UUID."""
    groups = _list_my_groups(settings)
    by_slug = {g["slug"]: g for g in groups}
    if slug not in by_slug:
        raise click.ClickException(
            f"Not a member of group {slug!r}. "
            f"Your groups: {', '.join(by_slug) or '(none)'}"
        )
    group_id = by_slug[slug]["id"]
    with _http(settings) as c:
        r = c.patch(
            _api_url(settings, "/users/me"),
            json={"home_group_id": group_id},
        )
    _raise_for_status(r, "set home group")
    click.echo(f"Home group set to {slug!r}.")


def _backfill_mapping(
    settings: SkillSettings, project_key: str
) -> int:
    encoded = quote(project_key, safe="")
    with _http(settings) as c:
        r = c.post(
            _api_url(
                settings,
                f"/users/me/projects/{encoded}/groups/backfill",
            ),
        )
    _raise_for_status(r, "backfill project mapping")
    return int(r.json()["data"]["rows_updated"])


# ── render helpers ────────────────────────────────────────────────


def _format_group_row(
    g: dict[str, Any], *, attributed: bool
) -> str:
    mark = "x" if attributed else " "
    role = g.get("role") or "?"
    status = g.get("status") or "?"
    return f"  [{mark}] {g['slug']:<24} ({role}, {status})"


# ── interactive picker ────────────────────────────────────────────


def _interactive_picker(
    settings: SkillSettings,
    *,
    project_key: str,
    groups: list[dict[str, Any]],
    current_ids: set[str],
) -> None:
    """Numeric-prompt picker. Avoids prompt_toolkit dep — same UX
    intent (toggle group N, save with `s`, cancel with `q`) but plain
    Click I/O so the install footprint stays small."""
    by_index = {i + 1: g for i, g in enumerate(groups)}
    # The picker shows slugs to the user but writes UUIDs to the
    # server (`group_ids[]` is a UUID column). The summary endpoint
    # exposes `id` so no slug→id round-trips are needed here.
    selected: set[str] = set(current_ids)

    while True:
        click.echo("")
        click.echo(f"Project: {project_key}")
        click.echo("Toggle which groups this project counts toward:")
        for i, g in by_index.items():
            attributed = g.get("id") in selected
            click.echo(f"  {i}. {_format_group_row(g, attributed=attributed)}")
        click.echo("")
        # Empty input re-renders the menu rather than silently saving
        # (Fix #14 — the previous default="s" meant pressing Enter
        # accidentally committed the current selection).
        choice = click.prompt(
            "Enter number to toggle, [s]ave, [q]uit, [c]lear-all",
            default="",
            show_default=False,
        ).strip().lower()

        if choice == "":
            continue
        if choice == "q":
            click.echo("Cancelled — no changes saved.")
            return
        if choice == "s":
            if not selected:
                # Saving with nothing selected → DELETE the mapping
                # (per FR-705 spec note). Leaves home_group_id as the
                # ingest fallback.
                _delete_mapping(settings, project_key)
                click.echo(
                    f"Cleared mapping for {project_key} — "
                    "ingest will fall back to home_group_id."
                )
                return
            _put_mapping(settings, project_key, sorted(selected))
            click.echo(
                f"Saved {len(selected)} group(s) for {project_key}."
            )
            return
        if choice == "c":
            selected.clear()
            continue
        if choice.isdigit():
            idx = int(choice)
            g = by_index.get(idx)
            if g is None:
                click.echo(f"Invalid number: {idx}")
                continue
            gid = g["id"]
            if gid in selected:
                selected.remove(gid)
            else:
                selected.add(gid)
            continue
        click.echo(f"Unknown command: {choice!r}")


# ── Click subcommand surface ──────────────────────────────────────


@click.command("groups")
@click.option(
    "--all",
    "show_all",
    is_flag=True,
    help="List every per-project mapping, not just current cwd.",
)
@click.option(
    "--primary",
    "primary_slug",
    metavar="SLUG",
    help="Set this group as your home group (FR-704).",
)
@click.option(
    "--backfill",
    is_flag=True,
    help="Re-stamp existing activities for the current cwd's project_key.",
)
def groups_cmd(
    show_all: bool,
    primary_slug: str | None,
    backfill: bool,
) -> None:
    """Manage per-project group attribution.

    With no flags → opens an interactive picker for the current cwd.
    """
    settings = load_settings()

    if primary_slug is not None:
        _set_home_group(settings, primary_slug)
        return

    cwd = os.getcwd()
    project_key = derive_project_key(cwd)
    if project_key is None:
        raise click.ClickException(
            "Could not derive a project_key from the current directory."
        )

    if backfill:
        count = _backfill_mapping(settings, project_key)
        click.echo(
            f"Backfill complete: {count} activities re-stamped for "
            f"{project_key}."
        )
        return

    if show_all:
        # Cheap implementation for v2.5 — list groups + the current
        # cwd's mapping. A real per-project list endpoint can land later.
        groups = _list_my_groups(settings)
        click.echo("Your groups:")
        for g in groups:
            click.echo(f"  {g['slug']:<24} ({g.get('role')}, {g.get('status')})")
        click.echo("")
        click.echo(f"Current cwd → {project_key}")
        current = _get_mapping(settings, project_key)
        if current is None:
            click.echo("  (no mapping; ingest uses home_group fallback)")
        else:
            click.echo(f"  Mapped to {len(current)} group(s): {current}")
        return

    groups = _list_my_groups(settings)
    if not groups:
        click.echo(
            "You're not in any groups yet. Create one or accept an "
            "invite first."
        )
        return

    current_ids: set[str] = set()
    existing = _get_mapping(settings, project_key)
    if existing is not None:
        current_ids = set(existing)

    _interactive_picker(
        settings,
        project_key=project_key,
        groups=groups,
        current_ids=current_ids,
    )
