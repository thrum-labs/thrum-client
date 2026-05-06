"""Thrum CLI — `thrum {init,run,status,logout}`.

`run` is a placeholder; the hook handler entry point is `thrum-hook`
(see `handler.py`).
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import click
import httpx

from thrum_client import __version__
from thrum_client.codex_config import (
    check_codex_version,
    codex_config_path,
    merge_codex_hooks,
    unmerge_codex_hooks,
)
from thrum_client.config import SkillSettings, load_settings
from thrum_client.cursor_config import (
    cursor_hooks_path,
    merge_cursor_hooks,
    unmerge_cursor_hooks,
)
from thrum_client.safe_log import safe_log
from thrum_client.settings_merge import merge_hooks, unmerge_hooks


def _write_token(path: Path, api_key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write then chmod so the final mode is 0600 even if umask is tight.
    path.write_text(api_key)
    os.chmod(path, 0o600)


def _hook_command() -> str:
    """Return the `command` string written into Claude's settings.json /
    Codex's config.toml. Both harnesses pass this value to `/bin/sh -c`,
    so an unquoted path containing whitespace ("…/thrum vibe/.venv/…")
    would be split into multiple arguments and fail with `No such file
    or directory`. `shlex.quote` is a no-op for safe paths and inserts
    the single-quote wrapper only when needed.
    """
    env_cmd = os.environ.get("THRUM_HOOK_CMD")
    if env_cmd:
        return env_cmd
    cmd = shutil.which("thrum-hook") or "thrum-hook"
    return shlex.quote(cmd) if any(c in cmd for c in " \t'\"") else cmd


def _maybe_register_cursor_hooks(settings: SkillSettings) -> None:
    """Optional Cursor IDE registration (FR-218f).

    Writes ~/.cursor/hooks.json with strict JSON. Cursor watches the
    file and hot-reloads on save. No version gate today — Cursor's
    `cursor --version` CLI is not consistently available across install
    flavours; we skip silently if `~/.cursor/` doesn't exist (i.e.
    Cursor not installed) and accept that broken Cursor versions will
    surface their own load errors in Cursor's Hooks settings panel.
    """
    cursor_dir = cursor_hooks_path().parent
    if not cursor_dir.exists():
        # Cursor not installed — skip with a one-line user-visible
        # message so the install flow doesn't appear to silently ignore
        # Cursor (review-flagged UX gap; was asymmetric with Codex's
        # explicit "skipping Codex hook registration" line).
        click.echo("Cursor hooks: not installed (no ~/.cursor/) — skipping")
        return
    try:
        merged = merge_cursor_hooks(cursor_hooks_path(), _hook_command())
    except Exception as exc:  # JSON parse / write errors must not break init
        safe_log(
            "init_cursor_error",
            log_path=settings.log_path,
            error_category=type(exc).__name__,
        )
        click.echo(f"Cursor hook registration skipped: {type(exc).__name__}")
        return
    safe_log(
        "init_cursor",
        log_path=settings.log_path,
        status="merged" if merged else "already_registered",
    )
    click.echo(
        f"Cursor hooks: {cursor_hooks_path()}"
        + ("" if merged else " (already registered)")
    )


def _maybe_register_codex_hooks(settings: SkillSettings) -> None:
    """Optional Codex CLI registration (FR-218b/c/d).

    Gated on `codex --version >= 0.124.0`. On a lower or missing version,
    skips silently with a clear message; never blocks the Claude flow.
    The TOML write uses tomlkit so unrelated tables (`[projects.<path>]`,
    `[tui.*]`) survive byte-for-byte.
    """
    if not check_codex_version():
        click.echo(
            "Codex CLI not found or version < 0.124.0 — skipping Codex hook "
            "registration. Run `codex --version` to check."
        )
        return
    cfg = codex_config_path()
    try:
        merged = merge_codex_hooks(cfg, _hook_command())
    except Exception as exc:  # tomlkit parse/write errors must not break init
        safe_log(
            "init_codex_error",
            log_path=settings.log_path,
            error_category=type(exc).__name__,
        )
        click.echo(f"Codex hook registration skipped: {type(exc).__name__}")
        return
    safe_log(
        "init_codex",
        log_path=settings.log_path,
        status="merged" if merged else "already_registered",
    )
    click.echo(
        f"Codex hooks: {cfg}"
        + ("" if merged else " (already registered)")
    )


@click.group()
@click.version_option(__version__)
def main() -> None:
    """Thrum client skill."""


# `groups` subcommand: lazy import of `groups_cli` (which pulls in
# httpx + the picker plumbing) deferred until the user actually runs
# `thrum groups`. Avoids ~50ms of import cost on every `thrum-hook`
# invocation. The eager import in the previous version sat in
# `_register_groups_command()` and ran at module load, defeating the
# lazy comment (Fix #14).
@main.command(
    "groups",
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
    },
    help="Manage per-project group attribution.",
    add_help_option=False,
)
@click.pass_context
def groups(ctx: click.Context) -> None:
    """Lazy proxy that imports the real `groups_cmd` and dispatches the
    forwarded argv to it on invocation."""
    from thrum_client.groups_cli import groups_cmd as _real

    with _real.make_context("groups", list(ctx.args), parent=ctx) as inner:
        return _real.invoke(inner)


def _run_backfill_pair(
    settings: SkillSettings,
    *,
    claude: bool,
    codex: bool,
    cursor: bool,
    force: bool,
) -> None:
    """Run the Claude / Codex / Cursor backfill loops, swallowing exceptions
    so the caller (`init` or the `backfill` subcommand) never crashes on
    a single bad transcript.

    The marker files (`.backfill_done` / `.codex_backfill_done` /
    `.cursor_backfill_done`) make each loop independently idempotent
    across re-runs — pass `force=True` to bypass them when a new code
    path lands (e.g. a new plan detector that re-attributes historical
    turns).
    """
    from thrum_client.backfill import (
        run_backfill,
        run_codex_backfill,
        run_cursor_backfill,
    )

    if claude:
        try:
            spans = run_backfill(settings, force=force)
        except Exception as exc:  # defensive — backfill must never crash init
            safe_log(
                "init_backfill_error",
                log_path=settings.log_path,
                error_category=type(exc).__name__,
            )
            click.echo(f"Backfill skipped: {type(exc).__name__}")
            spans = 0
        if spans:
            click.echo(f"Backfilled {spans} historical Claude turn(s).")

    if codex:
        try:
            codex_spans = run_codex_backfill(settings, force=force)
        except Exception as exc:
            safe_log(
                "init_codex_backfill_error",
                log_path=settings.log_path,
                error_category=type(exc).__name__,
            )
            click.echo(f"Codex backfill skipped: {type(exc).__name__}")
            codex_spans = 0
        if codex_spans:
            click.echo(f"Backfilled {codex_spans} historical Codex turn(s).")

    if cursor:
        try:
            cursor_spans = run_cursor_backfill(settings, force=force)
        except Exception as exc:
            safe_log(
                "init_cursor_backfill_error",
                log_path=settings.log_path,
                error_category=type(exc).__name__,
            )
            click.echo(f"Cursor backfill skipped: {type(exc).__name__}")
            cursor_spans = 0
        if cursor_spans:
            click.echo(f"Backfilled {cursor_spans} historical Cursor turn(s).")


@main.command()
def init() -> None:
    """Register as a guest, merge hooks, and run any pending backfill.

    Idempotent — re-running on an installed machine is safe and lets the
    backfill loops pick up new transcript sources (e.g. Codex rollouts
    accumulated since the original install) without an explicit
    `thrum backfill` invocation. Marker files prevent re-emitting turns
    that have already been ingested.
    """
    settings = load_settings()

    if settings.token_path.exists():
        merged = merge_hooks(settings.claude_settings_path, _hook_command())
        safe_log(
            "init_existing_token",
            log_path=settings.log_path,
            status="merged" if merged else "already_registered",
        )
        click.echo(f"Already installed — token at {settings.token_path}")
        click.echo(f"Hooks: {settings.claude_settings_path}")
        _maybe_register_codex_hooks(settings)
        _maybe_register_cursor_hooks(settings)
        # Fall through to the shared backfill block below — markers gate
        # idempotency, so this is a no-op when all loops have already run.
    else:
        url = settings.api_url.rstrip("/") + "/api/v1/auth/guest-register"
        try:
            resp = httpx.post(url, timeout=10.0)
        except httpx.HTTPError as exc:
            safe_log(
                "init_network_error",
                log_path=settings.log_path,
                error_category=type(exc).__name__,
            )
            raise click.ClickException(
                f"could not reach {url}: {type(exc).__name__}"
            )

        if resp.status_code != 201:
            raise click.ClickException(
                f"guest-register failed: HTTP {resp.status_code}"
            )

        data = resp.json()["data"]
        api_key = data["api_key"]
        username = data["username"]

        _write_token(settings.token_path, api_key)
        merged = merge_hooks(settings.claude_settings_path, _hook_command())

        safe_log(
            "init_ok",
            log_path=settings.log_path,
            status="merged" if merged else "already_registered",
        )

        click.echo(f"Thrum installed as {username}.")
        click.echo(f"Token: {settings.token_path}")
        click.echo(f"Hooks: {settings.claude_settings_path}")

        _maybe_register_codex_hooks(settings)
        _maybe_register_cursor_hooks(settings)

    _run_backfill_pair(
        settings, claude=True, codex=True, cursor=True, force=False
    )


@main.command()
@click.option(
    "--claude/--no-claude",
    default=None,
    help="Backfill Claude transcripts. Defaults to enabled when no source flag is given.",
)
@click.option(
    "--codex/--no-codex",
    default=None,
    help="Backfill Codex rollouts. Defaults to enabled when no source flag is given.",
)
@click.option(
    "--cursor/--no-cursor",
    default=None,
    help="Backfill Cursor transcripts. Defaults to enabled when no source flag is given.",
)
@click.option(
    "--all",
    "all_",
    is_flag=True,
    default=False,
    help="Run all backfill loops. Equivalent to `--claude --codex --cursor` (the default).",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help=(
        "Ignore the per-source marker files and re-emit every historical "
        "turn. Use after a parser change that re-attributes historical "
        "spans (e.g. a new plan detector)."
    ),
)
def backfill(
    claude: bool | None,
    codex: bool | None,
    cursor: bool | None,
    all_: bool,
    force: bool,
) -> None:
    """Replay historical transcripts / rollouts as backfill spans.

    With no flags all three sources (Claude, Codex, Cursor) run. Pass
    `--no-claude` / `--no-codex` / `--no-cursor` to scope. `--force`
    bypasses the marker files when you've changed a parser and want
    history re-attributed.
    """
    settings = load_settings()
    if not settings.token_path.exists():
        raise click.ClickException(
            "Not registered yet — run `thrum init` first."
        )
    # Default all ON when no source flag was supplied (or when `--all`
    # was passed). Click reports `None` for an unspecified flag thanks
    # to `default=None`, which lets us tell "user explicitly disabled"
    # apart from "user said nothing".
    if all_ or (claude is None and codex is None and cursor is None):
        claude = True
        codex = True
        cursor = True
    else:
        claude = bool(claude)
        codex = bool(codex)
        cursor = bool(cursor)
    if not (claude or codex or cursor):
        raise click.ClickException("Nothing to do — all sources disabled.")
    _run_backfill_pair(
        settings, claude=claude, codex=codex, cursor=cursor, force=force
    )


@main.command()
def status() -> None:
    """Print last event timestamp, token-file present, backfill state."""
    settings = load_settings()
    state = _read_state(settings.state_path)
    click.echo(f"Token present: {settings.token_path.exists()}")
    click.echo(f"Last event: {state.get('last_event_ts') or '(none)'}")
    click.echo(f"Backfill done: {settings.backfill_marker.exists()}")
    # H5 — group attribution for the current cwd. Best-effort: skip the
    # network call when the token is missing (would fail on `thrum init`
    # not yet run) or when the network is offline.
    if settings.token_path.exists():
        try:
            from thrum_client.groups_cli import (
                _get_mapping,
                derive_project_key,
            )

            cwd = os.getcwd()
            project_key = derive_project_key(cwd)
            click.echo(f"Project key: {project_key or '(none)'}")
            if project_key is not None:
                mapped = _get_mapping(settings, project_key)
                if mapped is None:
                    click.echo("Group attribution: (home_group fallback)")
                else:
                    click.echo(f"Group attribution: {len(mapped)} group(s)")
        except Exception as exc:  # network / parse failures are non-fatal
            click.echo(f"Group attribution: (lookup failed: {type(exc).__name__})")


@main.command()
def logout() -> None:
    """Remove local token. Server-side revoke is a future feature."""
    settings = load_settings()
    if settings.token_path.exists():
        settings.token_path.unlink()
        click.echo(f"Token removed: {settings.token_path}")
    else:
        click.echo("Already logged out.")
    click.echo(
        "Note: the API key is still valid server-side until revoke is built."
    )


@main.command()
def run() -> None:
    """Placeholder; hook handler is `thrum-hook`."""
    click.echo("The hook handler binary is `thrum-hook`; this command is a noop.")


@main.command()
@click.option(
    "--full",
    is_flag=True,
    help="Also remove buffers / logs / markers under ~/.config/thrum/.",
)
def uninstall(full: bool) -> None:
    """Unregister hooks from ~/.claude/settings.json and remove local state.

    Does NOT remove the CLI binary. Run `uv tool uninstall thrum-client`
    separately if you want that too.
    """
    settings = load_settings()

    removed = unmerge_hooks(settings.claude_settings_path)
    if removed:
        click.echo(f"Unregistered hooks in {settings.claude_settings_path}")
    else:
        click.echo(f"No thrum-hook entries found in {settings.claude_settings_path}")

    codex_cfg = codex_config_path()
    if codex_cfg.exists():
        try:
            codex_removed = unmerge_codex_hooks(codex_cfg)
        except Exception as exc:
            safe_log(
                "uninstall_codex_error",
                log_path=settings.log_path,
                error_category=type(exc).__name__,
            )
            click.echo(f"Codex unregister skipped: {type(exc).__name__}")
        else:
            if codex_removed:
                click.echo(f"Unregistered Codex hooks in {codex_cfg}")
            else:
                click.echo(f"No thrum-hook entries found in {codex_cfg}")

    cursor_cfg = cursor_hooks_path()
    if cursor_cfg.exists():
        try:
            cursor_removed = unmerge_cursor_hooks(cursor_cfg)
        except Exception as exc:
            safe_log(
                "uninstall_cursor_error",
                log_path=settings.log_path,
                error_category=type(exc).__name__,
            )
            click.echo(f"Cursor unregister skipped: {type(exc).__name__}")
        else:
            if cursor_removed:
                click.echo(f"Unregistered Cursor hooks in {cursor_cfg}")
            else:
                click.echo(f"No thrum-hook entries found in {cursor_cfg}")

    if settings.token_path.exists():
        settings.token_path.unlink()
        click.echo(f"Token removed: {settings.token_path}")

    if full:
        import shutil

        if settings.config_dir.exists():
            shutil.rmtree(settings.config_dir)
            click.echo(f"Removed {settings.config_dir}")
    else:
        if settings.config_dir.exists():
            click.echo(
                f"Kept buffers and logs at {settings.config_dir} "
                "(pass --full to remove)."
            )

    click.echo("\nThe CLI binaries remain on PATH. Remove with:")
    click.echo("  uv tool uninstall thrum-client    # or: pipx uninstall thrum-client")
    click.echo(
        "\nActivities already sent to your Thrum backend are not deleted."
    )


def _read_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text() or "{}")
    except json.JSONDecodeError:
        return {}


def _write_state(path: Path, **fields) -> None:
    """Helper for other modules (handler/emitter) to update state atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    current = _read_state(path)
    current.update(fields)
    current["updated_at"] = datetime.now(UTC).isoformat()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(current, separators=(",", ":")))
    tmp.replace(path)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
