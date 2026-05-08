# Thrum client

Privacy-respecting activity tracker for AI coding sessions. Captures one OTLP span per assistant turn from Claude Code, Codex CLI, and Cursor — and ships it to your Thrum backend. Prompts, completions, and tool I/O never leave your machine.

## What it does

- Hooks into Claude Code, Codex CLI, and Cursor lifecycle events.
- Emits one OTLP span per assistant turn: model, token counts, tools used, latency, plan boundaries, compact transitions.
- One-time backfill over existing transcripts on first install (idempotent).
- Per-project opt-out: `touch .thrum-disable` in any project (or ancestor) directory.

## Install

You'll need [uv](https://docs.astral.sh/uv/) (or [pipx](https://pipx.pypa.io/)) and a reachable Thrum backend.

```bash
uv tool install git+https://github.com/thrum-labs/thrum-client

thrum init
```

Don't have `uv`? `pipx install git+https://github.com/thrum-labs/thrum-client` works identically.

`thrum init` registers a guest user, merges hook entries into `~/.claude/settings.json` (Claude Code), optionally `~/.codex/config.toml` (Codex CLI) and `~/.cursor/hooks.json` (Cursor) when you opt in, and runs the one-time backfill.

## Verify

```bash
thrum status
```

Expect `Token present: True` and `Backfill done: True`.

## Uninstall

```bash
thrum uninstall            # removes hook entries + local token
thrum uninstall --full     # also removes ~/.config/thrum/
uv tool uninstall thrum-client   # removes the binaries (or `pipx uninstall thrum-client`)
```

## Configuration

Environment variables (read from `~/.config/thrum/env` or the shell):

| Variable | Default | Purpose |
|---|---|---|
| `THRUM_API_URL` | `https://thrumlabs.com` | Thrum backend (auth, ingest control plane). |
| `THRUM_COLLECTOR_URL` | `https://collector.thrumlabs.com/v1/traces` | OTLP/HTTP collector. |
| `THRUM_CONFIG_DIR` | `~/.config/thrum` | Token, buffers, logs, backfill marker. |
| `THRUM_CLAUDE_DIR` | `~/.claude` | Claude Code settings dir (only used when registering Claude Code hooks). |
| `CODEX_HOME` | `~/.codex` | Codex CLI config dir (only used when registering Codex hooks). Same env var Codex itself respects. |
| `CURSOR_HOME` | `~/.cursor` | Cursor config dir (only used when registering Cursor hooks). Cursor itself does not read this — useful mainly for tests. |

## Per-project opt-out

`touch .thrum-disable` in any project root (or any ancestor directory) silences that project — the hook handler exits early, before any buffer write or network call.

## Privacy

Prompts, completions, and tool I/O never leave the machine. Three allowlists enforce it: an ijson path allowlist in the parsers, an attribute allowlist in the span builder, and a field allowlist in `safe_log()`. A sentinel-fuzz test verifies the property across disk and OTLP wire.