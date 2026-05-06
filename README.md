# Thrum client

Privacy-respecting activity tracker for AI coding sessions. Captures one OTLP span per assistant turn from Claude Code, Codex CLI, and Cursor — and ships it to your Thrum backend. Prompts, completions, and tool I/O never leave your machine.

## What it does

- Hooks into Claude Code, Codex CLI, and Cursor lifecycle events.
- Emits one OTLP span per assistant turn: model, token counts, tools used, latency, plan boundaries, compact transitions.
- One-time backfill over existing transcripts on first install (idempotent).
- Per-project opt-out: `touch .thrum-disable` in any project (or ancestor) directory.

## Install

### Option A — From your AI coding tool (recommended)

This repo ships skill bundles for Claude Code, Codex CLI, and Cursor. Pick your tool:

**Claude Code:**

```text
/plugin marketplace add thrum-labs/thrum-client
/plugin install thrum-client@thrum-labs-thrum-client
```

**Codex CLI:**

```bash
codex plugin marketplace add thrum-labs/thrum-client
```

Then install via `codex /plugins`.

**Cursor:**

```bash
git clone https://github.com/thrum-labs/thrum-client.git ~/.cursor/plugins/local/thrum-client
```

Restart Cursor.

After install, three slash commands are available in your tool:

- `/thrum-client:thrum-install` — guides you through `uv tool install` + `thrum init`.
- `/thrum-client:thrum-status` — checks token, last event, backfill state.
- `/thrum-client:thrum-uninstall` — removes hooks + token (and optionally binaries).

Run `/thrum-client:thrum-install` and follow the prompts.

### Option B — Direct CLI install

You'll need [uv](https://docs.astral.sh/uv/) (or [pipx](https://pipx.pypa.io/)) and a reachable Thrum backend.

```bash
uv tool install git+https://github.com/thrum-labs/thrum-client

export THRUM_API_URL=http://127.0.0.1:8000   # your backend
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

If you installed via Option A (plugin install), also remove the plugin from your AI tool:

**Claude Code:**

```text
/plugin uninstall thrum-client@thrum-labs-thrum-client
/plugin marketplace remove thrum-labs-thrum-client
```

**Codex CLI:**

```bash
codex plugin marketplace remove thrum-labs/thrum-client
```

**Cursor:**

```bash
rm -rf ~/.cursor/plugins/local/thrum-client
```

## Configuration

Environment variables (read from `~/.config/thrum/env` or the shell):

| Variable | Default | Purpose |
|---|---|---|
| `THRUM_API_URL` | `http://127.0.0.1:8000` | Thrum backend (auth, ingest control plane). |
| `THRUM_COLLECTOR_URL` | `http://127.0.0.1:4318/v1/traces` | OTLP/HTTP collector. |
| `THRUM_CONFIG_DIR` | `~/.config/thrum` | Token, buffers, logs, backfill marker. |
| `THRUM_CLAUDE_DIR` | `~/.claude` | Claude Code settings dir (only used when registering Claude Code hooks). |
| `CODEX_HOME` | `~/.codex` | Codex CLI config dir (only used when registering Codex hooks). Same env var Codex itself respects. |
| `CURSOR_HOME` | `~/.cursor` | Cursor config dir (only used when registering Cursor hooks). Cursor itself does not read this — useful mainly for tests. |

## Per-project opt-out

`touch .thrum-disable` in any project root (or any ancestor directory) silences that project — the hook handler exits early, before any buffer write or network call.

## Privacy

Prompts, completions, and tool I/O never leave the machine. Three allowlists enforce it: an ijson path allowlist in the parsers, an attribute allowlist in the span builder, and a field allowlist in `safe_log()`. A sentinel-fuzz test verifies the property across disk and OTLP wire.

## Limitations

- One Thrum account = one machine. After `thrum init` on a second machine, claiming that guest creates a separate Thrum identity — there is no endpoint today that folds activity from a second guest into your already-claimed account. Linking is planned (see `temp/claim-ui-spike.md` "Known limitations after v1") but not yet implemented.
