---
name: thrum-install
description: Install the Thrum activity tracking client — runs `uv tool install`, calls `thrum init` to guest-register against the Thrum backend, and merges hooks. Use when the user asks to install Thrum, start tracking AI activity, or set up the Thrum client.
---

# Install Thrum

Sets up Thrum activity tracking on the user's machine.

## 1. Verify the backend

The client talks to a Thrum backend over HTTP. Confirm it's reachable (default is a local instance):

```bash
curl -s "${THRUM_API_URL:-http://127.0.0.1:8000}/health"
```

Expect `{"status":"ok"}`. If it fails, the user either needs to start their backend or point `THRUM_API_URL` at a hosted instance.

## 2. Install the CLI

Preferred (fastest):

```bash
uv tool install git+https://github.com/thrum-labs/thrum-client --reinstall
```

Fallback if the user doesn't have `uv` and doesn't want to install it (`curl -LsSf https://astral.sh/uv/install.sh | sh`):

```bash
pipx install --force git+https://github.com/thrum-labs/thrum-client
```

Either path puts two binaries on PATH: `thrum` (init / status / uninstall) and `thrum-hook` (the per-event hook handler that the host tool spawns once per fired hook).

If neither `uv` nor `pipx` is available, point the user at https://docs.astral.sh/uv/ or https://pipx.pypa.io/ and retry.

## 3. Configure + initialize

```bash
export THRUM_API_URL=http://127.0.0.1:8000
thrum init
```

`thrum init` does two things atomically:

1. **Guest-register.** POSTs `/api/v1/auth/guest-register`, gets back a `tk_…` API key, writes it to `~/.config/thrum/token` (mode 0600).
2. **Register hooks.** Additively merges entries for the supported lifecycle events into `~/.claude/settings.json` (Claude Code: `UserPromptSubmit`, `PreToolUse`, `PostToolUse`, `SubagentStop`, `PreCompact`, `Stop`, `SessionEnd`). Codex CLI and Cursor registration are offered on demand. Existing hook entries are preserved.

Backfill of existing transcripts is no longer run by `init`. Run `thrum backfill` separately to replay historical Claude / Codex / Cursor sessions (one-time, idempotent via marker files under `~/.config/thrum/`).

## 4. Confirm

```bash
thrum status
```

Expect `Token present: True`. `Backfill done` will read `False` until you run `thrum backfill`.

## 5. What changed

- The next assistant turn in any registered tool fires Thrum's hooks. The handler exits in under 100 ms and never blocks tool calls.
- **Per-project opt-out.** `touch .thrum-disable` in any project root (or any ancestor directory) silences that project — the hook handler exits early, before any buffer write or network call.
- **Remove.** `thrum uninstall` removes hook entries + the local token. `uv tool uninstall thrum-client` removes the binaries.
- **Privacy.** Prompts, completions, and tool I/O never leave the machine. Enforced by a four-layer defence: emission allowlist, streaming `ijson` extraction with content keys never bound, allowlisted logging, and collector + backend rejection of content keys. A CI sentinel-fuzz test proves this across OTLP wire bytes, buffer files, and logs.
