---
name: thrum-uninstall
description: Remove the Thrum client — unregisters hooks from ~/.claude/settings.json, removes the local token, and optionally deletes buffers/logs and the CLI binaries. Use when the user asks to uninstall Thrum, stop tracking, or remove the Thrum client.
---

# Uninstall Thrum

Reverses `/thrum-client:thrum-install`. Three depth levels:

| Level | What it removes | Command |
|---|---|---|
| **Standard** | Hook entries in `~/.claude/settings.json`, local token at `~/.config/thrum/token` | `thrum uninstall` |
| **Full** | Standard + `~/.config/thrum/` entirely (buffers, logs, backfill marker) | `thrum uninstall --full` |
| **Complete** | Full + `thrum` + `thrum-hook` binaries from PATH | `thrum uninstall --full` then `uv tool uninstall thrum-client` (or `pipx uninstall thrum-client`) |

Server-side activity rows captured from prior sessions are **not** touched. Those live in your Thrum backend.

## 1. Ask the user which depth they want

Default to **Standard** unless the user asks for more. Read back what it will do before running, so they can redirect.

## 2. Run

**Standard / Full:**

```bash
thrum uninstall            # standard
# or
thrum uninstall --full     # full
```

The command is idempotent — running a second time is safe.

**Complete removal (if they asked):**

```bash
uv tool uninstall thrum-client
# or, if installed via pipx:
pipx uninstall thrum-client
```

After this, `thrum` and `thrum-hook` are no longer on PATH. Subsequent sessions won't find `thrum-hook` to execute even if some orphan hook entry survived in `settings.json` — the hook simply fails with "command not found" and the host tool continues.

## 3. Verify

```bash
grep -c thrum-hook ~/.claude/settings.json 2>/dev/null || echo "no ~/.claude/settings.json"
test -f ~/.config/thrum/token && echo "token still present" || echo "token gone"
command -v thrum && echo "CLI still on PATH" || echo "CLI gone"
```

## 4. Reinstall path

If the user later wants to turn tracking back on, point them at `/thrum-client:thrum-install`.

## 5. Optional — remove the Claude Code plugin too

If they installed via `/plugin install`, the slash commands themselves stick around until the plugin is removed:

```text
/plugin uninstall thrum-client@thrum-labs-thrum-client
/plugin marketplace remove thrum-labs-thrum-client
```
