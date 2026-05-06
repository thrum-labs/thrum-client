---
name: thrum-status
description: Check Thrum's local state — token presence, last event, backfill status. Use when the user asks about Thrum status or whether tracking is working.
---

# Thrum status

Reports the client's local state.

## 1. CLI status

```bash
thrum status
```

Expected output:
- `Token present: True` — local `tk_…` guest-register key in place at `~/.config/thrum/token`.
- `Last event: <iso-timestamp>` — timestamp of the most recent emit from the client (or `(none)` if nothing has fired yet).
- `Backfill done: True` — the install-time backfill ran.

If `thrum` is not on PATH, point the user at `/thrum-client:thrum-install`.

## 2. If nothing is tracking

- **Hooks not firing.** Check `~/.claude/settings.json` has `thrum-hook` entries:

  ```bash
  grep -c thrum-hook ~/.claude/settings.json
  ```

  Expect 7 with the current Claude Code settings schema. If it's 0, run `/thrum-client:thrum-install` (or `thrum init`).

- **Hooks fire but nothing reaches the backend.** Check `~/.config/thrum/skill.log` for `emit_turn_error` events. Common causes:
  - The OTLP collector is not running or unreachable at `THRUM_COLLECTOR_URL`.
  - `THRUM_API_URL` points at a backend that doesn't have the user's guest token.
  - Network policy blocks outbound requests to the collector.

- **Activities attributed to the wrong user.** The collector must be configured to forward the client's `x-api-key` header (not stamp its own). If you've just changed collector config, restart the collector.
