# token-burn-forecaster

Forecast a Claude Code session's **final token burn from its first few turns**, using your own past sessions.

I run a lot of unattended Claude Code sessions, and the bill surprise comes from the few that run long. This reads
your transcripts (`~/.claude/projects/<project>/*.jsonl`), fits a tiny log-linear model on tokens-so-far and
tool-calls-so-far across your *other* sessions, and forecasts where the current one will end up.

```bash
uv run forecast.py ~/.claude/projects/<project>/<session>.jsonl          # history = the other sessions in that folder
uv run forecast.py <session>.jsonl --dir <history-folder> --n 5
uv run forecast.py --selfcheck
```

Example (a real session of mine):

```
after turn 5: 397,770 tokens so far -> forecast final ~3,202,199 tokens (fit on 13 past sessions; ...)
```

## How good is it?

Honestly: rough. Evaluated leave-one-out on 28 of my sessions (it never trains on the session it predicts),
forecasting from the first 5 turns, the **median error was 59.7%**; the mean is much worse because session lengths
are heavy-tailed. Use it as an early warning ("this one is heading long"), not a budget cutoff.

One Python file, numpy only, no network, no API key. Each assistant message's `usage` block is summed
(input + cache creation + cache read + output); streamed duplicate message ids are de-duplicated.

## Paid kit (optional)

The [$29 kit on Gumroad](https://vedovino.gumroad.com/l/token-burn-forecaster) adds a leave-one-out accuracy report
on **your** sessions and a live `--watch` mode that checks a running session every N seconds and flags it when the
forecast crosses your threshold. The forecaster above is the full model and stays free.

MIT licensed.
