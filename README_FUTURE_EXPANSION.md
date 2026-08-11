# Future Expansion Ideas

Not being worked on now — running fully local, everything below is "if/when."

## Going beyond local (only if you want the dashboard reachable off your own machine)

- **Render free tier** — 750 instance-hrs/month, free. Sleeps after 15 min idle,
  ~60s cold start on the next request. A total non-issue for "check it once a
  morning" traffic - probably the right call when this is wanted.
- **Render Starter ($7/mo)** — same platform, always-on, no cold start.
- **Small VPS (DigitalOcean/Linode, ~$4-6/mo)** — more setup, full control.
- **Railway** — no more ongoing free tier (one-time trial credit only), then
  usage-based (~$5/mo+). Cheaper than Render for bursty/idle traffic.
- **Cloudflare Tunnel (free)** — expose the already-local `uvicorn` server
  when you want remote access, no separate hosting bill at all. Only works
  while your own machine is on and the tunnel is running.

## Pipeline / model

- Investigate the open question from the handoff notes: feature importances
  are volatility-dominated (~0.55) - is the model mostly finding
  already-jumpy stocks rather than genuinely predicting *direction*? Check
  whether high-volatility misses gap down as often as up.
- Automate `build_watchlist.py` on a schedule (weekly/monthly) instead of
  running it manually.
- Automate `fill_actual_gap.py` + `merge_feedback.py` + a periodic
  `train_model.py` retrain, instead of running each by hand.
- Add a notification (email/SMS/push) for the daily top-5 instead of reading
  console output over coffee - useful once the "read it every morning"
  habit is established and feels like a chore.

## Once actually trading (paper or real)

- Track paper-trade results against `picks_log.csv`'s `actual_gap_pct` to see
  if acting on the picks (with real fills/slippage) matches the backtested
  Precision@5.
- Before any real money: position sizing / risk rules - none of this exists
  yet, and Precision@5 = 0.326 still means most days include a miss.

  ## Not yet built / open next steps

- **Scheduling `run_daily.py`** via cron/Task Scheduler - documented in
  its docstring, never actually set up.
- **Open model question, not yet investigated** (from the original
  handoff, still unresolved): feature importances are volatility-dominated
  (~0.55) - is the model mostly finding already-jumpy stocks rather than
  genuinely predicting direction? Worth checking whether high-volatility
  misses gap down as often as up.
- **Notification/alerting** instead of manually checking each morning -
  listed in `README_FUTURE_EXPANSION.md`, not started.
- **Position sizing / risk rules** - don't exist yet, relevant before any
  real (non-paper) money.
- **Hosting** - deliberately deferred, see `README_FUTURE_EXPANSION.md` (updates)
  for pricing/options if that changes.
- Whether the model's close-to-open label should eventually be
  supplemented or replaced by an entry-price-relative label, if
  `actual_return_from_entry_pct` data (now being collected) shows a
  persistent gap from `actual_gap_pct` over a larger sample.


## Gotchas worth knowing before touching this project again

- **PowerShell venv activation**: `.venv\Scripts\activate` (no extension)
  can resolve to the `.bat` version when run from PowerShell, which
  activates a child `cmd.exe` and doesn't persist to the parent shell.
  Use `..\.venv\Scripts\Activate.ps1` explicitly from PowerShell. Always
  worth a `python -c "import sys; print(sys.executable)"` check before a
  long-running command like `train_model.py`.
- **Webull Overnight (8pm-4am ET) vs Premarket (4am-9:30am ET) are
  different sessions.** The pipeline's timing assumptions are built
  around Premarket only. Confirm the clock before trusting a session.
- **Three different "did it work" columns, don't conflate them**:
  `actual_gap_pct` (model's own training-consistent metric),
  `actual_return_from_entry_pct` (real trading P&L question),
  `actual_best_return_pct` (best case that day). If entry-based returns
  start running meaningfully below the close-to-open numbers over time,
  that's a sign the model's optimization target and the real P&L target
  have drifted apart - worth revisiting the label definition itself at
  that point (not yet needed, small sample so far).
- **Comparing model retrains**: compare the printed metrics
  (train/test rows, Precision@5, days-with-a-hit), not the `.pkl` binary
  itself - there's nothing to usefully diff in a pickled model file.
- **`/api/dev/*` never logs. Only `/api/refresh` (and `run_daily.py`,
  which calls the same underlying `assemble_top_picks(..., log=True)`)
  writes to `picks_log.csv`.** Don't add logging to a dev/test endpoint.
- The `datetime.utcnow()` deprecation warning in `fill_actual_gap.py`'s
  console output is harmless and was left alone on purpose - fixing it
  changes the timestamp format, and real rows already exist in the old
  format; format churn wasn't worth it mid-testing. Fine to clean up
  later in a dedicated pass.