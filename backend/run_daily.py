#!/usr/bin/env python3
"""Runs the full top5predictdaily pipeline once, prints the top picks, and
logs them to data/picks_log.csv.

This is "next step 2" from the handoff notes: a scheduled script, not a
running server. It imports the pipeline directly - app.py's FastAPI server
does NOT need to be running for this to work.

SCHEDULING (run this before 9:30am ET so you have time to read it over
coffee and place any paper trades manually):

  cron (Linux/most servers) - crontab -e, then e.g. for 8:45am ET every
  weekday, using the TZ prefix so this doesn't drift with server-local time
  or DST:

      TZ=America/New_York
      45 8 * * 1-5 /usr/bin/python3 /path/to/backend/run_daily.py >> /path/to/logs/run_daily.log 2>&1

  macOS (launchd) - a ~/Library/LaunchAgents/*.plist with a
  <key>StartCalendarInterval</key> block set to hour 8 / minute 45, and
  <key>StandardOutPath</key> pointed at a log file. launchd uses local time,
  so this only needs adjusting if your Mac's system timezone isn't ET.

  Windows - Task Scheduler, a daily trigger at 8:45 AM, action
  `python.exe path\\to\\backend\\run_daily.py`. Task Scheduler triggers use
  the machine's local timezone, same caveat as above.

Exit code is 0 on success (including "ran fine, nothing cleared the bar
today") and 1 on any pipeline failure, so cron/launchd/Task Scheduler can
alert you (e.g. via its own failure-email/notification setting) on 502/503
class problems without you needing to watch it manually.
"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from scraper import refresh_data
from screener import MAX_PRICE
from predictor import ModelUnavailableError
from pipeline import assemble_top_picks, PICKS_LOG_FILE


def main(top_n: int = 5) -> int:
    print(f"[{datetime.now().isoformat()}] Running top5predictdaily refresh...")

    try:
        data = refresh_data(max_price=MAX_PRICE)
    except Exception as exc:
        print(f"FAILED: live data pull failed - {exc}", file=sys.stderr)
        return 1

    try:
        result = assemble_top_picks(data, top_n=top_n, log=True)
    except ModelUnavailableError as exc:
        print(f"FAILED: no usable model - {exc}", file=sys.stderr)
        print("Run train_model.py first.", file=sys.stderr)
        return 1

    picks = result["top_picks"]
    print(f"\n{result['candidates_scanned']} candidates scanned "
          f"({result['market_session']} session).\n")

    if not picks:
        print("No candidates cleared the filter today - nothing logged as a pick.")
        return 0

    print(f"Top {len(picks)} picks:\n")
    for i, pick in enumerate(picks, 1):
        print(f"{i}. {pick['ticker']}  ${pick['price']}  "
              f"confidence={pick['confidence_pct']}%  "
              f"expected_move={pick['expected_move_pct']}%")
        print(f"   {pick['explanation']}\n")

    print(f"Logged to {PICKS_LOG_FILE}. "
          "Run fill_actual_gap.py after the market closes to score these picks, "
          "and merge_feedback.py to fold resolved picks into the next training run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
