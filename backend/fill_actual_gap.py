#!/usr/bin/env python3
"""Fills in the outcome columns of data/picks_log.csv for picks whose
trading day has already happened, using yfinance daily history.

This is "next step 3" (part one) from the handoff notes.

Fills THREE different columns, because "was the model right" and "did I
make money" are different questions:

  - actual_gap_pct: pick date's Open vs the PRIOR trading day's Close.
    Kept consistent with train_model.py's label definition - this is what
    merge_feedback.py uses to build new training rows. Don't repurpose
    this one for judging your own trades; it doesn't know your entry price.

  - actual_return_from_entry_pct: pick date's Open vs the "price" column
    already logged for that pick (the premarket price at the moment the
    pick was made - i.e. roughly what you'd have actually paid). This is
    the number that answers "did I make 3%+ from where I'd have bought" -
    it can be very different from actual_gap_pct if the stock had already
    moved a lot premarket before the pick was logged.

  - actual_best_return_pct: pick date's High vs that same entry price -
    the best case if sold at the day's peak instead of exactly at the open.

Rows are only ever filled once - a row with all three already filled is
never touched again, so it's safe to run this daily (e.g. after market
close) via cron/launchd/Task Scheduler alongside run_daily.py.

CAVEAT: pick timestamps are logged via datetime.utcnow().isoformat() at
premarket pick time. The calendar date of that UTC timestamp matches the
trading day being predicted for in the overwhelming majority of cases
(premarket runs happen in the morning ET, comfortably inside the same UTC
calendar date). If you ever schedule run_daily.py very close to the
UTC/ET midnight boundary, double-check pick dates against actual run
times before trusting this blindly.
"""
from __future__ import annotations

import csv
import sys
from datetime import datetime, timedelta
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from pipeline import PICKS_LOG_FILE


def _load_rows() -> list[dict]:
    if not PICKS_LOG_FILE.exists():
        raise FileNotFoundError(
            f"{PICKS_LOG_FILE} doesn't exist yet - run app.py's /api/refresh "
            "or run_daily.py at least once first."
        )
    with open(PICKS_LOG_FILE, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_rows(rows: list[dict], fieldnames: list[str]) -> None:
    with open(PICKS_LOG_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _pick_date(row: dict):
    return datetime.fromisoformat(row["timestamp"]).date()


def fill_actual_gaps(min_days_old: int = 1) -> None:
    import yfinance as yf

    rows = _load_rows()
    if not rows:
        print("picks_log.csv is empty - nothing to do.")
        return

    fieldnames = list(rows[0].keys())
    today = datetime.utcnow().date()

    pending_by_ticker: dict[str, list[dict]] = {}
    already_filled = 0
    too_new = 0
    for row in rows:
        if row.get("actual_gap_pct") and row.get("actual_return_from_entry_pct"):
            already_filled += 1
            continue  # already filled - never overwrite
        pick_date = _pick_date(row)
        if (today - pick_date).days < min_days_old:
            too_new += 1
            continue  # market may not have opened/settled yet - try again tomorrow
        pending_by_ticker.setdefault(row["ticker"], []).append(row)

    if not pending_by_ticker:
        if too_new:
            print(f"Nothing to fill yet - {too_new} row(s) were logged too recently "
                  f"(need the trading day to be at least {min_days_old} day(s) old "
                  f"so the open/high are fully settled). Try again after that day closes.")
        elif already_filled:
            print("Nothing to fill - all resolvable rows already have outcomes filled in.")
        else:
            print("picks_log.csv has no rows to resolve.")
        return

    filled, skipped = 0, 0
    for ticker, ticker_rows in pending_by_ticker.items():
        dates = [_pick_date(r) for r in ticker_rows]
        start = min(dates) - timedelta(days=10)
        end = max(dates) + timedelta(days=3)
        try:
            hist = yf.download(
                ticker, start=start.isoformat(), end=end.isoformat(),
                interval="1d", auto_adjust=False, progress=False,
                multi_level_index=False,
            )
        except Exception as exc:
            print(f"  {ticker}: history fetch failed - {exc}, skipping its {len(ticker_rows)} row(s)")
            skipped += len(ticker_rows)
            continue

        if hist.empty:
            print(f"  {ticker}: no history returned, skipping its {len(ticker_rows)} row(s)")
            skipped += len(ticker_rows)
            continue

        hist = hist.sort_index()
        for row in ticker_rows:
            pick_date = _pick_date(row)
            matches = hist.index[hist.index.date == pick_date]
            if len(matches) == 0:
                skipped += 1
                continue
            idx = hist.index.get_loc(matches[0])
            if idx == 0:
                skipped += 1  # no prior trading day in this window
                continue
            open_price = float(hist.iloc[idx]["Open"])
            day_high = float(hist.iloc[idx]["High"])
            prev_close = float(hist.iloc[idx - 1]["Close"])
            entry_price = float(row["price"]) if row.get("price") else None

            if prev_close <= 0 or not entry_price or entry_price <= 0:
                skipped += 1
                continue

            row["actual_gap_pct"] = round((open_price - prev_close) / prev_close * 100.0, 2)
            row["actual_return_from_entry_pct"] = round((open_price - entry_price) / entry_price * 100.0, 2)
            row["actual_best_return_pct"] = round((day_high - entry_price) / entry_price * 100.0, 2)
            filled += 1

    _write_rows(rows, fieldnames)
    print(f"Filled {filled} row(s), skipped {skipped} (no matching trading-day data (yet)).")


if __name__ == "__main__":
    fill_actual_gaps()