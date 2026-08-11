"""Shared pipeline glue: feature building -> ranking -> explaining -> logging.

Pulled out of app.py so the same "run the whole thing and log it" logic can
be called from the FastAPI route (/api/refresh) AND from run_daily.py's
cron/scheduled-task entry point, without duplicating it or going through
HTTP just to talk to itself.

IMPORTANT SCHEMA NOTE: log_picks() writes the full feature vector
(predictor.FEATURE_ORDER) into picks_log.csv alongside each pick, not just
the prediction outputs. That's deliberate: to fold real outcomes back into
retraining later (see merge_feedback.py), you need the exact inputs the
model saw at prediction time - momentum, volatility, news_sentiment, etc.
at that moment can't be reconstructed after the fact (today's headlines are
gone tomorrow, and the premarket volume snapshot is gone once the market
opens). If you only log the derived prediction, picks_log.csv can tell you
"was the model right" but can never become new training data.

THREE outcome columns get filled in later by fill_actual_gap.py, and they
answer three DIFFERENT questions - don't conflate them:
  - actual_gap_pct: today's actual open vs YESTERDAY's close. This is the
    exact quantity train_model.py's label is defined on - keep this one for
    retraining/model-calibration purposes, not for judging your own P&L.
  - actual_return_from_entry_pct: today's actual open vs the "price" you
    were actually shown/would have bought at (the premarket price logged
    at pick time). THIS is the number that answers "did I make 3%+ from
    where I'd have bought" - it can differ a lot from actual_gap_pct if the
    stock had already moved a lot premarket before the pick was logged.
  - actual_best_return_pct: that day's high vs your entry price - the best
    case if you'd sold at the peak rather than exactly at the open.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import csv
from datetime import datetime, timezone

from feature_engineering import build_features
from predictor import rank_top_picks, FEATURE_ORDER
from screener import MAX_PRICE
from explainer import build_explanation

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PICKS_LOG_FILE = DATA_DIR / "picks_log.csv"
DATA_DIR.mkdir(parents=True, exist_ok=True)

PICKS_LOG_HEADER = (
    ["timestamp", "ticker", "price", "predicted_direction",
     "confidence_pct", "expected_move_pct", "model_type"]
    + FEATURE_ORDER
    + ["actual_gap_pct", "actual_return_from_entry_pct", "actual_best_return_pct"]
)


def log_picks(picks: List[Dict[str, Any]], feature_by_ticker: Dict[str, Dict[str, Any]]) -> None:
    """Append each top pick to picks_log.csv, including the feature vector
    that was fed to the model - so this row can later become a real
    training example once actual_gap_pct is filled in (see
    fill_actual_gap.py and merge_feedback.py)."""
    file_exists = PICKS_LOG_FILE.exists()
    with open(PICKS_LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(PICKS_LOG_HEADER)
        for pick in picks:
            features = feature_by_ticker.get(pick["ticker"], {})
            writer.writerow(
                [datetime.now(timezone.utc).isoformat(), pick["ticker"], pick["price"],
                 pick["predicted_direction"], pick["confidence_pct"],
                 pick["expected_move_pct"], pick["model_type"]]
                + [features.get(key, "") for key in FEATURE_ORDER]
                + ["", "", ""]  # actual_gap_pct, actual_return_from_entry_pct,
                                 # actual_best_return_pct - filled in later, never at pick time
            )


def assemble_top_picks(data: Dict[str, Any], top_n: int = 5, log: bool = False) -> Dict[str, Any]:
    """Runs feature building -> ranking -> explaining on `data`, and only
    logs to picks_log.csv when log=True. Dev/test endpoints working off
    sample_market_data.json must call this with log=False (the default) -
    otherwise fake test runs pollute the real feedback log that
    fill_actual_gap.py / merge_feedback.py later treat as ground truth.

    Raises ModelUnavailableError (propagated from rank_top_picks/score_ticker)
    if there's no trained model - same no-fallback contract as before.
    """
    feature_list = build_features(data)
    picks = rank_top_picks(feature_list, top_n=top_n, max_price=MAX_PRICE)

    feature_by_ticker = {f["ticker"]: f for f in feature_list}
    for pick in picks:
        pick["explanation"] = build_explanation(feature_by_ticker[pick["ticker"]], pick)

    if log:
        log_picks(picks, feature_by_ticker)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "market_session": data.get("market_session", "unknown"),
        "candidates_scanned": len(feature_list),
        "top_picks": picks,
    }