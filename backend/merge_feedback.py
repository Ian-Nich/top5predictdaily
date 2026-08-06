"""Turns resolved rows in data/picks_log.csv into real labeled training
examples, so live outcomes can feed back into retraining.

This is "next step 3" (part two) from the handoff notes - the actual
mechanism for "tie in whatever results I get to make this more accurate."

IMPORTANT - why this only works going forward, not retroactively:
A row can become a training example only if it has BOTH actual_gap_pct
(filled by fill_actual_gap.py) AND its original feature vector (momentum,
relative_volume, volatility, news_sentiment, etc. - logged by
pipeline.log_picks() at the moment the pick was made). The feature vector
can't be recomputed later: today's premarket headlines and volume snapshot
are gone once the day passes. Rows logged before pipeline.py started
persisting FEATURE_ORDER alongside each pick (i.e. anything logged by the
OLD app.py, before this refactor) will have actual_gap_pct but no usable
feature vector, and are skipped here - they're still fine for tracking
raw precision by hand, just not usable as new training rows.

Label definition matches train_model.py: 1 if actual_gap_pct >=
GAP_THRESHOLD_PCT, else 0. If you ever change train_model.GAP_THRESHOLD_PCT,
this picks it up automatically since it imports the constant rather than
hardcoding 3.0 - same "keep the label and the live filter in sync" lesson
from the handoff notes, just enforced by import instead of by memory.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from predictor import FEATURE_ORDER
from pipeline import PICKS_LOG_FILE
from train_model import GAP_THRESHOLD_PCT


def load_resolved_feedback() -> pd.DataFrame:
    """Returns a DataFrame with FEATURE_ORDER columns + a `label` column,
    built only from picks_log.csv rows that have both a filled
    actual_gap_pct and a complete original feature vector."""
    if not PICKS_LOG_FILE.exists():
        return pd.DataFrame(columns=FEATURE_ORDER + ["label"])

    usable, skipped_no_features, skipped_unresolved = [], 0, 0
    with open(PICKS_LOG_FILE, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get("actual_gap_pct"):
                skipped_unresolved += 1
                continue
            if any(row.get(col, "") == "" for col in FEATURE_ORDER):
                skipped_no_features += 1
                continue
            record = {col: float(row[col]) for col in FEATURE_ORDER}
            record["label"] = int(float(row["actual_gap_pct"]) >= GAP_THRESHOLD_PCT)
            usable.append(record)

    print(f"Feedback rows: {len(usable)} usable, "
          f"{skipped_no_features} skipped (pre-refactor rows with no logged "
          f"feature vector), {skipped_unresolved} skipped (actual_gap_pct not "
          f"filled in yet - run fill_actual_gap.py first)")

    return pd.DataFrame(usable, columns=FEATURE_ORDER + ["label"])


if __name__ == "__main__":
    df = load_resolved_feedback()
    if df.empty:
        print("No usable feedback rows yet.")
    else:
        print(df["label"].value_counts())
