"""Builds a labeled historical dataset and trains model.pkl.

Label definition: for each (ticker, day), label=1 if the NEXT trading day's
Open was >= 5% above that day's Close, else 0. This directly matches what
you're trying to predict live (a gap up at the 9:30am open).

Honesty about what this can and can't do:
- Free daily-bar history from yfinance is what this trains on: momentum,
  relative volume, volatility, cross-sectional strength vs the rest of the
  universe, and QQQ as a macro proxy.
- There is NO historical news feature here. Free news sources don't give
  clean per-date archives going back a year across many tickers, so
  news_sentiment/news_count are filled with neutral placeholders (0.5 / 0)
  for every training row. In live use (scraper.py), those same feature slots
  get real values from today's headlines - so the live feature vector will
  have information the model never saw in training. That's a real
  train/serve mismatch to be aware of, not a bug to silently ignore.
- A 5%+ overnight gap is a rare event for most tickers, so expect a heavily
  imbalanced dataset (few positives). scale_pos_weight is used to compensate,
  but treat any accuracy/precision number from a first pass as a starting
  point, not a green light to size a real position on.
- Split is by DATE (not random row split), so the model is validated on time
  periods it hasn't seen - a random row split would leak information since
  many tickers share the same calendar day's market-wide movement.

This could not be run against live Yahoo Finance from the sandbox this was
written in (no network access to Yahoo from that environment) - run it
yourself and inspect the printed metrics before trusting model.pkl.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predictor import FEATURE_ORDER  # keep vector order in sync with predictor.py
from screener import _load_watchlist

MODEL_OUT = Path(__file__).resolve().parent / "model.pkl"
LOOKBACK = "1y"
ROLL_WINDOW = 20
GAP_THRESHOLD_PCT = 5.0


def _per_ticker_frame(ticker: str) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(
        ticker, period=LOOKBACK, interval="1d", auto_adjust=False,
        progress=False, multi_level_index=False,
    )
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["ticker"] = ticker
    df["change_pct"] = df["Close"].pct_change() * 100.0
    df["avg_volume_20d"] = df["Volume"].rolling(ROLL_WINDOW).mean()
    df["relative_volume"] = df["Volume"] / df["avg_volume_20d"]
    df["volume_spike"] = (df["Volume"] > df["avg_volume_20d"] * 1.5).astype(float)
    df["volatility"] = df["change_pct"].abs() * 1.2 + df["relative_volume"].clip(upper=4.0)

    # label: next day's open vs today's close
    next_open = df["Open"].shift(-1)
    df["next_gap_pct"] = (next_open - df["Close"]) / df["Close"] * 100.0
    df["label"] = (df["next_gap_pct"] >= GAP_THRESHOLD_PCT).astype(int)

    df["news_sentiment"] = 0.5
    df["news_count"] = 0.0

    return df.reset_index()[
        ["Date", "ticker", "change_pct", "relative_volume", "volume_spike",
         "volatility", "news_sentiment", "news_count", "label"]
    ]


def build_dataset(tickers: list[str]) -> pd.DataFrame:
    frames = [f for t in tickers for f in [_per_ticker_frame(t)] if not f.empty]
    if not frames:
        raise RuntimeError("No data pulled for any ticker - check tickers/network.")

    all_df = pd.concat(frames, ignore_index=True)

    # cross-sectional context per date, computed across whatever tickers
    # actually have data for that date
    daily = all_df.groupby("Date")["change_pct"].agg(["mean"]).rename(
        columns={"mean": "universe_avg_change_pct"}
    )
    daily["universe_positive_ratio"] = all_df.groupby("Date")["change_pct"].apply(
        lambda s: (s > 0).mean()
    )
    all_df = all_df.merge(daily, on="Date", how="left")
    all_df["rel_strength_vs_universe"] = all_df["change_pct"] - all_df["universe_avg_change_pct"]

    # QQQ as macro proxy, aligned by date
    import yfinance as yf
    qqq = yf.download(
        "QQQ", period=LOOKBACK, interval="1d", auto_adjust=False,
        progress=False, multi_level_index=False,
    )
    qqq["qqq_change_pct"] = qqq["Close"].pct_change() * 100.0
    qqq_series = qqq.reset_index()[["Date", "qqq_change_pct"]]
    all_df = all_df.merge(qqq_series, on="Date", how="left")
    all_df["qqq_change_pct"] = all_df["qqq_change_pct"].fillna(0.0)

    all_df = all_df.rename(columns={"change_pct": "momentum"})
    all_df = all_df.dropna(subset=["momentum", "relative_volume", "volatility", "label"])
    return all_df


def train(tickers: list[str] | None = None, test_frac: float = 0.2):
    from xgboost import XGBClassifier
    from sklearn.metrics import classification_report

    tickers = tickers or _load_watchlist()
    df = build_dataset(tickers)

    df = df.sort_values("Date")
    split_idx = int(len(df) * (1 - test_frac))
    split_date = df.iloc[split_idx]["Date"]

    train_df = df[df["Date"] < split_date]
    test_df = df[df["Date"] >= split_date]

    X_train = train_df[FEATURE_ORDER].values
    y_train = train_df["label"].values
    X_test = test_df[FEATURE_ORDER].values
    y_test = test_df["label"].values

    pos = max(y_train.sum(), 1)
    neg = max(len(y_train) - y_train.sum(), 1)
    scale_pos_weight = neg / pos

    clf = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
    )
    clf.fit(X_train, y_train)

    if len(test_df) > 0:
        preds = clf.predict(X_test)
        print(f"Train rows: {len(train_df)}  Test rows: {len(test_df)}  "
              f"Positive rate (train): {y_train.mean():.3f}  (test): {y_test.mean():.3f}")
        print(classification_report(y_test, preds, zero_division=0))
    else:
        print("Not enough data for a held-out test split - training set only.")

    with open(MODEL_OUT, "wb") as f:
        pickle.dump(clf, f)
    print(f"Saved model to {MODEL_OUT}")

    return clf


if __name__ == "__main__":
    train()
