"""Prediction logic with optional ML model support.

Changed from "one score for the whole basket" to "score every candidate
ticker, then rank." feature_to_vector's key order is the CONTRACT between
this file and train_model.py - if you add/remove/reorder features here,
retrain the model (old model.pkl files trained on the previous 11-feature
basket schema will NOT work with this vector shape and should be deleted /
retrained).
"""

from __future__ import annotations

from typing import Any, Dict, List
import os
import pickle


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
model = None

if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    except Exception:
        model = None


FEATURE_ORDER = [
    "momentum",
    "relative_volume",
    "volume_spike",
    "volatility",
    "news_sentiment",
    "news_count",
    "rel_strength_vs_universe",
    "universe_positive_ratio",
    "qqq_change_pct",
]


def features_to_vector(features: Dict[str, Any]) -> List[float]:
    return [float(features.get(key, 0.0)) for key in FEATURE_ORDER]


def score_ticker(features: Dict[str, Any]) -> Dict[str, Any]:
    """Score one ticker's feature dict. Returns direction/confidence/expected
    move, same shape whether it came from the ML model or the rule fallback."""

    if model is not None:
        try:
            vector = features_to_vector(features)
            prob = model.predict_proba([vector])[0][1]

            predicted_direction = "UP" if prob > 0.5 else "DOWN"
            confidence = clamp(prob * 100, 52, 97)
            expected_move_pct = clamp(abs(prob - 0.5) * 20, 0.2, 10.0)

            return {
                "ticker": features["ticker"],
                "price": features.get("price"),
                "predicted_direction": predicted_direction,
                "confidence_pct": round(confidence, 1),
                "expected_move_pct": round(expected_move_pct, 2),
                "score": round(float(prob), 3),
                "model_type": "ML",
            }
        except Exception:
            pass  # fall through to rule-based

    # ORIGINAL RULE-BASED FALLBACK, adapted to per-ticker features
    score = 0.0
    score += features.get("momentum", 0.0) * 0.35
    score += (features.get("relative_volume", 1.0) - 1.0) * 1.8
    score += (features.get("news_sentiment", 0.5) - 0.5) * 7.0
    score += features.get("qqq_change_pct", 0.0) * 0.6
    score += features.get("rel_strength_vs_universe", 0.0) * 0.5
    score += features.get("volume_spike", 0.0) * 0.6

    if features.get("market_session") == "premarket":
        score += 0.4
    if features.get("volatility", 0.0) > 7.0:
        score -= 0.5

    predicted_direction = "UP" if score >= 0 else "DOWN"
    expected_move_pct = clamp(abs(score) * 0.9, 0.2, 10.0)
    confidence = clamp(50 + abs(score) * 8, 51, 96)

    return {
        "ticker": features["ticker"],
        "price": features.get("price"),
        "predicted_direction": predicted_direction,
        "confidence_pct": round(confidence, 1),
        "expected_move_pct": round(expected_move_pct, 2),
        "score": round(score, 3),
        "model_type": "rule-based",
    }


def rank_top_picks(
    feature_list: List[Dict[str, Any]],
    top_n: int = 5,
    max_price: float = 20.0,
    min_expected_move_pct: float = 5.0,
) -> List[Dict[str, Any]]:
    """Score every candidate, keep UP calls under max_price that clear the
    move threshold, and return the top_n by score."""
    scored = []
    for features in feature_list:
        if features.get("price") is not None and features["price"] > max_price:
            continue
        result = score_ticker(features)
        scored.append(result)

    picks = [
        r for r in scored
        if r["predicted_direction"] == "UP" and r["expected_move_pct"] >= min_expected_move_pct
    ]
    picks.sort(key=lambda r: (r["score"], r["expected_move_pct"]), reverse=True)
    return picks[:top_n]
