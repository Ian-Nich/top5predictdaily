"""Prediction logic. ML model REQUIRED - there is no rule-based fallback.

If model.pkl is missing, fails to load, or fails to predict, this raises
ModelUnavailableError rather than silently substituting a heuristic score.
app.py catches that and returns a clean failure response - it does not
degrade to a different (less accurate) computation and pass it off as normal
output. Train a model first: python train_model.py
"""

from __future__ import annotations

from typing import Any, Dict, List
import os
import pickle


class ModelUnavailableError(RuntimeError):
    """No usable trained model. By design there is no fallback for this -
    fix the underlying problem (train/retrain, or check the file), don't
    catch this and substitute a different scoring method."""


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
model = None
_load_error: Exception | None = None

if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    except Exception as exc:  # noqa: BLE001 - captured for the raised error's __cause__
        _load_error = exc
else:
    _load_error = FileNotFoundError(f"{MODEL_PATH} does not exist")


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
    """Score one ticker's feature dict using the trained model.

    Raises ModelUnavailableError if there's no model to use. Any error from
    the model itself (bad vector shape, corrupted model, etc.) propagates
    as-is - it is NOT caught and papered over.
    """
    if model is None:
        raise ModelUnavailableError(
            "No trained model available (model.pkl missing or failed to load). "
            "Run train_model.py - there is no rule-based fallback."
        ) from _load_error

    vector = features_to_vector(features)
    prob = float(model.predict_proba([vector])[0][1])  # cast immediately - numpy.float32 isn't JSON-serializable

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


def rank_top_picks(
    feature_list: List[Dict[str, Any]],
    top_n: int = 5,
    max_price: float = 20.0,
    min_expected_move_pct: float = 3.0,
) -> List[Dict[str, Any]]:
    """Score every candidate, keep UP calls under max_price that clear the
    move threshold, and return the top_n by score.

    Raises ModelUnavailableError (propagated from score_ticker) if there's no
    model - the whole batch fails rather than returning partial results.
    """
    scored = []
    for features in feature_list:
        if features.get("price") is not None and features["price"] > max_price:
            continue
        scored.append(score_ticker(features))

    picks = [
        r for r in scored
        if r["predicted_direction"] == "UP" and r["expected_move_pct"] >= min_expected_move_pct
    ]
    picks.sort(key=lambda r: (r["score"], r["expected_move_pct"]), reverse=True)
    return picks[:top_n]
