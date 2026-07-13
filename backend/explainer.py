"""Explanation layer.

The original explainer.py source was never shared with me - only a compiled
.pyc for the OLD basket-wide schema. I recovered its logic by disassembling
that bytecode (dis.dis), confirmed it reproduces sensible output, and
adapted the same style (short reason fragments, joined into one sentence) to
the new per-ticker feature schema from feature_engineering.py.
"""

from __future__ import annotations

from typing import Any, Dict


def build_explanation(features: Dict[str, Any], prediction: Dict[str, Any]) -> str:
    reasons = []

    if features.get("momentum", 0) > 0:
        reasons.append("it's already showing positive momentum")
    else:
        reasons.append("momentum is currently negative")

    if features.get("relative_volume", 1) > 1.2:
        reasons.append("relative volume is elevated")

    if features.get("volume_spike", 0):
        reasons.append("volume is spiking well above average")

    sentiment = features.get("news_sentiment", 0.5)
    if sentiment > 0.55:
        reasons.append("news tone is positive")
    elif sentiment < 0.45:
        reasons.append("news tone is negative")

    rel_strength = features.get("rel_strength_vs_universe", 0)
    if rel_strength > 0:
        reasons.append("it's outperforming the rest of today's candidates")
    else:
        reasons.append("it's underperforming the rest of today's candidates")

    qqq = features.get("qqq_change_pct", 0)
    if qqq > 0:
        reasons.append("broader tech is helping")
    elif qqq < 0:
        reasons.append("broader tech is acting as a headwind")

    joined = ", ".join(reasons[:-1]) + (", and " + reasons[-1] if len(reasons) > 1 else reasons[0])

    return (
        f"Model expects {prediction['predicted_direction']} movement of "
        f"~{prediction['expected_move_pct']}% because {joined}."
    )
