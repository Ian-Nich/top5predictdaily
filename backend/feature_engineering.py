"""Feature engineering layer.

Changed from basket-averaging (one feature set for all 4 tickers) to
per-ticker feature vectors, since the goal is now "rank N candidates and
return the top 5" rather than "call direction for one fixed basket."

Universe-level context (how many candidates are green, avg move size, QQQ
change) is computed once and attached to every ticker's row, since "the
whole market is ripping" is still a useful signal for an individual stock.
"""

from __future__ import annotations

from typing import Any, Dict, List


def _universe_context(stocks: List[Dict[str, Any]], macro: Dict[str, Any]) -> Dict[str, Any]:
    if not stocks:
        return {
            "universe_avg_change_pct": 0.0,
            "universe_positive_ratio": 0.0,
            "qqq_change_pct": round(float(macro.get("qqq_change_pct", 0.0)), 3),
        }

    avg_change = sum(s["change_pct"] for s in stocks) / len(stocks)
    positive_ratio = sum(1 for s in stocks if s["change_pct"] > 0) / len(stocks)

    return {
        "universe_avg_change_pct": round(avg_change, 3),
        "universe_positive_ratio": round(positive_ratio, 3),
        "qqq_change_pct": round(float(macro.get("qqq_change_pct", 0.0)), 3),
    }


def build_ticker_features(stock: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Feature vector for a single ticker. Keys here must match, in order,
    what predictor.features_to_vector() expects."""
    change_pct = stock.get("change_pct", 0.0)
    relative_volume = stock.get("relative_volume", stock.get("volume", 0) / max(stock.get("avg_volume", 1), 1))
    volatility = stock.get("volatility", 0.0)
    news_sentiment = stock.get("news_sentiment", 0.5)
    volume_spike = 1.0 if stock.get("volume", 0) > stock.get("avg_volume", 1) * 1.5 else 0.0
    rel_strength_vs_universe = change_pct - context["universe_avg_change_pct"]

    return {
        "ticker": stock["ticker"],
        "price": stock.get("price", 0.0),
        "momentum": round(change_pct, 3),
        "relative_volume": round(relative_volume, 3),
        "volume_spike": volume_spike,
        "volatility": round(volatility, 3),
        "news_sentiment": round(news_sentiment, 3),
        "news_count": stock.get("news_count", 0),
        "rel_strength_vs_universe": round(rel_strength_vs_universe, 3),
        "universe_positive_ratio": context["universe_positive_ratio"],
        "qqq_change_pct": context["qqq_change_pct"],
        "market_session": stock.get("market_session", "unknown"),
    }


def build_features(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Returns a list of per-ticker feature dicts (was: one basket-wide dict).

    NOTE: this is a breaking change in shape for anything downstream that
    expected build_features() to return a single dict - app.py and
    predictor.py in this project have been updated to match, but if you have
    other callers, update them too.
    """
    stocks: List[Dict[str, Any]] = data.get("stocks", [])
    macro: Dict[str, Any] = data.get("macro", {})

    context = _universe_context(stocks, macro)
    return [build_ticker_features(stock, context) for stock in stocks]
