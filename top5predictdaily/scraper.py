"""Data ingestion layer.

Changes from the original starter:
- Works over a dynamic list of tickers (from screener.get_universe()) instead
  of a hardcoded 4-stock basket.
- Prefers premarket price/change fields when available (relevant since the
  target is "gap at the 9:30am open", not intraday movement).
- Pulls real headlines via yfinance's Ticker.news and scores them with VADER
  (free, no API key) instead of the old placeholder-string sentiment stub.

IMPORTANT: this module was written without network access to Yahoo Finance
(the sandbox this was built in only allows a fixed set of package-registry
domains). The `.info` field names and the news article schema below are
based on the yfinance source and prior known behavior, not a live test run -
verify field names against a real response in your own environment before
trusting this for money. Wrap anything flaky in the try/except patterns
already used here and widen them if a field is missing/renamed.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from screener import get_universe, MAX_PRICE

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_market_data.json"

_analyzer = SentimentIntensityAnalyzer()


def load_sample_data() -> Dict[str, Any]:
    with open(DATA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def save_data(data: Dict[str, Any]) -> None:
    with open(DATA_PATH, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def _score_sentiment(headlines: List[str]) -> float:
    """VADER compound score in [-1, 1], remapped to [0, 1] to match the
    existing schema (0.5 = neutral) used by feature_engineering/predictor."""
    if not headlines:
        return 0.5
    scores = [_analyzer.polarity_scores(h)["compound"] for h in headlines]
    avg_compound = sum(scores) / len(scores)
    return round((avg_compound + 1) / 2, 3)


def _fetch_news_for_ticker(ticker_obj: Any, max_articles: int = 5) -> Dict[str, Any]:
    """Pull recent headlines for a ticker. Handles both known yfinance news
    article shapes (flat 'title' key, or nested under 'content') since the
    exact schema could not be verified live from this environment."""
    headlines: List[str] = []
    try:
        articles = ticker_obj.news or []
        for article in articles[:max_articles]:
            title = article.get("title")
            if not title and isinstance(article.get("content"), dict):
                title = article["content"].get("title")
            if title:
                headlines.append(title)
    except Exception:
        pass

    return {
        "headlines": headlines,
        "news_count": len(headlines),
        "news_sentiment": _score_sentiment(headlines),
    }


def _build_stock_record(ticker: str, info: Dict[str, Any], news: Dict[str, Any]) -> Dict[str, Any]:
    prev_close = float(info.get("previousClose") or 1.0)

    pre_market_price = info.get("preMarketPrice")
    pre_market_change_pct = info.get("preMarketChangePercent")

    if pre_market_price:
        price = float(pre_market_price)
        market_session = "premarket"
        change_pct = (
            float(pre_market_change_pct) * 100.0
            if pre_market_change_pct is not None
            else ((price - prev_close) / prev_close) * 100.0 if prev_close else 0.0
        )
    else:
        price = float(info.get("currentPrice") or info.get("regularMarketPrice") or prev_close)
        market_session = "regular"
        change_pct = ((price - prev_close) / prev_close) * 100.0 if prev_close else 0.0

    volume = int(info.get("volume") or info.get("regularMarketVolume") or 0)
    avg_volume = int(info.get("averageVolume") or max(volume, 1))
    relative_volume = volume / avg_volume if avg_volume else 1.0
    volatility = abs(change_pct) * 1.2 + min(relative_volume, 4.0)

    return {
        "ticker": ticker,
        "price": round(price, 2),
        "change_pct": round(change_pct, 2),
        "volume": volume,
        "avg_volume": avg_volume,
        "relative_volume": round(relative_volume, 3),
        "volatility": round(volatility, 2),
        "news_sentiment": news["news_sentiment"],
        "news_count": news["news_count"],
        "headline": news["headlines"][0] if news["headlines"] else f"No recent headline for {ticker}.",
        "market_session": market_session,
    }


def try_live_pull(max_price: float = MAX_PRICE) -> Dict[str, Any]:
    try:
        import yfinance as yf
    except Exception as exc:
        raise RuntimeError("yfinance is not installed.") from exc

    tickers = get_universe(max_price=max_price)

    stock_rows: List[Dict[str, Any]] = []
    for ticker in tickers:
        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info
            price = float(info.get("preMarketPrice") or info.get("currentPrice") or info.get("regularMarketPrice") or 0.0)
            if price <= 0 or price > max_price:
                continue  # enforce the price cap even if the screener missed it
            news = _fetch_news_for_ticker(ticker_obj)
            stock_rows.append(_build_stock_record(ticker, info, news))
        except Exception:
            continue

    if not stock_rows:
        raise RuntimeError("Could not fetch live market data for any candidate.")

    try:
        qqq_info = yf.Ticker("QQQ").info
        qqq_prev = float(qqq_info.get("previousClose") or 1.0)
        qqq_price = float(qqq_info.get("preMarketPrice") or qqq_info.get("regularMarketPrice") or qqq_prev)
        qqq_change_pct = ((qqq_price - qqq_prev) / qqq_prev) * 100.0 if qqq_prev else 0.0
    except Exception:
        qqq_change_pct = 0.0

    live_data = {
        "last_updated": datetime.utcnow().isoformat(),
        "market_session": stock_rows[0]["market_session"] if stock_rows else "unknown",
        "stocks": stock_rows,
        "macro": {"qqq_change_pct": round(qqq_change_pct, 3)},
    }
    return live_data


def refresh_data(max_price: float = MAX_PRICE) -> Dict[str, Any]:
    try:
        fresh = try_live_pull(max_price=max_price)
        save_data(fresh)
        return fresh
    except Exception:
        return load_sample_data()
