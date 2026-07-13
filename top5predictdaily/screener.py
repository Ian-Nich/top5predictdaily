"""Universe discovery layer.

Goal: produce a candidate list of tickers to score every morning, instead of
a hardcoded 4-ticker basket. Tries a live Yahoo Finance screen first
(price < MAX_PRICE, already showing positive momentum), and falls back to a
static watchlist file if the live screen fails, returns too few names, or
yfinance's screener changes shape on us.

NOTE: this module makes network calls to Yahoo Finance via yfinance. Those
calls could not be tested from the sandbox this was written in (no network
access to Yahoo from that environment) - run it locally / in your own infra
first and sanity check the output shape before wiring it into app.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

MAX_PRICE = 20.0
MIN_UNIVERSE_SIZE = 5  # if live screen returns fewer than this, fall back

WATCHLIST_PATH = Path(__file__).resolve().parent.parent / "data" / "watchlist.txt"


def _load_watchlist() -> List[str]:
    """Static fallback universe. Edit data/watchlist.txt to curate this."""
    if not WATCHLIST_PATH.exists():
        # bare-minimum fallback so the pipeline never crashes with an empty list
        return ["QBTS", "QUBT", "IONQ", "RGTI"]
    tickers = []
    with open(WATCHLIST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().upper()
            if line and not line.startswith("#"):
                tickers.append(line)
    return tickers


def _try_live_screen(max_price: float = MAX_PRICE, limit: int = 60) -> List[str]:
    """Ask Yahoo for stocks under max_price that are already showing gains.

    This does NOT guarantee premarket data - Yahoo's public screener reflects
    whatever session is currently active/most recent. Treat this as a way to
    narrow a large market down to a manageable candidate list, not as a
    premarket-specific feed.
    """
    import yfinance as yf

    query = yf.EquityQuery(
        "and",
        [
            yf.EquityQuery("is-in", ["exchange", "NMS", "NYQ", "NGM", "NCM"]),
            yf.EquityQuery("lt", ["intradayprice", max_price]),
            yf.EquityQuery("gt", ["percentchange", 2]),
        ],
    )

    result = yf.screen(query, sortField="percentchange", sortAsc=False, size=limit)
    quotes = result.get("quotes", []) if isinstance(result, dict) else []

    tickers = [q["symbol"] for q in quotes if q.get("symbol")]
    return tickers


def get_universe(max_price: float = MAX_PRICE) -> List[str]:
    """Return the list of tickers to run features/prediction on this morning."""
    try:
        live = _try_live_screen(max_price=max_price)
        if len(live) >= MIN_UNIVERSE_SIZE:
            return live
    except Exception:
        pass

    return _load_watchlist()
