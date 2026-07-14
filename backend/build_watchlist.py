"""Builds a broad BACKTEST/TRAINING universe by scanning the whole US-listed
common-stock market for names under a price cap, and writes the result to
data/watchlist.txt.

Different job than screener.py: screener.py does a LIVE daily premarket scan
for actually placing trades. This script is a one-off/periodic maintenance
step (run weekly/monthly, not every morning) to widen the OFFLINE universe
train_model.py learns and evaluates on - the bottleneck flagged after the
last training run (4 tickers -> ~26 positive examples wasn't enough).

Pipeline:
1. Pull the full list of NASDAQ + NYSE/AMEX/ARCA listed securities from
   Nasdaq Trader's public symbol directory (nasdaqtrader.com - static text
   files, no API key/auth needed).
2. Drop ETFs, test issues, and tickers with punctuation (warrants, units,
   preferred/when-issued share classes - these often don't map cleanly to a
   yfinance ticker anyway, and add noise rather than signal).
3. Batch-download ~1mo of price/volume via yfinance, in chunks (checking
   thousands of tickers one at a time would be far too slow and likely to
   get rate-limited).
4. Filter to price < MAX_PRICE AND avg volume >= MIN_AVG_VOLUME. The
   liquidity floor matters for two reasons: an illiquid name is (a) mostly
   noise for a model to learn from, and (b) not something you could actually
   get a real fill on anyway, so training on it doesn't help you.
5. Write survivors to data/watchlist.txt with a header comment, and print
   funnel counts at every stage so you can see where names got cut.

NOT TESTED LIVE: this sandbox has no network access to nasdaqtrader.com or
Yahoo Finance. The symbol-file parsing was tested against a synthetic file
matching the documented column layout, and the chunked yfinance handling
against a mocked MultiIndex DataFrame (verified against the installed
yfinance version's actual source, not guessed) - but the real symbol-file
column names could have drifted since this was written. Run it yourself and
read the printed counts/errors before trusting watchlist.txt; if the column
check below raises, the printed message tells you what columns it actually
found so you can fix the mapping fast.
"""

from __future__ import annotations

import time
from io import StringIO
from pathlib import Path
from typing import List

import pandas as pd
import requests

WATCHLIST_PATH = Path(__file__).resolve().parent.parent / "data" / "watchlist.txt"

NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

MAX_PRICE = 20.0
MIN_AVG_VOLUME = 300_000  # liquidity floor - tune this; higher = fewer, more tradeable names
CHUNK_SIZE = 200
CHUNK_DELAY_SECONDS = 1.0  # be polite to Yahoo between chunks


def _fetch_symbol_file(url: str) -> pd.DataFrame:
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    lines = resp.text.splitlines()
    # last line is a footer like "File Creation Time: ..." - drop it if present
    if lines and lines[-1].lower().startswith("file creation time"):
        lines = lines[:-1]
    return pd.read_csv(StringIO("\n".join(lines)), sep="|")


def _require_columns(df: pd.DataFrame, required: List[str], source: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"{source} is missing expected column(s) {missing}. "
            f"Columns actually found: {list(df.columns)}. "
            "Nasdaq's symbol directory format may have changed - update the "
            "column names in this function to match."
        )


def _is_plain_common_stock(symbol: str) -> bool:
    """Drops warrants/units/preferred/when-issued classes, which tend to use
    punctuation in the symbol and often don't map cleanly to a single
    yfinance ticker anyway."""
    if not isinstance(symbol, str) or not symbol:
        return False
    return symbol.isalpha() and symbol.isupper() and len(symbol) <= 5


def get_full_universe() -> List[str]:
    """Full list of plain-common-stock tickers across NASDAQ/NYSE/AMEX/ARCA,
    with ETFs and test issues already dropped."""
    nasdaq = _fetch_symbol_file(NASDAQ_LISTED_URL)
    other = _fetch_symbol_file(OTHER_LISTED_URL)

    _require_columns(nasdaq, ["Symbol", "Test Issue", "ETF"], "nasdaqlisted.txt")
    _require_columns(other, ["ACT Symbol", "Test Issue", "ETF"], "otherlisted.txt")

    print(f"nasdaqlisted.txt: {len(nasdaq)} rows, otherlisted.txt: {len(other)} rows")

    nasdaq = nasdaq[(nasdaq["Test Issue"] == "N") & (nasdaq["ETF"] == "N")]
    other = other[(other["Test Issue"] == "N") & (other["ETF"] == "N")]

    all_symbols = sorted(
        set(nasdaq["Symbol"].dropna().astype(str)) | set(other["ACT Symbol"].dropna().astype(str))
    )
    print(f"After dropping ETFs/test issues: {len(all_symbols)} symbols")

    plain_symbols = [s for s in all_symbols if _is_plain_common_stock(s)]
    print(f"After dropping warrants/units/preferred/etc (punctuation in symbol): {len(plain_symbols)} symbols")

    return plain_symbols


def _chunked(items: List[str], size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def filter_by_price_and_liquidity(
    symbols: List[str],
    max_price: float = MAX_PRICE,
    min_avg_volume: float = MIN_AVG_VOLUME,
) -> List[str]:
    import yfinance as yf

    survivors: List[str] = []
    checked = 0
    failed_chunks = 0

    for chunk in _chunked(symbols, CHUNK_SIZE):
        try:
            data = yf.download(
                chunk, period="1mo", interval="1d", group_by="ticker",
                threads=True, progress=False, auto_adjust=False,
                multi_level_index=False,
            )
        except Exception as exc:
            failed_chunks += 1
            print(f"Chunk starting at {chunk[0]} failed entirely: {exc}")
            continue

        for ticker in chunk:
            try:
                df = data if len(chunk) == 1 else data[ticker]
                df = df.dropna(subset=["Close", "Volume"])
                if df.empty:
                    continue
                last_price = float(df["Close"].iloc[-1])
                avg_volume = float(df["Volume"].mean())
                if last_price < max_price and avg_volume >= min_avg_volume:
                    survivors.append(ticker)
            except Exception:
                continue  # this one ticker had no usable data - skip it, don't abort the chunk

        checked += len(chunk)
        print(f"Checked {checked}/{len(symbols)} - {len(survivors)} survivors so far")
        time.sleep(CHUNK_DELAY_SECONDS)

    if failed_chunks:
        print(f"WARNING: {failed_chunks} chunk(s) failed entirely and were skipped.")

    return survivors


def build_watchlist(max_price: float = MAX_PRICE, min_avg_volume: float = MIN_AVG_VOLUME) -> None:
    universe = get_full_universe()
    survivors = filter_by_price_and_liquidity(universe, max_price=max_price, min_avg_volume=min_avg_volume)

    print(f"\nFinal watchlist: {len(survivors)} tickers "
          f"(price < ${max_price}, avg 1mo volume >= {min_avg_volume:,.0f})")

    with open(WATCHLIST_PATH, "w", encoding="utf-8") as f:
        f.write(f"# Auto-generated by build_watchlist.py - {len(survivors)} tickers\n")
        f.write(f"# Filter: price < ${max_price}, avg 1mo volume >= {min_avg_volume:,.0f}\n")
        f.write(f"# Regenerate periodically (weekly/monthly) - prices and liquidity drift.\n")
        for ticker in sorted(survivors):
            f.write(ticker + "\n")

    print(f"Wrote {WATCHLIST_PATH}")


if __name__ == "__main__":
    build_watchlist()
