from __future__ import annotations

from pathlib import Path
import sys
import csv
import os
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from scraper import load_sample_data, refresh_data
from feature_engineering import build_features
from predictor import score_ticker, rank_top_picks, ModelUnavailableError
from screener import MAX_PRICE
from explainer import build_explanation  # hard import - no fallback if this is missing


app = FastAPI(title="Quantum Stocks Predictor Starter API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PICKS_LOG_FILE = DATA_DIR / "picks_log.csv"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def log_picks(picks: list[dict]) -> None:
    """Append each top pick to a CSV so you can go back later and label
    whether it actually gapped up, for future retraining."""
    file_exists = PICKS_LOG_FILE.exists()
    with open(PICKS_LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "ticker", "price", "predicted_direction",
                "confidence_pct", "expected_move_pct", "model_type", "actual_gap_pct"
            ])
        for pick in picks:
            writer.writerow([
                datetime.utcnow().isoformat(),
                pick["ticker"], pick["price"], pick["predicted_direction"],
                pick["confidence_pct"], pick["expected_move_pct"], pick["model_type"], ""
            ])


def assemble_top_picks(data: dict, top_n: int = 5):
    feature_list = build_features(data)
    picks = rank_top_picks(feature_list, top_n=top_n, max_price=MAX_PRICE)

    feature_by_ticker = {f["ticker"]: f for f in feature_list}
    for pick in picks:
        pick["explanation"] = build_explanation(feature_by_ticker[pick["ticker"]], pick)

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "market_session": data.get("market_session", "unknown"),
        "candidates_scanned": len(feature_list),
        "top_picks": picks,
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "message": "Quantum predictor backend is running."}


@app.get("/api/data")
def get_data():
    return load_sample_data()


@app.get("/api/features")
def get_features():
    return build_features(load_sample_data())


@app.get("/api/predict")
def get_prediction():
    """Legacy-shaped endpoint: full feature list + per-ticker predictions
    for whatever's currently in sample_market_data.json (no live pull).
    Fails with 503 if there's no trained model - no rule-based fallback."""
    data = load_sample_data()
    feature_list = build_features(data)
    try:
        predictions = [score_ticker(f) for f in feature_list]
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"data": data, "features": feature_list, "predictions": predictions}


@app.get("/api/top-picks")
def get_top_picks(top_n: int = 5):
    """Ranks whatever's currently in sample_market_data.json and returns the
    top N candidates under the price cap that are predicted UP by >=5%.
    Fails with 503 if there's no trained model - no rule-based fallback."""
    data = load_sample_data()
    try:
        result = assemble_top_picks(data, top_n=top_n)
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    log_picks(result["top_picks"])
    return result


@app.post("/api/refresh")
def refresh(top_n: int = 5):
    """Pulls fresh live data (screener -> per-ticker premarket data + news),
    then returns the ranked top picks.

    No fallback: if the live pull fails (screener down, yfinance error, too
    few candidates) this returns 502. If there's no trained model, 503.
    Either way you get a clean failure, never stale/sample data passed off
    as live, and never a partial/rule-based result standing in for the ML
    prediction you were expecting.
    """
    try:
        new_data = refresh_data(max_price=MAX_PRICE)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Live data pull failed: {exc}") from exc

    try:
        result = assemble_top_picks(new_data, top_n=top_n)
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    log_picks(result["top_picks"])
    return result


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
