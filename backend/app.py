from __future__ import annotations

from pathlib import Path
import sys
import os

from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from scraper import load_sample_data, refresh_data
from feature_engineering import build_features
from predictor import score_ticker, ModelUnavailableError
from screener import MAX_PRICE
from pipeline import assemble_top_picks


app = FastAPI(title="top5predictdaily API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok", "message": "top5predictdaily backend is running."}


@app.post("/api/refresh")
def refresh(top_n: int = 5):
    """THE primary endpoint. Pulls fresh live data (screener -> per-ticker
    premarket data + news), ranks it, logs the picks (with their full
    feature vectors) to data/picks_log.csv, and returns the ranked top N.

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
        result = assemble_top_picks(new_data, top_n=top_n, log=True)
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return result


# ---------------------------------------------------------------------------
# Dev/test-only routes - work off data/sample_market_data.json, never touch
# live data, and never write to picks_log.csv (that log is real-outcome
# ground truth for retraining; sample-data test calls have no place in it).
#
# Gated behind ENABLE_DEV_ENDPOINTS so they don't accidentally ship live -
# default is on since that's almost certainly what you want while developing
# locally; set ENABLE_DEV_ENDPOINTS=false wherever this gets deployed.
# ---------------------------------------------------------------------------
ENABLE_DEV_ENDPOINTS = os.environ.get("ENABLE_DEV_ENDPOINTS", "true").lower() not in ("0", "false", "no")

dev_router = APIRouter(prefix="/api/dev", tags=["dev"])


@dev_router.get("/data")
def dev_get_data():
    """Raw contents of sample_market_data.json - no live pull."""
    return load_sample_data()


@dev_router.get("/features")
def dev_get_features():
    """Feature vectors built from sample_market_data.json - no live pull."""
    return build_features(load_sample_data())


@dev_router.get("/predict")
def dev_get_prediction():
    """Per-ticker predictions against sample_market_data.json. Fails with
    503 if there's no trained model - no rule-based fallback."""
    data = load_sample_data()
    feature_list = build_features(data)
    try:
        predictions = [score_ticker(f) for f in feature_list]
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"data": data, "features": feature_list, "predictions": predictions}


@dev_router.get("/top-picks")
def dev_get_top_picks(top_n: int = 5):
    """Same ranking/explanation logic as /api/refresh, but against
    sample_market_data.json and WITHOUT logging to picks_log.csv - this is
    for exercising the ranking logic against a fixture, not for anything
    that should count as a real pick."""
    data = load_sample_data()
    try:
        result = assemble_top_picks(data, top_n=top_n, log=False)
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return result


if ENABLE_DEV_ENDPOINTS:
    app.include_router(dev_router)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)