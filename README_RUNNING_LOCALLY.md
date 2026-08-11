# Running the app locally (backend + frontend)

Everything runs fully local for now - no hosting, no deployed URL.

# BEFORE RUNNING...

# Install with: python -m pip install -r requirements.txt (inside the activated venv)

fastapi
uvicorn[standard]
pandas
numpy
requests
yfinance
xgboost
scikit-learn
vaderSentiment

## 1. Confirm the frontend files are in place

`frontend/app.js` and `frontend/index.html` should already point at
`http://localhost:8000` and call the current API routes. Same folder as
the existing `style.css` - nothing else in `frontend/` needs to change.

## 2. Start the backend

Terminal 1 (venv activated):
```
cd backend
python app.py
```

## 3. Serve the frontend from a local server (don't just double-click the HTML)

Terminal 2:
```
cd frontend
python -m http.server 5500
```

Some browsers (Chrome especially) restrict network requests from a page
opened directly via `file://`, even when the API itself allows it. A real
localhost server avoids that entirely.

## 4. Open it

**http://localhost:5500/index.html**

## What you should see

- The page loads immediately and calls `/api/dev/top-picks` (sample data,
  safe - no live pull, nothing logged). You'll see either a populated
  table or "No candidates cleared the 3% / sub-$20 bar."
- Clicking **"Pull Live Data"** fires the real `POST /api/refresh`.
  During premarket hours (4:00-9:30am ET) that's a live scan and logs
  real picks. Outside those hours it may legitimately return few/no
  picks, or a 502 if the live screener comes back empty - same as
  `run_daily.py` would.

## If something doesn't work

Open the browser's dev console (F12 -> Console / Network tab) and check
the failed request there - it'll show the exact error (CORS, 404, 502,
connection refused, etc.) rather than guessing blind from the rendered
page alone.

## Keeping it updated

- **`fill_actual_gap.py` and `merge_feedback.py`** are safe to run
  repeatedly / on a cadence (e.g. daily after close) - idempotent, never
  overwrite already-resolved rows.
- **`train_model.py`** is a deliberate manual step, not automatic - rerun
  it periodically once enough real feedback has accumulated to matter
  (a handful of rows, like the current 10, is not yet worth a retrain by
  itself).
- **`build_watchlist.py`** (the broader training-backtest universe) is
  meant to be refreshed weekly/monthly, also still manual.
- **yfinance's live schema was explicitly flagged as unverified** in the
  original handoff and has now run successfully for real - but that
  remains the most likely thing to silently break from an external change
  (Yahoo altering field names/shapes), not from anything in this repo.
- Before any new work session, confirm the delivered files (`pipeline.py`,
  `app.py`, `run_daily.py`, `fill_actual_gap.py`, `merge_feedback.py`,
  `train_model.py`, `screener.py`) are actually the versions in place -
  several-day gaps between sessions are exactly when a fix can get
  half-applied or reverted without noticing.
