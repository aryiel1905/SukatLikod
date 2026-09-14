# Uprightly Random Forest Backend

The live posture result currently relies on the supplied three-class Random Forest. MediaPipe runs in the browser only to extract the eight normalized points required by the `uprightly_8point_v1` feature schema; camera images and video are not sent to this API.

## Install and run

Use Python dependencies compatible with the saved model (notably scikit-learn 1.6.1):

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r backend/requirements.txt
uvicorn backend.api:app --reload --host 127.0.0.1 --port 8000
```

Configure the frontend in `.env`:

```bash
VITE_ML_API_URL=http://127.0.0.1:8000
```

## API contract

- `GET /health` returns service readiness.
- `POST /predict` accepts the 35 numeric features listed in `uprightly_random_forest_meta.json` in a flat JSON object.
- The response contains `label`, `confidence`, class `probabilities`, probability-derived `score`, and `feedback`.

Supported labels are `neutral_posture`, `mild_asymmetry`, and `severe_misalignment`.

The active artifact is `backend/model/uprightly_random_forest_final.joblib`. Startup fails if its embedded feature order or classes do not match the API contract.
