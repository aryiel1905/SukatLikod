# SukatLikod Python ML Backend

## 1) Prepare your dataset
Create `backend/data/posture_dataset.csv` with these columns:

- `trunk_angle`
- `head_forward`
- `shoulder_tilt`
- `trunk_variance`
- `label` (`proper` or `needs_correction`)

Use `backend/data/posture_dataset.sample.csv` as a format template.

## 2) Install dependencies
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r backend/requirements.txt
```

## 3) Train model
```bash
python backend/train.py --data backend/data/posture_dataset.csv
```

Artifacts:
- `backend/model/posture_model.joblib`
- `backend/model/model_meta.json`

## 4) Run API
```bash
uvicorn backend.api:app --reload --host 127.0.0.1 --port 8000
```

Health check:
`GET http://127.0.0.1:8000/health`

Prediction endpoint:
`POST http://127.0.0.1:8000/predict`

Request body:
```json
{
  "trunk_angle": 14.3,
  "head_forward": 0.09,
  "shoulder_tilt": 0.03,
  "trunk_variance": 1.2
}
```
