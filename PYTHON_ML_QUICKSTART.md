# Python ML Quickstart

## 1) Create your dataset
Copy `backend/data/posture_dataset.sample.csv` to `backend/data/posture_dataset.csv`, then replace with your real labeled data.

Required columns:
- `trunk_angle`
- `head_forward`
- `shoulder_tilt`
- `trunk_variance`
- `label` (`proper` or `needs_correction`)

## 2) Train in Python
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r backend/requirements.txt
python backend/train.py --data backend/data/posture_dataset.csv
```

## 3) Start ML API
```bash
uvicorn backend.api:app --reload --host 127.0.0.1 --port 8000
```

## 4) Connect frontend
Create `.env` in project root:
```bash
VITE_ML_API_URL=http://127.0.0.1:8000
```

Then run:
```bash
npm run dev
```

If API is not reachable, the app automatically falls back to local rule-based scoring.
