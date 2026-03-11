# SukatLikod: System Architecture and Process

## 1. Overview

SukatLikod has two connected layers:

- **Frontend (React + MediaPipe)**: captures webcam posture data, computes features, and displays real-time feedback.
- **Backend (Python + FastAPI + scikit-learn)**: receives features, runs ML prediction, and returns posture classification + guidance.

Think of it as:

- Frontend = **sensor + dashboard**
- Backend = **ML decision engine**

## 2. Frontend Responsibilities

The frontend:

1. Opens webcam stream.
2. Runs MediaPipe Pose to get body landmarks.
3. Computes posture features:
   - `trunk_angle`
   - `head_forward`
   - `shoulder_tilt`
   - `trunk_variance`
4. Sends these required features to backend `/predict`.
5. Displays returned prediction and corrective feedback.

## 3. Backend Responsibilities

The backend:

1. Loads trained model artifacts:
   - `backend/model/posture_model.joblib`
   - `backend/model/model_meta.json`
2. Exposes API endpoints:
   - `GET /health` for service status
   - `POST /predict` for posture inference
3. Returns:
   - label (`proper` or `needs_correction`)
   - confidence
   - feedback text

Prediction contract note:
- `trunk_variance` is required in the current v1 backend contract.

## 4. Training Pipeline

Training uses **tabular features + labels**, not raw images directly.

### Input dataset

`backend/data/posture_dataset.csv` with columns:

- `trunk_angle`
- `head_forward`
- `shoulder_tilt`
- `trunk_variance`
- `label`

### Training command

```powershell
.\.venv\Scripts\python.exe backend/train.py --data backend/data/posture_dataset.csv
```

### Output artifacts

- `backend/model/posture_model.joblib`
- `backend/model/model_meta.json`

## 5. Runtime Data Flow

Live session flow:

1. Webcam frame captured in frontend.
2. Pose landmarks extracted.
3. Numeric features computed.
4. Features sent to backend `/predict`.
5. Model predicts posture class.
6. UI shows score + corrective message.

## 6. Deployment Architecture

- **Frontend**: Vercel (`https://sukat-likod.vercel.app`)
- **Backend**: Render (`https://sukatlikod.onrender.com`)

### Connection setting

In Vercel env vars:
`VITE_ML_API_URL=https://sukatlikod.onrender.com`

### CORS

Backend allows frontend origin:
`https://sukat-likod.vercel.app`

## 7. What Is Already Set Up

- Frontend posture UI and webcam processing
- Feature extraction and feedback display
- Python training and inference API
- Frontend-backend integration via env var
- Render + Vercel deployment path
- Documentation/guide pages

## 8. What Is Left To Improve

1. Collect more balanced labeled data.
2. Improve label consistency.
3. Retrain model after dataset updates.
4. Redeploy backend with latest model.
5. Validate in real sessions and iterate.

## 9. Critical Understanding

Model quality depends mostly on the dataset:

- More accurate labels -> better predictions
- More varied data -> better generalization
- Continuous retraining loop -> steady improvement
