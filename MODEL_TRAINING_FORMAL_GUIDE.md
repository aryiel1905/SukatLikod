# SukatLikod Formal Guide: From Photos to Trained ML Model

## 1. Purpose of This Guide
This document explains, in beginner-friendly steps, how to:
- Gather posture data (videos/frames)
- Label samples (`proper` vs `needs_correction`)
- Build a dataset for machine learning
- Train and retrain the Python model
- Connect the model to the React app

This guide follows the current project setup in this repository.

## 2. Important Concept (Before You Start)
The model is trained on **numeric posture features**, not directly on raw photos.

Photos/frames are used to extract keypoints and compute features such as:
- `trunk_angle`
- `head_forward`
- `shoulder_tilt`
- `trunk_variance`

These feature values + a label become training rows in `posture_dataset.csv`.

## 3. Initial Folder Setup
Create these folders in the project root:

- `dataset/raw_videos/`
- `dataset/frames/`
- `dataset/labeled/proper/`
- `dataset/labeled/needs_correction/`
- `dataset/labeled/skip/`

Use `skip` for unusable samples (blur, occlusion, wrong view, cropped body).

## 4. Define Labeling Rules First
Before labeling, write clear criteria:

- `proper`: neutral posture, head not significantly forward, shoulders mostly level
- `needs_correction`: visible slouch, forward head, uneven shoulder alignment
- `skip`: poor quality or invalid framing

Keep these rules fixed while labeling to reduce inconsistencies.

## 5. Gather Data (Videos Recommended)
Collect short posture clips (10 to 30 seconds each) with variety:

- Proper posture clips
- Poor posture clips
- Different people (if available)
- Different lighting and camera distances

Recommended starting target:
- 20 to 40 clips total
- Roughly balanced between classes

## 6. Convert Videos to Frames
Extract frames from each video (example: 2 FPS to avoid duplicates):

```bash
ffmpeg -i input.mp4 -vf fps=2 dataset/frames/clip01_%04d.jpg
```

Repeat for each raw video.

## 7. Manually Label Frames
Open frames and move each file into:

- `dataset/labeled/proper/`
- `dataset/labeled/needs_correction/`
- `dataset/labeled/skip/`

Only include confident samples in the first two folders.

## 8. Build `posture_dataset.csv`
Create `backend/data/posture_dataset.csv` with columns:

- `trunk_angle`
- `head_forward`
- `shoulder_tilt`
- `trunk_variance`
- `label`

Reference format:
- `backend/data/posture_dataset.sample.csv`

At this stage, you can:
- Fill values manually if you have measurements, or
- Use a feature-extraction script (recommended) to compute values from labeled frames

## 9. Train the Model (First-Time Setup)
Run these commands once for initial environment setup:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r backend/requirements.txt
python backend/train.py --data backend/data/posture_dataset.csv
```

Output files:
- `backend/model/posture_model.joblib`
- `backend/model/model_meta.json`

## 10. Run Inference API
Start backend API:

```bash
uvicorn backend.api:app --reload --host 127.0.0.1 --port 8000
```

Health check:
- `http://127.0.0.1:8000/health`

## 11. Connect Frontend to ML API
Create `.env` in root:

```bash
VITE_ML_API_URL=http://127.0.0.1:8000
```

Run frontend:

```bash
npm run dev
```

The app will call `/predict` automatically.

## 12. Retraining (Repeatable Workflow)
Yes, retraining is expected and should be repeated as you improve data.

Retrain whenever:
- You add new labeled data
- You correct wrong labels
- You improve feature engineering

For most retraining sessions, run only:

```bash
.venv\Scripts\activate
python backend/train.py --data backend/data/posture_dataset.csv
```

You do **not** need to recreate venv every time.

## 13. When to Re-run Full Setup Commands
Run full setup again only if:
- `.venv` was deleted
- You switched machines
- Requirements changed
- Environment is broken

## 14. Suggested Iteration Cycle
Use this cycle repeatedly:

1. Gather more videos
2. Extract frames
3. Label frames
4. Build/update `posture_dataset.csv`
5. Retrain model
6. Run API and test in app
7. Review errors and improve data

## 15. Using Codex Effectively
You can ask Codex to automate repetitive work, for example:

- Generate a script to extract features from labeled frames
- Create a labeling helper utility
- Validate class balance in `posture_dataset.csv`
- Summarize training metrics after each run

Recommended approach: do one clear step at a time, then verify output before moving on.
