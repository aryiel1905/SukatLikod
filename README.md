# SukatLikod

SukatLikod is a posture-analysis prototype with a React/Vite frontend and a Python/FastAPI ML backend.

The current product flow is:

1. capture webcam posture input
2. extract pose landmarks and posture features
3. send features to the backend classifier
4. return `proper` or `needs_correction`
5. show actionable feedback in the UI

## Main Areas

- `src/`: React frontend, live posture UI, MediaPipe/browser logic
- `backend/`: FastAPI inference, training scripts, dataset preparation
- `public/models/`: browser pose model asset
- `skills/`: repo-specific Codex skills for workflow, frontend, backend, dataset/eval, and training

## Local Run

Frontend:

```bash
npm install
npm run dev
```

Backend:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r backend/requirements.txt
uvicorn backend.api:app --reload --host 127.0.0.1 --port 8000
```

Set the frontend API URL in `.env`:

```bash
VITE_ML_API_URL=http://127.0.0.1:8000
```

## Skills

Repo-specific skills live under [`skills/`](c:/Users/Aryiel Joshua/OneDrive/Desktop/SukatLikod/skills). Start with [skills/INDEX.md](c:/Users/Aryiel Joshua/OneDrive/Desktop/SukatLikod/skills/INDEX.md) to see which skill to use for a given task.
