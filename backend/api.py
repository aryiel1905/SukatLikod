from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

MODEL_PATH = Path("backend/model/posture_model.joblib")
META_PATH = Path("backend/model/model_meta.json")
DEFAULT_FEATURES = [
    "trunk_angle",
    "head_forward",
    "shoulder_tilt",
    "trunk_variance",
    "neck_forward_contour",
    "upper_back_curvature",
    "torso_outline_angle",
    "silhouette_stability",
]
PROPER_CONFIDENCE_THRESHOLD = 0.70
PREDICTION_VOTE_WINDOW = 7

app = FastAPI(title="SukatLikod ML API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://sukat-likod.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    trunk_angle: float = Field(ge=0)
    head_forward: float = Field(ge=0)
    shoulder_tilt: float = Field(ge=0)
    trunk_variance: float = Field(ge=0)
    neck_forward_contour: float = Field(default=0, ge=0)
    upper_back_curvature: float = Field(default=0, ge=0)
    torso_outline_angle: float = Field(default=0, ge=0)
    silhouette_stability: float = Field(default=0, ge=0)


class PredictResponse(BaseModel):
    label: str
    confidence: float
    probabilities: dict[str, float]
    feedback: str


def build_feedback(label: str, body: PredictRequest) -> str:
    if label == "proper":
        return "Good posture - keep it."

    candidates = [
        ("trunk_angle", body.trunk_angle, "Straighten your back (reduce trunk lean)."),
        ("head_forward", body.head_forward, "Bring your head back (avoid head-forward)."),
        ("shoulder_tilt", body.shoulder_tilt, "Level your shoulders."),
    ]
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][2]


_model = None
_features = DEFAULT_FEATURES
_label_window: deque[str] = deque(maxlen=PREDICTION_VOTE_WINDOW)
_proper_prob_window: deque[float] = deque(maxlen=PREDICTION_VOTE_WINDOW)


def gated_label(probabilities: dict[str, float]) -> str:
    proper_prob = float(probabilities.get("proper", 0.0))
    needs_prob = float(probabilities.get("needs_correction", 0.0))

    # Be strict: only return "proper" when confidence is high enough.
    if proper_prob >= PROPER_CONFIDENCE_THRESHOLD and proper_prob >= needs_prob:
        return "proper"
    return "needs_correction"


def vote_label() -> str:
    if not _label_window:
        return "needs_correction"

    proper = sum(1 for x in _label_window if x == "proper")
    needs = len(_label_window) - proper
    if proper == needs:
        return _label_window[-1]
    return "proper" if proper > needs else "needs_correction"


def smoothed_confidence(label: str) -> float:
    if not _proper_prob_window:
        return 0.0

    proper_mean = float(sum(_proper_prob_window) / len(_proper_prob_window))
    if label == "proper":
        return proper_mean
    return 1.0 - proper_mean


@app.on_event("startup")
def load_model() -> None:
    global _model, _features

    if not MODEL_PATH.exists():
        raise RuntimeError(
            "Model file not found. Train first with: "
            "python backend/train.py --data backend/data/posture_dataset.csv"
        )

    _model = joblib.load(MODEL_PATH)

    if META_PATH.exists():
        data = json.loads(META_PATH.read_text(encoding="utf-8"))
        if isinstance(data.get("features"), list) and data["features"]:
            _features = data["features"]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest) -> PredictResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    row = [[getattr(body, name) for name in _features]]
    X = np.array(row, dtype=float)

    proba = _model.predict_proba(X)[0]
    classes = _model.classes_
    probs = {str(classes[i]): float(proba[i]) for i in range(len(classes))}
    current_label = gated_label(probs)
    _label_window.append(current_label)
    _proper_prob_window.append(float(probs.get("proper", 0.0)))

    pred = vote_label()
    confidence = smoothed_confidence(pred)

    return PredictResponse(
        label=str(pred),
        confidence=confidence,
        probabilities=probs,
        feedback=build_feedback(str(pred), body),
    )
