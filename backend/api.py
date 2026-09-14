from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict

MODEL_PATH = Path("backend/model/uprightly_random_forest_final.joblib")
META_PATH = Path("backend/model/uprightly_random_forest_meta.json")
SCHEMA_ID = "uprightly_8point_v1"
FEATURES = [
    "shoulder_tilt_angle_degrees",
    "shoulder_height_difference_normalized_signed",
    "eye_tilt_angle_degrees",
    "ear_tilt_angle_degrees",
    "head_axis_tilt_from_vertical_degrees",
    "eye_span_to_shoulder_ratio",
    "ear_span_to_shoulder_ratio",
    "nose_chin_distance_to_shoulder_ratio",
    "left_ear_to_shoulder_distance_ratio",
    "right_ear_to_shoulder_distance_ratio",
    "ear_shoulder_symmetry_difference_normalized_signed",
    "nose_horizontal_offset_from_shoulders_normalized_signed",
    "nose_vertical_offset_from_shoulders_normalized_signed",
    "chin_horizontal_offset_from_shoulders_normalized_signed",
    "chin_vertical_offset_from_shoulders_normalized_signed",
    "eye_mid_horizontal_offset_from_shoulders_normalized_signed",
    "eye_mid_vertical_offset_from_shoulders_normalized_signed",
    "ear_mid_horizontal_offset_from_shoulders_normalized_signed",
    "ear_mid_vertical_offset_from_shoulders_normalized_signed",
    "nose_horizontal_offset_from_eye_mid_normalized_signed",
    "nose_vertical_offset_from_eye_mid_normalized_signed",
    "chin_horizontal_offset_from_eye_mid_normalized_signed",
    "chin_vertical_offset_from_eye_mid_normalized_signed",
    "nose_horizontal_offset_from_ear_mid_normalized_signed",
    "nose_vertical_offset_from_ear_mid_normalized_signed",
    "chin_horizontal_offset_from_ear_mid_normalized_signed",
    "chin_vertical_offset_from_ear_mid_normalized_signed",
    "eye_mid_horizontal_offset_from_ear_mid_normalized_signed",
    "eye_mid_vertical_offset_from_ear_mid_normalized_signed",
    "left_eye_to_ear_distance_ratio",
    "right_eye_to_ear_distance_ratio",
    "eye_ear_symmetry_difference_normalized_signed",
    "left_nose_to_ear_distance_ratio",
    "right_nose_to_ear_distance_ratio",
    "nose_ear_symmetry_difference_normalized_signed",
]
CLASSES = ["mild_asymmetry", "neutral_posture", "severe_misalignment"]
FEEDBACK = {
    "neutral_posture": "Your posture appears balanced.",
    "mild_asymmetry": "A small alignment adjustment may help.",
    "severe_misalignment": "Return toward a centered, comfortable upright posture.",
}

app = FastAPI(title="Uprightly posture model API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://uprightly-prmsu.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    shoulder_tilt_angle_degrees: float
    shoulder_height_difference_normalized_signed: float
    eye_tilt_angle_degrees: float
    ear_tilt_angle_degrees: float
    head_axis_tilt_from_vertical_degrees: float
    eye_span_to_shoulder_ratio: float
    ear_span_to_shoulder_ratio: float
    nose_chin_distance_to_shoulder_ratio: float
    left_ear_to_shoulder_distance_ratio: float
    right_ear_to_shoulder_distance_ratio: float
    ear_shoulder_symmetry_difference_normalized_signed: float
    nose_horizontal_offset_from_shoulders_normalized_signed: float
    nose_vertical_offset_from_shoulders_normalized_signed: float
    chin_horizontal_offset_from_shoulders_normalized_signed: float
    chin_vertical_offset_from_shoulders_normalized_signed: float
    eye_mid_horizontal_offset_from_shoulders_normalized_signed: float
    eye_mid_vertical_offset_from_shoulders_normalized_signed: float
    ear_mid_horizontal_offset_from_shoulders_normalized_signed: float
    ear_mid_vertical_offset_from_shoulders_normalized_signed: float
    nose_horizontal_offset_from_eye_mid_normalized_signed: float
    nose_vertical_offset_from_eye_mid_normalized_signed: float
    chin_horizontal_offset_from_eye_mid_normalized_signed: float
    chin_vertical_offset_from_eye_mid_normalized_signed: float
    nose_horizontal_offset_from_ear_mid_normalized_signed: float
    nose_vertical_offset_from_ear_mid_normalized_signed: float
    chin_horizontal_offset_from_ear_mid_normalized_signed: float
    chin_vertical_offset_from_ear_mid_normalized_signed: float
    eye_mid_horizontal_offset_from_ear_mid_normalized_signed: float
    eye_mid_vertical_offset_from_ear_mid_normalized_signed: float
    left_eye_to_ear_distance_ratio: float
    right_eye_to_ear_distance_ratio: float
    eye_ear_symmetry_difference_normalized_signed: float
    left_nose_to_ear_distance_ratio: float
    right_nose_to_ear_distance_ratio: float
    nose_ear_symmetry_difference_normalized_signed: float


class PredictResponse(BaseModel):
    label: str
    confidence: float
    probabilities: dict[str, float]
    score: int
    feedback: str


_model = None


def score_from_probabilities(probabilities: dict[str, float]) -> int:
    score = (
        100 * probabilities.get("neutral_posture", 0.0)
        + 65 * probabilities.get("mild_asymmetry", 0.0)
        + 25 * probabilities.get("severe_misalignment", 0.0)
    )
    return max(0, min(100, int(score + 0.5)))


def validate_model(model: object) -> None:
    model_features = list(getattr(model, "feature_names_in_", []))
    if model_features and model_features != FEATURES:
        raise RuntimeError("The model feature schema does not match uprightly_8point_v1.")

    model_classes = {str(value) for value in getattr(model, "classes_", [])}
    if model_classes != set(CLASSES):
        raise RuntimeError("The model classes do not match Uprightly's three states.")


@app.on_event("startup")
def load_model() -> None:
    global _model
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")

    _model = joblib.load(MODEL_PATH)
    validate_model(_model)
    if META_PATH.exists():
        metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
        if metadata.get("schema_id") != SCHEMA_ID:
            raise RuntimeError("The model metadata schema ID is not supported.")
        if metadata.get("features") != FEATURES:
            raise RuntimeError("The model metadata feature order is not supported.")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest) -> PredictResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    row = body.model_dump()
    frame = pd.DataFrame([[row[name] for name in FEATURES]], columns=FEATURES)
    proba = _model.predict_proba(frame)[0]
    classes = [str(value) for value in _model.classes_]
    probabilities = {
        class_name: float(proba[index])
        for index, class_name in enumerate(classes)
    }
    label = str(_model.predict(frame)[0])
    if label not in FEEDBACK:
        raise HTTPException(status_code=500, detail="Model returned an unknown class.")

    return PredictResponse(
        label=label,
        confidence=probabilities[label],
        probabilities=probabilities,
        score=score_from_probabilities(probabilities),
        feedback=FEEDBACK[label],
    )
