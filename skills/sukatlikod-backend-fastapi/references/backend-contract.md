# Backend Contract

## Current Files

Primary backend files at the time this skill was written:

- `backend/api.py`
- `backend/train.py`
- `backend/requirements.txt`

Treat `backend/api.py` as the active inference entry point unless the repo structure changes.

## Core Endpoints

- `GET /health`
- `POST /predict`

`GET /health` should return a minimal readiness-style response.

`POST /predict` is the canonical inference route for the frontend.

## Current Request Shape

The backend currently expects these fields:

- `trunk_angle`
- `head_forward`
- `shoulder_tilt`
- `trunk_variance`
- `neck_forward_contour`
- `upper_back_curvature`
- `torso_outline_angle`
- `silhouette_stability`

All are numeric and currently validated as non-negative in the API layer.

## Current Response Shape

The backend currently returns:

- `label`
- `confidence`
- `probabilities`
- `feedback`

Keep this stable unless the frontend is updated in the same change.

## Label Semantics

Canonical labels:

- `proper`
- `needs_correction`

Do not introduce new posture classes casually. That is a product and dataset change, not just an API change.

## Feedback Semantics

Backend feedback should be brief and actionable:

- trunk issue -> straighten the back
- head-forward issue -> bring the head back
- shoulder issue -> level the shoulders

If returning `proper`, keep the feedback short and non-clinical.

## Smoothing And Gating

The current backend includes:

- a strict confidence threshold before returning `proper`
- a recent-label vote window
- smoothed confidence derived from recent probabilities

These are product-facing behaviors. If you change them, treat that as a behavioral change, not a refactor.

## Artifact Rules

Expected artifacts:

- `backend/model/posture_model.joblib`
- `backend/model/model_meta.json`

Prefer metadata-driven feature ordering when metadata exists.

## Anti-Patterns

Avoid:

- changing feature names only in the API model
- hardcoding a new feature order that disagrees with training metadata
- returning free-form responses with unstable keys
- mixing frontend phrasing concerns into low-level model utilities
- hiding missing-model failures behind generic 500 behavior

## Safe Change Pattern

When changing backend inference behavior:

1. inspect frontend payload generation
2. inspect training feature list
3. inspect model metadata expectations
4. update the API schema and helpers
5. verify the user-visible result semantics
