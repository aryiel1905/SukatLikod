# Workflow Contract

## Canonical User Flow

1. User opens SukatLikod.
2. App guides framing for an allowed capture mode.
3. Webcam session starts from a single RGB camera.
4. MediaPipe Pose extracts landmarks.
5. Frontend buffers a short pose sequence.
6. Frontend computes posture features from the sequence.
7. Frontend sends the feature payload to backend `/predict`.
8. Backend returns posture label, confidence, and feedback.
9. UI shows a stable result and one clear corrective suggestion.

## MVP Capture Modes

- Required now: Front, Side
- Deferred: Back

Prefer guided capture. Do not assume arbitrary framing is valid input.

## Canonical Vocabulary

- `proper`
- `needs_correction`
- `confidence`
- `feedback`
- `probabilities`

Use these terms consistently across UI copy, API schemas, and review comments unless the task explicitly changes the product language.

## Feature Contract

Current feature set used across training and inference:

- `trunk_angle`
- `head_forward`
- `shoulder_tilt`
- `trunk_variance`
- `neck_forward_contour`
- `upper_back_curvature`
- `torso_outline_angle`
- `silhouette_stability`

The minimum authoritative v1 contract documented in project notes is:

- `trunk_angle`
- `head_forward`
- `shoulder_tilt`
- `trunk_variance`

When changing the feature list, update all affected layers together:

1. frontend feature computation
2. backend request schema
3. training feature list
4. model metadata

## Backend Contract

Current stable endpoints:

- `GET /health`
- `POST /predict`

`POST /predict` should return structured posture output, centered on:

- `label`
- `confidence`
- `feedback`

Include `probabilities` when the backend already exposes them.

## Feedback Mapping

Map dominant issues to direct instructions:

- high `trunk_angle` -> "Straighten your back"
- high `head_forward` -> "Bring your head back"
- high `shoulder_tilt` -> "Level your shoulders"

Prefer one dominant instruction over multiple competing corrections.

## Non-Goals

Do not introduce these without an explicit scope change:

- medical diagnosis
- full-body biomechanics platform
- multi-camera capture
- 3D reconstruction
- broad patient analytics

## Acceptance Checks

- Frontend and backend exchange the same feature names.
- Missing landmarks do not crash the session loop.
- Sequence buffering exists where live inference depends on stability.
- Output stays within the `proper` / `needs_correction` posture framing for MVP.
- User-facing feedback stays brief and actionable.
