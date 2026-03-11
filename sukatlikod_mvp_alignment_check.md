# SukatLikod - Thesis-Aligned MVP Alignment Check (Authoritative v2)

Date: 2026-03-06  
Status: Active baseline for MVP review and implementation checks.

## 1. Core Goal of the MVP
Build a camera-based posture assessment system that:

1. captures a user with a single RGB webcam
2. extracts human pose landmarks
3. computes posture alignment features
4. classifies posture as:
   - `proper`
   - `needs_correction`
5. provides simple corrective feedback

The system should operate on guided capture and short pose sequences.

## 2. Simplified System Architecture

```text
User Webcam
   |
   v
React Frontend (Video capture + UI)
   |
   v
Pose Estimation (MediaPipe Pose)
   |
   v
Pose Sequence Buffer (WINDOW=30)
   |
   v
Feature Extraction
(trunk_angle, head_forward, shoulder_tilt, trunk_variance)
   |
   v
Python Backend (ML Classification)
   |
   v
Posture Result (proper / needs_correction)
   |
   v
Feedback UI
```

## 3. Guided Capture Scope
The system should guide user posture capture rather than accept arbitrary framing.

### Allowed capture modes
- Front view
- Side view
- Back view

### MVP implementation scope
- Required now: Front + Side
- Deferred: Back view

## 4. Pose Estimation Layer
Use MediaPipe Pose Landmarker with key landmarks:
- nose
- left/right shoulder
- left/right hip
- left/right ear

### Model choice: `lite` vs `full` (decision guidance)
Current implementation target:
- `pose_landmarker_lite.task` for MVP real-time responsiveness

Why `lite` is acceptable for MVP:
- lower latency on commodity devices
- smoother frame-by-frame UX
- simpler deployment behavior in browser

When to consider `full`:
- if posture feature stability is insufficient
- if controlled tests show material accuracy gain
- if hardware budget allows lower FPS / higher compute

If switching, compare:
1. average FPS
2. inference latency
3. posture classification impact on held-out data

## 5. Pose Sequence Buffer
Do not rely on single-frame decisions.

- `WINDOW = 30` frames
- per-frame features: `trunk_angle`, `head_forward`, `shoulder_tilt`
- sequence statistics: mean + variance

Purpose:
- reduce noise
- improve stability of predictions and feedback

## 6. Feature Extraction
Core features:

1. `trunk_angle`
```text
vector = shoulder_midpoint - hip_midpoint
angle = deviation from vertical
```

2. `head_forward`
```text
front mode: abs(nose.z - shoulder_mid.z)
side mode:  abs(nose.x - shoulder_mid.x) [normalized landmarks]
```

3. `shoulder_tilt`
```text
abs(left_shoulder.y - right_shoulder.y)
```

4. `trunk_variance`
```text
variance of trunk_angle over WINDOW
```

## 7. Machine Learning Classifier Contract (v1)
Backend request fields are **required**:
- `trunk_angle`
- `head_forward`
- `shoulder_tilt`
- `trunk_variance`

Example request:

```json
{
  "trunk_angle": 12.5,
  "head_forward": 0.06,
  "shoulder_tilt": 0.03,
  "trunk_variance": 1.2
}
```

Example response:

```json
{
  "label": "needs_correction",
  "confidence": 0.82,
  "feedback": "Straighten your back and pull your head slightly backward."
}
```

## 8. Feedback System
Feedback must be actionable and feature-linked:
- high trunk angle -> "Straighten your back"
- high head-forward -> "Bring your head back"
- high shoulder tilt -> "Level your shoulders"

UI should display:
- posture label
- confidence or score (0-100)
- short corrective suggestion

## 9. Minimal User Flow
1. User opens SukatLikod.
2. App shows framing instructions.
3. User starts posture session.
4. App buffers sequence (`WINDOW=30`).
5. Features extracted and sent to backend.
6. Backend predicts posture class.
7. UI presents score + correction message.

## 10. Out-of-Scope for MVP
Do not include yet:
- multi-camera capture
- 3D reconstruction pipeline
- clinical diagnosis claims
- full-body biomechanics analysis

## 11. MVP Completion Definition
MVP is complete when the app can:
- capture webcam video
- detect pose landmarks
- buffer sequence frames
- compute required feature set
- send features to Python backend
- classify posture into two classes
- render score + corrective feedback

## 12. Measurable Acceptance Criteria
1. `GET /health` returns `{"status":"ok"}` locally and in production.
2. `POST /predict` accepts required v1 fields and returns label/confidence/feedback.
3. Front view and side view both produce live inference (not blocked).
4. Decisions are sequence-based (`WINDOW=30`), not single-frame only.
5. Missing landmarks do not crash the session loop.
6. Training run outputs held-out classification metrics.
7. Deployed frontend can call deployed backend without CORS failure.

## 13. Codex Verification Checklist
Use this checklist against current code:

1. Single RGB webcam input?
2. Pose landmarks detected from live stream?
3. Guided capture behavior present?
4. Front + side supported for MVP inference?
5. Sequence buffer used (`WINDOW=30`)?
6. Required features computed (`trunk_angle`, `head_forward`, `shoulder_tilt`, `trunk_variance`)?
7. Features sent to Python backend?
8. Backend returns `proper` or `needs_correction`?
9. UI shows posture score + correction feedback?
10. Out-of-scope complexity deferred?

## 14. Requested Review Output Template
When asking Codex for review, request:
- what already aligns
- what partially aligns
- what is missing
- what should be removed/deferred
- overall thesis alignment judgment
