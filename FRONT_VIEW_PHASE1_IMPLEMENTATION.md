# Front View Phase 1 Implementation

## Goal

Strengthen front-only posture detection without assuming hips are always visible in a normal desk-camera setup.

## Scope

Phase 1 supports two front-facing capture tiers in the frontend:

- `full_front`
  - nose visible
  - both shoulders visible
  - both hips visible
  - suitable for full posture assessment and backend inference

- `upper_front`
  - nose visible
  - both shoulders visible
  - both ears or both eyes visible
  - hips optional
  - suitable for limited local posture assessment only

## What This Phase Does

1. Keeps side and back framing blocked from posture assessment.
2. Replaces the old hip-dependent front gate with tiered front capture logic.
3. Allows upper-body-only framing to continue as `upper_front` instead of failing immediately.
4. Uses backend `/predict` only for `full_front`.
5. Uses local-only posture rules for `upper_front`.
6. Integrates MediaPipe `FaceLandmarker` so `upper_front` can use a real chin landmark.

## Face Landmarker Integration

The frontend now runs two MediaPipe tasks on the same video stream:

- `PoseLandmarker` for shoulders, hips, and torso posture signals
- `FaceLandmarker` for chin and facial alignment signals

The main purpose of `FaceLandmarker` in this phase is to strengthen `upper_front`
assessment by using a real chin landmark instead of a pseudo-chin estimate.

### Chin-assisted upper-front features

- chin center offset relative to shoulder midpoint
- chin forward lean relative to shoulder depth
- mouth line tilt as an additional facial symmetry cue

These are used only to improve front-facing upper-body assessment. They do not
change the backend contract in this phase.

## Screenshot-Based Threshold Tuning

The current upper-front thresholds were tuned against real screenshots captured
from the user camera in three states:

- neutral
- mild slouch
- clear slouch

Observed behavior from those samples:

- `chin lean` and `upper lean` were the strongest separators
- `nose offset` had weak supporting value
- `shoulder level`, `mouth tilt`, and `eye/ear tilt` were not reliable primary
  slouch indicators for this camera setup

As a result, the current upper-front score now:

- prioritizes forward lean
- treats head offset as a secondary cue
- avoids letting shoulder-level noise dominate the slouch decision
- introduces a more explicit severe slouch band

## Front Capture Rules

### Shared front requirements

- face visible enough
- both shoulders visible
- shoulder width above a minimum normalized width
- left/right shoulder depth difference small enough to count as front-facing

### `full_front` additional requirements

- both hips visible
- hip width above minimum
- shoulder-midpoint to hip-midpoint torso length above minimum

### `upper_front` fallback

- accepted when upper-body evidence is strong but hip evidence is weak or missing

## Decision Behavior

### `full_front`

- local smoothing and posture scoring
- backend inference enabled

### `upper_front`

- local smoothing and posture scoring
- backend inference skipped
- messaging should describe this as limited posture assessment

### invalid front framing

- guidance only
- no posture judgment

## Files Changed In Phase 1

- `src/App.tsx`

## Follow-Up Work

1. Extract front-capture classification into a dedicated frontend module.
2. Rename upper-front proxy metrics so the UI does not imply full torso analysis when hips are missing.
3. Update product docs to explicitly mention `full_front`, `upper_front`, and chin-assisted face tracking.
4. Decide whether the backend and model should gain dedicated `upper_front` and chin-based features.
5. Add a test checklist for laptop-camera framing, partial torso framing, weak lighting, and slight user rotation.

## Notes

- Phase 1 is intentionally conservative.
- It improves real-world usability first.
- It does not yet expand the backend contract or retrain the model.
