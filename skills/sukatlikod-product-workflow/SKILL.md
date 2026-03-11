---
name: sukatlikod-product-workflow
description: End-to-end workflow guidance for the SukatLikod posture-analysis product across React frontend, MediaPipe pose extraction, FastAPI inference, and ML result handling. Use when Codex is implementing or reviewing webcam capture flow, sequence buffering, feature extraction, frontend/backend prediction contracts, posture result UI, feedback logic, or any cross-layer change that affects how SukatLikod turns camera input into posture classification and guidance.
---

# Sukatlikod Product Workflow

## Overview

Follow this skill when a task spans product behavior rather than a single file. Preserve the canonical SukatLikod flow: guided capture, pose landmarks, buffered feature extraction, backend prediction, and actionable feedback.

Read [references/workflow-contract.md](references/workflow-contract.md) when a task needs the exact v1 vocabulary, request/response fields, or capture-mode constraints.

## Core Workflow

Treat this as the canonical product pipeline:

1. Guide the user into an allowed capture mode.
2. Acquire pose landmarks from a single RGB webcam session.
3. Buffer short pose sequences instead of trusting a single frame.
4. Compute posture features from the buffered sequence.
5. Send the feature payload to the backend prediction contract.
6. Convert the prediction into a stable, understandable result.
7. Show actionable, feature-linked feedback in the UI.

Do not skip intermediate layers without a clear product decision. For example, do not couple the UI directly to raw landmark geometry when the current contract is feature-based inference.

## Product Rules

Keep these rules stable unless the task explicitly changes product scope:

- Use guided capture, not arbitrary camera intake.
- Treat Front and Side as MVP capture modes.
- Keep Back view deferred unless the task explicitly expands scope.
- Prefer sequence-based decisions over per-frame snap judgments.
- Keep the MVP output classes to `proper` and `needs_correction`.
- Keep feedback short, actionable, and tied to the strongest posture issue.
- Preserve privacy-aware behavior: local frame processing by default and minimal logging.
- Avoid medical or diagnostic claims in labels, messages, or UI framing.

## Contract Priorities

When frontend, backend, and model logic disagree, resolve in this order:

1. Product workflow contract
2. Backend request/response schema
3. Training feature list and model metadata
4. UI presentation details

Do not add or remove prediction fields in only one layer. If a feature changes, update training, inference, and frontend payload generation together.

## Frontend Guidance

Use the frontend as sensor plus dashboard.

- Keep camera permissions, framing guidance, buffering, loading, and inference results as separate states.
- Show clear blocked states for missing camera permission, missing landmarks, unstable framing, and backend unavailability.
- Prefer interpretable result panels over decorative scoring.
- When confidence is weak, avoid overstating certainty in the UI copy.
- If MediaPipe behavior changes, verify that computed features still match the backend contract.

## Backend Guidance

Use the backend as decision engine, not UI policy store.

- Keep `/health` simple and deterministic.
- Keep `/predict` centered on normalized feature inputs and structured outputs.
- Load model artifacts on startup and fail clearly when artifacts are missing.
- Keep response fields stable: `label`, `confidence`, `probabilities` when available, and `feedback`.
- Gate and smooth predictions when stability is part of the current behavior.

## Feedback Guidance

Feedback must explain what to correct, not just what was detected.

- High `trunk_angle`: tell the user to straighten the back.
- High `head_forward`: tell the user to bring the head back.
- High `shoulder_tilt`: tell the user to level the shoulders.
- Prefer one primary instruction over a paragraph of mixed advice.
- If posture is `proper`, keep the message brief and non-clinical.

## Scope Boundaries

Defer these unless the task explicitly expands the MVP:

- Multi-camera capture
- 3D reconstruction
- Clinical diagnosis language
- Full-body biomechanics analysis
- Unbounded analytics or user-history systems

## Review Checklist

When reviewing a change, verify:

1. Does it preserve the capture-to-prediction pipeline?
2. Does it respect the current allowed capture modes?
3. Does it keep inference sequence-based where required?
4. Does it preserve the canonical feature vocabulary?
5. Does frontend/backend data exchange still match?
6. Does the UI present actionable feedback instead of raw internals?
7. Does the change avoid expanding scope accidentally?

## Resources

### references/workflow-contract.md

Read this file when you need the exact current MVP contract, capture modes, required feature list, and acceptance criteria.
