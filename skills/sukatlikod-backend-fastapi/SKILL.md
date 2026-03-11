---
name: sukatlikod-backend-fastapi
description: Backend guidance for the SukatLikod FastAPI inference service, including prediction request and response schemas, model artifact loading, feedback generation, confidence handling, CORS setup, and coordination with frontend and training code. Use when Codex is editing `backend/api.py`, adjusting inference behavior, changing feature contracts, restructuring backend modules, adding endpoints, or reviewing whether backend changes still align with SukatLikod product workflow and model metadata.
---

# Sukatlikod Backend Fastapi

## Overview

Use this skill to keep the backend predictable while the product evolves. Preserve the API as a stable decision layer between frontend feature extraction and trained model artifacts, and avoid backend changes that silently desynchronize training, inference, and UI behavior.

Read [references/backend-contract.md](references/backend-contract.md) when you need the exact endpoint expectations, feature alignment rules, or safe refactor targets based on the current backend.

## Backend Role

Treat the backend as the inference decision engine:

- Load trained model artifacts.
- Validate posture feature payloads.
- Produce stable posture classifications.
- Return concise machine-readable output plus brief human feedback.

Do not let the backend become a duplicate frontend or a second training script.

## Core Priorities

When making backend changes, optimize in this order:

1. Preserve the prediction contract.
2. Keep feature names aligned with training metadata and frontend payloads.
3. Fail clearly when artifacts or inputs are invalid.
4. Keep inference behavior stable and explainable.
5. Add backend structure only when it reduces risk or confusion.

## Endpoint Rules

Keep the API small and explicit.

- `GET /health` should remain deterministic and trivial.
- `POST /predict` is the main product endpoint and must stay stable.
- New endpoints should only be added when they serve a clear operational need such as diagnostics, model metadata, or version reporting.

Do not introduce backend surface area that the product does not use.

## Schema Rules

- Keep request fields explicit with Pydantic models.
- Use non-negative validation where the feature semantics require it.
- Keep response fields stable: `label`, `confidence`, `feedback`, and `probabilities` when available.
- Preserve the canonical label vocabulary: `proper` and `needs_correction`.
- If adding fields, ensure the frontend can ignore them safely or update the frontend in the same change.

## Feature Alignment Rules

The backend must stay aligned with:

1. frontend payload generation
2. `backend/train.py` feature order
3. `backend/model/model_meta.json`

If the feature list changes, update all three layers together. Do not rely on memory or comments alone; verify the actual feature names in code and metadata.

## Model Loading Rules

- Load artifacts on startup.
- Fail early with a clear error if the model file is missing.
- Prefer metadata-driven feature order when metadata is present.
- Keep artifact paths explicit and local to the backend directory structure.
- When adding model versioning, make the active version discoverable without changing the prediction shape.

## Prediction Behavior

Inference behavior can include policy, not just raw model output.

- Gating logic is acceptable when it prevents overconfident `proper` classifications.
- Smoothing and vote windows are acceptable when they improve live-session stability.
- Feedback mapping should stay short and tied to dominant posture issues.
- Do not bury product behavior in opaque heuristics; keep helpers readable and isolated.

## Refactor Guidance

If the backend grows, split by responsibility:

- schema models
- model loading and artifact utilities
- prediction service logic
- route definitions
- feedback helpers

Do not split a small backend into many files without reducing actual complexity.

## Operational Rules

- Keep CORS aligned with local development and deployed frontend origins.
- Return `503` or another clear operational error when the model is unavailable.
- Avoid broad exception swallowing around prediction code.
- Keep logs minimal and useful; do not log sensitive or unnecessary payload detail by default.

## Review Checklist

When reviewing backend changes, verify:

1. Does `POST /predict` still accept the expected posture payload?
2. Does the feature order still match training and metadata?
3. Does startup behavior fail clearly if artifacts are missing?
4. Are label, confidence, and feedback semantics still stable?
5. Did smoothing or gating behavior change user-visible outcomes?
6. Are CORS and environment assumptions still correct?
7. Did the change avoid expanding backend scope unnecessarily?

## Resources

### references/backend-contract.md

Read this file for the current API contract, feature list, and backend-specific anti-patterns.
