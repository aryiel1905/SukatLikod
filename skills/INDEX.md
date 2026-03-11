# Skills Index

This file explains which SukatLikod skill to use for which kind of task.

## Recommended Order

Use the skills in this order when a task spans multiple areas:

1. `sukatlikod-product-workflow`
2. `sukatlikod-frontend`
3. `sukatlikod-backend-fastapi`
4. `sukatlikod-dataset-eval`
5. `sukatlikod-trainer`

Reason:

- `sukatlikod-product-workflow` defines the end-to-end product contract.
- The frontend and backend skills implement that contract.
- The dataset/eval and trainer skills protect model quality and compatibility.

## Which Skill To Use

### `sukatlikod-product-workflow`

Use this first when the task affects overall product behavior across layers.

Examples:

- changing the capture-to-feedback flow
- changing posture result semantics
- changing what the frontend, backend, and model must agree on

Entry point:

- [sukatlikod-product-workflow/SKILL.md](c:/Users/Aryiel Joshua/OneDrive/Desktop/SukatLikod/skills/sukatlikod-product-workflow/SKILL.md)

### `sukatlikod-frontend`

Use this when the task is mostly in `src/`.

Examples:

- refactoring `App.tsx`
- improving webcam UI states
- cleaning up CSS structure
- improving result cards, logs, calibration panels, and responsive behavior

Entry point:

- [sukatlikod-frontend/SKILL.md](c:/Users/Aryiel Joshua/OneDrive/Desktop/SukatLikod/skills/sukatlikod-frontend/SKILL.md)

### `sukatlikod-backend-fastapi`

Use this when the task is mostly in `backend/api.py` or inference-serving behavior.

Examples:

- changing `POST /predict`
- changing model loading
- changing response schema
- reviewing CORS or backend startup behavior

Entry point:

- [sukatlikod-backend-fastapi/SKILL.md](c:/Users/Aryiel Joshua/OneDrive/Desktop/SukatLikod/skills/sukatlikod-backend-fastapi/SKILL.md)

### `sukatlikod-dataset-eval`

Use this when the task is about datasets, labels, balancing, or evaluation quality.

Examples:

- preparing datasets from reference sources
- checking label mappings
- balancing class ratios
- comparing dataset variants
- reviewing whether a reported model improvement is trustworthy

Entry point:

- [sukatlikod-dataset-eval/SKILL.md](c:/Users/Aryiel Joshua/OneDrive/Desktop/SukatLikod/skills/sukatlikod-dataset-eval/SKILL.md)

### `sukatlikod-trainer`

Use this when the task is about `backend/train.py` and model artifact generation.

Examples:

- changing the feature list
- changing the estimator or preprocessing pipeline
- changing artifact metadata
- reviewing training-time schema validation

Entry point:

- [sukatlikod-trainer/SKILL.md](c:/Users/Aryiel Joshua/OneDrive/Desktop/SukatLikod/skills/sukatlikod-trainer/SKILL.md)

## Common Routing

- UI-only change: `sukatlikod-frontend`
- API-only change: `sukatlikod-backend-fastapi`
- Training-only change: `sukatlikod-trainer`
- Dataset or metrics question: `sukatlikod-dataset-eval`
- Anything cross-layer: start with `sukatlikod-product-workflow`, then add the specific implementation skill

## Current Important Repo Reality

The live code currently treats the eight-feature training schema as authoritative, even though some older docs still mention a reduced four-feature schema.

For tasks involving model inputs, trust:

1. `backend/train.py`
2. `backend/model/model_meta.json`
3. `backend/api.py`

Only trust older simplified docs if the code is being intentionally updated to match them.
