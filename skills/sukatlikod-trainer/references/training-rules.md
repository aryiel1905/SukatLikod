# Training Rules

## Active Training Entry Point

Current authoritative training script:

- `backend/train.py`

Treat this file as the baseline unless the repo is restructured.

## Active Feature List

The current training code expects these features in this order:

- `trunk_angle`
- `head_forward`
- `shoulder_tilt`
- `trunk_variance`
- `neck_forward_contour`
- `upper_back_curvature`
- `torso_outline_angle`
- `silhouette_stability`

This order matters because the backend prediction path reads model metadata and constructs rows accordingly.

## Label Column

Current label column:

- `label`

Current class vocabulary:

- `proper`
- `needs_correction`

## Baseline Model Setup

Current baseline behavior in `backend/train.py`:

- train/test split with stratification
- `StandardScaler`
- `RandomForestClassifier`
- fixed random seed
- classification report output

Treat this as the baseline to beat, not just code to replace casually.

## Artifact Contract

Current expected outputs:

- `backend/model/posture_model.joblib`
- `backend/model/model_meta.json`

Current metadata includes:

- `features`
- `classes`
- `model_type`

If training output changes, keep inference compatibility in view.

## Important Mismatch To Watch

Some repo documentation still describes a smaller four-feature dataset:

- `trunk_angle`
- `head_forward`
- `shoulder_tilt`
- `trunk_variance`

But the current training code expects eight posture features.

Trust the current code and artifacts unless the task explicitly updates the training contract.

## Safe Experiment Guidance

When trying a new model or feature set:

1. keep the old baseline measurable
2. change one major variable at a time when practical
3. record what changed
4. compare held-out metrics
5. verify backend compatibility before adopting the new artifact

## Anti-Patterns

Avoid:

- changing the feature list without updating metadata
- saving artifacts without enough metadata to reconstruct expectations
- calling a model better based only on one metric or one split
- making estimator changes and dataset changes at the same time without separating causes
- assuming the frontend will adapt automatically to new training outputs
