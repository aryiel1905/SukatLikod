# Dataset Rules

## Current Dataset Scripts

The repo currently includes these dataset preparation scripts:

- `backend/prepare_reference_dataset.py`
- `backend/prepare_archives_dataset.py`
- `backend/prepare_balanced_dataset.py`

Prefer using or adapting these over manual CSV editing.

## Current Dataset Variants

Examples of generated dataset files already present in `backend/data/`:

- `posture_dataset.csv`
- `posture_dataset.sample.csv`
- `posture_dataset.from_reference.csv`
- `posture_dataset.from_archives.csv`
- `posture_dataset.balanced.csv`

Treat these as distinct artifacts with different provenance. Do not assume they are interchangeable without checking schema and label counts.

## Current Training Schema

The active training code expects:

- `trunk_angle`
- `head_forward`
- `shoulder_tilt`
- `trunk_variance`
- `neck_forward_contour`
- `upper_back_curvature`
- `torso_outline_angle`
- `silhouette_stability`
- `label`

This is stricter than some simplified project notes. When in doubt, trust the current training code.

## Known Source Label Mappings

From `prepare_reference_dataset.py`:

- `TUP` -> `proper`
- `TLF` -> `needs_correction`
- `TLB` -> `needs_correction`
- `TLR` -> `needs_correction`
- `TLL` -> `needs_correction`

From `prepare_archives_dataset.py`:

- `[correct_posture]` -> `proper`
- `[INcorrect_posture]` -> `needs_correction`

If a new dataset source is added, define mappings explicitly.

## Important Mismatch To Watch

Some preparation paths currently emit only the reduced feature subset plus `label`, while `backend/train.py` expects the full eight-feature schema.

That means dataset work must answer one of these explicitly:

1. compute the missing features
2. backfill them deliberately
3. update training expectations
4. reject the dataset variant for current training

Do not leave this implicit.

## Balancing Guidance

The balancing script currently enforces a configurable majority-to-minority ratio.

When using it, record:

- input files
- pre-balance class counts
- post-balance class counts
- chosen ratio
- random seed

## Evaluation Minimum

Before claiming improvement, check:

- class distribution
- held-out classification report
- minority-class precision and recall
- whether the comparison used the same split assumptions

If those are not controlled, the result is not strong enough to call an improvement.

## Anti-Patterns

Avoid:

- mixing hand-edited and generated rows without documenting it
- changing labels and balancing in one step without separate evaluation
- treating placeholder zero-filled features as equivalent to measured features
- comparing metrics from different dataset schemas as if they were the same experiment
- assuming more rows automatically means better data
