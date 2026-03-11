---
name: sukatlikod-dataset-eval
description: Dataset preparation and evaluation guidance for SukatLikod posture classification, including source-data conversion, label mapping, schema validation, class balancing, feature consistency, and interpretation of training metrics. Use when Codex is editing dataset scripts in `backend/`, reviewing CSV schema changes, preparing new posture datasets, reconciling label definitions, comparing dataset variants, or assessing whether a model-training change reflects real improvement rather than data drift.
---

# Sukatlikod Dataset Eval

## Overview

Use this skill to keep dataset work reproducible and evaluation claims credible. Prefer explicit schema checks, stable label definitions, and careful comparison of dataset variants over ad hoc CSV edits or vague accuracy discussions.

Read [references/dataset-rules.md](references/dataset-rules.md) when you need the current dataset sources, label mappings, or safe evaluation workflow based on this repo.

## Current Reality

The repo already contains:

- multiple dataset conversion scripts
- multiple CSV variants in `backend/data/`
- a training pipeline expecting eight posture features
- older docs that sometimes describe a smaller minimum schema

Treat dataset work as an integration problem, not just a CSV-editing task.

## Priorities

When making dataset or evaluation changes, optimize in this order:

1. Preserve label meaning.
2. Preserve training-schema compatibility.
3. Avoid hidden class imbalance or leakage.
4. Compare changes with explicit metrics.
5. Keep dataset lineage understandable.

## Source Data Rules

Prefer generating datasets through the existing scripts rather than hand-editing CSV files.

- Use reference conversion scripts when source material comes from `Reference/data.csv` or `Reference/archives_data/...`.
- Use balancing scripts when class ratio is the issue, not feature computation.
- Keep generated dataset outputs in `backend/data/`.
- If adding a new source pipeline, document how its labels map to `proper` and `needs_correction`.

## Schema Rules

The authoritative training schema currently includes:

- `trunk_angle`
- `head_forward`
- `shoulder_tilt`
- `trunk_variance`
- `neck_forward_contour`
- `upper_back_curvature`
- `torso_outline_angle`
- `silhouette_stability`
- `label`

Do not assume the smaller four-feature examples in older notes are sufficient for the current training code. Verify the real schema in `backend/train.py`.

## Label Rules

- Keep the final training labels binary for MVP:
  - `proper`
  - `needs_correction`
- When raw source labels are more granular, map them explicitly and consistently.
- Do not mix different label semantics into the same final class without noting the tradeoff.
- If a source has uncertain labels, prefer excluding low-trust rows over pretending they are clean ground truth.

## Balancing Rules

- Check class counts before and after balancing.
- Balance by controlled sampling, not by silently dropping random rows.
- Record which inputs were merged and what ratio was targeted.
- Do not confuse balancing with evaluation improvement; balanced training data can still generalize poorly.

## Evaluation Rules

- Report at least class-aware metrics, not just a single accuracy number.
- Compare held-out performance before claiming improvement.
- Watch for regressions in minority-class recall.
- Treat label remapping, feature changes, and balancing changes as separate causes until verified.
- If evaluation data overlaps too closely with training generation logic, call out leakage risk explicitly.

## Feature Engineering Rules

- If a new feature is added to datasets, update:
  1. dataset-generation scripts
  2. training feature list
  3. model metadata
  4. backend prediction schema
  5. frontend payload generation when relevant
- If a feature is unavailable for one source dataset, do not silently omit the column. Decide whether to backfill with an explicit default, compute it, or reject the dataset for current training.

## Review Checklist

When reviewing dataset or evaluation work, verify:

1. Does the dataset match the actual training schema?
2. Are labels mapped consistently into `proper` and `needs_correction`?
3. Are class counts and balancing choices explicit?
4. Is there a clear lineage from source files to generated CSVs?
5. Are evaluation claims supported by held-out metrics?
6. Did the change introduce leakage, duplication, or schema drift?
7. If features changed, were all dependent layers updated?

## Resources

### references/dataset-rules.md

Read this file for the current source pipelines, label mappings, and evaluation anti-patterns in this repo.
