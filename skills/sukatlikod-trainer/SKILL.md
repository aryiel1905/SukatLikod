---
name: sukatlikod-trainer
description: Training workflow guidance for the SukatLikod posture classifier, including dataset validation, feature-list management, model pipeline changes, artifact generation, metadata updates, and interpretation of training output. Use when Codex is editing `backend/train.py`, changing model features or estimators, adjusting train/test split behavior, reviewing retraining steps, or verifying that new model artifacts remain compatible with backend inference and frontend payload expectations.
---

# Sukatlikod Trainer

## Overview

Use this skill to keep model training changes controlled and compatible with the rest of the product. Treat `backend/train.py` as the authoritative training entry point and assume that any feature-list change is a cross-layer change until proven otherwise.

Read [references/training-rules.md](references/training-rules.md) when you need the active feature list, artifact expectations, or safe experiment workflow based on the current repo.

## Trainer Role

Treat the trainer as responsible for:

- validating training data schema
- defining feature order
- fitting the model pipeline
- reporting held-out metrics
- exporting model and metadata artifacts

Do not overload training code with frontend concerns, API serving concerns, or dataset collection workflows.

## Priorities

When modifying training logic, optimize in this order:

1. Preserve feature-schema correctness.
2. Preserve artifact compatibility with inference.
3. Preserve evaluation credibility.
4. Keep model behavior explainable enough for the current product.
5. Increase complexity only when it is justified by data or evaluation results.

## Feature List Rules

Treat the feature list in `backend/train.py` as authoritative for training.

- Keep feature order stable unless there is a deliberate change.
- If a feature is added, removed, or renamed, update:
  1. dataset generation paths
  2. training validation
  3. model metadata output
  4. backend request handling
  5. frontend payload generation where relevant
- Do not rely on outdated docs that mention a reduced schema if the code expects more.

## Dataset Validation Rules

- Validate required columns before fitting.
- Validate that at least two classes are present.
- Prefer explicit failure over quietly training on malformed data.
- If new data checks are added, keep them deterministic and easy to interpret.

## Model Pipeline Rules

- Keep the pipeline explicit.
- Make estimator changes visible in metadata.
- Do not add model complexity without a comparison against the current baseline.
- Preserve reproducibility where practical through fixed random seeds.
- If preprocessing changes, ensure inference artifacts still work with the backend load path.

## Evaluation Rules

- Use held-out evaluation, not just training fit.
- Report class-aware metrics through the classification report or stronger equivalents.
- Treat class imbalance as a first-class issue when reading results.
- If metrics improve after schema or label changes, do not attribute the gain to the model alone.
- If split assumptions change, note that comparisons to earlier results are weaker.

## Artifact Rules

- Save a model artifact and metadata artifact together.
- Keep artifact paths explicit.
- Metadata should include, at minimum:
  - feature list
  - class list
  - model type
- Do not ship a new model artifact without metadata that reflects the same training run.

## Refactor Guidance

- Keep the training entry point easy to run from the command line.
- Split helper functions only when it clarifies validation, feature handling, or artifact export.
- Avoid turning a short training script into a framework without clear payoff.
- If experiment tracking grows, add it deliberately rather than scattering print statements and one-off files.

## Safe Change Pattern

When changing training behavior:

1. inspect the active dataset schema
2. inspect the current feature list
3. change training code
4. update metadata export if needed
5. verify backend inference compatibility
6. verify frontend payload compatibility if features changed
7. review held-out metrics before claiming improvement

## Review Checklist

When reviewing training changes, verify:

1. Does the dataset validation match the actual required schema?
2. Does the feature list still align with backend and frontend expectations?
3. Did the model pipeline change in a deliberate, reviewable way?
4. Are artifacts and metadata written together?
5. Are metric comparisons fair given the data and split assumptions?
6. Did the change improve clarity or only add complexity?
7. Can the training command still be run straightforwardly?

## Resources

### references/training-rules.md

Read this file for the current training baseline, artifact contract, and common trainer anti-patterns in this repo.
