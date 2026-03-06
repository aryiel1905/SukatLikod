# Sukat-Likod Project Goal

## What We Are Building
Sukat-Likod is a camera-based, machine learning system that automatically assesses back posture from pose sequences and gives real-time corrective feedback.

## Core Goal
Build a practical prototype that can:
- Capture live posture data from a webcam feed.
- Extract upper-body pose keypoints and compute posture features (for example trunk inclination, head-forward distance, shoulder alignment, and temporal stability).
- Classify posture states as `proper` or `needs correction` using a supervised ML model.
- Show clear, immediate guidance messages tied to detected posture issues.

## Why This Matters
The system targets early posture correction during study/work sessions to reduce prolonged poor posture behavior linked to low back pain and musculoskeletal strain.

## Product Direction for This Repository
- Real-time posture monitoring interface.
- Session controls (`start/stop`) and live status indicators.
- Interpretable feedback instead of opaque scores.
- Privacy-aware processing by default (local frame processing, minimal non-identifying logs).
- Validation focus on accuracy, precision, recall, F1, and robustness under realistic camera/lighting conditions.
