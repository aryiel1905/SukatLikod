# SukatLikod Thesis-Alignment Review Prompt for Codex

Use this file to evaluate whether the current SukatLikod project fits the thesis direction below.

## Thesis-Aligned Technical Direction

The project should stay closely aligned with this direction:

1. Use **MediaPipe Pose / BlazePose** as the primary pose estimation front-end.
2. Use **short pose sequences** as the input representation instead of relying on only a single frame.
3. Compute **clinically inspired posture features**, such as:
   - trunk inclination
   - shoulder levelness
   - hip levelness
   - head/neck alignment
   - temporal stability
4. Train or use a **supervised ML classifier** on those extracted features.
5. Add **orientation/view detection** so the system knows whether the subject is in:
   - front view
   - side view
   - back view

The system should then apply the correct posture rules depending on the detected view.

## Thesis Scope Guidance

The project should be framed as:

**guided multi-view back posture assessment from monocular RGB pose sequences**

It should **not** be framed as:

**unconstrained any-angle posture detection**

## Model Recommendation

For thesis alignment, the recommended primary model choice is:

- **MediaPipe Pose Landmarker Full or Heavy**

If higher accuracy is being explored, the safest thesis-aligned approach is:

- keep **MediaPipe / BlazePose** as the primary pose estimator
- optionally compare it against **RTMPose** only as an experimental benchmark

Important note:

If the project fully switches to **RTMPose** or **ViTPose** as the main core, the thesis may shift away from:
- accessible deployment
- consumer hardware suitability
- lightweight practical usability

and more toward:
- accuracy-first backend inference
- heavier research-oriented architecture

## What Codex Should Check

Review the current codebase and determine:

1. Does the project currently use **MediaPipe Pose / BlazePose** as the main pose estimator?
2. If yes, is it using a thesis-aligned version such as **Full** or **Heavy**, rather than only a lightweight version?
3. Does the project process **short pose sequences** or only individual frames?
4. Does it compute posture features similar to:
   - trunk inclination
   - shoulder levelness
   - hip levelness
   - head/neck alignment
   - temporal stability
5. Does it include **orientation/view detection** for front, side, and back?
6. Does it apply **different posture logic depending on view**, or is it still mainly front-only?
7. Does the current system architecture still fit the thesis framing of:
   - guided capture
   - monocular RGB input
   - posture assessment from pose sequences
8. Is anything in the current project pushing it away from the thesis scope?
9. What parts already align well?
10. What parts should be changed, deferred, or removed to stay thesis-aligned?

## Requested Output Format

Please return the review in this structure:

### 1. Overall Verdict
- Is the current project aligned with the thesis direction?
- Fully aligned / partially aligned / misaligned

### 2. What Already Fits
List the current parts of the project that already match the thesis direction.

### 3. What Partially Fits
List the parts that are present but need adjustment.

### 4. What Is Missing
List the thesis-aligned requirements not yet implemented.

### 5. What May Conflict With the Thesis
List any design choices that could shift the project away from the intended thesis scope.

### 6. Recommended Next Steps
Give the best next implementation steps to make the current project match the thesis direction more closely.

## Extra Instruction for Codex

Do not judge the project based on “most advanced possible posture AI.”
Judge it based on whether it fits this thesis goal:

**a practical, guided, monocular RGB, thesis-aligned posture assessment system using MediaPipe-centered pose extraction, short pose sequences, extracted posture features, supervised classification, and orientation-aware rules.**
