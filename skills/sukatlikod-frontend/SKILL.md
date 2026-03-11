---
name: sukatlikod-frontend
description: Frontend guidance for the SukatLikod React and Vite application, including webcam capture UI, MediaPipe integration in the browser, live posture-analysis states, metric cards, session logs, calibration controls, and CSS structure. Use when Codex is editing `src/` files, refactoring the posture monitoring interface, improving responsive behavior, cleaning up `App.tsx`, refining result presentation, or aligning frontend behavior with the SukatLikod workflow contract.
---

# Sukatlikod Frontend

## Overview

Use this skill to keep the frontend understandable while the product remains real-time and state-heavy. Favor clean UI state boundaries, predictable posture result presentation, and incremental extraction from large components instead of adding more behavior directly into one file.

Read [references/frontend-patterns.md](references/frontend-patterns.md) when you need concrete direction for component boundaries, UI states, and CSS rules derived from the current app.

## Current Reality

The current frontend has these constraints:

- `src/App.tsx` currently carries camera setup, MediaPipe processing, feature extraction, feedback logic, rendering, and control UI.
- `src/index.css` provides global page shell rules.
- The live UI uses many utility classes directly in JSX.
- The product includes dense transient states: idle, loading, camera active, weak tracking, calibration, prediction, and error handling.

Work with those constraints pragmatically. Prefer staged cleanup over a large rewrite.

## Frontend Priorities

When making frontend changes, optimize in this order:

1. Preserve the capture-to-feedback workflow.
2. Keep UI states explicit and legible.
3. Prevent `App.tsx` from growing further when extraction is feasible.
4. Keep result panels interpretable.
5. Maintain responsive behavior on desktop and mobile.
6. Improve polish only after state and structure are sound.

## State Model

Model the UI around distinct states, not loose booleans spread everywhere.

- Idle: session not started, camera inactive, prompt user to begin.
- Loading: camera or landmarker is initializing.
- Live tracking: landmarks and metrics are updating normally.
- Calibration: user is being guided to hold a supported pose.
- Weak tracking: session is active but landmarks are unreliable.
- Predicting: frontend is sending or smoothing inference results.
- Error: camera denied, model failed to load, or backend unavailable.

If a UI element behaves differently across states, centralize the decision instead of scattering conditional strings and classes.

## Component Boundaries

When refactoring, extract by responsibility:

- Camera stage: video, canvas overlay, active/idle overlay.
- Session chrome: header, start or stop controls, live badges.
- Results panel: score, label, feedback, confidence.
- Metrics strip: trunk angle, head forward, shoulder tilt, derived indicators.
- Session log: time-ordered feedback history.
- Calibration panel: thresholds, audio mode, reset actions.

Do not extract tiny presentational fragments with no reuse value. Extract pieces that reduce cognitive load or isolate state transitions.

## React Guidance

- Keep long-running browser resources in refs and lifecycle-aware effects.
- Keep real-time loop code separate from presentational JSX when possible.
- Prefer derived helpers for posture labels, score colors, and metric display logic.
- Avoid mixing feature-computation math with card rendering if extraction is practical.
- If a function is domain logic rather than view logic, consider moving it to a frontend utility module.

## CSS Guidance

Use `src/index.css` for global shell behavior only:

- viewport sizing
- root layout
- body background
- global reset rules

If shared UI styling grows beyond inline utility classes, move repeated patterns into a dedicated CSS file or component-level abstraction instead of bloating global CSS.

Do not put feature-specific posture styles into `index.css` unless they truly affect the whole app shell.

## Result Presentation Rules

- Show actionable feedback before technical detail.
- Keep the main score secondary to the plain-language posture result.
- Use warning language for `needs_correction`, not alarmist language.
- Treat confidence as supporting context, not the headline.
- Surface tracking quality and stability separately from posture quality.

## Responsive Rules

- Preserve a usable mobile path even if desktop remains the primary experience.
- Avoid panels that permanently consume viewport width on small screens.
- Keep primary session controls reachable with one thumb on mobile.
- Do not hide essential posture status behind hover-only interactions.

## Review Checklist

When reviewing a frontend change, verify:

1. Does the UI still match the SukatLikod workflow contract?
2. Are session states explicit and understandable?
3. Did the change reduce or increase pressure on `App.tsx`?
4. Are camera, tracking, and backend failures still visible to the user?
5. Is the posture result clear without reading raw metrics?
6. Does the layout still work on narrow screens?
7. Did CSS remain scoped and intentional?

## Resources

### references/frontend-patterns.md

Read this file for concrete refactor targets and frontend conventions derived from the current codebase.
