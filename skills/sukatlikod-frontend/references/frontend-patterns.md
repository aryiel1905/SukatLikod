# Frontend Patterns

## Current Hotspots

These areas deserve special attention in the current frontend:

- `src/App.tsx`: oversized, mixes domain math, browser APIs, async inference, and JSX layout
- `src/index.css`: should stay limited to app-shell and reset responsibilities
- `src/App.css`: useful as a staging area for repeated UI patterns if utility-class duplication starts growing

## Recommended Extraction Order

If refactoring incrementally, split in this order:

1. extract posture math and display helpers
2. extract session log and metric cards
3. extract calibration panel
4. extract camera stage and overlays
5. extract session controller hook if loop complexity keeps growing

This order reduces JSX weight first, then isolates behavior.

## Suggested Feature Areas

When structure expansion is justified, prefer folders like:

- `src/features/posture/`
- `src/features/session/`
- `src/components/`
- `src/lib/`

Keep domain-specific logic out of a generic `utils` dump.

## State Separation

Keep these concerns conceptually separate even if they temporarily live in one file:

- media permissions and stream lifecycle
- MediaPipe model loading
- pose tracking quality
- buffered metric computation
- backend prediction state
- UI panel visibility
- audio feedback rules

Avoid a single blob state object unless the transitions are explicitly modeled.

## UI Behavior Rules

- Idle state should explain the next action.
- Loading state should communicate what is loading.
- Weak tracking state should explain whether the user should hold still, reframe, or improve visibility.
- Feedback log entries should remain short and scannable.
- Calibration controls should not overpower the main session workflow.

## Design Direction

Keep the look intentional and instrument-like:

- dark operational canvas
- bright contrast for critical posture status
- muted chrome around secondary controls
- limited accent colors with clear meaning

Prefer consistency over adding more visual effects.

## Anti-Patterns

Avoid:

- adding more domain logic directly inside large JSX returns
- duplicating threshold logic in multiple UI sections
- mixing transport failures with posture warnings
- using one score to represent tracking, confidence, and posture all at once
- turning `index.css` into a dump for component styling

## Done Criteria For Frontend Changes

A frontend change is in good shape when:

- the main session path is easier to follow than before
- result messaging is clearer
- component or helper extraction reduces cognitive load
- responsive behavior does not regress
- workflow contract terms stay consistent with backend and product rules
