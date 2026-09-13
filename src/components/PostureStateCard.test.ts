import { describe, expect, it } from "vitest";
import { resolvePostureState } from "../features/posture/postureState";

describe("resolvePostureState", () => {
  it("keeps inactive, loading, and camera error states distinct", () => {
    expect(
      resolvePostureState({
        status: "idle",
        isActive: false,
        metricsPaused: true,
        score: 0,
      }),
    ).toBe("inactive");
    expect(
      resolvePostureState({
        status: "loading",
        isActive: false,
        metricsPaused: true,
        score: 0,
      }),
    ).toBe("analyzing");
    expect(
      resolvePostureState({
        status: "error",
        isActive: false,
        metricsPaused: true,
        score: 0,
      }),
    ).toBe("unavailable");
  });

  it("shows neutral only after a confirmed good result", () => {
    expect(
      resolvePostureState({
        status: "detecting",
        isActive: true,
        metricsPaused: true,
        score: 100,
      }),
    ).toBe("analyzing");
    expect(
      resolvePostureState({
        status: "good",
        isActive: true,
        metricsPaused: false,
        score: 100,
      }),
    ).toBe("neutral");
  });

  it("separates mild and severe corrections at 60 points", () => {
    expect(
      resolvePostureState({
        status: "fix",
        isActive: true,
        metricsPaused: false,
        score: 61,
      }),
    ).toBe("mild");
    expect(
      resolvePostureState({
        status: "fix",
        isActive: true,
        metricsPaused: false,
        score: 60,
      }),
    ).toBe("severe");
  });
});
