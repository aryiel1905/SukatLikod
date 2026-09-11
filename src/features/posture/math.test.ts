import { describe, expect, it } from "vitest";
import {
  average,
  clamp,
  metricQuality,
  stabilityFromVariance,
  variance,
} from "./math";

describe("posture math", () => {
  it("clamps values to the supported interval", () => {
    expect(clamp(-4, 0, 100)).toBe(0);
    expect(clamp(42, 0, 100)).toBe(42);
    expect(clamp(140, 0, 100)).toBe(100);
  });

  it("computes averages and population variance", () => {
    expect(average([])).toBeNull();
    expect(average([2, 4, 6])).toBe(4);
    expect(variance([2, 4, 6])).toBeCloseTo(8 / 3);
  });

  it("maps stable sequences to higher scores", () => {
    expect(stabilityFromVariance(0)).toBe(100);
    expect(stabilityFromVariance(1)).toBe(65);
    expect(stabilityFromVariance(10)).toBe(0);
  });

  it("maps threshold-relative metrics without exceeding bounds", () => {
    expect(metricQuality(10, 10)).toBe(100);
    expect(metricQuality(20, 10)).toBe(30);
    expect(metricQuality(100, 10)).toBe(0);
    expect(metricQuality(10, 0)).toBe(0);
  });
});
