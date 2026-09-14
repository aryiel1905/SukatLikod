import { describe, expect, it } from "vitest";
import {
  extractRandomForestFeatures,
  RANDOM_FOREST_FEATURE_NAMES,
  validateRandomForestFrame,
} from "./randomForestFeatures";

const symmetricPoints = {
  LS: { x: 0.3, y: 0.7 },
  RS: { x: 0.7, y: 0.7 },
  N: { x: 0.5, y: 0.3 },
  C: { x: 0.5, y: 0.5 },
  LE: { x: 0.44, y: 0.35 },
  RE: { x: 0.56, y: 0.35 },
  LA: { x: 0.38, y: 0.4 },
  RA: { x: 0.62, y: 0.4 },
};

describe("extractRandomForestFeatures", () => {
  it("reproduces the uprightly_8point_v1 normalized geometry", () => {
    const result = extractRandomForestFeatures(symmetricPoints);

    expect(result).not.toBeNull();
    expect(Object.keys(result ?? {})).toEqual([...RANDOM_FOREST_FEATURE_NAMES]);
    expect(result?.shoulder_tilt_angle_degrees).toBeCloseTo(0);
    expect(result?.eye_tilt_angle_degrees).toBeCloseTo(0);
    expect(result?.ear_tilt_angle_degrees).toBeCloseTo(0);
    expect(result?.head_axis_tilt_from_vertical_degrees).toBeCloseTo(0);
    expect(result?.eye_span_to_shoulder_ratio).toBeCloseTo(0.3);
    expect(result?.ear_span_to_shoulder_ratio).toBeCloseTo(0.6);
    expect(result?.nose_chin_distance_to_shoulder_ratio).toBeCloseTo(0.5);
    expect(
      result?.nose_horizontal_offset_from_shoulders_normalized_signed,
    ).toBeCloseTo(0);
    expect(result?.nose_vertical_offset_from_shoulders_normalized_signed).toBeCloseTo(
      -1,
    );
    expect(
      result?.ear_shoulder_symmetry_difference_normalized_signed,
    ).toBeCloseTo(0);
  });

  it("preserves signed tilt and offset direction", () => {
    const result = extractRandomForestFeatures({
      ...symmetricPoints,
      RS: { x: 0.7, y: 0.74 },
      N: { x: 0.54, y: 0.3 },
    });

    expect(result?.shoulder_tilt_angle_degrees).toBeGreaterThan(0);
    expect(
      result?.shoulder_height_difference_normalized_signed,
    ).toBeLessThan(0);
    expect(
      result?.nose_horizontal_offset_from_shoulders_normalized_signed,
    ).toBeGreaterThan(0);
  });

  it("rejects invalid or unusably narrow shoulder geometry", () => {
    expect(
      extractRandomForestFeatures({
        ...symmetricPoints,
        RS: { x: 0.31, y: 0.7 },
      }),
    ).toBeNull();
    expect(
      extractRandomForestFeatures({
        ...symmetricPoints,
        N: { x: Number.NaN, y: 0.3 },
      }),
    ).toBeNull();
  });

  it("accepts posture tilt while retaining the canonical eight-point topology", () => {
    const result = validateRandomForestFrame({
      ...symmetricPoints,
      LS: { x: 0.3, y: 0.76 },
      RS: { x: 0.7, y: 0.68 },
      LE: { x: 0.44, y: 0.31 },
      RE: { x: 0.56, y: 0.39 },
      N: { x: 0.5, y: 0.37 },
      C: { x: 0.5, y: 0.59 },
    });

    expect(result).toEqual({ valid: true });
  });

  it("only rejects unusable capture geometry, not a posture the model should classify", () => {
    expect(
      validateRandomForestFrame({
        ...symmetricPoints,
        RE: { x: 1.3, y: 0.35 },
      }),
    ).toEqual({ valid: false, issue: "outside_frame" });

    expect(
      validateRandomForestFrame({
        ...symmetricPoints,
        C: { x: 0.5, y: 0.28 },
      }),
    ).toEqual({ valid: true });
  });
});
