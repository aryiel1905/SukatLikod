export const RANDOM_FOREST_FEATURE_NAMES = [
  "shoulder_tilt_angle_degrees",
  "shoulder_height_difference_normalized_signed",
  "eye_tilt_angle_degrees",
  "ear_tilt_angle_degrees",
  "head_axis_tilt_from_vertical_degrees",
  "eye_span_to_shoulder_ratio",
  "ear_span_to_shoulder_ratio",
  "nose_chin_distance_to_shoulder_ratio",
  "left_ear_to_shoulder_distance_ratio",
  "right_ear_to_shoulder_distance_ratio",
  "ear_shoulder_symmetry_difference_normalized_signed",
  "nose_horizontal_offset_from_shoulders_normalized_signed",
  "nose_vertical_offset_from_shoulders_normalized_signed",
  "chin_horizontal_offset_from_shoulders_normalized_signed",
  "chin_vertical_offset_from_shoulders_normalized_signed",
  "eye_mid_horizontal_offset_from_shoulders_normalized_signed",
  "eye_mid_vertical_offset_from_shoulders_normalized_signed",
  "ear_mid_horizontal_offset_from_shoulders_normalized_signed",
  "ear_mid_vertical_offset_from_shoulders_normalized_signed",
  "nose_horizontal_offset_from_eye_mid_normalized_signed",
  "nose_vertical_offset_from_eye_mid_normalized_signed",
  "chin_horizontal_offset_from_eye_mid_normalized_signed",
  "chin_vertical_offset_from_eye_mid_normalized_signed",
  "nose_horizontal_offset_from_ear_mid_normalized_signed",
  "nose_vertical_offset_from_ear_mid_normalized_signed",
  "chin_horizontal_offset_from_ear_mid_normalized_signed",
  "chin_vertical_offset_from_ear_mid_normalized_signed",
  "eye_mid_horizontal_offset_from_ear_mid_normalized_signed",
  "eye_mid_vertical_offset_from_ear_mid_normalized_signed",
  "left_eye_to_ear_distance_ratio",
  "right_eye_to_ear_distance_ratio",
  "eye_ear_symmetry_difference_normalized_signed",
  "left_nose_to_ear_distance_ratio",
  "right_nose_to_ear_distance_ratio",
  "nose_ear_symmetry_difference_normalized_signed",
] as const;

export type RandomForestFeatureName =
  (typeof RANDOM_FOREST_FEATURE_NAMES)[number];
export type RandomForestFeatureVector = Record<RandomForestFeatureName, number>;

export type Point2 = { x: number; y: number };

export type RandomForestEightPoints = {
  N: Point2;
  LE: Point2;
  RE: Point2;
  LA: Point2;
  RA: Point2;
  C: Point2;
  LS: Point2;
  RS: Point2;
};

export type RandomForestFrameIssue =
  | "missing_landmark"
  | "outside_frame"
  | "shoulders_too_narrow";

export type RandomForestFrameValidation =
  | { valid: true }
  | { valid: false; issue: RandomForestFrameIssue };

const distance = (a: Point2, b: Point2) => Math.hypot(a.x - b.x, a.y - b.y);
const midpoint = (a: Point2, b: Point2): Point2 => ({
  x: (a.x + b.x) / 2,
  y: (a.y + b.y) / 2,
});
const lineAngle = (a: Point2, b: Point2) =>
  (Math.atan2(b.y - a.y, b.x - a.x) * 180) / Math.PI;

/**
 * Validates whether an eight-point camera frame has the same canonical topology
 * used to construct the trained feature schema. This is deliberately a capture
 * quality check, not a posture classifier: it does not constrain tilt, leaning,
 * or the model's eventual severity label.
 */
export function validateRandomForestFrame(
  points: RandomForestEightPoints,
): RandomForestFrameValidation {
  const allPoints = Object.values(points);
  if (
    allPoints.some(
      (point) => !Number.isFinite(point.x) || !Number.isFinite(point.y),
    )
  ) {
    return { valid: false, issue: "missing_landmark" };
  }

  // MediaPipe can predict slightly outside a frame at the edge, but a point
  // farther away is not a usable camera capture.
  if (
    allPoints.some(
      (point) =>
        point.x < -0.15 || point.x > 1.15 || point.y < -0.15 || point.y > 1.15,
    )
  ) {
    return { valid: false, issue: "outside_frame" };
  }

  const shoulderWidth = distance(points.LS, points.RS);
  if (shoulderWidth < 0.04) {
    return { valid: false, issue: "shoulders_too_narrow" };
  }

  return { valid: true };
}

export function extractRandomForestFeatures(
  points: RandomForestEightPoints,
): RandomForestFeatureVector | null {
  if (!validateRandomForestFrame(points).valid) return null;

  const shoulderWidth = distance(points.LS, points.RS);

  const shoulderMid = midpoint(points.LS, points.RS);
  const eyeMid = midpoint(points.LE, points.RE);
  const earMid = midpoint(points.LA, points.RA);
  const ratio = (value: number) => value / shoulderWidth;
  const xOffset = (point: Point2, reference: Point2) =>
    ratio(point.x - reference.x);
  const yOffset = (point: Point2, reference: Point2) =>
    ratio(point.y - reference.y);

  const leftEarShoulder = distance(points.LA, points.LS);
  const rightEarShoulder = distance(points.RA, points.RS);
  const leftEyeEar = distance(points.LE, points.LA);
  const rightEyeEar = distance(points.RE, points.RA);
  const leftNoseEar = distance(points.N, points.LA);
  const rightNoseEar = distance(points.N, points.RA);

  return {
    shoulder_tilt_angle_degrees: lineAngle(points.LS, points.RS),
    shoulder_height_difference_normalized_signed: ratio(
      points.LS.y - points.RS.y,
    ),
    eye_tilt_angle_degrees: lineAngle(points.LE, points.RE),
    ear_tilt_angle_degrees: lineAngle(points.LA, points.RA),
    head_axis_tilt_from_vertical_degrees:
      (Math.atan2(points.N.x - points.C.x, points.C.y - points.N.y) * 180) /
      Math.PI,
    eye_span_to_shoulder_ratio: ratio(distance(points.LE, points.RE)),
    ear_span_to_shoulder_ratio: ratio(distance(points.LA, points.RA)),
    nose_chin_distance_to_shoulder_ratio: ratio(distance(points.N, points.C)),
    left_ear_to_shoulder_distance_ratio: ratio(leftEarShoulder),
    right_ear_to_shoulder_distance_ratio: ratio(rightEarShoulder),
    ear_shoulder_symmetry_difference_normalized_signed: ratio(
      leftEarShoulder - rightEarShoulder,
    ),
    nose_horizontal_offset_from_shoulders_normalized_signed: xOffset(
      points.N,
      shoulderMid,
    ),
    nose_vertical_offset_from_shoulders_normalized_signed: yOffset(
      points.N,
      shoulderMid,
    ),
    chin_horizontal_offset_from_shoulders_normalized_signed: xOffset(
      points.C,
      shoulderMid,
    ),
    chin_vertical_offset_from_shoulders_normalized_signed: yOffset(
      points.C,
      shoulderMid,
    ),
    eye_mid_horizontal_offset_from_shoulders_normalized_signed: xOffset(
      eyeMid,
      shoulderMid,
    ),
    eye_mid_vertical_offset_from_shoulders_normalized_signed: yOffset(
      eyeMid,
      shoulderMid,
    ),
    ear_mid_horizontal_offset_from_shoulders_normalized_signed: xOffset(
      earMid,
      shoulderMid,
    ),
    ear_mid_vertical_offset_from_shoulders_normalized_signed: yOffset(
      earMid,
      shoulderMid,
    ),
    nose_horizontal_offset_from_eye_mid_normalized_signed: xOffset(
      points.N,
      eyeMid,
    ),
    nose_vertical_offset_from_eye_mid_normalized_signed: yOffset(
      points.N,
      eyeMid,
    ),
    chin_horizontal_offset_from_eye_mid_normalized_signed: xOffset(
      points.C,
      eyeMid,
    ),
    chin_vertical_offset_from_eye_mid_normalized_signed: yOffset(
      points.C,
      eyeMid,
    ),
    nose_horizontal_offset_from_ear_mid_normalized_signed: xOffset(
      points.N,
      earMid,
    ),
    nose_vertical_offset_from_ear_mid_normalized_signed: yOffset(
      points.N,
      earMid,
    ),
    chin_horizontal_offset_from_ear_mid_normalized_signed: xOffset(
      points.C,
      earMid,
    ),
    chin_vertical_offset_from_ear_mid_normalized_signed: yOffset(
      points.C,
      earMid,
    ),
    eye_mid_horizontal_offset_from_ear_mid_normalized_signed: xOffset(
      eyeMid,
      earMid,
    ),
    eye_mid_vertical_offset_from_ear_mid_normalized_signed: yOffset(
      eyeMid,
      earMid,
    ),
    left_eye_to_ear_distance_ratio: ratio(leftEyeEar),
    right_eye_to_ear_distance_ratio: ratio(rightEyeEar),
    eye_ear_symmetry_difference_normalized_signed: ratio(
      leftEyeEar - rightEyeEar,
    ),
    left_nose_to_ear_distance_ratio: ratio(leftNoseEar),
    right_nose_to_ear_distance_ratio: ratio(rightNoseEar),
    nose_ear_symmetry_difference_normalized_signed: ratio(
      leftNoseEar - rightNoseEar,
    ),
  };
}
