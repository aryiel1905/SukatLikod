export function clamp(value: number, minimum: number, maximum: number) {
  return Math.max(minimum, Math.min(maximum, value));
}

export function average(values: number[]) {
  if (values.length === 0) return null;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

export function variance(values: number[]) {
  if (values.length < 2) return 0;
  const mean = average(values) ?? 0;
  return (
    values.reduce((sum, value) => sum + (value - mean) ** 2, 0) /
    values.length
  );
}

export function stabilityFromVariance(trunkVariance: number) {
  return Math.round(clamp(100 - trunkVariance * 35, 0, 100));
}

export function metricQuality(value: number, threshold: number) {
  if (threshold <= 0) return 0;
  const ratio = value / threshold;
  return Math.round(clamp(100 - (ratio - 1) * 70, 0, 100));
}
