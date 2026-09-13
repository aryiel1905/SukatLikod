export type PostureStateKind =
  | "inactive"
  | "analyzing"
  | "neutral"
  | "mild"
  | "severe"
  | "unavailable";

type ResolvePostureStateInput = {
  status: "idle" | "loading" | "detecting" | "good" | "fix" | "error";
  isActive: boolean;
  metricsPaused: boolean;
  score: number;
};

export function resolvePostureState({
  status,
  isActive,
  metricsPaused,
  score,
}: ResolvePostureStateInput): PostureStateKind {
  if (status === "error") return "unavailable";
  if (status === "loading" || (isActive && metricsPaused)) return "analyzing";
  if (!isActive) return "inactive";
  if (status === "good") return "neutral";
  if (status === "fix") return score <= 60 ? "severe" : "mild";
  return "analyzing";
}
