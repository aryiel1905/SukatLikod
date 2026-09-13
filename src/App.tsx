import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
} from "react";
import { createPortal } from "react-dom";
import { useGSAP } from "@gsap/react";
import { gsap } from "gsap";
import {
  Camera,
  VideoOff,
  Settings,
  Activity,
  AlertCircle,
  CheckCircle2,
  ChevronRight,
  Maximize2,
  Bell,
  X,
  PanelRightOpen,
  Moon,
  Sun,
  Monitor,
  Volume2,
  BookOpen,
  ShieldCheck,
} from "lucide-react";
import type {
  FaceLandmarker,
  FaceLandmarkerResult,
  PoseLandmarker,
  PoseLandmarkerResult,
} from "@mediapipe/tasks-vision";
import { MetricCard } from "./components/MetricCard";
import {
  average as avg,
  clamp,
  metricQuality,
  stabilityFromVariance,
  variance,
} from "./features/posture/math";
import "./features/settings/settings-scroll.css";

gsap.registerPlugin(useGSAP);

const IDX = {
  NOSE: 0,
  L_EYE: 2,
  R_EYE: 5,
  L_EAR: 7,
  R_EAR: 8,
  L_ELBOW: 13,
  R_ELBOW: 14,
  L_SHOULDER: 11,
  R_SHOULDER: 12,
  L_HIP: 23,
  R_HIP: 24,
} as const;

const FACE_IDX = {
  NOSE_TIP: 1,
  L_EYE_OUTER: 33,
  R_EYE_OUTER: 263,
  L_MOUTH: 61,
  R_MOUTH: 291,
  CHIN: 152,
} as const;

type Pill = "idle" | "loading" | "detecting" | "good" | "fix" | "error";
type Point3 = { x: number; y: number; z: number };
type FeedbackType = "info" | "warning" | "critical" | "success";
type OrientationKind =
  | "front"
  | "back"
  | "side_left"
  | "side_right"
  | "unknown";
type FrontCaptureTier = "full_front" | "upper_front";
type BaselineMetrics = {
  trunk: number;
  head: number;
  shoulder: number;
};

type FeedbackItem = {
  id: number;
  type: FeedbackType;
  title: string;
  color: string;
  bg: string;
  text: string;
  time: string;
};

type Sensitivity = {
  trunkAngle: number;
  headDistance: number;
  shoulderTilt: number;
};

type DominantIssue = "trunk" | "head" | "shoulder" | null;

type SilhouetteMetrics = {
  neckForwardContour: number;
  upperBackCurvature: number;
  torsoOutlineAngle: number;
  silhouetteStability: number;
};

type AudioMode = "off" | "voice";
type ThemeMode = "dark" | "light";
type AutoPipStatus = "supported" | "manual_only" | "blocked" | "unsupported";
type FloatingWindowOrigin = "auto" | "manual" | null;
type CameraDevice = {
  id: string;
  label: string;
};
type SpeechStatus = "unsupported" | "blocked" | "ready" | "loading";
type MlStatus = "checking" | "connected" | "degraded" | "unavailable";
type TutorialTarget =
  | "start-session"
  | "camera-stage"
  | "posture-score"
  | "session-log"
  | "settings-panel";
type TutorialStep = {
  target: TutorialTarget;
  title: string;
  body: string;
};
type GuidedTrialStepId =
  | "baseline"
  | "forward"
  | "shoulders"
  | "framing"
  | "neutral"
  | "floating";
type GuidedTrialStep = {
  id: GuidedTrialStepId;
  title: string;
  instruction: string;
  expected: string;
};
type TutorialRect = {
  top: number;
  left: number;
  right: number;
  bottom: number;
  width: number;
  height: number;
};
type DebugMetrics = {
  chinCenterOffset: number;
  chinForwardLean: number;
  chinLiftProxy: number;
  noseCenterOffset: number;
  mouthLineTilt: number;
  eyeOrEarTilt: number;
  upperForwardLean: number;
  upperShoulderTilt: number;
};

type MlPrediction = {
  label: "proper" | "needs_correction" | string;
  confidence: number;
  probabilities: Record<string, number>;
  feedback: string;
};

type FeedbackPresentation = {
  type: FeedbackType;
  title: string;
  color: string;
  bg: string;
  text: string;
  audio: string;
};

type DocumentPictureInPictureApi = {
  requestWindow(options?: { width?: number; height?: number }): Promise<Window>;
  window?: Window | null;
};

const AUTO_PIP_ACTION = "enterpictureinpicture" as MediaSessionAction;
const WINDOW = 30;
const EMA_ALPHA = 0.25;
const VIS_THRESHOLD = 0.35;
const DRAW_VIS_THRESHOLD = 0.12;
const HOLD_STILL_MS = 300;
const PREDICTION_VOTE_WINDOW = 3;
const AUDIO_COOLDOWN_MS = 5000;
const HEAD_FORWARD_GRACE_RATIO = 1.2;
const FRONT_FACE_VISIBILITY_MIN = 0.4;
const FRONT_SHOULDER_WIDTH_MIN = 0.16;
const FRONT_SHOULDER_DEPTH_DIFF_MAX = 0.09;
const FRONT_HIP_WIDTH_MIN = 0.12;
const FRONT_TORSO_LENGTH_MIN = 0.18;
const UPPER_FRONT_HEAD_OFFSET_THRESHOLD = 0.18;
const UPPER_FRONT_FORWARD_LEAN_THRESHOLD = 0.18;
const UPPER_FRONT_FORWARD_LEAN_SEVERE = 0.45;
const UPPER_FRONT_SHOULDER_TILT_THRESHOLD = 0.12;
const UPPER_FRONT_SHOULDER_TILT_RECOVERY = 0.09;
const CHIN_LIFT_PROXY_NEUTRAL = 0.55;
const CHIN_LIFT_PROXY_TO_HEAD_LEAN_SCALE = 0.45;
const CHIN_LIFT_PROXY_THRESHOLD = 0.95;
const CHIN_LIFT_PROXY_SEVERE = 1.15;
const UPPER_FRONT_TRACKING_MIN = 62;
const UPPER_FRONT_FRAME_MARGIN = 0.08;
const UPPER_FRONT_SCORE_CAP = 86;
const THEME_STORAGE_KEY = "sukatlikod-theme";
const TUTORIAL_SEEN_STORAGE_KEY = "uprightly-tutorial-seen";
const PRIVACY_NOTICE_STORAGE_KEY = "uprightly-privacy-notice";
const FLOATING_WINDOW_STORAGE_KEY = "uprightly-auto-floating-window-v2";
const GUIDED_TRIAL_STORAGE_KEY = "uprightly-guided-trial-seen-v1";
const PRIVACY_NOTICE_VERSION = "2";
const PRIVACY_POLICY_UPDATED = "September 12, 2026";
const DEFAULT_SENSITIVITY: Sensitivity = {
  trunkAngle: 18,
  headDistance: 0.1,
  shoulderTilt: 0.05,
};
const DEFAULT_SILHOUETTE_METRICS: SilhouetteMetrics = {
  neckForwardContour: 0,
  upperBackCurvature: 0,
  torsoOutlineAngle: 0,
  silhouetteStability: 0,
};
const DEFAULT_DEBUG_METRICS: DebugMetrics = {
  chinCenterOffset: 0,
  chinForwardLean: 0,
  chinLiftProxy: 0,
  noseCenterOffset: 0,
  mouthLineTilt: 0,
  eyeOrEarTilt: 0,
  upperForwardLean: 0,
  upperShoulderTilt: 0,
};
const TUTORIAL_STEPS: TutorialStep[] = [
  {
    target: "start-session",
    title: "Begin when you're ready",
    body: "Start Session opens your camera and begins live posture guidance. The same control ends the session whenever you need a break.",
  },
  {
    target: "camera-stage",
    title: "Frame your upper body",
    body: "Keep your head and shoulders visible inside the guide. A steady, front-facing view gives Uprightly the clearest posture signal.",
  },
  {
    target: "posture-score",
    title: "Read the overall signal",
    body: "The score summarizes your current posture. The supporting measurements show whether your head, trunk, or shoulders need attention.",
  },
  {
    target: "session-log",
    title: "Notice patterns, not moments",
    body: "The session log collects meaningful posture changes so you can follow the pattern without reacting to every small movement.",
  },
  {
    target: "settings-panel",
    title: "Tune the experience",
    body: "Choose your camera and theme, manage voice feedback, replay this tour, or keep status visible in a floating window.",
  },
];
const GUIDED_TRIAL_STEPS: GuidedTrialStep[] = [
  {
    id: "baseline",
    title: "Establish your starting posture",
    instruction:
      "Sit comfortably upright, face the camera, and keep your head and shoulders visible. Hold still for a few seconds.",
    expected: "Tracking becomes stable and Uprightly establishes a reference.",
  },
  {
    id: "forward",
    title: "Try a gentle forward posture",
    instruction:
      "Move your head and upper body slightly forward, then hold that position briefly.",
    expected: "Look for Bring your head back or Sit straighter.",
  },
  {
    id: "shoulders",
    title: "Change your shoulder alignment",
    instruction:
      "Gently raise or lower one shoulder while keeping both shoulders visible.",
    expected: "Look for Level your shoulders.",
  },
  {
    id: "framing",
    title: "Test camera framing",
    instruction:
      "Turn slightly away or move one shoulder partly outside the camera frame.",
    expected: "Uprightly asks you to face the camera or keep both shoulders visible.",
  },
  {
    id: "neutral",
    title: "Return to neutral",
    instruction:
      "Return to your original comfortable upright position and hold still.",
    expected: "The score and guidance recover after the reading stabilizes.",
  },
  {
    id: "floating",
    title: "Work in another tab",
    instruction:
      "Switch to another tab or application while the session remains active, then return here.",
    expected: "On supported browsers, the floating posture status appears.",
  },
];

type SideKind = "left" | "right";

function pushLimited(arr: number[], x: number) {
  arr.push(x);
  if (arr.length > WINDOW) arr.shift();
}

function findVisibleTourElement(target: TutorialTarget): HTMLElement | null {
  if (typeof document === "undefined") return null;

  const elements = Array.from(
    document.querySelectorAll<HTMLElement>(`[data-tour="${target}"]`),
  );

  for (const element of elements) {
    const rect = element.getBoundingClientRect();
    const style = window.getComputedStyle(element);
    const isVisible =
      rect.width > 0 &&
      rect.height > 0 &&
      style.display !== "none" &&
      style.visibility !== "hidden" &&
      style.opacity !== "0";

    if (isVisible) {
      return element;
    }
  }

  return null;
}

function findVisibleTourTarget(target: TutorialTarget): TutorialRect | null {
  const element = findVisibleTourElement(target);
  if (!element) return null;

  const rect = element.getBoundingClientRect();
  return {
    top: rect.top,
    left: rect.left,
    right: rect.right,
    bottom: rect.bottom,
    width: rect.width,
    height: rect.height,
  };
}

function vsub(a: Point3, b: Point3) {
  return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
}

function vlen(v: Point3) {
  return Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

function trunkAngleDeg(midShoulder: Point3, midHip: Point3) {
  const v = vsub(midShoulder, midHip);
  const L = vlen(v);
  if (L < 1e-6) return 0;
  const cos = Math.abs(v.y) / L;
  const rad = Math.acos(clamp(cos, 0, 1));
  return (rad * 180) / Math.PI;
}

function trunkAngleSignedDeg(midShoulder: Point3, midHip: Point3) {
  const v = vsub(midShoulder, midHip);
  return (Math.atan2(v.x, -v.y) * 180) / Math.PI;
}

function headForwardM(nose: Point3, midShoulder: Point3) {
  return Math.abs((nose?.z ?? 0) - (midShoulder?.z ?? 0));
}

function headForwardSignedM(nose: Point3, midShoulder: Point3) {
  return (nose?.z ?? 0) - (midShoulder?.z ?? 0);
}

function shoulderTiltM(ls: Point3, rs: Point3) {
  return Math.abs((ls?.y ?? 0) - (rs?.y ?? 0));
}

function shoulderTiltSignedM(ls: Point3, rs: Point3) {
  return (ls?.y ?? 0) - (rs?.y ?? 0);
}

function ema(prev: number | null, next: number, alpha = EMA_ALPHA) {
  if (prev == null) return next;
  return prev + alpha * (next - prev);
}

function visOk(
  p?: {
    x: number;
    y: number;
    z: number;
    visibility?: number;
  },
  min = VIS_THRESHOLD,
) {
  return !!p && (p.visibility ?? 1) >= min;
}

function avgVisibility(
  points: Array<{ visibility?: number } | undefined>,
  min = VIS_THRESHOLD,
) {
  const valid = points.filter(Boolean) as Array<{ visibility?: number }>;
  if (valid.length === 0) return 0;
  const mean =
    valid.reduce((s, p) => s + (p.visibility ?? 0), 0) / valid.length;
  if (mean < min) return 0;
  return mean;
}

function planarDistance(a: Point3, b: Point3) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function midpoint(a: Point3, b: Point3): Point3 {
  return {
    x: (a.x + b.x) / 2,
    y: (a.y + b.y) / 2,
    z: (a.z + b.z) / 2,
  };
}

function normalizedDepthDelta(a: Point3, b: Point3, scale: number) {
  return Math.abs(a.z - b.z) / Math.max(scale, 1e-3);
}

function dominantSideFromNorm(
  norm: { x: number; y: number; z: number; visibility?: number }[],
): SideKind {
  const lScore = avgVisibility([
    norm[IDX.L_SHOULDER],
    norm[IDX.L_HIP],
    norm[IDX.L_EAR],
    norm[IDX.L_EYE],
  ]);
  const rScore = avgVisibility([
    norm[IDX.R_SHOULDER],
    norm[IDX.R_HIP],
    norm[IDX.R_EAR],
    norm[IDX.R_EYE],
  ]);
  return lScore >= rScore ? "left" : "right";
}

function detectOrientation(
  world: { x: number; y: number; z: number }[],
  norm?: { x: number; y: number; z: number; visibility?: number }[],
): { kind: OrientationKind; label: string } {
  const lsW = world[IDX.L_SHOULDER];
  const rsW = world[IDX.R_SHOULDER];
  const lsN = norm?.[IDX.L_SHOULDER];
  const rsN = norm?.[IDX.R_SHOULDER];
  if (!lsW || !rsW || !lsN || !rsN)
    return { kind: "unknown", label: "Unknown" };

  const shoulderDepthDiff = Math.abs(lsW.z - rsW.z);
  const shoulderWidth2D = Math.abs(lsN.x - rsN.x);
  const sideLike = shoulderDepthDiff > 0.11 || shoulderWidth2D < 0.16;

  if (sideLike) {
    const leftCloser = lsW.z < rsW.z;
    return leftCloser
      ? { kind: "side_left", label: "Side (Left)" }
      : { kind: "side_right", label: "Side (Right)" };
  }

  const facePoints = [
    norm?.[IDX.NOSE],
    norm?.[IDX.L_EYE],
    norm?.[IDX.R_EYE],
    norm?.[IDX.L_EAR],
    norm?.[IDX.R_EAR],
  ].filter(Boolean) as { visibility?: number }[];

  const faceVis =
    facePoints.length > 0
      ? facePoints.reduce((s, p) => s + (p.visibility ?? 0), 0) /
        facePoints.length
      : 0;

  if (faceVis < 0.35) return { kind: "back", label: "Back" };
  return { kind: "front", label: "Front" };
}

function classifyFrontCapture(
  world: { x: number; y: number; z: number }[],
  norm: { x: number; y: number; z: number; visibility?: number }[],
): {
  tier: FrontCaptureTier | null;
  faceVisible: boolean;
  upperVisible: boolean;
  hipsVisible: boolean;
} {
  const noseN = norm[IDX.NOSE];
  const lsN = norm[IDX.L_SHOULDER];
  const rsN = norm[IDX.R_SHOULDER];
  const lhN = norm[IDX.L_HIP];
  const rhN = norm[IDX.R_HIP];
  const leN = norm[IDX.L_EYE];
  const reN = norm[IDX.R_EYE];
  const lEarN = norm[IDX.L_EAR];
  const rEarN = norm[IDX.R_EAR];
  const lsW = world[IDX.L_SHOULDER];
  const rsW = world[IDX.R_SHOULDER];

  const earsVisible = visOk(lEarN) && visOk(rEarN);
  const eyesVisible = visOk(leN) && visOk(reN);
  const faceVisible =
    visOk(noseN) &&
    (earsVisible || eyesVisible) &&
    avgVisibility([noseN, leN, reN, lEarN, rEarN], FRONT_FACE_VISIBILITY_MIN) >
      0;
  const upperVisible = visOk(lsN) && visOk(rsN) && faceVisible;

  if (!upperVisible || !lsW || !rsW || !lsN || !rsN) {
    return { tier: null, faceVisible, upperVisible, hipsVisible: false };
  }

  const shoulderWidth = Math.abs(lsN.x - rsN.x);
  const shoulderDepthDiff = Math.abs(lsW.z - rsW.z);
  const frontAligned =
    shoulderWidth >= FRONT_SHOULDER_WIDTH_MIN &&
    shoulderDepthDiff <= FRONT_SHOULDER_DEPTH_DIFF_MAX;
  const upperBodyInsideFrame =
    noseN.x >= UPPER_FRONT_FRAME_MARGIN &&
    noseN.x <= 1 - UPPER_FRONT_FRAME_MARGIN &&
    lsN.x >= UPPER_FRONT_FRAME_MARGIN / 2 &&
    rsN.x <= 1 - UPPER_FRONT_FRAME_MARGIN / 2 &&
    lsN.y >= UPPER_FRONT_FRAME_MARGIN / 2 &&
    rsN.y >= UPPER_FRONT_FRAME_MARGIN / 2;

  if (!frontAligned || !upperBodyInsideFrame) {
    return { tier: null, faceVisible, upperVisible, hipsVisible: false };
  }

  const hipsVisible = visOk(lhN) && visOk(rhN);
  if (!hipsVisible || !lhN || !rhN) {
    return { tier: "upper_front", faceVisible, upperVisible, hipsVisible };
  }

  const hipWidth = Math.abs(lhN.x - rhN.x);
  const midShoulderN: Point3 = {
    x: (lsN.x + rsN.x) / 2,
    y: (lsN.y + rsN.y) / 2,
    z: (lsN.z + rsN.z) / 2,
  };
  const midHipN: Point3 = {
    x: (lhN.x + rhN.x) / 2,
    y: (lhN.y + rhN.y) / 2,
    z: (lhN.z + rhN.z) / 2,
  };
  const torsoLength = planarDistance(midShoulderN, midHipN);

  if (
    hipWidth >= FRONT_HIP_WIDTH_MIN &&
    torsoLength >= FRONT_TORSO_LENGTH_MIN
  ) {
    return { tier: "full_front", faceVisible, upperVisible, hipsVisible };
  }

  return { tier: "upper_front", faceVisible, upperVisible, hipsVisible };
}

function getPrimaryFaceLandmarks(
  faceResult?: FaceLandmarkerResult,
): Point3[] | null {
  const landmarks = faceResult?.faceLandmarks?.[0];
  if (!landmarks || landmarks.length === 0) return null;
  return landmarks as Point3[];
}

function getNaturalAudioFromMessage(
  msg: string,
  dominant: DominantIssue,
): string {
  const lower = msg.toLowerCase();
  if (lower.includes("move into view")) return "Move a little more into view.";
  if (lower.includes("face the camera")) return "Face the camera a bit more.";
  if (lower.includes("turn and face"))
    return "Turn a little and face the camera.";
  if (lower.includes("hold still")) return "Hold still for a moment.";
  if (lower.includes("level your shoulders") || dominant === "shoulder") {
    return "Relax and level your shoulders.";
  }
  if (lower.includes("lower your chin")) {
    return "Lower your chin a little.";
  }
  if (lower.includes("center your head") || dominant === "trunk") {
    return "Center your head a bit more.";
  }
  if (
    lower.includes("bring your head back") ||
    lower.includes("sit straighter") ||
    dominant === "head"
  ) {
    return "Bring your head back a little.";
  }
  if (lower.includes("good posture") || lower.includes("looking good")) {
    return "That looks good. Keep it there.";
  }
  return "Adjust your posture a little.";
}

function getFeedbackPresentation(
  scoreValue: number,
  msg: string,
  h: number,
  dominant: DominantIssue,
  headThreshold: number,
): FeedbackPresentation {
  let type: FeedbackType = "info";
  let title = "Looking Good";
  let color = "text-[#91a889]";
  let bg = "bg-[#91a889]/10 border-[#91a889]/25";
  let text = msg;
  let audio = "That looks good. Keep it there.";
  const lower = msg.toLowerCase();

  if (scoreValue < 60) {
    type = "critical";
    title = "Sit Straighter";
    color = "text-rose-400";
    bg = "bg-rose-500/10 border-rose-500/20";
    text = msg;
    audio = getNaturalAudioFromMessage(msg, dominant);
  } else if (lower.includes("lower your chin")) {
    type = "warning";
    title = "Lower Your Chin";
    color = "text-amber-400";
    bg = "bg-amber-500/10 border-amber-500/20";
    text = "Lower your chin and keep your head level.";
    audio = "Lower your chin a little.";
  } else if (dominant === "head" && h > headThreshold) {
    type = "warning";
    title = "Bring Your Head Back";
    color = "text-amber-400";
    bg = "bg-amber-500/10 border-amber-500/20";
    text =
      "Lift through the crown of your head and keep your chin gently tucked.";
    audio = "Bring your head back a little.";
  } else if (dominant === "shoulder") {
    type = "warning";
    title = "Level Your Shoulders";
    color = "text-amber-400";
    bg = "bg-amber-500/10 border-amber-500/20";
    text = "Relax your neck and level both shoulders.";
    audio = "Relax and level your shoulders.";
  } else if (scoreValue > 85) {
    type = "success";
    title = "Good Posture";
    color = "text-[#91a889]";
    bg = "bg-[#91a889]/10 border-[#91a889]/25";
    text = "Nice posture. Keep it steady.";
    audio = "That looks good. Keep it there.";
  }

  return { type, title, color, bg, text, audio };
}

const DESKTOP_VIEWPORT_QUERY = "(min-width: 1024px)";

function hasAcknowledgedPrivacyNotice(): boolean {
  return (
    typeof window !== "undefined" &&
    window.localStorage.getItem(PRIVACY_NOTICE_STORAGE_KEY) ===
      PRIVACY_NOTICE_VERSION
  );
}

function useDesktopViewport() {
  const [isDesktopViewport, setIsDesktopViewport] = useState(() =>
    typeof window === "undefined"
      ? true
      : window.matchMedia(DESKTOP_VIEWPORT_QUERY).matches,
  );

  useEffect(() => {
    const mediaQuery = window.matchMedia(DESKTOP_VIEWPORT_QUERY);
    const handleChange = (event: MediaQueryListEvent) =>
      setIsDesktopViewport(event.matches);

    mediaQuery.addEventListener("change", handleChange);
    return () => mediaQuery.removeEventListener("change", handleChange);
  }, []);

  return isDesktopViewport;
}

function MobileCompatibilityNotice() {
  useEffect(() => {
    document.documentElement.style.colorScheme = "dark";
    document.body.style.background = "#10100f";
  }, []);

  return (
    <main className="relative flex min-h-dvh w-full max-w-full items-center justify-center overflow-hidden bg-[#10100f] px-6 py-12 font-sans text-[#f4f0e8]">
      <div
        className="pointer-events-none absolute inset-0"
        aria-hidden="true"
        style={{
          background:
            "radial-gradient(circle at 50% 18%, rgba(211,154,56,0.14), transparent 34%), radial-gradient(circle at 82% 88%, rgba(145,168,137,0.07), transparent 28%)",
        }}
      />
      <section
        className="relative w-full max-w-md rounded-[2rem] border border-white/10 bg-[#171715]/95 p-7 text-center shadow-[0_30px_90px_-35px_rgba(0,0,0,0.9)] backdrop-blur-xl sm:p-9"
        aria-labelledby="desktop-required-title"
        aria-describedby="desktop-required-description"
      >
        <p className="text-3xl font-bold tracking-[-0.045em]">Uprightly</p>
        <div className="mx-auto mt-7 flex h-16 w-16 items-center justify-center rounded-2xl border border-white/25 bg-white/10 text-[#f1f0ec]">
          <Monitor size={28} strokeWidth={1.8} aria-hidden="true" />
        </div>
        <h1
          id="desktop-required-title"
          className="mx-auto mt-6 max-w-md text-2xl font-bold leading-tight tracking-[-0.025em]"
        >
          Designed for computers
        </h1>
        <p
          id="desktop-required-description"
          className="mx-auto mt-3 max-w-sm text-[0.9375rem] leading-6 text-white/60"
        >
          Open Uprightly on a laptop or desktop with a camera to begin posture
          monitoring.
        </p>
      </section>
    </main>
  );
}

export default function App() {
  const isDesktopViewport = useDesktopViewport();
  return isDesktopViewport ? <DesktopApp /> : <MobileCompatibilityNotice />;
}

function DesktopApp() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const chatEndRef = useRef<HTMLDivElement | null>(null);
  const sessionLogStackRef = useRef<HTMLDivElement | null>(null);
  const tutorialCardRef = useRef<HTMLDivElement | null>(null);
  const tutorialContentRef = useRef<HTMLDivElement | null>(null);
  const tutorialReturnFocusRef = useRef<HTMLElement | null>(null);
  const guidedTrialIntroRef = useRef<HTMLDivElement | null>(null);
  const purposeNoticeRef = useRef<HTMLElement | null>(null);
  const privacyNoticeRef = useRef<HTMLElement | null>(null);
  const privacyPolicyRef = useRef<HTMLDivElement | null>(null);
  const privacyReturnFocusRef = useRef<HTMLElement | null>(null);
  const tutorialPanelStateRef = useRef({
    showSettings: false,
    showSessionLog: false,
  });

  const poseRef = useRef<PoseLandmarker | null>(null);
  const faceRef = useRef<FaceLandmarker | null>(null);
  const rafRef = useRef<number | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const lastVideoTimeRef = useRef<number>(-1);
  const lastFeedbackRef = useRef<string>("");
  const lastSpokenMessageRef = useRef<string>("");
  const lastInferTsRef = useRef<number>(0);
  const inferInFlightRef = useRef<boolean>(false);
  const lastAudioEventRef = useRef<{ key: string; at: number }>({
    key: "",
    at: 0,
  });
  const lastAnnouncedStateRef = useRef<"good" | "fix" | "idle">("idle");
  const baselineMetricsRef = useRef<
    Record<FrontCaptureTier, BaselineMetrics | null>
  >({
    full_front: null,
    upper_front: null,
  });
  const loadedModelPathRef = useRef<string | null>(null);
  const loadedFaceModelPathRef = useRef<string | null>(null);
  const landmarkerLoadPromiseRef = useRef<Promise<void> | null>(null);
  const holdStillStartRef = useRef<number>(0);
  const floatingWindowRef = useRef<Window | null>(null);
  const floatingRootRef = useRef<HTMLDivElement | null>(null);
  const floatingWindowOriginRef = useRef<FloatingWindowOrigin>(null);
  const floatingWindowOpeningRef = useRef<Promise<void> | null>(null);
  const floatingWindowRequestIdRef = useRef(0);
  const emaRef = useRef<{
    trunk: number | null;
    head: number | null;
    shoulder: number | null;
    contour: number | null;
    curvature: number | null;
    outline: number | null;
  }>({
    trunk: null,
    head: null,
    shoulder: null,
    contour: null,
    curvature: null,
    outline: null,
  });
  const predictionVotesRef = useRef<boolean[]>([]);
  const shoulderWarningActiveRef = useRef(false);
  const lastSmoothedRef = useRef<{
    trunk: number;
    head: number;
    shoulder: number;
    contour: number;
    curvature: number;
    outline: number;
  } | null>(null);

  const buffersRef = useRef<{
    trunk: number[];
    head: number[];
    shoulder: number[];
    contour: number[];
    curvature: number[];
    outline: number[];
  }>({
    trunk: [],
    head: [],
    shoulder: [],
    contour: [],
    curvature: [],
    outline: [],
  });

  const [isActive, setIsActive] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [showSessionLog, setShowSessionLog] = useState(false);
  const [showPrivacyNotice, setShowPrivacyNotice] = useState(
    () => !hasAcknowledgedPrivacyNotice(),
  );
  const [startupNoticeStep, setStartupNoticeStep] = useState<
    "purpose" | "privacy"
  >("purpose");
  const [rememberPrivacyNotice, setRememberPrivacyNotice] = useState(false);
  const [showPrivacyPolicy, setShowPrivacyPolicy] = useState(false);
  const [showTutorial, setShowTutorial] = useState(() => {
    if (typeof window === "undefined") return false;
    return (
      hasAcknowledgedPrivacyNotice() &&
      window.localStorage.getItem(TUTORIAL_SEEN_STORAGE_KEY) !== "true"
    );
  });
  const [tutorialStepIndex, setTutorialStepIndex] = useState(0);
  const [tutorialTargetRect, setTutorialTargetRect] =
    useState<TutorialRect | null>(null);
  const [tutorialCardSize, setTutorialCardSize] = useState({
    width: 384,
    height: 320,
  });
  const [isCompactTutorial, setIsCompactTutorial] = useState(false);
  const [showGuidedTrialIntro, setShowGuidedTrialIntro] = useState(false);
  const [showGuidedTrial, setShowGuidedTrial] = useState(false);
  const [guidedTrialPending, setGuidedTrialPending] = useState(false);
  const [guidedTrialStepIndex, setGuidedTrialStepIndex] = useState(0);
  const [completedGuidedTrialSteps, setCompletedGuidedTrialSteps] = useState<
    Set<GuidedTrialStepId>
  >(() => new Set());
  const [floatingWindowEnabled, setFloatingWindowEnabled] = useState(() => {
    if (typeof window === "undefined") return true;
    return window.localStorage.getItem(FLOATING_WINDOW_STORAGE_KEY) !== "false";
  });
  const [floatingWindowReady, setFloatingWindowReady] = useState(false);
  const [autoPipStatus, setAutoPipStatus] =
    useState<AutoPipStatus>("unsupported");
  const [isPageFocused, setIsPageFocused] = useState(() =>
    typeof document === "undefined"
      ? true
      : document.hasFocus() && !document.hidden,
  );
  const [pill, setPill] = useState<Pill>("idle");
  const [theme, setTheme] = useState<ThemeMode>(() => {
    if (typeof window === "undefined") return "dark";
    const storedTheme = window.localStorage.getItem(THEME_STORAGE_KEY);
    return storedTheme === "light" ? "light" : "dark";
  });

  const [score, setScore] = useState(0);
  const [feedback, setFeedback] = useState("Press Start Session to begin.");
  const [feedbacks, setFeedbacks] = useState<FeedbackItem[]>([]);

  const [metrics, setMetrics] = useState({
    trunkAngle: 0,
    headForward: 0,
    shoulderTilt: 0,
  });
  const [signedMetrics, setSignedMetrics] = useState({
    trunkAngle: 0,
    headForward: 0,
    shoulderTilt: 0,
  });
  const [, setSilhouetteMetrics] = useState<SilhouetteMetrics>(
    DEFAULT_SILHOUETTE_METRICS,
  );
  const [audioMode, setAudioMode] = useState<AudioMode>("voice");
  const [speechStatus, setSpeechStatus] = useState<SpeechStatus>("loading");
  const [availableVoices, setAvailableVoices] = useState(0);
  const [cameraDevices, setCameraDevices] = useState<CameraDevice[]>([]);
  const [selectedCameraId, setSelectedCameraId] = useState("");
  const [assessmentTier, setAssessmentTier] = useState<FrontCaptureTier | null>(
    null,
  );
  const [, setDebugMetrics] = useState<DebugMetrics>(DEFAULT_DEBUG_METRICS);

  const [sensitivity] = useState<Sensitivity>(DEFAULT_SENSITIVITY);
  const [stabilityScore, setStabilityScore] = useState(0);
  const [trackingHealth, setTrackingHealth] = useState(0);
  const [mlStatus, setMlStatus] = useState<MlStatus>("checking");
  const overlayDetail = "detailed" as const;

  const modelPath = useMemo(() => "/models/pose_landmarker_lite.task", []);
  const faceModelPath = useMemo(() => "/models/face_landmarker.task", []);
  const mlApiUrl = useMemo(
    () => (import.meta.env.VITE_ML_API_URL as string | undefined)?.trim() ?? "",
    [],
  );

  const resetBuffers = useCallback(() => {
    buffersRef.current.trunk = [];
    buffersRef.current.head = [];
    buffersRef.current.shoulder = [];
    buffersRef.current.contour = [];
    buffersRef.current.curvature = [];
    buffersRef.current.outline = [];
    lastVideoTimeRef.current = -1;
    lastFeedbackRef.current = "";
    lastSpokenMessageRef.current = "";
    holdStillStartRef.current = 0;
    lastAudioEventRef.current = { key: "", at: 0 };
    lastAnnouncedStateRef.current = "idle";
    baselineMetricsRef.current = {
      full_front: null,
      upper_front: null,
    };
    shoulderWarningActiveRef.current = false;
    emaRef.current = {
      trunk: null,
      head: null,
      shoulder: null,
      contour: null,
      curvature: null,
      outline: null,
    };
    lastSmoothedRef.current = null;
    predictionVotesRef.current = [];
  }, []);

  const stop = useCallback(() => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }

    if (videoRef.current) videoRef.current.srcObject = null;

    if (typeof window !== "undefined" && "speechSynthesis" in window) {
      window.speechSynthesis.cancel();
    }

    const c = canvasRef.current;
    if (c) c.getContext("2d")?.clearRect(0, 0, c.width, c.height);

    setIsActive(false);
    setPill("idle");
    setFeedback("Press Start Session to begin.");
    setFeedbacks([]);
    setScore(0);
    setMetrics({ trunkAngle: 0, headForward: 0, shoulderTilt: 0 });
    setSignedMetrics({ trunkAngle: 0, headForward: 0, shoulderTilt: 0 });
    setAssessmentTier(null);
    setDebugMetrics(DEFAULT_DEBUG_METRICS);
    setSilhouetteMetrics(DEFAULT_SILHOUETTE_METRICS);
    setStabilityScore(0);
    setTrackingHealth(0);
    resetBuffers();
  }, [resetBuffers]);

  const ensureLandmarker = useCallback(async () => {
    if (
      poseRef.current &&
      faceRef.current &&
      loadedModelPathRef.current === modelPath &&
      loadedFaceModelPathRef.current === faceModelPath
    ) {
      return;
    }
    if (landmarkerLoadPromiseRef.current)
      return landmarkerLoadPromiseRef.current;
    poseRef.current = null;
    faceRef.current = null;
    loadedModelPathRef.current = null;
    loadedFaceModelPathRef.current = null;

    landmarkerLoadPromiseRef.current = (async () => {
      const { FaceLandmarker, FilesetResolver, PoseLandmarker } = await import(
        "@mediapipe/tasks-vision"
      );
      const vision = await FilesetResolver.forVisionTasks(
        "/mediapipe/wasm",
      );

      poseRef.current = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: modelPath, delegate: "GPU" },
        runningMode: "VIDEO",
        numPoses: 1,
        minPoseDetectionConfidence: 0.5,
        minPosePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });
      loadedModelPathRef.current = modelPath;

      faceRef.current = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: faceModelPath, delegate: "GPU" },
        runningMode: "VIDEO",
        numFaces: 1,
        minFaceDetectionConfidence: 0.5,
        minFacePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
        outputFaceBlendshapes: false,
        outputFacialTransformationMatrixes: false,
      });
      loadedFaceModelPathRef.current = faceModelPath;
    })();

    try {
      await landmarkerLoadPromiseRef.current;
    } catch (error) {
      poseRef.current = null;
      faceRef.current = null;
      loadedModelPathRef.current = null;
      loadedFaceModelPathRef.current = null;
      throw error;
    } finally {
      landmarkerLoadPromiseRef.current = null;
    }
  }, [faceModelPath, modelPath]);

  const computeDecision = useCallback(
    (
      captureTier: FrontCaptureTier,
      thresholds?: Sensitivity,
      baseline?: BaselineMetrics | null,
    ) => {
      const rawT = avg(buffersRef.current.trunk);
      const rawH = avg(buffersRef.current.head);
      const rawS = avg(buffersRef.current.shoulder);

      if (rawT == null || rawH == null || rawS == null) {
        return {
          ok: false,
          score: null as number | null,
          msg: "Hold still...",
          dominant: null as DominantIssue,
          t: rawT,
          h: rawH,
          s: rawS,
          rawT,
          rawH,
          rawS,
          tRatio: 0,
          hRatio: 0,
          sRatio: 0,
        };
      }

      const t =
        captureTier === "full_front"
          ? baseline
            ? Math.abs(rawT - baseline.trunk)
            : rawT
          : baseline
            ? Math.abs(rawT - baseline.trunk)
            : rawT;
      const h =
        captureTier === "full_front"
          ? baseline
            ? Math.max(0, rawH - baseline.head)
            : rawH
          : baseline
            ? Math.max(0, rawH - baseline.head)
            : rawH;
      const s = baseline ? Math.abs(rawS - baseline.shoulder) : rawS;

      const tThr = thresholds?.trunkAngle ?? sensitivity.trunkAngle;
      const hThr = thresholds?.headDistance ?? sensitivity.headDistance;
      const sThr = thresholds?.shoulderTilt ?? sensitivity.shoulderTilt;
      const shoulderRecoveryThreshold =
        captureTier === "upper_front"
          ? Math.min(sThr, UPPER_FRONT_SHOULDER_TILT_RECOVERY)
          : sThr * 0.78;
      const shoulderWarnActive = shoulderWarningActiveRef.current;
      const shoulderOutOfRange = shoulderWarnActive
        ? s >= shoulderRecoveryThreshold
        : s >= sThr;
      shoulderWarningActiveRef.current = shoulderOutOfRange;
      const tRatio = t / tThr;
      const hRatio = h / hThr;
      const sRatio = shoulderOutOfRange ? s / sThr : 0;
      const worst = Math.max(tRatio, hRatio, sRatio);
      const dominant: DominantIssue =
        captureTier === "upper_front"
          ? hRatio >= sRatio && hRatio >= tRatio
            ? "head"
            : sRatio >= tRatio
              ? "shoulder"
              : "trunk"
          : worst === tRatio
            ? "trunk"
            : worst === hRatio
              ? "head"
              : worst === sRatio
                ? "shoulder"
                : null;

      const nextScore =
        captureTier === "upper_front"
          ? Math.round(
              clamp(
                100 -
                  Math.max(0, hRatio - 1) * 52 -
                  Math.max(0, sRatio - 1) * 54 -
                  Math.max(0, tRatio - 1) * 22,
                0,
                100,
              ),
            )
          : Math.round(clamp(100 - (worst - 1) * 45, 0, 100));
      const mildHeadOnly =
        captureTier === "upper_front"
          ? false
          : dominant === "head" &&
            hRatio <= HEAD_FORWARD_GRACE_RATIO &&
            tRatio <= 1 &&
            sRatio <= 1;
      const ok =
        captureTier === "upper_front"
          ? hRatio <= 1 && tRatio <= 1.1 && sRatio <= 1
          : worst <= 1 || mildHeadOnly;

      let msg =
        captureTier === "full_front" ? "Good posture." : "Looking good.";
      if (!ok) {
        if (captureTier === "full_front") {
          if (dominant === "trunk") msg = "Sit straighter.";
          else if (dominant === "head") msg = "Bring your head back a little.";
          else msg = "Level your shoulders.";
        } else {
          if (h >= UPPER_FRONT_FORWARD_LEAN_SEVERE) {
            msg = "Sit straighter and bring your head back.";
          } else if (dominant === "shoulder") {
            msg = "Level your shoulders.";
          } else if (dominant === "trunk") {
            msg = "Center your head.";
          } else if (dominant === "head") {
            msg = "Bring your head back a little.";
          }
        }
      } else if (mildHeadOnly) {
        msg = captureTier === "full_front" ? "Good posture." : "Looking good.";
      }

      return {
        ok,
        score: nextScore,
        msg,
        t,
        h,
        s,
        rawT,
        rawH,
        rawS,
        dominant,
        tRatio,
        hRatio,
        sRatio,
      };
    },
    [
      sensitivity.headDistance,
      sensitivity.shoulderTilt,
      sensitivity.trunkAngle,
    ],
  );

  const pushFeedback = useCallback(
    (
      scoreValue: number,
      msg: string,
      _t: number,
      h: number,
      dominant: DominantIssue,
      headThreshold = sensitivity.headDistance,
    ) => {
      if (lastFeedbackRef.current === msg) return;
      lastFeedbackRef.current = msg;

      const now = Date.now();

      const time = new Date(now).toLocaleTimeString([], {
        hour12: false,
        hour: "2-digit",
        minute: "2-digit",
      });
      const presentation = getFeedbackPresentation(
        scoreValue,
        msg,
        h,
        dominant,
        headThreshold,
      );

      setFeedbacks((prev) =>
        [
          ...prev,
          {
            id: now,
            type: presentation.type,
            title: presentation.title,
            color: presentation.color,
            bg: presentation.bg,
            text: presentation.text,
            time,
          },
        ].slice(-50),
      );
    },
    [sensitivity.headDistance],
  );

  const inferMl = useCallback(
    async (payload: {
      trunk_angle: number;
      head_forward: number;
      shoulder_tilt: number;
      trunk_variance: number;
      neck_forward_contour: number;
      upper_back_curvature: number;
      torso_outline_angle: number;
      silhouette_stability: number;
    }): Promise<MlPrediction | null> => {
      if (!mlApiUrl) {
        setMlStatus("unavailable");
        return null;
      }
      if (inferInFlightRef.current) return null;

      const now = Date.now();
      if (now - lastInferTsRef.current < 800) return null;

      inferInFlightRef.current = true;
      lastInferTsRef.current = now;
      try {
        const res = await fetch(`${mlApiUrl}/predict`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });

        if (!res.ok) {
          setMlStatus("degraded");
          return null;
        }
        const data = (await res.json()) as MlPrediction;
        setMlStatus("connected");
        return data;
      } catch {
        setMlStatus("unavailable");
        return null;
      } finally {
        inferInFlightRef.current = false;
      }
    },
    [mlApiUrl],
  );

  const refreshCameraDevices = useCallback(async () => {
    if (
      typeof navigator === "undefined" ||
      !navigator.mediaDevices?.enumerateDevices
    ) {
      return;
    }

    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const cameras = devices
        .filter((device) => device.kind === "videoinput")
        .map((device, index) => ({
          id: device.deviceId,
          label: device.label || `Camera ${index + 1}`,
        }));

      setCameraDevices(cameras);
      setSelectedCameraId((current) => {
        if (current && cameras.some((camera) => camera.id === current)) {
          return current;
        }
        return cameras[0]?.id ?? "";
      });
    } catch {
      setCameraDevices([]);
    }
  }, []);

  const refreshSpeechSupport = useCallback(() => {
    if (typeof window === "undefined" || !("speechSynthesis" in window)) {
      setSpeechStatus("unsupported");
      setAvailableVoices(0);
      return;
    }

    const voices = window.speechSynthesis.getVoices();
    setAvailableVoices(voices.length);
    setSpeechStatus(voices.length > 0 ? "ready" : "loading");
  }, []);

  const applyPredictionVote = useCallback((ok: boolean) => {
    predictionVotesRef.current.push(ok);
    if (predictionVotesRef.current.length > PREDICTION_VOTE_WINDOW) {
      predictionVotesRef.current.shift();
    }
    const good = predictionVotesRef.current.filter(Boolean).length;
    const bad = predictionVotesRef.current.length - good;
    return good >= bad;
  }, []);

  const speakFeedback = useCallback(
    (nextState: "good" | "fix", message: string, eventKey: string) => {
      if (audioMode === "off") return;
      if (typeof window === "undefined" || !("speechSynthesis" in window)) {
        setSpeechStatus("unsupported");
        return;
      }

      if (lastSpokenMessageRef.current === message) {
        return;
      }

      const now = Date.now();
      const last = lastAudioEventRef.current;
      const stateChanged = lastAnnouncedStateRef.current !== nextState;
      if (
        !stateChanged &&
        last.key === eventKey &&
        now - last.at < AUDIO_COOLDOWN_MS
      ) {
        return;
      }

      window.speechSynthesis.resume();
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(message);
      const voices = window.speechSynthesis.getVoices();
      if (voices.length > 0) {
        utterance.voice = voices[0];
        setSpeechStatus("ready");
        setAvailableVoices(voices.length);
      } else {
        setSpeechStatus("blocked");
      }
      utterance.rate = 1;
      utterance.pitch = 1;
      utterance.volume = 0.9;
      window.speechSynthesis.speak(utterance);

      lastSpokenMessageRef.current = message;
      lastAudioEventRef.current = { key: eventKey, at: now };
      lastAnnouncedStateRef.current = nextState;
    },
    [audioMode],
  );

  const draw = useCallback(
    (result: PoseLandmarkerResult, faceResult?: FaceLandmarkerResult) => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      ctx.save();
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const landmarks = result.landmarks?.[0];
      const world = result.worldLandmarks?.[0];
      const faceLandmarks = getPrimaryFaceLandmarks(faceResult);
      if (landmarks && world) {
        const orient = detectOrientation(world as Point3[], landmarks);
        const dominantSide = dominantSideFromNorm(landmarks);
        const neckNorm: Point3 = {
          x: (landmarks[IDX.L_SHOULDER].x + landmarks[IDX.R_SHOULDER].x) / 2,
          y: (landmarks[IDX.L_SHOULDER].y + landmarks[IDX.R_SHOULDER].y) / 2,
          z: 0,
        };
        const sidePrimary =
          dominantSide === "left"
            ? {
                shoulder: IDX.L_SHOULDER,
                hip: IDX.L_HIP,
                ear: IDX.L_EAR,
                eye: IDX.L_EYE,
                elbow: IDX.L_ELBOW,
              }
            : {
                shoulder: IDX.R_SHOULDER,
                hip: IDX.R_HIP,
                ear: IDX.R_EAR,
                eye: IDX.R_EYE,
                elbow: IDX.R_ELBOW,
              };

        type NodeRef = number | Point3;
        let nodes: NodeRef[] = [];
        let links: Array<[NodeRef, NodeRef]> = [];

        if (orient.kind === "front") {
          nodes =
            overlayDetail === "detailed"
              ? [
                  IDX.NOSE,
                  IDX.L_EYE,
                  IDX.R_EYE,
                  IDX.L_EAR,
                  IDX.R_EAR,
                  IDX.L_SHOULDER,
                  IDX.R_SHOULDER,
                  IDX.L_ELBOW,
                  IDX.R_ELBOW,
                  IDX.L_HIP,
                  IDX.R_HIP,
                  ...(faceLandmarks
                    ? [{ ...faceLandmarks[FACE_IDX.CHIN] }]
                    : []),
                ]
              : [
                  IDX.NOSE,
                  IDX.L_EAR,
                  IDX.R_EAR,
                  IDX.L_SHOULDER,
                  IDX.R_SHOULDER,
                  IDX.L_HIP,
                  IDX.R_HIP,
                  ...(faceLandmarks
                    ? [{ ...faceLandmarks[FACE_IDX.CHIN] }]
                    : []),
                ];
          links =
            overlayDetail === "detailed"
              ? [
                  [IDX.NOSE, IDX.L_EYE],
                  [IDX.NOSE, IDX.R_EYE],
                  [IDX.L_EYE, IDX.L_EAR],
                  [IDX.R_EYE, IDX.R_EAR],
                  [IDX.L_SHOULDER, IDX.R_SHOULDER],
                  [IDX.L_SHOULDER, IDX.L_ELBOW],
                  [IDX.R_SHOULDER, IDX.R_ELBOW],
                  [IDX.L_SHOULDER, IDX.L_HIP],
                  [IDX.R_SHOULDER, IDX.R_HIP],
                  [IDX.L_HIP, IDX.R_HIP],
                  ...(faceLandmarks
                    ? [
                        [IDX.NOSE, { ...faceLandmarks[FACE_IDX.CHIN] }] as [
                          NodeRef,
                          NodeRef,
                        ],
                      ]
                    : []),
                ]
              : [
                  [IDX.NOSE, IDX.L_EAR],
                  [IDX.NOSE, IDX.R_EAR],
                  [IDX.L_SHOULDER, IDX.R_SHOULDER],
                  [IDX.L_SHOULDER, IDX.L_HIP],
                  [IDX.R_SHOULDER, IDX.R_HIP],
                  [IDX.L_HIP, IDX.R_HIP],
                  ...(faceLandmarks
                    ? [
                        [IDX.NOSE, { ...faceLandmarks[FACE_IDX.CHIN] }] as [
                          NodeRef,
                          NodeRef,
                        ],
                      ]
                    : []),
                ];
        } else if (
          orient.kind === "side_left" ||
          orient.kind === "side_right"
        ) {
          nodes =
            overlayDetail === "detailed"
              ? [
                  IDX.NOSE,
                  sidePrimary.eye,
                  sidePrimary.ear,
                  neckNorm,
                  IDX.L_SHOULDER,
                  IDX.R_SHOULDER,
                  sidePrimary.shoulder,
                  IDX.L_HIP,
                  IDX.R_HIP,
                  sidePrimary.elbow,
                  sidePrimary.hip,
                ]
              : [
                  IDX.NOSE,
                  sidePrimary.ear,
                  sidePrimary.shoulder,
                  sidePrimary.hip,
                ];
          links =
            overlayDetail === "detailed"
              ? [
                  [IDX.NOSE, sidePrimary.eye],
                  [sidePrimary.eye, sidePrimary.ear],
                  [sidePrimary.ear, neckNorm],
                  [IDX.L_SHOULDER, IDX.R_SHOULDER],
                  [IDX.NOSE, neckNorm],
                  [neckNorm, sidePrimary.shoulder],
                  [IDX.L_HIP, IDX.R_HIP],
                  [sidePrimary.shoulder, sidePrimary.elbow],
                  [sidePrimary.shoulder, sidePrimary.hip],
                ]
              : [
                  [IDX.NOSE, sidePrimary.ear],
                  [IDX.NOSE, sidePrimary.shoulder],
                  [sidePrimary.shoulder, sidePrimary.hip],
                ];
        } else {
          // Back or unknown
          nodes =
            overlayDetail === "detailed"
              ? [
                  IDX.L_SHOULDER,
                  IDX.R_SHOULDER,
                  IDX.L_ELBOW,
                  IDX.R_ELBOW,
                  IDX.L_HIP,
                  IDX.R_HIP,
                ]
              : [IDX.L_SHOULDER, IDX.R_SHOULDER, IDX.L_HIP, IDX.R_HIP];
          links =
            overlayDetail === "detailed"
              ? [
                  [IDX.L_SHOULDER, IDX.R_SHOULDER],
                  [IDX.L_SHOULDER, IDX.L_ELBOW],
                  [IDX.R_SHOULDER, IDX.R_ELBOW],
                  [IDX.L_SHOULDER, IDX.L_HIP],
                  [IDX.R_SHOULDER, IDX.R_HIP],
                  [IDX.L_HIP, IDX.R_HIP],
                ]
              : [
                  [IDX.L_SHOULDER, IDX.R_SHOULDER],
                  [IDX.L_SHOULDER, IDX.L_HIP],
                  [IDX.R_SHOULDER, IDX.R_HIP],
                  [IDX.L_HIP, IDX.R_HIP],
                ];
        }

        const getPoint = (ref: NodeRef) => {
          if (typeof ref === "number") return landmarks[ref];
          return ref;
        };
        const isRefVisible = (ref: NodeRef) => {
          if (typeof ref !== "number") return true;
          return visOk(landmarks[ref], DRAW_VIS_THRESHOLD);
        };

        const drawNode = (ref: NodeRef) => {
          const p = getPoint(ref);
          if (!p) return;
          if (!isRefVisible(ref)) return;
          if (p.x < 0 || p.x > 1 || p.y < 0 || p.y > 1) return;
          const x = p.x * canvas.width;
          const y = p.y * canvas.height;
          ctx.beginPath();
          ctx.arc(x, y, 4, 0, Math.PI * 2);
          ctx.fillStyle = "#e2e8f0";
          ctx.fill();
        };

        const drawLink = (a: NodeRef, b: NodeRef) => {
          const p1 = getPoint(a);
          const p2 = getPoint(b);
          if (!p1 || !p2) return;
          if (!isRefVisible(a) || !isRefVisible(b)) return;
          if (
            p1.x < 0 ||
            p1.x > 1 ||
            p1.y < 0 ||
            p1.y > 1 ||
            p2.x < 0 ||
            p2.x > 1 ||
            p2.y < 0 ||
            p2.y > 1
          ) {
            return;
          }
          ctx.beginPath();
          ctx.moveTo(p1.x * canvas.width, p1.y * canvas.height);
          ctx.lineTo(p2.x * canvas.width, p2.y * canvas.height);
          ctx.strokeStyle = "#22d3ee";
          ctx.lineWidth = 2;
          ctx.stroke();
        };

        links.forEach(([a, b]) => drawLink(a, b));
        nodes.forEach((idx) => drawNode(idx));
      }

      ctx.restore();
    },
    [overlayDetail],
  );

  const process = useCallback(
    (result: PoseLandmarkerResult, faceResult?: FaceLandmarkerResult) => {
      const world = result.worldLandmarks?.[0];
      const norm = result.landmarks?.[0];
      const faceLandmarks = getPrimaryFaceLandmarks(faceResult);
      if (!world) {
        setPill("detecting");
        setFeedback("Make sure your upper body is visible.");
        return;
      }

      const nose = world[IDX.NOSE];
      const ls = world[IDX.L_SHOULDER];
      const rs = world[IDX.R_SHOULDER];
      const lh = world[IDX.L_HIP];
      const rh = world[IDX.R_HIP];
      const noseN = norm?.[IDX.NOSE];
      const leN = norm?.[IDX.L_EYE];
      const reN = norm?.[IDX.R_EYE];
      const lEarN = norm?.[IDX.L_EAR];
      const rEarN = norm?.[IDX.R_EAR];
      const lsN = norm?.[IDX.L_SHOULDER];
      const rsN = norm?.[IDX.R_SHOULDER];
      const lhN = norm?.[IDX.L_HIP];
      const rhN = norm?.[IDX.R_HIP];
      if (
        !nose ||
        !ls ||
        !rs ||
        !noseN ||
        !lsN ||
        !rsN ||
        !lEarN ||
        !rEarN ||
        !leN ||
        !reN
      ) {
        return;
      }

      const health = Math.round(
        avgVisibility([noseN, lsN, rsN, lEarN, rEarN, leN, reN, lhN, rhN]) *
          100,
      );
      setTrackingHealth(health);

      if (!visOk(noseN) || !visOk(lsN) || !visOk(rsN) || health < 45) {
        setPill("detecting");
        setFeedback(
          "Low landmark confidence. Improve lighting and hold still.",
        );
        return;
      }

      const orient = detectOrientation(world as Point3[], norm);
      const frontCapture = classifyFrontCapture(
        world as Point3[],
        norm as { x: number; y: number; z: number; visibility?: number }[],
      );

      if (orient.kind === "unknown") {
        holdStillStartRef.current = 0;
        lastSmoothedRef.current = null;
        setPill("detecting");
        setScore(0);
        setMetrics({ trunkAngle: 0, headForward: 0, shoulderTilt: 0 });
        setSignedMetrics({ trunkAngle: 0, headForward: 0, shoulderTilt: 0 });
        setAssessmentTier(null);
        setDebugMetrics(DEFAULT_DEBUG_METRICS);
        setSilhouetteMetrics(DEFAULT_SILHOUETTE_METRICS);
        setStabilityScore(0);
        setFeedback("Move into view.");
        return;
      }

      if (orient.kind !== "front") {
        holdStillStartRef.current = 0;
        lastSmoothedRef.current = null;
        setPill("detecting");
        setScore(0);
        setMetrics({ trunkAngle: 0, headForward: 0, shoulderTilt: 0 });
        setSignedMetrics({ trunkAngle: 0, headForward: 0, shoulderTilt: 0 });
        setAssessmentTier(null);
        setDebugMetrics(DEFAULT_DEBUG_METRICS);
        setSilhouetteMetrics(DEFAULT_SILHOUETTE_METRICS);
        setStabilityScore(0);
        setFeedback(
          orient.kind === "back"
            ? "Face the camera."
            : "Turn and face the camera.",
        );
        return;
      }

      if (!frontCapture.tier) {
        holdStillStartRef.current = 0;
        lastSmoothedRef.current = null;
        setPill("detecting");
        setScore(0);
        setMetrics({ trunkAngle: 0, headForward: 0, shoulderTilt: 0 });
        setSignedMetrics({ trunkAngle: 0, headForward: 0, shoulderTilt: 0 });
        setAssessmentTier(null);
        setDebugMetrics(DEFAULT_DEBUG_METRICS);
        setSilhouetteMetrics(DEFAULT_SILHOUETTE_METRICS);
        setStabilityScore(0);
        setFeedback(
          frontCapture.upperVisible
            ? "Face the camera more directly."
            : "Keep your face and shoulders visible.",
        );
        return;
      }

      const captureTier = frontCapture.tier;
      setAssessmentTier(captureTier);
      const tierLabel =
        captureTier === "full_front" ? "Front view" : "Upper-front view";
      if (captureTier === "upper_front" && health < UPPER_FRONT_TRACKING_MIN) {
        holdStillStartRef.current = 0;
        lastSmoothedRef.current = null;
        setPill("detecting");
        setScore(0);
        setMetrics({ trunkAngle: 0, headForward: 0, shoulderTilt: 0 });
        setSignedMetrics({ trunkAngle: 0, headForward: 0, shoulderTilt: 0 });
        setFeedback("Keep both shoulders in view.");
        return;
      }
      const hipsReady =
        captureTier === "full_front" && !!lh && !!rh && !!lhN && !!rhN;

      const midShoulder = midpoint(ls as Point3, rs as Point3);
      const midHip: Point3 = hipsReady
        ? midpoint(lh as Point3, rh as Point3)
        : midShoulder;
      const shoulderWidth = Math.max(Math.abs(lsN.x - rsN.x), 1e-3);
      const shoulderWorldWidth = Math.max(
        planarDistance(ls as Point3, rs as Point3),
        1e-3,
      );
      const shoulderMidX = (lsN.x + rsN.x) / 2;
      const noseCenterOffset = Math.abs(noseN.x - shoulderMidX) / shoulderWidth;
      const chinPoint = faceLandmarks?.[FACE_IDX.CHIN];
      const chinCenterOffset = chinPoint
        ? Math.abs(chinPoint.x - shoulderMidX) / shoulderWidth
        : noseCenterOffset;
      const eyeOrEarTilt =
        visOk(leN) && visOk(reN)
          ? Math.abs(leN.y - reN.y) / shoulderWidth
          : Math.abs(lEarN.y - rEarN.y) / shoulderWidth;
      const chinForwardLean = chinPoint
        ? normalizedDepthDelta(chinPoint, midShoulder, shoulderWorldWidth)
        : 0;
      const chinLiftProxy = chinPoint
        ? Math.max(0, (chinPoint.y - noseN.y) / shoulderWidth)
        : 0;
      const upperBackwardLean =
        Math.max(0, chinLiftProxy - CHIN_LIFT_PROXY_NEUTRAL) *
        CHIN_LIFT_PROXY_TO_HEAD_LEAN_SCALE;
      const mouthLineTilt =
        faceLandmarks?.[FACE_IDX.L_MOUTH] && faceLandmarks?.[FACE_IDX.R_MOUTH]
          ? Math.abs(
              faceLandmarks[FACE_IDX.L_MOUTH].y -
                faceLandmarks[FACE_IDX.R_MOUTH].y,
            ) / shoulderWidth
          : 0;
      const upperForwardLean = Math.max(
        normalizedDepthDelta(nose as Point3, midShoulder, shoulderWorldWidth),
        chinForwardLean,
        eyeOrEarTilt,
        upperBackwardLean,
      );
      const upperShoulderTilt = Math.max(
        Math.abs(lsN.y - rsN.y) / shoulderWidth,
        mouthLineTilt,
      );
      setDebugMetrics({
        chinCenterOffset,
        chinForwardLean,
        chinLiftProxy,
        noseCenterOffset,
        mouthLineTilt,
        eyeOrEarTilt,
        upperForwardLean,
        upperShoulderTilt,
      });
      const tRaw =
        captureTier === "full_front" ? trunkAngleDeg(midShoulder, midHip) : 0;
      const hRaw =
        captureTier === "full_front"
          ? headForwardM(nose as Point3, midShoulder)
          : upperForwardLean;
      const sRaw =
        captureTier === "full_front"
          ? shoulderTiltM(ls as Point3, rs as Point3)
          : upperShoulderTilt;
      const alignmentRaw =
        captureTier === "upper_front"
          ? Math.max(noseCenterOffset, chinCenterOffset)
          : tRaw;
      const contourRaw = 0;
      const curvatureRaw = 0;
      const outlineRaw = 0;

      const tDeg = ema(emaRef.current.trunk, alignmentRaw);
      const hM = ema(emaRef.current.head, hRaw);
      const sM = ema(emaRef.current.shoulder, sRaw);
      const contour = ema(emaRef.current.contour, contourRaw);
      const curvature = ema(emaRef.current.curvature, curvatureRaw);
      const outline = ema(emaRef.current.outline, outlineRaw);
      emaRef.current = {
        trunk: tDeg,
        head: hM,
        shoulder: sM,
        contour,
        curvature,
        outline,
      };
      const tSigned =
        captureTier === "full_front"
          ? trunkAngleSignedDeg(midShoulder, midHip)
          : ((chinPoint?.x ?? noseN.x) - shoulderMidX) / shoulderWidth;
      const hSigned =
        captureTier === "full_front"
          ? headForwardSignedM(nose as Point3, midShoulder)
          : (() => {
              const forwardComponent = Math.max(
                normalizedDepthDelta(
                  nose as Point3,
                  midShoulder,
                  shoulderWorldWidth,
                ),
                chinForwardLean,
              );
              return forwardComponent >= upperBackwardLean
                ? forwardComponent
                : -upperBackwardLean;
            })();
      const sSigned =
        captureTier === "full_front"
          ? shoulderTiltSignedM(ls as Point3, rs as Point3)
          : (lsN.y - rsN.y) / shoulderWidth;

      const now = performance.now();
      if (!baselineMetricsRef.current[captureTier]) {
        baselineMetricsRef.current[captureTier] = {
          trunk: tDeg,
          head: hM,
          shoulder: sM,
        };
        setFeedback(
          captureTier === "upper_front"
            ? "Tracking upper posture. Stay centered."
            : `Tracking ${tierLabel}. Hold your position.`,
        );
      }

      pushLimited(buffersRef.current.trunk, tDeg);
      pushLimited(buffersRef.current.head, hM);
      pushLimited(buffersRef.current.shoulder, sM);
      pushLimited(buffersRef.current.contour, contour);
      pushLimited(buffersRef.current.curvature, curvature);
      pushLimited(buffersRef.current.outline, outline);
      const trunkVar = variance(buffersRef.current.trunk);
      const silhouetteStability = 0;
      setStabilityScore(stabilityFromVariance(trunkVar));
      const prevSmoothed = lastSmoothedRef.current;
      if (prevSmoothed) {
        const moved =
          Math.abs(tDeg - prevSmoothed.trunk) > 1.8 ||
          Math.abs(hM - prevSmoothed.head) > 0.02 ||
          Math.abs(sM - prevSmoothed.shoulder) > 0.01 ||
          Math.abs(contour - prevSmoothed.contour) > 0.025 ||
          Math.abs(curvature - prevSmoothed.curvature) > 0.025 ||
          Math.abs(outline - prevSmoothed.outline) > 2;
        if (moved) holdStillStartRef.current = 0;
      }
      lastSmoothedRef.current = {
        trunk: tDeg,
        head: hM,
        shoulder: sM,
        contour,
        curvature,
        outline,
      };
      if (holdStillStartRef.current === 0) holdStillStartRef.current = now;
      const holdReady = now - holdStillStartRef.current >= HOLD_STILL_MS;

      const effectiveSensitivity: Sensitivity =
        captureTier === "full_front"
          ? sensitivity
          : {
              trunkAngle: UPPER_FRONT_HEAD_OFFSET_THRESHOLD,
              headDistance: UPPER_FRONT_FORWARD_LEAN_THRESHOLD,
              shoulderTilt: UPPER_FRONT_SHOULDER_TILT_THRESHOLD,
            };

      const baseline =
        baselineMetricsRef.current[captureTier] ??
        (baselineMetricsRef.current[captureTier] = {
          trunk: tDeg,
          head: hM,
          shoulder: sM,
        });

      const dBase = computeDecision(
        captureTier,
        effectiveSensitivity,
        baseline,
      );
      const lookUpDetected = chinLiftProxy >= CHIN_LIFT_PROXY_THRESHOLD;
      const severeLookUp = chinLiftProxy >= CHIN_LIFT_PROXY_SEVERE;
      const d = lookUpDetected
        ? {
            ...dBase,
            ok: false,
            score: Math.min(dBase.score ?? 100, severeLookUp ? 52 : 68),
            msg: severeLookUp
              ? "Lower your chin and sit straighter."
              : "Lower your chin a little.",
            dominant: "head" as DominantIssue,
            h: Math.max(
              dBase.h ?? 0,
              effectiveSensitivity.headDistance * (severeLookUp ? 1.8 : 1.3),
            ),
            hRatio: Math.max(dBase.hRatio, severeLookUp ? 1.8 : 1.3),
          }
        : dBase;
      setMetrics({
        trunkAngle: d.t ?? 0,
        headForward: d.h ?? 0,
        shoulderTilt: d.s ?? 0,
      });
      setSignedMetrics({
        trunkAngle: tSigned,
        headForward: hSigned,
        shoulderTilt: sSigned,
      });
      setSilhouetteMetrics({
        neckForwardContour: 0,
        upperBackCurvature: 0,
        torsoOutlineAngle: 0,
        silhouetteStability,
      });

      if (!holdReady) {
        const clearIssue =
          d.hRatio >= 1.6 || d.sRatio >= 1.5 || d.tRatio >= 1.4;
        if (clearIssue) {
          const quickAudioPrompt =
            d.dominant === "shoulder"
              ? "Relax and level your shoulders."
              : d.dominant === "trunk"
                ? "Center your head a bit more."
                : "Bring your head back a little.";
          speakFeedback("fix", quickAudioPrompt, `quick-${quickAudioPrompt}`);
        }
        setPill("detecting");
        setFeedback(
          captureTier === "upper_front"
            ? "Hold still and keep both shoulders visible..."
            : "Hold still for stable reading...",
        );
        return;
      }

      const nextScore =
        captureTier === "upper_front"
          ? Math.min(d.score ?? 0, UPPER_FRONT_SCORE_CAP)
          : (d.score ?? 0);
      const votedOk = applyPredictionVote(d.ok);
      const stablePresentation = getFeedbackPresentation(
        nextScore,
        d.msg,
        d.h ?? 0,
        votedOk ? null : d.dominant,
        effectiveSensitivity.headDistance,
      );
      setScore(nextScore);
      setFeedback(d.msg);
      setPill(votedOk ? "good" : "fix");
      const stablePrompt = votedOk
        ? captureTier === "full_front"
          ? "Good posture."
          : "Looking good. Keep your head centered and shoulders level."
        : d.msg;
      if (votedOk) {
        speakFeedback("good", stablePresentation.audio, `good-${captureTier}`);
      } else {
        speakFeedback(
          "fix",
          stablePresentation.audio,
          stablePresentation.audio,
        );
      }
      if (votedOk) {
        setFeedback(stablePrompt);
      }

      if (d.t != null && d.h != null) {
        const logMsg = votedOk ? stablePrompt : d.msg;
        pushFeedback(
          nextScore,
          logMsg,
          d.t,
          d.h,
          votedOk ? null : d.dominant,
          effectiveSensitivity.headDistance,
        );
      }

      if (
        captureTier === "full_front" &&
        d.t != null &&
        d.h != null &&
        d.s != null &&
        d.rawT != null &&
        d.rawH != null &&
        d.rawS != null
      ) {
        const dT = d.rawT;
        const dH = d.rawH;
        const dS = d.rawS;
        void inferMl({
          trunk_angle: dT,
          head_forward: dH,
          shoulder_tilt: dS,
          trunk_variance: trunkVar,
          neck_forward_contour: 0,
          upper_back_curvature: 0,
          torso_outline_angle: 0,
          silhouette_stability: silhouetteStability,
        }).then((pred) => {
          if (!pred) return;

          const mlOk = pred.label === "proper";
          const votedMlOk = applyPredictionVote(mlOk);
          const mlScore = Math.round(clamp(pred.confidence * 100, 0, 100));
          const mlMsg =
            pred.feedback ||
            (mlOk ? "Good posture - keep it." : "Needs correction.");
          const headOnlyLocalWarning =
            d.dominant === "head" &&
            d.hRatio <= HEAD_FORWARD_GRACE_RATIO &&
            d.tRatio <= 1 &&
            d.sRatio <= 1;
          const localBlocksMl = !d.ok && !headOnlyLocalWarning;
          const finalOk = localBlocksMl ? false : votedMlOk;
          const finalScore = finalOk ? Math.min(nextScore, mlScore) : nextScore;
          const finalMsg = finalOk ? mlMsg : localBlocksMl ? d.msg : mlMsg;
          const finalDominant = finalOk || mlOk ? null : d.dominant;
          const finalPresentation = getFeedbackPresentation(
            finalScore,
            finalMsg,
            dH,
            finalDominant,
            effectiveSensitivity.headDistance,
          );

          setScore(finalScore);
          setFeedback(finalMsg);
          setPill(finalOk ? "good" : "fix");
          if (finalOk) {
            speakFeedback(
              "good",
              finalPresentation.audio,
              `good-${captureTier}-ml`,
            );
          } else {
            speakFeedback(
              "fix",
              finalPresentation.audio,
              finalPresentation.audio,
            );
          }
          pushFeedback(finalScore, finalMsg, dT, dH, finalDominant);
        });
      }
    },
    [
      applyPredictionVote,
      computeDecision,
      inferMl,
      pushFeedback,
      sensitivity,
      speakFeedback,
    ],
  );

  const loop = useCallback(
    function tick(): void {
      const pose = poseRef.current;
      const video = videoRef.current;
      if (!pose || !video) return;

      if (video.currentTime !== lastVideoTimeRef.current) {
        const now = performance.now();
        pose.detectForVideo(video, now, (result) => {
          const faceResult = faceRef.current?.detectForVideo(video, now);
          draw(result, faceResult);
          process(result, faceResult);
        });
        lastVideoTimeRef.current = video.currentTime;
      }

      rafRef.current = requestAnimationFrame(tick);
    },
    [draw, process],
  );

  const closeFloatingWindow = useCallback(() => {
    floatingWindowRequestIdRef.current += 1;
    floatingRootRef.current = null;
    setFloatingWindowReady(false);
    floatingWindowOriginRef.current = null;

    const pipWindow = floatingWindowRef.current;
    floatingWindowRef.current = null;

    if (pipWindow && !pipWindow.closed) {
      pipWindow.close();
    }
  }, []);

  const openFloatingWindow = useCallback(async (origin: Exclude<FloatingWindowOrigin, null>) => {
    if (typeof window === "undefined") return;

    const pipApi = (
      window as Window & {
        documentPictureInPicture?: DocumentPictureInPictureApi;
      }
    ).documentPictureInPicture;

    if (!pipApi?.requestWindow) return;

    if (floatingWindowRef.current && !floatingWindowRef.current.closed) {
      if (origin === "manual") {
        floatingWindowRef.current.focus();
      }
      setFloatingWindowReady(true);
      return;
    }

    if (floatingWindowOpeningRef.current) {
      await floatingWindowOpeningRef.current;
      return;
    }

    const requestId = ++floatingWindowRequestIdRef.current;
    floatingWindowOriginRef.current = origin;
    const opening = (async () => {
      try {
        const pipWindow = await pipApi.requestWindow({
          width: 456,
          height: 535,
        });

        if (requestId !== floatingWindowRequestIdRef.current) {
          pipWindow.close();
          return;
        }

        floatingWindowRef.current = pipWindow;
        pipWindow.document.title = "Uprightly Status";
        pipWindow.document.body.innerHTML = "";
        pipWindow.document.body.style.margin = "0";
        pipWindow.document.documentElement.style.width = "100%";
        pipWindow.document.documentElement.style.height = "100%";
        pipWindow.document.body.style.width = "100%";
        pipWindow.document.body.style.height = "100%";
        pipWindow.document.body.style.minHeight = "0";
        pipWindow.document.body.style.background = "transparent";
        pipWindow.document.body.style.overflow = "hidden";

        Array.from(
          document.querySelectorAll("style, link[rel='stylesheet']"),
        ).forEach((node) => {
          pipWindow.document.head.appendChild(node.cloneNode(true));
        });

        const root = pipWindow.document.createElement("div");
        root.id = "floating-status-root";
        root.style.width = "100%";
        root.style.height = "100%";
        pipWindow.document.body.appendChild(root);
        floatingRootRef.current = root;
        setFloatingWindowReady(true);

        pipWindow.addEventListener("pagehide", () => {
          if (floatingWindowRef.current !== pipWindow) return;
          floatingWindowRef.current = null;
          floatingRootRef.current = null;
          floatingWindowOriginRef.current = null;
          setFloatingWindowReady(false);
        });
      } catch (error) {
        if (requestId === floatingWindowRequestIdRef.current) {
          floatingWindowOriginRef.current = null;
          setFloatingWindowReady(false);
        }
        throw error;
      }
    })();

    floatingWindowOpeningRef.current = opening;
    try {
      await opening;
    } finally {
      if (floatingWindowOpeningRef.current === opening) {
        floatingWindowOpeningRef.current = null;
      }
    }
  }, []);

  const start = useCallback(
    async (cameraId = selectedCameraId) => {
      try {
        setPill("loading");
        setFeedback("Loading pose model...");
        await ensureLandmarker();

        setFeedback("Requesting webcam...");
        const stream = await navigator.mediaDevices.getUserMedia({
          video: cameraId
            ? {
                deviceId: { exact: cameraId },
                width: { ideal: 960 },
                height: { ideal: 540 },
                frameRate: { ideal: 30, max: 30 },
              }
            : {
                facingMode: { ideal: "user" },
                width: { ideal: 960 },
                height: { ideal: 540 },
                frameRate: { ideal: 30, max: 30 },
              },
          audio: false,
        });

        streamRef.current = stream;
        const video = videoRef.current;
        if (!video) return;

        video.srcObject = stream;
        await new Promise<void>((resolve) => {
          video.onloadedmetadata = () => resolve();
        });

        const canvas = canvasRef.current;
        if (!canvas) return;

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        resetBuffers();
        await refreshCameraDevices();
        setIsActive(true);
        setPill("detecting");
        setFeedback("Face the camera. Keep your head and shoulders visible.");
        rafRef.current = requestAnimationFrame(loop);
      } catch (error) {
        console.error(error);
        setIsActive(false);
        setPill("error");
        setFeedback("Failed to start. Check camera permission and reload.");
      }
    },
    [
      ensureLandmarker,
      loop,
      refreshCameraDevices,
      resetBuffers,
      selectedCameraId,
    ],
  );

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [feedbacks]);

  useEffect(() => stop, [stop]);

  useEffect(() => {
    if (typeof window !== "undefined") {
      window.localStorage.setItem(THEME_STORAGE_KEY, theme);
    }
    document.documentElement.style.colorScheme = theme;
    document.body.style.background = theme === "dark" ? "#10100f" : "#f2efe7";
    if (floatingWindowRef.current?.document?.body) {
      floatingWindowRef.current.document.body.style.background = "transparent";
    }
  }, [theme]);

  useEffect(() => {
    window.localStorage.setItem(
      FLOATING_WINDOW_STORAGE_KEY,
      String(floatingWindowEnabled),
    );
  }, [floatingWindowEnabled]);

  useEffect(() => {
    if (typeof window === "undefined") return;

    const syncPageFocus = () => {
      setIsPageFocused(document.hasFocus() && !document.hidden);
    };

    syncPageFocus();
    document.addEventListener("visibilitychange", syncPageFocus);
    window.addEventListener("focus", syncPageFocus);
    window.addEventListener("blur", syncPageFocus);

    return () => {
      document.removeEventListener("visibilitychange", syncPageFocus);
      window.removeEventListener("focus", syncPageFocus);
      window.removeEventListener("blur", syncPageFocus);
    };
  }, []);

  useEffect(() => {
    if (
      !floatingWindowEnabled &&
      floatingWindowOriginRef.current === "auto"
    ) {
      closeFloatingWindow();
    }
  }, [closeFloatingWindow, floatingWindowEnabled]);

  useEffect(() => closeFloatingWindow, [closeFloatingWindow]);

  useEffect(() => {
    if (!isActive && floatingWindowOriginRef.current !== null) {
      closeFloatingWindow();
    }
  }, [closeFloatingWindow, isActive]);

  useEffect(() => {
    if (typeof window === "undefined") return;

    const pipApi = (
      window as Window & {
        documentPictureInPicture?: DocumentPictureInPictureApi;
      }
    ).documentPictureInPicture;
    const hasDocumentPip = !!pipApi?.requestWindow;

    if (!hasDocumentPip) {
      setAutoPipStatus("unsupported");
      return;
    }

    if (!window.isSecureContext || !("mediaSession" in navigator)) {
      setAutoPipStatus("manual_only");
      return;
    }

    if (typeof navigator.mediaSession.setCameraActive !== "function") {
      setAutoPipStatus("manual_only");
      return;
    }

    try {
      navigator.mediaSession.setActionHandler(AUTO_PIP_ACTION, () => {});
      navigator.mediaSession.setActionHandler(AUTO_PIP_ACTION, null);
      setAutoPipStatus("supported");
    } catch {
      setAutoPipStatus("blocked");
    }
  }, []);

  useEffect(() => {
    if (
      typeof window === "undefined" ||
      !("mediaSession" in navigator) ||
      typeof navigator.mediaSession.setCameraActive !== "function"
    ) {
      return;
    }

    try {
      navigator.mediaSession.setCameraActive(isActive);
    } catch (error) {
      console.warn(
        "Unable to update camera activity for media session:",
        error,
      );
    }

    return () => {
      try {
        navigator.mediaSession.setCameraActive(false);
      } catch {
        // Ignore cleanup failures in browsers with partial Media Session support.
      }
    };
  }, [isActive]);

  useEffect(() => {
    const supportsFloatingWindow =
      typeof window !== "undefined" &&
      !!(
        window as Window & {
          documentPictureInPicture?: DocumentPictureInPictureApi;
        }
      ).documentPictureInPicture?.requestWindow;

    if (
      typeof window === "undefined" ||
      !("mediaSession" in navigator) ||
      !supportsFloatingWindow ||
      !isActive ||
      !floatingWindowEnabled
    ) {
      return;
    }

    try {
      navigator.mediaSession.setActionHandler(AUTO_PIP_ACTION, () => {
        void openFloatingWindow("auto").catch((error) => {
          console.error("Automatic floating window failed:", error);
        });
      });
    } catch (error) {
      console.warn("Automatic floating window is not supported here:", error);
      return;
    }

    return () => {
      try {
        navigator.mediaSession.setActionHandler(AUTO_PIP_ACTION, null);
      } catch {
        // Ignore cleanup failures in browsers that partially expose Media Session.
      }
    };
  }, [floatingWindowEnabled, isActive, openFloatingWindow]);

  useEffect(() => {
    if (!isActive || !floatingWindowEnabled) return;

    const closeAutomaticWindowOnReturn = () => {
      if (
        !document.hidden &&
        document.hasFocus() &&
        floatingWindowOriginRef.current === "auto"
      ) {
        closeFloatingWindow();
      }
    };

    document.addEventListener(
      "visibilitychange",
      closeAutomaticWindowOnReturn,
    );
    window.addEventListener("focus", closeAutomaticWindowOnReturn);

    return () => {
      document.removeEventListener(
        "visibilitychange",
        closeAutomaticWindowOnReturn,
      );
      window.removeEventListener("focus", closeAutomaticWindowOnReturn);
    };
  }, [closeFloatingWindow, floatingWindowEnabled, isActive]);

  const beginGuidedTrial = useCallback(() => {
    window.localStorage.setItem(GUIDED_TRIAL_STORAGE_KEY, "true");
    setCompletedGuidedTrialSteps(new Set());
    setGuidedTrialStepIndex(0);
    setGuidedTrialPending(false);
    setShowGuidedTrialIntro(false);
    setShowGuidedTrial(true);
    setShowSettings(false);
    setShowSessionLog(false);
  }, []);

  const dismissGuidedTrialIntro = useCallback(() => {
    window.localStorage.setItem(GUIDED_TRIAL_STORAGE_KEY, "true");
    setGuidedTrialPending(false);
    setShowGuidedTrialIntro(false);
  }, []);

  const finishGuidedTrial = useCallback(() => {
    window.localStorage.setItem(GUIDED_TRIAL_STORAGE_KEY, "true");
    setShowGuidedTrial(false);
  }, []);

  const requestGuidedTrial = useCallback(() => {
    setGuidedTrialPending(false);
    setShowSettings(false);
    setShowSessionLog(false);
    setShowGuidedTrial(false);
    setShowGuidedTrialIntro(true);
  }, []);

  const confirmGuidedTrial = useCallback(() => {
    if (isActive) {
      beginGuidedTrial();
      return;
    }

    setGuidedTrialPending(true);
    setShowGuidedTrialIntro(false);
    void start();
  }, [beginGuidedTrial, isActive, start]);

  useEffect(() => {
    if (
      !isActive ||
      trackingHealth < 45 ||
      showTutorial ||
      showPrivacyNotice ||
      showPrivacyPolicy ||
      showGuidedTrial ||
      showGuidedTrialIntro ||
      window.localStorage.getItem(GUIDED_TRIAL_STORAGE_KEY) === "true"
    ) {
      return;
    }

    const timer = window.setTimeout(() => setShowGuidedTrialIntro(true), 650);
    return () => window.clearTimeout(timer);
  }, [
    isActive,
    showGuidedTrial,
    showGuidedTrialIntro,
    showPrivacyNotice,
    showPrivacyPolicy,
    showTutorial,
    trackingHealth,
  ]);

  useEffect(() => {
    if (!guidedTrialPending) return;

    if (pill === "error") {
      const errorTimer = window.setTimeout(() => {
        setGuidedTrialPending(false);
        setShowGuidedTrialIntro(true);
      }, 0);
      return () => window.clearTimeout(errorTimer);
    }

    if (!isActive || trackingHealth < 45) return;
    const readyTimer = window.setTimeout(beginGuidedTrial, 350);
    return () => window.clearTimeout(readyTimer);
  }, [
    beginGuidedTrial,
    guidedTrialPending,
    isActive,
    pill,
    trackingHealth,
  ]);

  useEffect(() => {
    if (!showGuidedTrial) return;
    const stepId = GUIDED_TRIAL_STEPS[guidedTrialStepIndex]?.id;
    if (!stepId || completedGuidedTrialSteps.has(stepId)) return;

    const normalizedFeedback = feedback.toLowerCase();
    const detected =
      (stepId === "baseline" && trackingHealth >= 45) ||
      (stepId === "forward" &&
        (normalizedFeedback.includes("head back") ||
          normalizedFeedback.includes("sit straighter"))) ||
      (stepId === "shoulders" &&
        normalizedFeedback.includes("level your shoulders")) ||
      (stepId === "framing" &&
        (normalizedFeedback.includes("face the camera") ||
          normalizedFeedback.includes("shoulders in view") ||
          normalizedFeedback.includes("shoulders visible"))) ||
      (stepId === "neutral" && pill === "good" && score >= 80) ||
      (stepId === "floating" && floatingWindowReady);

    if (!detected) return;
    const detectedTimer = window.setTimeout(() => {
      setCompletedGuidedTrialSteps((completed) => {
        const next = new Set(completed);
        next.add(stepId);
        return next;
      });
    }, 0);
    return () => window.clearTimeout(detectedTimer);
  }, [
    completedGuidedTrialSteps,
    feedback,
    floatingWindowReady,
    guidedTrialStepIndex,
    pill,
    score,
    showGuidedTrial,
    trackingHealth,
  ]);

  useEffect(() => {
    if (!showGuidedTrialIntro) return;
    const focusFrame = window.requestAnimationFrame(() => {
      guidedTrialIntroRef.current
        ?.querySelector<HTMLElement>("[data-guided-trial-autofocus]")
        ?.focus();
    });
    const containGuidedTrialFocus = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        dismissGuidedTrialIntro();
        return;
      }
      if (event.key !== "Tab" || !guidedTrialIntroRef.current) return;
      const focusable = Array.from(
        guidedTrialIntroRef.current.querySelectorAll<HTMLElement>(
          "button:not([disabled]), [href], [tabindex]:not([tabindex='-1'])",
        ),
      );
      if (focusable.length === 0) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    window.addEventListener("keydown", containGuidedTrialFocus);
    return () => {
      window.cancelAnimationFrame(focusFrame);
      window.removeEventListener("keydown", containGuidedTrialFocus);
    };
  }, [dismissGuidedTrialIntro, showGuidedTrialIntro]);

  useEffect(() => {
    if (!mlApiUrl) {
      setMlStatus("unavailable");
      return;
    }

    const controller = new AbortController();
    setMlStatus("checking");
    void fetch(`${mlApiUrl.replace(/\/$/, "")}/health`, {
      signal: controller.signal,
    })
      .then((response) => {
        setMlStatus(response.ok ? "connected" : "degraded");
      })
      .catch((error: unknown) => {
        if ((error as { name?: string }).name !== "AbortError") {
          setMlStatus("unavailable");
        }
      });

    return () => controller.abort();
  }, [mlApiUrl]);

  useEffect(() => {
    if (
      isActive &&
      typeof window !== "undefined" &&
      window.matchMedia("(min-width: 1024px)").matches
    ) {
      setShowSessionLog(true);
    }
  }, [isActive]);

  useEffect(() => {
    void refreshCameraDevices();

    if (
      typeof navigator === "undefined" ||
      !navigator.mediaDevices?.addEventListener
    ) {
      return;
    }

    const handleDeviceChange = () => {
      void refreshCameraDevices();
    };

    navigator.mediaDevices.addEventListener("devicechange", handleDeviceChange);
    return () => {
      navigator.mediaDevices.removeEventListener(
        "devicechange",
        handleDeviceChange,
      );
    };
  }, [refreshCameraDevices]);

  useEffect(() => {
    if (audioMode !== "off") return;
    if (typeof window !== "undefined" && "speechSynthesis" in window) {
      window.speechSynthesis.cancel();
    }
  }, [audioMode]);

  useEffect(() => {
    refreshSpeechSupport();
    if (typeof window === "undefined" || !("speechSynthesis" in window)) {
      return;
    }

    const handleVoicesChanged = () => {
      refreshSpeechSupport();
    };

    window.speechSynthesis.addEventListener(
      "voiceschanged",
      handleVoicesChanged,
    );
    return () => {
      window.speechSynthesis.removeEventListener(
        "voiceschanged",
        handleVoicesChanged,
      );
    };
  }, [refreshSpeechSupport]);

  const openPrivacyPolicy = useCallback(() => {
    privacyReturnFocusRef.current =
      document.activeElement instanceof HTMLElement
        ? document.activeElement
        : null;
    setShowPrivacyPolicy(true);
  }, []);

  const closePrivacyPolicy = useCallback(() => {
    setShowPrivacyPolicy(false);
    const returnTarget = privacyReturnFocusRef.current;
    privacyReturnFocusRef.current = null;
    window.requestAnimationFrame(() => returnTarget?.focus());
  }, []);

  const continueFromPrivacyNotice = useCallback(() => {
    if (rememberPrivacyNotice) {
      window.localStorage.setItem(
        PRIVACY_NOTICE_STORAGE_KEY,
        PRIVACY_NOTICE_VERSION,
      );
    }

    setShowPrivacyNotice(false);
    if (window.localStorage.getItem(TUTORIAL_SEEN_STORAGE_KEY) !== "true") {
      window.setTimeout(() => setShowTutorial(true), 0);
    }
  }, [rememberPrivacyNotice]);

  const continueToPrivacyNotice = useCallback(() => {
    setStartupNoticeStep("privacy");
    window.setTimeout(() => {
      privacyNoticeRef.current
        ?.querySelector<HTMLElement>("[data-privacy-autofocus]")
        ?.focus();
    }, 0);
  }, []);

  const reviewPrivacyNotice = useCallback(() => {
    window.localStorage.removeItem(PRIVACY_NOTICE_STORAGE_KEY);
    setStartupNoticeStep("purpose");
    setRememberPrivacyNotice(false);
    setShowPrivacyPolicy(false);
    setShowPrivacyNotice(true);
  }, []);

  useEffect(() => {
    const dialog = showPrivacyPolicy
      ? privacyPolicyRef.current
      : showPrivacyNotice
        ? startupNoticeStep === "purpose"
          ? purposeNoticeRef.current
          : privacyNoticeRef.current
        : null;
    if (!dialog) return;

    const focusFrame = window.requestAnimationFrame(() => {
      dialog.querySelector<HTMLElement>("[data-privacy-autofocus]")?.focus();
    });

    const containFocus = (event: KeyboardEvent) => {
      if (event.key === "Escape" && showPrivacyPolicy) {
        event.preventDefault();
        event.stopImmediatePropagation();
        closePrivacyPolicy();
        return;
      }
      if (event.key !== "Tab") return;

      const focusable = Array.from(
        dialog.querySelectorAll<HTMLElement>(
          "button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex='-1'])",
        ),
      );
      if (focusable.length === 0) return;

      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };

    window.addEventListener("keydown", containFocus, true);
    return () => {
      window.cancelAnimationFrame(focusFrame);
      window.removeEventListener("keydown", containFocus, true);
    };
  }, [
    closePrivacyPolicy,
    showPrivacyNotice,
    showPrivacyPolicy,
    startupNoticeStep,
  ]);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (
        showTutorial ||
        showGuidedTrialIntro ||
        showGuidedTrial ||
        showPrivacyNotice ||
        showPrivacyPolicy ||
        e.code !== "Space" ||
        e.repeat
      )
        return;

      const target = e.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.tagName === "SELECT" ||
          target.isContentEditable)
      ) {
        return;
      }

      e.preventDefault();
      if (pill === "loading") return;
      if (isActive) stop();
      else void start();
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [
    isActive,
    pill,
    showPrivacyNotice,
    showPrivacyPolicy,
    showGuidedTrial,
    showGuidedTrialIntro,
    showTutorial,
    start,
    stop,
  ]);

  useGSAP(
    () => {
      const noticeRoot =
        startupNoticeStep === "purpose"
          ? purposeNoticeRef.current
          : privacyNoticeRef.current;
      if (!showPrivacyNotice || showPrivacyPolicy || !noticeRoot) {
        return;
      }

      const reduceMotion = window.matchMedia(
        "(prefers-reduced-motion: reduce)",
      ).matches;
      const revealItems = noticeRoot.querySelectorAll(
        "[data-onboarding-reveal]",
      );
      const ambientShapes = noticeRoot.querySelectorAll(
        "[data-onboarding-ambient]",
      );

      if (reduceMotion) {
        gsap.set(revealItems, { autoAlpha: 1, y: 0, scale: 1 });
        gsap.set(ambientShapes, { autoAlpha: 1, scale: 1 });
        return;
      }

      const timeline = gsap.timeline({ defaults: { ease: "power3.out" } });
      timeline
        .fromTo(
          ambientShapes,
          { autoAlpha: 0, scale: 0.72 },
          { autoAlpha: 1, scale: 1, duration: 1.05, stagger: 0.1 },
        )
        .fromTo(
          revealItems,
          { autoAlpha: 0, y: 22, scale: 0.985 },
          {
            autoAlpha: 1,
            y: 0,
            scale: 1,
            duration: 0.58,
            stagger: 0.09,
          },
          0.12,
        );
      timeline.eventCallback("onComplete", () => {
        noticeRoot
          .querySelector<HTMLElement>("[data-privacy-autofocus]")
          ?.focus();
      });
    },
    {
      dependencies: [
        showPrivacyNotice,
        showPrivacyPolicy,
        startupNoticeStep,
      ],
    },
  );

  const currentTutorialStep = TUTORIAL_STEPS[tutorialStepIndex];
  const currentGuidedTrialStep = GUIDED_TRIAL_STEPS[guidedTrialStepIndex];
  const currentGuidedTrialDetected = completedGuidedTrialSteps.has(
    currentGuidedTrialStep.id,
  );

  const closeTutorial = useCallback(() => {
    window.localStorage.setItem(TUTORIAL_SEEN_STORAGE_KEY, "true");
    setShowTutorial(false);
    setShowSettings(tutorialPanelStateRef.current.showSettings);
    setShowSessionLog(tutorialPanelStateRef.current.showSessionLog);
  }, []);

  const openTutorial = useCallback(() => {
    tutorialPanelStateRef.current = { showSettings, showSessionLog };
    tutorialReturnFocusRef.current =
      document.activeElement instanceof HTMLElement
        ? document.activeElement
        : null;
    setTutorialStepIndex(0);
    setShowTutorial(true);
  }, [showSessionLog, showSettings]);

  const goToNextTutorialStep = useCallback(() => {
    if (tutorialStepIndex >= TUTORIAL_STEPS.length - 1) {
      closeTutorial();
      return;
    }
    setTutorialStepIndex((index) => index + 1);
  }, [closeTutorial, tutorialStepIndex]);

  const goToPreviousTutorialStep = useCallback(() => {
    setTutorialStepIndex((index) => Math.max(index - 1, 0));
  }, []);

  useEffect(() => {
    if (!showTutorial) return;

    if (currentTutorialStep.target === "settings-panel") {
      setShowSettings(true);
      setShowSessionLog(false);
    } else if (currentTutorialStep.target === "session-log") {
      setShowSettings(false);
      setShowSessionLog(true);
    } else {
      setShowSettings(false);
      setShowSessionLog(false);
    }
  }, [currentTutorialStep.target, showTutorial]);

  useEffect(() => {
    if (!showTutorial) return;

    const mediaQuery = window.matchMedia(
      "(max-width: 639px), (max-height: 680px)",
    );
    const updateViewportMode = () => setIsCompactTutorial(mediaQuery.matches);
    updateViewportMode();
    mediaQuery.addEventListener("change", updateViewportMode);
    return () => mediaQuery.removeEventListener("change", updateViewportMode);
  }, [showTutorial]);

  useEffect(() => {
    if (!showTutorial || !tutorialCardRef.current) return;

    const observer = new ResizeObserver(([entry]) => {
      if (!entry) return;
      setTutorialCardSize({
        width: entry.borderBoxSize?.[0]?.inlineSize ?? entry.contentRect.width,
        height: entry.borderBoxSize?.[0]?.blockSize ?? entry.contentRect.height,
      });
    });
    observer.observe(tutorialCardRef.current);
    return () => observer.disconnect();
  }, [showTutorial, tutorialStepIndex]);

  useEffect(() => {
    if (!showTutorial) {
      setTutorialTargetRect(null);
      return;
    }

    let frame = 0;

    const measure = () => {
      window.cancelAnimationFrame(frame);
      frame = window.requestAnimationFrame(() => {
        setTutorialTargetRect(
          findVisibleTourTarget(currentTutorialStep.target),
        );
      });
    };

    const targetElement = findVisibleTourElement(currentTutorialStep.target);
    if (targetElement) {
      const targetRect = targetElement.getBoundingClientRect();
      const isOutsideViewport =
        targetRect.top < 12 || targetRect.bottom > window.innerHeight - 12;
      if (isOutsideViewport) {
        targetElement.scrollIntoView({
          behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches
            ? "auto"
            : "smooth",
          block: "nearest",
          inline: "nearest",
        });
      }
    }

    measure();
    const deferredMeasure = window.setTimeout(measure, 280);

    window.addEventListener("resize", measure);
    window.addEventListener("scroll", measure, true);

    return () => {
      window.clearTimeout(deferredMeasure);
      window.cancelAnimationFrame(frame);
      window.removeEventListener("resize", measure);
      window.removeEventListener("scroll", measure, true);
    };
  }, [currentTutorialStep.target, showSettings, showSessionLog, showTutorial]);

  useEffect(() => {
    if (!showTutorial) return;

    if (!tutorialReturnFocusRef.current) {
      tutorialReturnFocusRef.current =
        document.activeElement instanceof HTMLElement
          ? document.activeElement
          : null;
    }

    const focusFrame = window.requestAnimationFrame(() => {
      tutorialCardRef.current
        ?.querySelector<HTMLElement>("[data-tutorial-primary]")
        ?.focus();
    });

    return () => {
      window.cancelAnimationFrame(focusFrame);
      const returnTarget = tutorialReturnFocusRef.current;
      tutorialReturnFocusRef.current = null;
      window.requestAnimationFrame(() => returnTarget?.focus());
    };
  }, [showTutorial]);

  useEffect(() => {
    const onTransientKeyDown = (event: KeyboardEvent) => {
      if (showPrivacyNotice || showPrivacyPolicy) return;

      if (!showTutorial) {
        if (event.key !== "Escape") return;
        if (showSettings) setShowSettings(false);
        else if (showSessionLog) setShowSessionLog(false);
        return;
      }

      if (event.repeat) return;
      const target = event.target as HTMLElement | null;
      const isEditable =
        target?.matches("input, textarea, select, [contenteditable='true']") ??
        false;

      if (event.key === "Escape") {
        event.preventDefault();
        closeTutorial();
        return;
      }

      if (event.key === "Tab") {
        const focusable = Array.from(
          tutorialCardRef.current?.querySelectorAll<HTMLElement>(
            "button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex='-1'])",
          ) ?? [],
        );
        if (focusable.length === 0) return;
        const first = focusable[0];
        const last = focusable[focusable.length - 1];
        if (event.shiftKey && document.activeElement === first) {
          event.preventDefault();
          last.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
          event.preventDefault();
          first.focus();
        }
        return;
      }

      if (isEditable) return;

      if (event.key === "ArrowLeft") {
        event.preventDefault();
        goToPreviousTutorialStep();
        return;
      }

      const isSpace = event.code === "Space" || event.key === " ";
      if (isSpace && target?.closest("button, a, [role='button']")) return;

      if (event.key === "ArrowRight" || isSpace) {
        event.preventDefault();
        goToNextTutorialStep();
      }
    };

    window.addEventListener("keydown", onTransientKeyDown);
    return () => window.removeEventListener("keydown", onTransientKeyDown);
  }, [
    closeTutorial,
    goToNextTutorialStep,
    goToPreviousTutorialStep,
    showPrivacyNotice,
    showPrivacyPolicy,
    showSessionLog,
    showSettings,
    showTutorial,
  ]);

  useGSAP(
    () => {
      if (
        !showTutorial ||
        !tutorialContentRef.current ||
        window.matchMedia("(prefers-reduced-motion: reduce)").matches
      ) {
        return;
      }

      gsap.fromTo(
        tutorialContentRef.current.querySelectorAll("[data-tutorial-reveal]"),
        { autoAlpha: 0, y: 10 },
        {
          autoAlpha: 1,
          y: 0,
          duration: 0.36,
          stagger: 0.055,
          ease: "power2.out",
        },
      );
    },
    {
      dependencies: [showTutorial, tutorialStepIndex],
      scope: tutorialCardRef,
    },
  );

  useGSAP(
    () => {
      if (
        !showSessionLog ||
        feedbacks.length === 0 ||
        window.matchMedia("(prefers-reduced-motion: reduce)").matches
      ) {
        return;
      }

      const newestCard = sessionLogStackRef.current?.querySelector(
        "[data-feedback-card]",
      );
      if (!newestCard) return;
      gsap.fromTo(
        newestCard,
        { autoAlpha: 0, y: 18, scale: 0.97 },
        {
          autoAlpha: 1,
          y: 0,
          scale: 1,
          duration: 0.42,
          ease: "power3.out",
        },
      );
    },
    {
      dependencies: [feedbacks.length, showSessionLog],
      scope: sessionLogStackRef,
    },
  );

  const getScoreColor = (s: number) => {
    if (s > 80) return "text-[#91a889]";
    if (s > 60) return "text-amber-400";
    return "text-rose-400";
  };
  const metricMeta =
    assessmentTier === "upper_front"
      ? {
          trunk: { label: "Head Offset", unit: "norm" },
          head: { label: "Head Lean", unit: "norm" },
          shoulder: { label: "Shoulder Level", unit: "norm" },
          thresholds: {
            trunk: UPPER_FRONT_HEAD_OFFSET_THRESHOLD,
            head: UPPER_FRONT_FORWARD_LEAN_THRESHOLD,
            shoulder: UPPER_FRONT_SHOULDER_TILT_THRESHOLD,
          },
        }
      : {
          trunk: { label: "Trunk Angle", unit: "deg" },
          head: { label: "Head Forward", unit: "m" },
          shoulder: { label: "Shoulder Tilt", unit: "m" },
          thresholds: {
            trunk: sensitivity.trunkAngle,
            head: sensitivity.headDistance,
            shoulder: sensitivity.shoulderTilt,
          },
        };

  const isLoading = pill === "loading";
  const isDarkTheme = theme === "dark";
  const shellClass = isDarkTheme
    ? "uprightly-shell-dark bg-[#10100f] text-[#f4f0e8]"
    : "uprightly-shell-light bg-[#f2efe7] text-[#1c1b19]";
  const heroCardClass = isDarkTheme
    ? "bg-gradient-to-br from-white/[0.08] to-transparent border-white/10 hover:bg-white/[0.08]"
    : "bg-gradient-to-br from-white to-stone-50 border-stone-200 hover:bg-white";
  const stageClass = isDarkTheme
    ? "bg-[#151412] border-white/8"
    : isActive
      ? "bg-[#fffdf8] border-[#ded8cc]"
      : "bg-[#eef4fa] border-[#cbdbea]";
  const stageGlassClass = isDarkTheme
    ? "bg-black/45 backdrop-blur-md border-white/10"
    : "bg-white/88 backdrop-blur-md border-stone-200";
  const settingsPanelClass = isDarkTheme
    ? "border-white/10 bg-[#171715]/95 backdrop-blur-2xl"
    : "border-stone-200 bg-white/95 backdrop-blur-2xl";
  const subtleTextClass = isDarkTheme ? "text-white/60" : "text-stone-500";
  const mutedTextClass = isDarkTheme ? "text-white/60" : "text-stone-500";
  const quietTextClass = isDarkTheme ? "text-white/70" : "text-stone-600";
  const iconButtonClass = isDarkTheme
    ? "border-white/30 bg-white/10 text-white/80 hover:bg-white hover:text-black"
    : "border-stone-200 bg-white text-stone-700 hover:bg-stone-100";
  const sessionLogIconButtonClass = isDarkTheme
    ? "border-white/10 bg-black/30 text-white/80 hover:bg-black/45 hover:text-white"
    : "border-stone-200 bg-white/75 text-stone-700 hover:bg-white hover:text-stone-900";
  const primaryButtonClass = isDarkTheme
    ? "bg-[#e8e7e2] text-[#171612] hover:bg-white shadow-lg"
    : "bg-[#0A3A72] text-white hover:bg-[#082f5d] shadow-lg";
  const selectedControlClass = isDarkTheme
    ? "bg-[#e8e7e2] text-[#171612]"
    : "bg-[#0A3A72] text-white";
  const accentSoftClass = isDarkTheme
    ? "border-white/30 bg-white/10 text-[#f1f0ec]"
    : "border-[#0A3A72]/35 bg-[#0A3A72]/10 text-[#0A3A72]";
  const accentColor = isDarkTheme ? "#e8e7e2" : "#0A3A72";
  const tutorialOverlayClass = isDarkTheme ? "bg-[#10100f]/82" : "bg-[#f2efe7]/84";
  const themeVars = {
    color: isDarkTheme ? "#f4f0e8" : "#1c1b19",
    "--uprightly-accent": accentColor,
  } as CSSProperties;
  const floatingWindowSupported =
    typeof window !== "undefined" &&
    !!(
      window as Window & {
        documentPictureInPicture?: DocumentPictureInPictureApi;
      }
    ).documentPictureInPicture?.requestWindow;

  const metricsPaused =
    !isActive || pill === "detecting" || trackingHealth < 45;

  const autoPipStatusLabel =
    autoPipStatus === "supported"
      ? "Auto PiP supported"
      : autoPipStatus === "manual_only"
        ? "Manual only"
        : autoPipStatus === "blocked"
          ? "Auto PiP blocked"
          : "Unsupported";
  const mlStatusLabel =
    mlStatus === "connected"
      ? "ML connected"
      : mlStatus === "checking"
        ? "Checking ML"
        : mlStatus === "degraded"
          ? "ML degraded"
          : "ML unavailable";
  const autoPipStatusMessage =
    autoPipStatus === "supported"
      ? "Opens automatically when an active session moves out of focus."
      : autoPipStatus === "manual_only"
        ? "Available manually; automatic opening is limited in this browser."
        : autoPipStatus === "blocked"
          ? "Available manually, but automatic opening is currently blocked."
          : "Not supported here. Try a Chromium browser over HTTPS.";
  const tutorialCardWidth =
    typeof window === "undefined" ? 384 : Math.min(384, window.innerWidth - 24);
  const tutorialHighlightStyle: CSSProperties | undefined = tutorialTargetRect
    ? {
        top: tutorialTargetRect.top - 10,
        left: tutorialTargetRect.left - 10,
        width: tutorialTargetRect.width + 20,
        height: tutorialTargetRect.height + 20,
      }
    : undefined;
  const tutorialCardStyle: CSSProperties = (() => {
    if (typeof window === "undefined") {
      return { width: tutorialCardWidth, left: 16, top: 16 };
    }

    if (isCompactTutorial) {
      return {
        width: tutorialCardWidth,
        left: Math.max(12, (window.innerWidth - tutorialCardWidth) / 2),
        bottom: 12,
        maxHeight: "calc(100dvh - 24px)",
        overflowY: "auto",
      };
    }

    if (tutorialTargetRect) {
      return (() => {
          const gap = 18;
          const cardHeight = Math.max(280, tutorialCardSize.height);
          const viewportPadding = 16;
          const viewportWidth = window.innerWidth;
          const viewportHeight = window.innerHeight;
          const fitsRight =
            tutorialTargetRect.right + gap + tutorialCardWidth <=
            viewportWidth - viewportPadding;
          const fitsLeft =
            tutorialTargetRect.left - gap - tutorialCardWidth >=
            viewportPadding;
          const fitsBelow =
            tutorialTargetRect.bottom + gap + cardHeight <=
            viewportHeight - viewportPadding;
          const fitsAbove =
            tutorialTargetRect.top - gap - cardHeight >= viewportPadding;

          let left = clamp(
            tutorialTargetRect.left,
            viewportPadding,
            viewportWidth - tutorialCardWidth - viewportPadding,
          );
          let top = clamp(
            tutorialTargetRect.bottom + gap,
            viewportPadding,
            viewportHeight - cardHeight - viewportPadding,
          );

          if (fitsRight || fitsLeft) {
            left = fitsRight
              ? tutorialTargetRect.right + gap
              : tutorialTargetRect.left - tutorialCardWidth - gap;
            top = clamp(
              tutorialTargetRect.top +
                tutorialTargetRect.height / 2 -
                cardHeight / 2,
              viewportPadding,
              viewportHeight - cardHeight - viewportPadding,
            );
          } else if (fitsBelow || fitsAbove) {
            top = fitsBelow
              ? tutorialTargetRect.bottom + gap
              : tutorialTargetRect.top - cardHeight - gap;
            left = clamp(
              tutorialTargetRect.left,
              viewportPadding,
              viewportWidth - tutorialCardWidth - viewportPadding,
            );
          }

          return {
            width: tutorialCardWidth,
            left,
            top,
          };
        })();
    }

    return {
      width: tutorialCardWidth,
      left: Math.max(16, (window.innerWidth - tutorialCardWidth) / 2),
      top: Math.max(16, (window.innerHeight - tutorialCardSize.height) / 2),
    };
  })();
  const visibleFeedbacks = feedbacks.slice(-5).reverse();
  const isSessionLogIdle = !isActive && feedbacks.length === 0;

  return (
    <main
      className={`uprightly-shell min-h-dvh w-full max-w-full overflow-x-hidden px-3 py-3 font-sans transition-colors duration-300 sm:p-4 lg:flex lg:items-center lg:justify-center lg:p-8 ${shellClass}`}
      style={themeVars}
    >
      <div className="flex min-h-[calc(100dvh-1.5rem)] w-full flex-col gap-4 sm:min-h-[calc(100dvh-2rem)] lg:h-[calc(100dvh-4rem)] lg:max-h-[56rem] lg:min-h-0 lg:gap-6">
        <div className="flex flex-none min-h-0 gap-4 lg:flex-1 lg:gap-6">
          <aside
            data-testid="desktop-rail"
            className="uprightly-desktop-rail hidden h-full min-h-0 w-60 flex-shrink-0 flex-col gap-4 overflow-hidden lg:flex xl:w-64"
          >
            <div className="relative z-10 flex w-full flex-col items-center gap-2 overflow-visible px-1 pt-1 text-center">
              <h1
                className={`relative z-10 w-full overflow-visible whitespace-nowrap pb-1 text-[2.7rem] font-bold leading-[1.08] tracking-[-0.045em] xl:text-[2.85rem] ${
                  isDarkTheme ? "text-[#f4f0e8]" : "text-[#1c1b19]"
                }`}
              >
                Uprightly
              </h1>
              <p className={`text-sm font-medium ${mutedTextClass}`}>
                Calm, real-time posture guidance
              </p>
            </div>

            <button
              onClick={isActive ? stop : () => void start()}
              disabled={isLoading}
              data-tour="start-session"
              className={`w-full flex items-center justify-center gap-2 px-5 py-2.5 rounded-full font-semibold transition-all ${isActive ? "bg-rose-500/20 text-rose-400 border border-rose-500/30 hover:bg-rose-500/30" : primaryButtonClass} ${isLoading ? "opacity-50 cursor-not-allowed" : ""}`}
            >
              {isActive ? <VideoOff size={18} /> : <Camera size={18} />}
              {isActive ? "Stop" : "Start Session"}
            </button>

            <button
              onClick={openTutorial}
              className={`w-full flex items-center justify-center gap-2 px-5 py-2.5 rounded-full font-semibold border transition-all ${iconButtonClass}`}
            >
              <BookOpen size={18} />
              Open Tutorial
            </button>

            <div className="uprightly-desktop-metrics mt-auto flex min-h-0 flex-col gap-4">
              <div
                data-tour="posture-score"
                className={`backdrop-blur-md border rounded-2xl p-4 flex flex-col gap-1 transition-all relative overflow-hidden group ${heroCardClass}`}
              >
                <div
                  className={`flex items-center justify-between mb-1 z-10 ${subtleTextClass}`}
                >
                  <span className="text-xs font-medium tracking-wide">
                    Posture Score
                  </span>
                  {score > 70 ? (
                    <CheckCircle2 size={14} className="text-[#91a889]" />
                  ) : (
                    <AlertCircle
                      size={14}
                      className={isDarkTheme ? "text-white" : "text-stone-600"}
                    />
                  )}
                </div>

                <div className="flex items-center justify-between mt-1 z-10">
                  <div className="flex items-baseline gap-1">
                    <span
                      className={`text-3xl font-black tracking-tight ${getScoreColor(score)}`}
                    >
                      {score}
                    </span>
                    <span className={`text-xs font-medium ${mutedTextClass}`}>
                      / 100
                    </span>
                  </div>
                </div>

                <div className="absolute right-4 top-1/2 -translate-y-1/2 opacity-80 group-hover:opacity-95 transition-opacity pointer-events-none">
                  <svg className="w-20 h-20 transform -rotate-90">
                    <circle
                      cx="40"
                      cy="40"
                      r="33"
                      stroke="currentColor"
                      strokeWidth="8"
                      fill="transparent"
                      className={
                        isDarkTheme ? "text-white/10" : "text-stone-200"
                      }
                    />
                    <circle
                      cx="40"
                      cy="40"
                      r="33"
                      stroke="currentColor"
                      strokeWidth="8"
                      fill="transparent"
                      strokeDasharray={207.3}
                      strokeDashoffset={207.3 - (207.3 * score) / 100}
                      className={`${getScoreColor(score)} transition-all duration-1000 ease-out`}
                    />
                  </svg>
                </div>
              </div>

              <MetricCard
                paused={metricsPaused}
                theme={theme}
                label={metricMeta.trunk.label}
                value={metrics.trunkAngle.toFixed(1)}
                unit={metricMeta.trunk.unit}
                icon={Activity}
                variant="trunk"
                rawValue={metrics.trunkAngle}
                signedValue={signedMetrics.trunkAngle}
                threshold={metricMeta.thresholds.trunk}
                progress={metricQuality(
                  metrics.trunkAngle,
                  metricMeta.thresholds.trunk,
                )}
              />
              <MetricCard
                paused={metricsPaused}
                theme={theme}
                label={metricMeta.head.label}
                value={metrics.headForward.toFixed(2)}
                unit={metricMeta.head.unit}
                icon={ChevronRight}
                variant="head"
                rawValue={metrics.headForward}
                signedValue={signedMetrics.headForward}
                threshold={metricMeta.thresholds.head}
                progress={metricQuality(
                  metrics.headForward,
                  metricMeta.thresholds.head,
                )}
              />
              <MetricCard
                paused={metricsPaused}
                theme={theme}
                label={metricMeta.shoulder.label}
                value={metrics.shoulderTilt.toFixed(2)}
                unit={metricMeta.shoulder.unit}
                icon={Maximize2}
                variant="shoulder"
                rawValue={metrics.shoulderTilt}
                signedValue={signedMetrics.shoulderTilt}
                threshold={metricMeta.thresholds.shoulder}
                progress={metricQuality(
                  metrics.shoulderTilt,
                  metricMeta.thresholds.shoulder,
                )}
              />
            </div>
          </aside>

          <div
            className="relative flex min-h-0 min-w-0 flex-1"
          >
            <div
              data-tour="camera-stage"
              className={`group relative aspect-[4/5] min-h-[30rem] flex-1 overflow-hidden rounded-[2rem] border shadow-2xl transition-colors duration-300 motion-reduce:transition-none lg:aspect-auto lg:min-h-0 ${stageClass}`}
            >
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="absolute inset-0 w-full h-full object-cover -scale-x-100"
              />
              <canvas
                ref={canvasRef}
                className="absolute inset-0 w-full h-full object-cover -scale-x-100"
              />

              {!isActive ? (
                <div
                  className={`absolute inset-0 z-10 flex flex-col items-center justify-center backdrop-blur-sm ${isDarkTheme ? "bg-black/45" : "bg-[#eef4fa]/85"}`}
                >
                  <div
                    className={`mb-4 flex h-20 w-20 items-center justify-center rounded-full ${isDarkTheme ? "bg-white/5" : "bg-white/65"}`}
                  >
                    <Camera
                      size={32}
                      className={
                        isDarkTheme ? "text-white/20" : "text-[#0A3A72]/45"
                      }
                    />
                  </div>
                  <p
                    className={`font-medium ${
                      isDarkTheme ? mutedTextClass : "text-[#0A3A72]/65"
                    }`}
                  >
                    Camera Feed Inactive
                  </p>
                </div>
              ) : null}

              {showGuidedTrial && isActive ? (
                <aside
                  aria-label="Guided posture check"
                  className={`absolute right-5 top-5 z-30 w-[min(22rem,calc(100%-2.5rem))] overflow-hidden rounded-[1.5rem] border p-5 shadow-2xl backdrop-blur-2xl ${
                    isDarkTheme
                      ? "border-white/12 bg-[#171715]/95 text-[#f4f0e8]"
                      : "border-[#cbdbea] bg-white/95 text-[#1c1b19]"
                  }`}
                >
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <p className={`text-[11px] font-semibold uppercase tracking-[0.14em] ${mutedTextClass}`}>
                        Guided posture check
                      </p>
                      <p className={`mt-1 text-xs ${mutedTextClass}`}>
                        {guidedTrialStepIndex + 1} of {GUIDED_TRIAL_STEPS.length}
                      </p>
                    </div>
                    <button
                      type="button"
                      onClick={finishGuidedTrial}
                      aria-label="End guided trial"
                      className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-full border transition-colors ${
                        isDarkTheme
                          ? "border-white/15 text-white/65 hover:bg-white/10 hover:text-white"
                          : "border-stone-200 text-stone-500 hover:bg-stone-100 hover:text-stone-900"
                      }`}
                    >
                      <X size={16} aria-hidden="true" />
                    </button>
                  </div>

                  <div className="mt-4 flex gap-1.5" aria-label="Trial progress">
                    {GUIDED_TRIAL_STEPS.map((step, index) => (
                      <span
                        key={step.id}
                        className={`h-1.5 flex-1 rounded-full transition-colors ${
                          index <= guidedTrialStepIndex
                            ? isDarkTheme
                              ? "bg-[#e8e7e2]"
                              : "bg-[#0A3A72]"
                            : isDarkTheme
                              ? "bg-white/15"
                              : "bg-[#0A3A72]/15"
                        }`}
                      />
                    ))}
                  </div>

                  <div className="mt-5 flex items-start justify-between gap-3">
                    <h2 className="text-lg font-bold leading-tight tracking-[-0.02em]">
                      {currentGuidedTrialStep.title}
                    </h2>
                    <span
                      role="status"
                      aria-live="polite"
                      className={`shrink-0 rounded-full border px-2 py-1 text-[10px] font-semibold ${
                        currentGuidedTrialDetected
                          ? "border-[#91a889]/30 bg-[#91a889]/12 text-[#91a889]"
                          : isDarkTheme
                            ? "border-white/10 text-white/45"
                            : "border-stone-200 text-stone-500"
                      }`}
                    >
                      {currentGuidedTrialDetected ? "Detected" : "Try it"}
                    </span>
                  </div>
                  <p className={`mt-3 text-sm leading-6 ${quietTextClass}`}>
                    {currentGuidedTrialStep.instruction}
                  </p>
                  <div
                    className={`mt-4 border-l-2 pl-3 text-xs leading-5 ${
                      isDarkTheme
                        ? "border-white/20 text-white/60"
                        : "border-[#0A3A72]/30 text-stone-600"
                    }`}
                  >
                    <span className="font-semibold">Expected: </span>
                    {currentGuidedTrialStep.expected}
                  </div>
                  <p className={`mt-4 text-[11px] leading-4 ${mutedTextClass}`}>
                    Use only gentle movements. Stop if anything feels uncomfortable.
                  </p>

                  <div className="mt-5 grid grid-cols-2 gap-2">
                    <button
                      type="button"
                      onClick={() =>
                        setGuidedTrialStepIndex((index) => Math.max(0, index - 1))
                      }
                      disabled={guidedTrialStepIndex === 0}
                      className={`min-h-10 rounded-xl border px-3 text-xs font-semibold transition-colors ${
                        guidedTrialStepIndex === 0
                          ? "cursor-not-allowed opacity-35"
                          : isDarkTheme
                            ? "border-white/15 text-white/75 hover:bg-white/10"
                            : "border-stone-200 text-stone-600 hover:bg-stone-100"
                      }`}
                    >
                      Previous
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        if (
                          guidedTrialStepIndex ===
                          GUIDED_TRIAL_STEPS.length - 1
                        ) {
                          finishGuidedTrial();
                          return;
                        }
                        setGuidedTrialStepIndex((index) => index + 1);
                      }}
                      className={`min-h-10 rounded-xl px-3 text-xs font-semibold ${primaryButtonClass}`}
                    >
                      {guidedTrialStepIndex === GUIDED_TRIAL_STEPS.length - 1
                        ? "Finish trial"
                        : "Next"}
                    </button>
                  </div>
                </aside>
              ) : null}

              <div
                className={`absolute left-3 right-3 top-3 z-20 flex min-h-14 items-center justify-end gap-2 rounded-2xl border px-2.5 py-2 lg:hidden ${stageGlassClass}`}
              >
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => {
                      setShowSettings(false);
                      setShowSessionLog(true);
                    }}
                    className={`flex h-11 items-center justify-center gap-2 rounded-full border px-3.5 text-sm font-semibold transition-all ${iconButtonClass}`}
                    title="Open activity"
                    aria-label="Open activity"
                  >
                    <Bell size={18} />
                    Activity
                  </button>
                  <button
                    onClick={isActive ? stop : () => void start()}
                    disabled={isLoading}
                    data-tour="start-session"
                    className={`flex min-h-11 items-center gap-2 rounded-full px-4 py-2.5 text-sm font-semibold transition-all lg:hidden ${isActive ? "bg-rose-500/20 text-rose-400 border border-rose-500/30 hover:bg-rose-500/30" : primaryButtonClass} ${isLoading ? "opacity-50 cursor-not-allowed" : ""}`}
                  >
                    {isActive ? <VideoOff size={18} /> : <Camera size={18} />}
                    {isActive ? "Stop" : "Start Session"}
                  </button>
                </div>
              </div>

              <div
                className={`absolute inset-0 transition-opacity duration-1000 ${isActive ? "opacity-100" : "opacity-0"}`}
              >
                <div
                  className={`absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-48 h-64 border-2 border-dashed rounded-full ${isDarkTheme ? "border-white/20" : "border-stone-300"}`}
                />
              </div>

              <div className="absolute left-6 bottom-6 z-20">
                {isActive ? (
                  <>
                    <div className="text-sm font-bold tracking-wide">
                      Stability {stabilityScore}%
                    </div>
                    <div
                      className={`text-[11px] font-semibold tracking-wide ${quietTextClass}`}
                    >
                      Tracking {trackingHealth}%
                    </div>
                  </>
                ) : null}
              </div>

              <div
                data-tour="session-log"
                className={`z-40 transition-all duration-300 ease-out motion-reduce:transition-none ${
                  showSessionLog
                    ? `fixed inset-x-3 bottom-3 lg:absolute lg:inset-y-0 lg:left-auto lg:right-0 lg:h-auto lg:w-[23rem] xl:w-96 ${
                        isSessionLogIdle
                          ? "h-[min(42dvh,22rem)]"
                          : "h-[min(60dvh,32rem)]"
                      }`
                    : showSettings
                      ? `pointer-events-none fixed inset-x-3 bottom-3 opacity-0 lg:absolute lg:inset-y-0 lg:left-auto lg:right-0 lg:h-auto lg:w-[23rem] xl:w-96 ${
                          isSessionLogIdle
                            ? "h-[min(42dvh,22rem)]"
                            : "h-[min(60dvh,32rem)]"
                        }`
                    : "pointer-events-none absolute right-0 top-0 h-16 w-16 lg:bottom-0 lg:h-auto"
                }`}
              >
                {showSessionLog ? (
                  <section
                    className={`relative h-full overflow-hidden rounded-[1.75rem] border shadow-2xl backdrop-blur-2xl ${
                      isDarkTheme
                        ? "border-white/10 bg-[#171715]/95"
                        : "border-stone-200 bg-white/95"
                    }`}
                    aria-label="Activity"
                  >
                    <div
                      className="pointer-events-none absolute inset-0"
                      style={{
                        background: isDarkTheme
                          ? "radial-gradient(circle at 100% 0%, rgba(211,154,56,0.10), transparent 34%)"
                          : "radial-gradient(circle at 100% 0%, rgba(211,154,56,0.08), transparent 34%)",
                      }}
                    />
                    <div className="pointer-events-none absolute left-4 right-4 top-4 z-10">
                      <div
                        className={`pointer-events-auto flex w-full items-center rounded-xl border p-1 ${
                          isDarkTheme
                            ? "border-white/10 bg-black/25"
                            : "border-stone-200 bg-white/75"
                        }`}
                        role="group"
                        aria-label="Panel controls"
                      >
                        <div
                          className="grid min-w-0 flex-1 grid-cols-2 gap-1"
                          role="tablist"
                          aria-label="Utility panel"
                        >
                          <button
                            type="button"
                            role="tab"
                            aria-selected="true"
                            className={`flex min-h-11 w-full items-center justify-center gap-2 rounded-lg px-3 text-sm font-semibold ${selectedControlClass}`}
                          >
                            <Bell size={16} aria-hidden="true" />
                            Activity
                          </button>
                          <button
                            type="button"
                            role="tab"
                            aria-selected="false"
                            onClick={() => {
                              setShowSessionLog(false);
                              setShowSettings(true);
                            }}
                            className={`flex min-h-11 w-full items-center justify-center gap-2 rounded-lg px-3 text-sm font-semibold transition-colors ${
                              isDarkTheme
                                ? "text-white/65 hover:bg-white/10 hover:text-white"
                                : "text-stone-600 hover:bg-stone-100 hover:text-stone-900"
                            }`}
                          >
                            <Settings size={16} aria-hidden="true" />
                            Settings
                          </button>
                        </div>
                        <span
                          className={`mx-1 h-7 w-px shrink-0 ${
                            isDarkTheme ? "bg-white/10" : "bg-stone-200"
                          }`}
                          aria-hidden="true"
                        />
                        <button
                          onClick={() => setShowSessionLog(false)}
                          className={`flex h-11 w-11 shrink-0 items-center justify-center rounded-lg border-0 transition-colors ${sessionLogIconButtonClass}`}
                          title="Close utility panel"
                          aria-label="Close utility panel"
                        >
                          <X size={19} aria-hidden="true" />
                        </button>
                      </div>
                      <p
                        className={`mt-2 pl-1 text-[11px] ${
                          isDarkTheme ? "text-white/45" : "text-stone-500"
                        }`}
                        aria-live="polite"
                      >
                        {isSessionLogIdle
                          ? "Standby"
                          : `${feedbacks.length} ${feedbacks.length === 1 ? "event" : "events"}`}
                      </p>
                    </div>

                    <div
                        className={`absolute bottom-4 left-4 right-4 top-24 z-10 flex ${
                        isSessionLogIdle
                          ? "items-center justify-center"
                          : "pointer-events-none items-end justify-end"
                      }`}
                      style={
                        isSessionLogIdle
                          ? undefined
                          : {
                              maskImage:
                                "linear-gradient(to top, black 0%, black 58%, rgba(0,0,0,0.78) 72%, rgba(0,0,0,0.28) 84%, transparent 100%)",
                              WebkitMaskImage:
                                "linear-gradient(to top, black 0%, black 58%, rgba(0,0,0,0.78) 72%, rgba(0,0,0,0.28) 84%, transparent 100%)",
                            }
                      }
                    >
                      {isSessionLogIdle ? (
                        <div
                          className="flex w-full max-w-[18rem] flex-col items-center px-3 pb-1 pt-2 text-center"
                        >
                          <div
                            className={`relative flex h-14 w-14 items-center justify-center rounded-full border ${
                              isDarkTheme
                                ? "border-amber-400/20 bg-amber-400/[0.07] text-amber-300"
                                : "border-amber-200 bg-amber-50 text-amber-600"
                            }`}
                            aria-hidden="true"
                          >
                            <span className="absolute inset-1 rounded-full border border-current opacity-20 motion-safe:animate-pulse" />
                            <Activity size={22} />
                          </div>
                          <h3 className="mt-3 text-base font-semibold">
                            No activity yet
                          </h3>
                          <p
                            className={`mt-1 max-w-[15rem] text-sm leading-relaxed ${
                              isDarkTheme ? "text-white/55" : "text-stone-500"
                            }`}
                          >
                            Posture feedback will appear here when your session
                            begins.
                          </p>
                          <button
                            onClick={() => void start()}
                            disabled={isLoading}
                            className={`mt-4 flex min-h-11 w-full items-center justify-center gap-2 rounded-full px-4 py-2.5 text-sm font-semibold transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--uprightly-accent)] focus-visible:ring-offset-2 focus-visible:ring-offset-transparent ${primaryButtonClass} ${
                              isLoading ? "cursor-not-allowed opacity-50" : ""
                            }`}
                          >
                            <Camera size={17} aria-hidden="true" />
                            {isLoading ? "Preparing Camera" : "Start Session"}
                          </button>
                        </div>
                      ) : (
                        <div
                          ref={sessionLogStackRef}
                          className="flex w-full max-w-[22.5rem] flex-col-reverse gap-3"
                        >
                          {visibleFeedbacks.map((f, index) => {
                            const opacity =
                              index === 0
                                ? 1
                                : index === 1
                                  ? 0.92
                                  : index === 2
                                    ? 0.78
                                    : index === 3
                                      ? 0.58
                                      : 0.38;
                            const translateY =
                              index === 0
                                ? 0
                                : index === 1
                                  ? -2
                                  : index === 2
                                    ? -6
                                    : index === 3
                                      ? -10
                                      : -14;

                            return (
                              <div
                                key={f.id}
                                data-feedback-card
                                className={`rounded-2xl border px-4 py-3 transition-all ${
                                  isDarkTheme
                                    ? "border-transparent bg-black/70"
                                    : "border-stone-300 bg-white shadow-[0_22px_45px_-30px_rgba(15,23,42,0.24)]"
                                }`}
                                style={{
                                  opacity,
                                  transform: `translateY(${translateY}px) scale(${1 - index * 0.02})`,
                                }}
                              >
                                <div className="flex justify-between items-center mb-1.5 gap-3">
                                  <div className="flex items-center gap-2 min-w-0">
                                    {f.type === "success" ? (
                                      <CheckCircle2
                                        size={14}
                                        className="flex-shrink-0 text-[#91a889]"
                                      />
                                    ) : f.type === "warning" ? (
                                      <AlertCircle
                                        size={14}
                                        className="text-amber-400 flex-shrink-0"
                                      />
                                    ) : f.type === "critical" ? (
                                      <AlertCircle
                                        size={14}
                                        className="text-rose-400 flex-shrink-0"
                                      />
                                    ) : (
                                      <Bell
                                        size={14}
                                        className="text-amber-400 flex-shrink-0"
                                      />
                                    )}
                                    <span
                                      className={`text-xs font-bold tracking-wide ${
                                        isDarkTheme
                                          ? "text-white/92"
                                          : "text-stone-800"
                                      }`}
                                    >
                                      {f.title}
                                    </span>
                                  </div>
                                  <span
                                    className={`text-[10px] ${
                                      isDarkTheme
                                        ? "text-white/45"
                                        : "text-stone-500"
                                    }`}
                                  >
                                    {f.time}
                                  </span>
                                </div>
                                <p
                                  className={`text-[13px] leading-relaxed ${
                                    isDarkTheme
                                      ? "text-white/92"
                                      : "text-stone-700"
                                  }`}
                                >
                                  {f.text}
                                </p>
                              </div>
                            );
                          })}
                          <div ref={chatEndRef} />
                        </div>
                      )}
                    </div>
                  </section>
                ) : !showSettings ? (
                  <nav
                    aria-label="Workspace panels"
                    className="pointer-events-auto absolute right-4 top-4 z-10 hidden items-center gap-2 lg:flex"
                  >
                    <button
                      onClick={() => {
                        setShowSettings(false);
                        setShowSessionLog(true);
                      }}
                      className={`flex min-h-11 items-center justify-center gap-2 rounded-xl border px-3.5 text-sm font-semibold shadow-lg backdrop-blur-xl transition-all ${
                        isDarkTheme
                          ? "border-white/10 bg-[#171715]/88 text-white/72 hover:bg-white/10 hover:text-white"
                          : "border-[#ded8cc] bg-[#fffdf8]/90 text-stone-600 hover:bg-stone-100 hover:text-stone-900"
                      }`}
                      title="Open activity"
                      aria-label="Open activity"
                      aria-pressed="false"
                    >
                      <PanelRightOpen size={17} aria-hidden="true" />
                      <span>Activity</span>
                    </button>
                    <button
                      onClick={() => {
                        setShowSessionLog(false);
                        setShowSettings(true);
                      }}
                      className={`flex min-h-11 items-center justify-center gap-2 rounded-xl border px-3.5 text-sm font-semibold shadow-lg backdrop-blur-xl transition-all ${
                        isDarkTheme
                          ? "border-white/10 bg-[#171715]/88 text-white/72 hover:bg-white/10 hover:text-white"
                          : "border-[#ded8cc] bg-[#fffdf8]/90 text-stone-600 hover:bg-stone-100 hover:text-stone-900"
                      }`}
                      title="Open settings"
                      aria-label="Open settings"
                      aria-pressed="false"
                    >
                      <Settings size={17} aria-hidden="true" />
                      <span>Settings</span>
                    </button>
                  </nav>
                ) : null}
              </div>
            </div>

            <div
              className={`flex-shrink-0 overflow-hidden transition-[width,opacity] duration-200 ease-in-out motion-reduce:transition-none ${
                showSettings
                  ? "fixed inset-x-3 bottom-3 z-50 h-[min(72dvh,42rem)] w-auto opacity-100 lg:absolute lg:inset-y-0 lg:left-auto lg:right-0 lg:h-auto lg:w-[23rem] xl:w-96"
                  : showSessionLog
                    ? "pointer-events-none fixed inset-x-3 bottom-3 z-50 h-[min(72dvh,42rem)] w-auto opacity-0 lg:absolute lg:inset-y-0 lg:left-auto lg:right-0 lg:h-auto lg:w-[23rem] xl:w-96"
                    : "hidden w-0 opacity-0 lg:block"
              }`}
              inert={!showSettings}
              style={!showSettings && !showSessionLog ? { display: "none" } : undefined}
            >
              <div
                aria-hidden={!showSettings}
                data-tour="settings-panel"
                className={`settings-panel-shell ${isDarkTheme ? "settings-panel-dark" : "settings-panel-light"} pointer-events-auto flex h-full w-full flex-col overflow-hidden rounded-[1.75rem] border shadow-2xl lg:w-[23rem] xl:w-96 ${settingsPanelClass}`}
              >
                <div className="flex flex-shrink-0 items-center px-5 pb-4 pt-5">
                  <div
                    className={`flex w-full min-w-0 items-center rounded-xl border p-1 ${
                      isDarkTheme
                        ? "border-white/10 bg-black/25"
                        : "border-stone-200 bg-white/75"
                    }`}
                    role="group"
                    aria-label="Panel controls"
                  >
                    <div
                      className="grid min-w-0 flex-1 grid-cols-2 gap-1"
                      role="tablist"
                      aria-label="Utility panel"
                    >
                      <button
                        type="button"
                        role="tab"
                        aria-selected="false"
                        onClick={() => {
                          setShowSettings(false);
                          setShowSessionLog(true);
                        }}
                        className={`flex min-h-11 w-full items-center justify-center gap-2 rounded-lg px-3 text-sm font-semibold transition-colors ${
                          isDarkTheme
                            ? "text-white/65 hover:bg-white/10 hover:text-white"
                            : "text-stone-600 hover:bg-stone-100 hover:text-stone-900"
                        }`}
                      >
                        <Bell size={16} aria-hidden="true" />
                        Activity
                      </button>
                      <button
                        type="button"
                        role="tab"
                        aria-selected="true"
                        className={`flex min-h-11 w-full items-center justify-center gap-2 rounded-lg px-3 text-sm font-semibold ${selectedControlClass}`}
                      >
                        <Settings size={16} aria-hidden="true" />
                        Settings
                      </button>
                    </div>
                    <span
                      className={`mx-1 h-7 w-px shrink-0 ${
                        isDarkTheme ? "bg-white/10" : "bg-stone-200"
                      }`}
                      aria-hidden="true"
                    />
                    <button
                      onClick={() => setShowSettings(false)}
                      className={`flex h-11 w-11 shrink-0 items-center justify-center rounded-lg border-0 transition-colors ${sessionLogIconButtonClass}`}
                      title="Close utility panel"
                      aria-label="Close utility panel"
                    >
                      <X size={20} aria-hidden="true" />
                    </button>
                  </div>
                </div>

                <div className="settings-scroll-frame relative min-h-0 flex-1">
                  <div
                    data-testid="settings-scroll-area"
                    className="settings-scroll-area h-full overflow-y-auto px-5 pb-5 pt-3"
                    role="region"
                    aria-label="Settings controls"
                    tabIndex={0}
                  >
                    <div className="space-y-3">
                  <section className={`space-y-3 rounded-2xl border p-4 ${isDarkTheme ? "border-white/8 bg-white/[0.025]" : "border-stone-200 bg-white/70"}`}>
                    <div className="flex items-center justify-between gap-3">
                      <label
                        htmlFor="camera-source"
                        className={`flex items-center gap-2 text-xs font-semibold ${subtleTextClass}`}
                      >
                        <Camera size={15} aria-hidden="true" />
                        Camera
                      </label>
                      <span
                        className={`rounded-full border px-2 py-1 text-[10px] font-semibold ${isDarkTheme ? "border-white/10 bg-white/5" : "border-stone-200 bg-stone-50"} ${mutedTextClass}`}
                      >
                        {cameraDevices.length || 0} detected
                      </span>
                    </div>
                    <select
                      id="camera-source"
                      value={selectedCameraId}
                      onChange={(e) => {
                        const nextCameraId = e.target.value;
                        setSelectedCameraId(nextCameraId);
                        if (isActive) {
                          stop();
                          window.setTimeout(() => {
                            void start(nextCameraId);
                          }, 0);
                        }
                      }}
                      className={`w-full rounded-xl border px-3 py-2.5 text-sm font-semibold outline-none transition-colors ${isDarkTheme ? "border-white/15 bg-white/5 text-white hover:bg-white/10" : "border-stone-200 bg-white text-stone-900 hover:bg-stone-50"}`}
                    >
                      {cameraDevices.length === 0 ? (
                        <option value="">No camera detected yet</option>
                      ) : null}
                      {cameraDevices.map((camera) => (
                        <option
                          key={camera.id}
                          value={camera.id}
                          className={
                            isDarkTheme
                              ? "bg-stone-900 text-white"
                              : "bg-white text-stone-900"
                          }
                        >
                          {camera.label}
                        </option>
                      ))}
                    </select>
                    <p
                      className={`text-[11px] leading-relaxed ${mutedTextClass}`}
                    >
                      Camera names appear after permission is granted.
                    </p>
                  </section>

                  <section className={`space-y-3 rounded-2xl border p-4 ${isDarkTheme ? "border-white/8 bg-white/[0.025]" : "border-stone-200 bg-white/70"}`}>
                    <div className="flex items-center justify-between gap-3">
                      <div
                        className={`flex items-center gap-2 text-xs font-semibold ${subtleTextClass}`}
                      >
                        {isDarkTheme ? (
                          <Moon size={15} aria-hidden="true" />
                        ) : (
                          <Sun size={15} aria-hidden="true" />
                        )}
                        Theme
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-2">
                      {(
                        [
                          ["dark", "Dark", Moon],
                          ["light", "Light", Sun],
                        ] as const
                      ).map(([mode, label, Icon]) => (
                        <button
                          key={mode}
                          onClick={() => setTheme(mode)}
                          className={`rounded-xl border px-3 py-2.5 text-sm font-semibold transition-colors flex items-center justify-center gap-2 ${
                            theme === mode
                              ? accentSoftClass
                              : isDarkTheme
                                ? "border-white/15 bg-white/5 text-white/70 hover:bg-white/10 hover:text-white"
                                : "border-stone-200 bg-white text-stone-600 hover:bg-stone-50 hover:text-stone-900"
                          }`}
                        >
                          <Icon size={16} />
                          {label}
                        </button>
                      ))}
                    </div>
                  </section>

                  <section className={`space-y-3 rounded-2xl border p-4 ${isDarkTheme ? "border-white/8 bg-white/[0.025]" : "border-stone-200 bg-white/70"}`}>
                    <div className="flex items-center justify-between gap-4">
                      <div>
                        <div
                          className={`flex items-center gap-2 text-sm font-semibold ${subtleTextClass}`}
                        >
                          <Volume2 size={16} aria-hidden="true" />
                          Voice prompts
                        </div>
                        <p className={`mt-1 text-[11px] ${mutedTextClass}`}>
                          Speak stable posture changes
                        </p>
                      </div>
                      <button
                        type="button"
                        role="switch"
                        aria-checked={audioMode === "voice"}
                        aria-label="Voice prompts"
                        onClick={() =>
                          setAudioMode(audioMode === "voice" ? "off" : "voice")
                        }
                        className={`relative h-7 w-12 shrink-0 overflow-hidden rounded-full border transition-colors ${
                          audioMode === "voice"
                            ? isDarkTheme
                              ? "border-[#e8e7e2] bg-[#e8e7e2]"
                              : "border-[#0A3A72] bg-[#0A3A72]"
                            : isDarkTheme
                              ? "border-white/15 bg-white/10"
                              : "border-stone-300 bg-stone-200"
                        }`}
                      >
                        <span
                          className={`absolute left-1 top-1/2 h-[1.125rem] w-[1.125rem] -translate-y-1/2 rounded-full shadow-sm transition-transform ${
                            audioMode === "voice"
                              ? `${isDarkTheme ? "bg-[#171612]" : "bg-white"} translate-x-[1.375rem]`
                              : "translate-x-0 bg-white"
                          }`}
                        />
                      </button>
                    </div>
                    <div
                      className={`flex justify-between items-center px-1 text-[11px] ${mutedTextClass}`}
                    >
                      <span>
                        Status:{" "}
                        {speechStatus === "ready"
                          ? "Ready"
                          : speechStatus === "loading"
                            ? "Loading voices"
                            : speechStatus === "blocked"
                              ? "No voice loaded"
                              : "Unsupported"}
                      </span>
                      <span>{availableVoices} voice(s)</span>
                    </div>
                    <button
                      onClick={() => {
                        setAudioMode("voice");
                        speakFeedback(
                          "good",
                          "That looks good. Keep it there.",
                          "test-voice",
                        );
                      }}
                      className={`w-full rounded-xl border text-sm font-semibold py-2.5 transition-colors flex items-center justify-center gap-2 ${isDarkTheme ? "border-white/15 bg-white/5 hover:bg-white/10 text-white/80 hover:text-white" : "border-stone-200 bg-white hover:bg-stone-50 text-stone-700 hover:text-stone-900"}`}
                    >
                      <Volume2 size={16} />
                      Test Voice
                    </button>
                    <p
                      className={`text-[11px] leading-relaxed ${mutedTextClass}`}
                    >
                      Prompts play only after a stable posture change.
                    </p>
                  </section>

                  <section className={`space-y-3 rounded-2xl border p-4 ${isDarkTheme ? "border-white/8 bg-white/[0.025]" : "border-stone-200 bg-white/70"}`}>
                    <div className="flex items-center justify-between gap-4">
                      <div>
                        <div
                          className={`flex items-center gap-2 text-sm font-semibold ${subtleTextClass}`}
                        >
                          <Monitor size={16} aria-hidden="true" />
                          Automatic floating window
                        </div>
                        <p className={`mt-1 text-[11px] ${mutedTextClass}`}>
                          Show posture status when you move away
                        </p>
                      </div>
                      <button
                        type="button"
                        role="switch"
                        aria-checked={floatingWindowEnabled}
                        aria-label="Automatic floating window"
                        onClick={() =>
                          floatingWindowSupported &&
                          setFloatingWindowEnabled((value) => !value)
                        }
                        disabled={!floatingWindowSupported}
                          className={`relative h-7 w-12 shrink-0 overflow-hidden rounded-full border transition-colors ${
                          !floatingWindowSupported
                            ? "cursor-not-allowed border-stone-300 bg-stone-200 opacity-50"
                            : floatingWindowEnabled
                              ? isDarkTheme
                                ? "border-[#e8e7e2] bg-[#e8e7e2]"
                                : "border-[#0A3A72] bg-[#0A3A72]"
                              : isDarkTheme
                                ? "border-white/15 bg-white/10"
                                : "border-stone-300 bg-stone-200"
                        }`}
                      >
                        <span
                          className={`absolute left-1 top-1/2 h-[1.125rem] w-[1.125rem] -translate-y-1/2 rounded-full shadow-sm transition-transform ${
                            floatingWindowEnabled
                              ? `${isDarkTheme ? "bg-[#171612]" : "bg-white"} translate-x-[1.375rem]`
                              : "translate-x-0 bg-white"
                          }`}
                        />
                      </button>
                    </div>
                    {floatingWindowSupported ? (
                      <p
                        className={`text-[11px] leading-relaxed ${mutedTextClass}`}
                      >
                        During an active session, it opens after you switch tabs,
                        minimize Uprightly, or change to another app.
                      </p>
                    ) : null}
                    <div
                      className={`flex items-start justify-between gap-3 text-[11px] leading-relaxed ${mutedTextClass}`}
                    >
                      <span>{autoPipStatusMessage}</span>
                      <span className="shrink-0">
                        {floatingWindowReady
                          ? "Open"
                          : isActive
                            ? floatingWindowEnabled
                              ? "Automatic"
                              : "Ready"
                            : autoPipStatusLabel}
                      </span>
                    </div>
                    {floatingWindowSupported ? (
                      <button
                        type="button"
                        onClick={() => {
                          void openFloatingWindow("manual").catch((error) => {
                            console.error("Floating window failed:", error);
                          });
                        }}
                        disabled={!isActive}
                        className={`flex min-h-10 w-full items-center justify-center gap-2 rounded-xl border px-3.5 text-xs font-semibold transition-colors ${
                          !isActive
                            ? "cursor-not-allowed border-stone-300 bg-stone-200 text-stone-500 opacity-60"
                            : isDarkTheme
                              ? "border-white/15 bg-white/5 text-white/80 hover:bg-white/10 hover:text-white"
                              : "border-[#0A3A72]/20 bg-white/70 text-[#0A3A72] hover:bg-white"
                        }`}
                      >
                        <PanelRightOpen size={15} aria-hidden="true" />
                        Open floating window now
                      </button>
                    ) : null}
                  </section>

                  <section
                    className={`space-y-3 rounded-2xl border p-4 ${
                      isDarkTheme
                        ? "border-white/8 bg-white/[0.025]"
                        : "border-stone-200 bg-white/70"
                    }`}
                  >
                    <div>
                      <div
                        className={`flex items-center gap-2 text-sm font-semibold ${subtleTextClass}`}
                      >
                        <Camera size={16} aria-hidden="true" />
                        Capture lab
                      </div>
                      <p
                        className={`mt-1 text-[11px] leading-relaxed ${mutedTextClass}`}
                      >
                        Take local photos with landmark nodes and exact values.
                      </p>
                    </div>
                    <a
                      href="/capture-lab"
                      className={`flex min-h-10 w-full items-center justify-center rounded-xl border px-3.5 text-xs font-semibold transition-colors ${
                        isDarkTheme
                          ? "border-white/15 bg-white/5 text-white/80 hover:bg-white/10 hover:text-white"
                          : "border-[#0A3A72]/20 bg-white text-[#0A3A72] hover:bg-[#eef4fa]"
                      }`}
                    >
                      Open capture lab
                    </a>
                  </section>

                  <section
                    className={`space-y-3 rounded-2xl border p-4 ${
                      isDarkTheme
                        ? "border-white/8 bg-white/[0.025]"
                        : "border-stone-200 bg-white/70"
                    }`}
                  >
                    <div>
                      <div
                        className={`flex items-center gap-2 text-sm font-semibold ${subtleTextClass}`}
                      >
                        <Activity size={16} aria-hidden="true" />
                        Guided posture check
                      </div>
                      <p className={`mt-1 text-[11px] leading-relaxed ${mutedTextClass}`}>
                        Try each posture signal and see how Uprightly responds.
                      </p>
                    </div>
                    <button
                      type="button"
                      onClick={requestGuidedTrial}
                      className={`flex min-h-10 w-full items-center justify-center rounded-xl border px-3.5 text-xs font-semibold transition-colors ${
                        isDarkTheme
                          ? "border-white/15 bg-white/5 text-white/80 hover:bg-white/10 hover:text-white"
                          : "border-[#0A3A72]/20 bg-white text-[#0A3A72] hover:bg-[#eef4fa]"
                      }`}
                    >
                      Run guided trial
                    </button>
                  </section>

                  <section
                    className={`space-y-3 rounded-2xl border p-4 ${
                      isDarkTheme
                        ? "border-white/8 bg-white/[0.025]"
                        : "border-stone-200 bg-white/70"
                    }`}
                  >
                    <div>
                      <div
                        className={`flex items-center gap-2 text-sm font-semibold ${subtleTextClass}`}
                      >
                        <ShieldCheck size={16} aria-hidden="true" />
                        Privacy &amp; data
                      </div>
                      <p className={`mt-1 text-[11px] leading-relaxed ${mutedTextClass}`}>
                        Review how camera frames and local preferences are handled.
                      </p>
                    </div>
                    <button
                      type="button"
                      onClick={openPrivacyPolicy}
                      className={`flex w-full items-center justify-center gap-2 rounded-xl border py-2.5 text-sm font-semibold transition-colors ${
                        isDarkTheme
                          ? "border-white/15 bg-white/5 text-white/80 hover:bg-white/10 hover:text-white"
                          : "border-stone-200 bg-white text-stone-700 hover:bg-stone-50 hover:text-stone-900"
                      }`}
                    >
                      View Privacy &amp; Data Use
                    </button>
                    <button
                      type="button"
                      onClick={reviewPrivacyNotice}
                      className={`w-full rounded-xl px-3 py-2 text-xs font-semibold transition-colors ${
                        isDarkTheme
                          ? "text-white/55 hover:bg-white/5 hover:text-white/80"
                          : "text-stone-500 hover:bg-stone-100 hover:text-stone-800"
                      }`}
                    >
                      Review startup notice
                    </button>
                  </section>
                    </div>
                  </div>
              </div>
            </div>
          </div>
        </div>
        </div>

        <div className="grid flex-shrink-0 grid-flow-dense grid-cols-2 gap-3 lg:hidden">
          <div
            data-tour="posture-score"
            className={`backdrop-blur-md border rounded-2xl p-4 flex flex-col gap-1 transition-all relative overflow-hidden group ${heroCardClass}`}
          >
            <div
              className={`flex items-center justify-between z-10 ${subtleTextClass}`}
            >
              <span className="text-xs font-medium tracking-wide">
                Posture Score
              </span>
              {score > 70 ? (
                <CheckCircle2 size={14} className="text-[#91a889]" />
              ) : (
                <AlertCircle size={14} className="text-amber-400" />
              )}
            </div>

            <div className="flex items-center justify-between mt-1 z-10">
              <div className="flex items-baseline gap-1">
                <span
                  className={`text-3xl font-black tracking-tight ${getScoreColor(score)}`}
                >
                  {score}
                </span>
                <span className={`text-xs font-medium ${mutedTextClass}`}>
                  / 100
                </span>
              </div>
            </div>

            <div className="absolute right-4 top-1/2 -translate-y-1/2 opacity-80 group-hover:opacity-95 transition-opacity pointer-events-none">
              <svg className="w-20 h-20 transform -rotate-90">
                <circle
                  cx="40"
                  cy="40"
                  r="33"
                  stroke="currentColor"
                  strokeWidth="8"
                  fill="transparent"
                  className={isDarkTheme ? "text-white/10" : "text-stone-200"}
                />
                <circle
                  cx="40"
                  cy="40"
                  r="33"
                  stroke="currentColor"
                  strokeWidth="8"
                  fill="transparent"
                  strokeDasharray={207.3}
                  strokeDashoffset={207.3 - (207.3 * score) / 100}
                  className={`${getScoreColor(score)} transition-all duration-1000 ease-out`}
                />
              </svg>
            </div>
          </div>

          <MetricCard
            paused={metricsPaused}
            theme={theme}
            label={metricMeta.trunk.label}
            value={metrics.trunkAngle.toFixed(1)}
            unit={metricMeta.trunk.unit}
            icon={Activity}
            variant="trunk"
            rawValue={metrics.trunkAngle}
            signedValue={signedMetrics.trunkAngle}
            threshold={metricMeta.thresholds.trunk}
            progress={metricQuality(
              metrics.trunkAngle,
              metricMeta.thresholds.trunk,
            )}
          />
          <MetricCard
            paused={metricsPaused}
            theme={theme}
            label={metricMeta.head.label}
            value={metrics.headForward.toFixed(2)}
            unit={metricMeta.head.unit}
            icon={ChevronRight}
            variant="head"
            rawValue={metrics.headForward}
            signedValue={signedMetrics.headForward}
            threshold={metricMeta.thresholds.head}
            progress={metricQuality(
              metrics.headForward,
              metricMeta.thresholds.head,
            )}
          />
          <MetricCard
            paused={metricsPaused}
            theme={theme}
            label={metricMeta.shoulder.label}
            value={metrics.shoulderTilt.toFixed(2)}
            unit={metricMeta.shoulder.unit}
            icon={Maximize2}
            variant="shoulder"
            rawValue={metrics.shoulderTilt}
            signedValue={signedMetrics.shoulderTilt}
            threshold={metricMeta.thresholds.shoulder}
            progress={metricQuality(
              metrics.shoulderTilt,
              metricMeta.thresholds.shoulder,
            )}
          />
        </div>
      </div>

      {floatingRootRef.current
        ? createPortal(
            <FloatingStatusPanel
              compact={isActive && isPageFocused && !floatingWindowReady}
              isActive={isActive}
              pill={pill}
              score={score}
              feedback={feedback}
            />,
            floatingRootRef.current,
          )
        : null}

      {showGuidedTrialIntro &&
      !showPrivacyNotice &&
      !showPrivacyPolicy &&
      !showTutorial ? (
        <div
          className={`fixed inset-0 z-[65] flex items-center justify-center p-4 backdrop-blur-md ${tutorialOverlayClass}`}
        >
          <div
            ref={guidedTrialIntroRef}
            role="dialog"
            aria-modal="true"
            aria-labelledby="guided-trial-intro-title"
            aria-describedby="guided-trial-intro-description"
            className={`w-full max-w-lg rounded-[1.75rem] border p-6 shadow-[0_28px_90px_-28px_rgba(0,0,0,0.7)] sm:p-8 ${
              isDarkTheme
                ? "border-white/12 bg-[#171715] text-[#f4f0e8]"
                : "border-[#cbdbea] bg-white text-[#1c1b19]"
            }`}
          >
            <p className={`text-xs font-semibold uppercase tracking-[0.16em] ${mutedTextClass}`}>
              First-session practice
            </p>
            <h2
              id="guided-trial-intro-title"
              className="mt-3 text-3xl font-bold tracking-[-0.035em]"
            >
              Try a guided posture check
            </h2>
            <p
              id="guided-trial-intro-description"
              className={`mt-4 text-sm leading-6 ${quietTextClass}`}
            >
              Start from a comfortable upright position. Uprightly will guide
              you through gentle posture and framing changes so you can see how
              its feedback responds.
            </p>
            <div
              className={`mt-5 border-y py-4 text-xs leading-5 ${
                isDarkTheme
                  ? "border-white/10 text-white/60"
                  : "border-[#0A3A72]/15 text-stone-600"
              }`}
            >
              The check covers forward posture, shoulder alignment, camera
              framing, recovery, and the floating status window. It is guidance
              only, not a medical assessment.
            </div>
            {!isActive ? (
              <p className={`mt-4 text-xs ${mutedTextClass}`}>
                Your camera will start after you continue and request permission
                if needed.
              </p>
            ) : null}
            <div className="mt-6 grid gap-2 sm:grid-cols-[auto_1fr]">
              <button
                type="button"
                onClick={dismissGuidedTrialIntro}
                className={`min-h-11 rounded-xl border px-5 py-2.5 text-sm font-semibold transition-colors ${
                  isDarkTheme
                    ? "border-white/15 text-white/70 hover:bg-white/10 hover:text-white"
                    : "border-stone-200 text-stone-600 hover:bg-stone-100 hover:text-stone-900"
                }`}
              >
                Not now
              </button>
              <button
                type="button"
                data-guided-trial-autofocus
                onClick={confirmGuidedTrial}
                className={`min-h-11 rounded-xl px-5 py-2.5 text-sm font-semibold ${primaryButtonClass}`}
              >
                {isActive ? "Start guided check" : "Start session and begin"}
              </button>
            </div>
          </div>
        </div>
      ) : null}

      {showPrivacyNotice &&
      !showPrivacyPolicy &&
      startupNoticeStep === "purpose" ? (
        <div
          className={`fixed inset-0 z-[70] overflow-x-hidden overflow-y-auto ${
            isDarkTheme ? "bg-[#10100f]" : "bg-[#f2efe7]"
          }`}
        >
          <main
            ref={purposeNoticeRef}
            role="dialog"
            aria-modal="true"
            aria-label="Welcome to Uprightly"
            aria-describedby="purpose-notice-description"
            className={`relative flex min-h-dvh w-full max-w-full items-center justify-center overflow-hidden px-6 py-12 text-center sm:px-10 lg:px-16 ${
              isDarkTheme ? "text-[#f4f0e8]" : "text-[#1c1b19]"
            }`}
          >
            <div
              data-onboarding-ambient
              aria-hidden="true"
              className={`pointer-events-none absolute left-1/2 top-1/2 h-[34rem] w-[34rem] -translate-x-1/2 -translate-y-1/2 rounded-full blur-3xl ${
                isDarkTheme ? "bg-white/[0.035]" : "bg-[#0A3A72]/[0.055]"
              }`}
            />
            <div
              data-onboarding-ambient
              aria-hidden="true"
              className={`pointer-events-none absolute inset-x-0 top-0 h-px ${
                isDarkTheme ? "bg-white/15" : "bg-[#0A3A72]/20"
              }`}
            />

            <div className="relative mx-auto flex w-full max-w-6xl flex-col items-center">
              <p
                data-onboarding-reveal
                className={`text-[clamp(1rem,1.65vw,1.35rem)] font-bold uppercase tracking-[0.24em] ${
                  isDarkTheme ? "text-white/82" : "text-[#0A3A72]/80"
                }`}
              >
                Welcome to
              </p>
              <h1
                data-onboarding-reveal
                className={`mt-1 w-full max-w-6xl text-[clamp(4rem,10vw,8.5rem)] font-black uppercase leading-[0.82] tracking-[-0.065em] ${
                  isDarkTheme ? "text-[#e8e7e2]" : "text-[#0A3A72]"
                }`}
              >
                Uprightly
              </h1>

              <p
                data-onboarding-reveal
                className="mt-10 text-[clamp(1.05rem,1.7vw,1.35rem)] font-semibold tracking-[-0.02em]"
              >
                Designed to work alongside you
              </p>

              <p
                id="purpose-notice-description"
                data-onboarding-reveal
                className={`mt-4 max-w-2xl text-sm leading-6 sm:text-base sm:leading-7 ${quietTextClass}`}
              >
                Start a session, then continue working or studying in another
                tab. On supported browsers, Uprightly keeps your posture
                feedback visible in a small floating window.
              </p>

              <p
                data-onboarding-reveal
                className={`mt-5 text-xs font-medium ${mutedTextClass}`}
              >
                Posture guidance only—not medical advice.
              </p>

              <button
                type="button"
                autoFocus
                data-privacy-autofocus
                data-onboarding-reveal
                onClick={continueToPrivacyNotice}
                className={`mt-8 min-h-12 w-full max-w-sm rounded-xl px-6 py-3 text-sm font-semibold transition-[background-color,transform] hover:-translate-y-0.5 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--uprightly-accent)] focus-visible:ring-offset-2 focus-visible:ring-offset-transparent motion-reduce:transform-none ${primaryButtonClass}`}
              >
                Continue to privacy
              </button>
            </div>
          </main>
        </div>
      ) : null}

      {showPrivacyNotice &&
      !showPrivacyPolicy &&
      startupNoticeStep === "privacy" ? (
        <div
          className={`fixed inset-0 z-[70] overflow-x-hidden overflow-y-auto ${
            isDarkTheme ? "bg-[#10100f]" : "bg-[#f2efe7]"
          }`}
        >
          <main
            ref={privacyNoticeRef}
            role="dialog"
            aria-modal="true"
            aria-labelledby="privacy-notice-title"
            aria-describedby="privacy-notice-description"
            className={`relative flex min-h-dvh w-full max-w-full items-center overflow-hidden px-6 py-10 sm:px-10 lg:px-16 ${
              isDarkTheme ? "text-[#f4f0e8]" : "text-[#1c1b19]"
            }`}
          >
            <div
              data-onboarding-ambient
              aria-hidden="true"
              className={`pointer-events-none absolute -right-40 top-1/2 h-[30rem] w-[30rem] -translate-y-1/2 rounded-full blur-3xl ${
                isDarkTheme ? "bg-white/[0.035]" : "bg-[#0A3A72]/[0.055]"
              }`}
            />

            <div className="relative mx-auto grid w-full max-w-6xl gap-10 lg:grid-cols-[0.9fr_1.1fr] lg:items-center lg:gap-20">
              <section>
                <p
                  data-onboarding-reveal
                  className={`text-xs font-semibold uppercase tracking-[0.18em] ${mutedTextClass}`}
                >
                  Privacy before posture guidance
                </p>
                <h2
                  id="privacy-notice-title"
                  data-onboarding-reveal
                  className="mt-3 max-w-xl text-[clamp(2.6rem,5vw,5.25rem)] font-bold leading-[0.94] tracking-[-0.055em]"
                >
                  Your camera stays private
                </h2>
                <p
                  id="privacy-notice-description"
                  data-onboarding-reveal
                  className={`mt-6 max-w-lg text-base leading-7 ${quietTextClass}`}
                >
                  Uprightly analyzes your camera feed live for posture guidance.
                  It does not record, upload, or save video clips.
                </p>
                <p
                  data-onboarding-reveal
                  className={`mt-5 text-xs font-medium ${mutedTextClass}`}
                >
                  Uprightly is a guidance tool, not medical care.
                </p>
              </section>

              <section data-onboarding-reveal>
                <div
                  className={`divide-y border-y text-sm leading-6 ${
                    isDarkTheme
                      ? "divide-white/10 border-white/15"
                      : "divide-[#0A3A72]/15 border-[#0A3A72]/20"
                  }`}
                >
                  <div className="grid gap-1 py-4 sm:grid-cols-[10rem_1fr] sm:gap-5">
                    <strong>Processed temporarily</strong>
                    <p className={quietTextClass}>
                      Camera frames are processed temporarily in your browser
                      while posture guidance is active.
                    </p>
                  </div>
                  <div className="grid gap-1 py-4 sm:grid-cols-[10rem_1fr] sm:gap-5">
                    <strong>Measurements only</strong>
                    <p className={quietTextClass}>
                      Numerical posture measurements may be sent to the analysis
                      service—never images or video.
                    </p>
                  </div>
                  <div className="grid gap-1 py-4 sm:grid-cols-[10rem_1fr] sm:gap-5">
                    <strong>Saved on this device</strong>
                    <p className={quietTextClass}>
                      Preferences stay in this browser. Uprightly does not
                      currently create tracking cookies.
                    </p>
                  </div>
                </div>

                <label
                  className={`mt-5 flex cursor-pointer items-start gap-3 border-b pb-5 text-sm ${
                    isDarkTheme
                      ? "border-white/10 text-white/75"
                      : "border-[#0A3A72]/15 text-stone-700"
                  }`}
                >
                  <input
                    type="checkbox"
                    checked={rememberPrivacyNotice}
                    onChange={(event) =>
                      setRememberPrivacyNotice(event.target.checked)
                    }
                    className={`mt-0.5 h-4 w-4 shrink-0 ${
                      isDarkTheme ? "accent-[#e8e7e2]" : "accent-[#0A3A72]"
                    }`}
                  />
                  <span>
                    <span className="font-semibold">
                      Don&apos;t show these opening screens again
                    </span>
                    <span className={`mt-0.5 block text-xs ${mutedTextClass}`}>
                      You can revisit this information from Settings at any time.
                    </span>
                  </span>
                </label>

                <div className="mt-5 grid gap-2 sm:grid-cols-[auto_1fr]">
                  <button
                    type="button"
                    onClick={() => setStartupNoticeStep("purpose")}
                    className={`min-h-12 rounded-xl border px-5 py-3 text-sm font-semibold transition-colors ${
                      isDarkTheme
                        ? "border-white/15 bg-transparent text-white/75 hover:bg-white/5 hover:text-white"
                        : "border-[#0A3A72]/20 bg-transparent text-stone-600 hover:bg-white/50 hover:text-stone-900"
                    }`}
                  >
                    Back
                  </button>
                  <button
                    type="button"
                    autoFocus
                    data-privacy-autofocus
                    onClick={continueFromPrivacyNotice}
                    className={`min-h-12 rounded-xl px-6 py-3 text-sm font-semibold transition-[background-color,transform] hover:-translate-y-0.5 motion-reduce:transform-none ${primaryButtonClass}`}
                  >
                    Continue to Uprightly
                  </button>
                  <button
                    type="button"
                    onClick={openPrivacyPolicy}
                    className={`min-h-11 rounded-xl px-4 py-2.5 text-sm font-semibold underline-offset-4 transition-colors hover:underline sm:col-span-2 ${
                      isDarkTheme
                        ? "text-white/70 hover:text-white"
                        : "text-[#0A3A72] hover:text-[#082f5d]"
                    }`}
                  >
                    Read Privacy &amp; Data Use
                  </button>
                </div>
              </section>
            </div>
          </main>
        </div>
      ) : null}

      {showPrivacyPolicy ? (
        <div
          className={`fixed inset-0 z-[80] flex items-center justify-center p-4 backdrop-blur-md ${tutorialOverlayClass}`}
        >
          <div
            ref={privacyPolicyRef}
            role="dialog"
            aria-modal="true"
            aria-labelledby="privacy-policy-title"
            aria-describedby="privacy-policy-summary"
            className={`flex max-h-[calc(100dvh-2rem)] w-full max-w-2xl flex-col overflow-hidden rounded-[1.75rem] border shadow-[0_28px_90px_-28px_rgba(0,0,0,0.7)] ${
              isDarkTheme
                ? "border-white/12 bg-[#171715] text-[#f4f0e8]"
                : "border-[#ded8cc] bg-[#fffdf8] text-[#1c1b19]"
            }`}
          >
            <div className={`flex shrink-0 items-start justify-between gap-4 border-b px-5 py-5 sm:px-7 ${isDarkTheme ? "border-white/8" : "border-stone-200"}`}>
              <div className="flex items-center gap-3">
                <span className={`flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl border ${accentSoftClass}`}>
                  <ShieldCheck size={21} aria-hidden="true" />
                </span>
                <div>
                  <h2
                    id="privacy-policy-title"
                    className="text-xl font-bold tracking-[-0.02em]"
                  >
                    Privacy &amp; Data Use
                  </h2>
                  <p className={`mt-0.5 text-xs ${mutedTextClass}`}>
                    Version {PRIVACY_NOTICE_VERSION} · Updated {PRIVACY_POLICY_UPDATED}
                  </p>
                </div>
              </div>
              <button
                type="button"
                data-privacy-autofocus
                onClick={closePrivacyPolicy}
                aria-label="Close privacy policy"
                className={`flex h-11 w-11 shrink-0 items-center justify-center rounded-xl border transition-colors ${
                  isDarkTheme
                    ? "border-white/15 bg-white/5 text-white/70 hover:bg-white/10 hover:text-white"
                    : "border-[#ded8cc] bg-white text-stone-600 hover:bg-[#f2efe7] hover:text-stone-900"
                }`}
              >
                <X size={18} aria-hidden="true" />
              </button>
            </div>

            <div className="overflow-y-auto px-5 py-5 sm:px-7">
              <p id="privacy-policy-summary" className={`text-sm leading-6 ${quietTextClass}`}>
                This notice explains how Uprightly uses your camera and handles
                information while providing live posture guidance.
              </p>

              <div className="mt-6 space-y-6">
                <section>
                  <h3 className="font-semibold">Camera access</h3>
                  <p className={`mt-1.5 text-sm leading-6 ${quietTextClass}`}>
                    Camera access begins only after you press Start Session and
                    approve the browser permission. Uprightly requests video only,
                    not microphone audio. Stopping the session ends the active
                    camera stream.
                  </p>
                </section>

                <section>
                  <h3 className="font-semibold">Video and image handling</h3>
                  <p className={`mt-1.5 text-sm leading-6 ${quietTextClass}`}>
                    Camera frames are analyzed temporarily in your browser to find
                    posture landmarks. Uprightly does not record, upload, or save
                    video clips or camera images.
                  </p>
                </section>

                <section>
                  <h3 className="font-semibold">Posture measurements</h3>
                  <p className={`mt-1.5 text-sm leading-6 ${quietTextClass}`}>
                    Derived numerical measurements—such as trunk angle, head
                    position, shoulder tilt, and stability—may be sent to the
                    configured analysis service. These requests do not contain
                    images, video, or audio.
                  </p>
                </section>

                <section>
                  <h3 className="font-semibold">Cookies and local preferences</h3>
                  <p className={`mt-1.5 text-sm leading-6 ${quietTextClass}`}>
                    Uprightly does not currently create cookies. Theme choice,
                    tutorial completion, and your privacy-notice preference are
                    stored locally in this browser. Clearing site data removes
                    these preferences.
                  </p>
                </section>

                <section>
                  <h3 className="font-semibold">Your control</h3>
                  <p className={`mt-1.5 text-sm leading-6 ${quietTextClass}`}>
                    You can stop a session at any time and revoke camera access in
                    your browser settings. You can also reopen this notice from
                    Settings whenever you want.
                  </p>
                </section>
              </div>
            </div>

            <div className={`shrink-0 border-t px-5 py-4 sm:px-7 ${isDarkTheme ? "border-white/8" : "border-stone-200"}`}>
              <button
                type="button"
                onClick={closePrivacyPolicy}
                className={`min-h-11 w-full rounded-xl px-5 py-2.5 text-sm font-semibold transition-colors ${primaryButtonClass}`}
              >
                Done
              </button>
            </div>
          </div>
        </div>
      ) : null}

      {showTutorial ? (
        <div className="pointer-events-none fixed inset-0 z-50">
          {tutorialHighlightStyle ? (
            <div
              className="fixed rounded-[1.75rem] border-2 transition-all duration-300 motion-reduce:transition-none"
              style={{
                ...tutorialHighlightStyle,
                borderColor: accentColor,
                boxShadow: `0 0 0 9999px ${isDarkTheme ? "rgba(16, 16, 15, 0.82)" : "rgba(242, 239, 231, 0.84)"}`,
              }}
              aria-hidden="true"
            />
          ) : (
            <div
              className={`fixed inset-0 backdrop-blur-md ${tutorialOverlayClass}`}
              aria-hidden="true"
            />
          )}

          <div
            ref={tutorialCardRef}
            role="dialog"
            aria-modal="true"
            aria-labelledby="tutorial-title"
            aria-describedby="tutorial-description tutorial-keyboard-hint"
            className={`pointer-events-auto fixed rounded-[1.75rem] border p-5 shadow-[0_28px_90px_-28px_rgba(0,0,0,0.65)] transition-[left,top,bottom] duration-300 motion-reduce:transition-none sm:p-6 ${isDarkTheme ? "border-white/12 bg-[#171715] text-[#f4f0e8]" : "border-[#ded8cc] bg-[#fffdf8] text-[#1c1b19]"}`}
            style={tutorialCardStyle}
          >
            <div className="flex items-start justify-between gap-4">
              <div
                className={`flex items-center gap-2 text-sm font-semibold ${
                  isDarkTheme ? "text-[#e8e7e2]" : "text-[#0A3A72]"
                }`}
              >
                <span className={`flex h-9 w-9 items-center justify-center rounded-full border ${accentSoftClass}`}>
                  <BookOpen size={16} aria-hidden="true" />
                </span>
                Quick tour
              </div>
              <button
                onClick={closeTutorial}
                className={`flex h-11 w-11 items-center justify-center rounded-full border transition-colors ${isDarkTheme ? "border-white/15 bg-white/5 text-white/70 hover:bg-white/10 hover:text-white" : "border-[#ded8cc] bg-white text-[#6f6a61] hover:bg-[#f2efe7] hover:text-[#1c1b19]"}`}
                aria-label="Close tutorial"
                title="Skip tutorial"
              >
                <X size={18} aria-hidden="true" />
              </button>
            </div>

            <div ref={tutorialContentRef} key={tutorialStepIndex}>
              <div
                data-tutorial-reveal
                className={`mt-5 text-xs font-semibold ${mutedTextClass}`}
              >
                {tutorialStepIndex + 1} / {TUTORIAL_STEPS.length}
              </div>
              <h2
                id="tutorial-title"
                data-tutorial-reveal
                className="mt-2 max-w-5xl text-2xl font-bold leading-tight tracking-[-0.025em]"
              >
                {currentTutorialStep.title}
              </h2>
              <p
                id="tutorial-description"
                data-tutorial-reveal
                className={`mt-3 text-[0.9375rem] leading-6 ${quietTextClass}`}
              >
                {currentTutorialStep.body}
              </p>
            </div>

            <p
              id="tutorial-keyboard-hint"
              className={`mt-5 rounded-xl border px-3 py-2 text-xs ${
                isDarkTheme
                  ? "border-white/10 bg-black/20 text-white/60"
                  : "border-[#e5dfd4] bg-[#f6f2ea] text-[#6f6a61]"
              }`}
            >
              Space / → Next · ← Back · Esc Skip
            </p>

            <div className="mt-4 flex items-center justify-center">
              <div className="flex items-center" aria-label="Tutorial progress">
                {TUTORIAL_STEPS.map((step, index) => (
                  <button
                    key={step.target}
                    onClick={() => setTutorialStepIndex(index)}
                    className="group flex h-11 w-11 items-center justify-center rounded-full"
                    aria-label={`Go to tutorial step ${index + 1}`}
                    aria-current={index === tutorialStepIndex ? "step" : undefined}
                  >
                    <span
                      className={`block h-2.5 rounded-full transition-all duration-300 ${
                        index === tutorialStepIndex
                          ? isDarkTheme
                            ? "w-7 bg-[#e8e7e2]"
                            : "w-7 bg-[#0A3A72]"
                          : isDarkTheme
                            ? "w-2.5 bg-white/25 group-hover:bg-white/45"
                            : "w-2.5 bg-[#c9c2b6] group-hover:bg-[#938b80]"
                      }`}
                    />
                  </button>
                ))}
              </div>
            </div>

            <div className="mt-3 grid grid-cols-2 gap-2">
              <button
                onClick={goToPreviousTutorialStep}
                disabled={tutorialStepIndex === 0}
                className={`min-h-11 rounded-xl border px-4 py-2.5 text-sm font-semibold transition-colors ${
                  tutorialStepIndex === 0
                    ? "cursor-not-allowed opacity-40"
                    : isDarkTheme
                      ? "border-white/15 bg-white/5 text-white/80 hover:bg-white/10 hover:text-white"
                      : "border-[#ded8cc] bg-white text-[#47433d] hover:bg-[#f2efe7] hover:text-[#1c1b19]"
                }`}
              >
                Back
              </button>
              <button
                data-tutorial-primary
                onClick={goToNextTutorialStep}
                className={`min-h-11 rounded-xl px-4 py-2.5 text-sm font-semibold transition-colors ${primaryButtonClass}`}
              >
                {tutorialStepIndex === TUTORIAL_STEPS.length - 1
                  ? "Finish"
                  : "Next"}
              </button>
            </div>

            <div className="sr-only" aria-live="polite" aria-atomic="true">
              Tutorial step {tutorialStepIndex + 1} of {TUTORIAL_STEPS.length}: {currentTutorialStep.title}
            </div>
          </div>
        </div>
      ) : null}
      <div className="sr-only" role="status" aria-live="polite" aria-atomic="true">
        {feedback}. Backend status: {mlStatusLabel}.
      </div>
    </main>
  );
}

function FloatingStatusPanel({
  compact,
  isActive,
  pill,
  score,
  feedback,
}: {
  compact: boolean;
  isActive: boolean;
  pill: Pill;
  score: number;
  feedback: string;
}) {
  const boundedScore = clamp(score, 0, 100);
  const radius = 48;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (boundedScore / 100) * circumference;

  const scoreTone =
    score > 80
      ? {
          from: "#91a889",
          to: "#607a62",
          cardBorder: "#dfe8dc",
          cardGlow: "#f1f5ec",
          dot: "#718f73",
          title: "Looking good!",
        }
      : score > 60
        ? {
            from: "#fbbf24",
            to: "#d97706",
            cardBorder: "#fef3c7",
            cardGlow: "#fffbeb",
            dot: "#f59e0b",
            title: "Needs attention",
          }
        : {
            from: "#fb7185",
            to: "#e11d48",
            cardBorder: "#ffe4e6",
            cardGlow: "#fff1f2",
            dot: "#f43f5e",
            title: "Adjust posture",
          };

  const feedbackTitle =
    pill === "good"
      ? "Looking good!"
      : pill === "fix"
        ? "Needs attention"
        : pill === "error"
          ? "Camera issue"
          : isActive
            ? scoreTone.title
            : "Ready to monitor";

  const feedbackMessage = isActive
    ? feedback
    : "Start a session to begin posture feedback.";

  if (compact) {
    return (
      <div
        style={{
          alignItems: "center",
          background:
            "linear-gradient(135deg, rgba(255,255,255,0.98), rgba(248,250,252,0.94))",
          boxSizing: "border-box",
          color: "#111827",
          display: "flex",
          fontFamily:
            'ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
          height: "100dvh",
          justifyContent: "center",
          overflow: "hidden",
          padding: 18,
          width: "100dvw",
        }}
      >
        <div
          style={{
            alignItems: "center",
            background: "#ffffff",
            border: "1px solid rgba(226, 232, 240, 0.95)",
            borderRadius: 22,
            boxShadow: "0 20px 40px -28px rgba(15, 23, 42, 0.35)",
            boxSizing: "border-box",
            display: "flex",
            gap: 14,
            minHeight: 0,
            padding: "16px 18px",
            width: "100%",
          }}
        >
          <span
            style={{
              background: isActive ? scoreTone.dot : "#d4cec2",
              borderRadius: 999,
              display: "block",
              flexShrink: 0,
              height: 10,
              width: 10,
            }}
          />
          <div style={{ minWidth: 0 }}>
            <div
              style={{
                color: "#1c1b19",
                fontSize: 14,
                fontWeight: 800,
                letterSpacing: "-0.01em",
                lineHeight: 1.2,
              }}
            >
              SukatLikod is active
            </div>
            <div
              style={{
                color: "#6f6a61",
                fontSize: 12.5,
                fontWeight: 600,
                lineHeight: 1.4,
                marginTop: 3,
                whiteSpace: "nowrap",
              }}
            >
              Return here when the browser is in the background.
            </div>
          </div>
          <div
            style={{
              color: "#1c1b19",
              flexShrink: 0,
              fontSize: 28,
              fontWeight: 900,
              letterSpacing: "-0.05em",
              lineHeight: 1,
              marginLeft: "auto",
            }}
          >
            {boundedScore}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div
      style={{
        alignItems: "center",
        background: "#ffffff",
        boxSizing: "border-box",
        color: "#111827",
        display: "flex",
        fontFamily:
          'ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
        height: "100dvh",
        justifyContent: "center",
        overflow: "hidden",
        padding: 0,
        width: "100dvw",
      }}
    >
      <div
        style={{
          background: "#ffffff",
          border: 0,
          borderRadius: 0,
          boxShadow: "none",
          boxSizing: "border-box",
          display: "flex",
          flexDirection: "column",
          height: "100%",
          maxWidth: "none",
          overflow: "hidden",
          width: "100%",
        }}
      >
        <div
          style={{
            alignItems: "center",
            display: "flex",
            justifyContent: "space-between",
            padding: "28px 28px 24px",
          }}
        >
          <div style={{ alignItems: "center", display: "flex", gap: 10 }}>
            <span
              style={{
                background: isActive ? scoreTone.dot : "#d4cec2",
                borderRadius: 999,
                display: "block",
                height: 7,
                width: 7,
              }}
            />
            <span
              style={{
                color: "#47433d",
                fontSize: 15,
                fontWeight: 700,
                letterSpacing: "-0.01em",
              }}
            >
              Posture Monitor
            </span>
          </div>
        </div>

        <div
          style={{
            alignItems: "center",
            boxSizing: "border-box",
            display: "flex",
            flex: 1,
            flexDirection: "column",
            minHeight: 0,
            overflow: "hidden",
            padding: "10px 32px 32px",
          }}
        >
          <div
            style={{
              alignItems: "center",
              display: "flex",
              flexShrink: 0,
              justifyContent: "center",
              position: "relative",
            }}
          >
            <svg
              style={{
                height: "min(44vmin, 188px)",
                transform: "rotate(-90deg)",
                width: "min(44vmin, 188px)",
              }}
              viewBox="0 0 120 120"
              aria-hidden="true"
            >
              <defs>
                <linearGradient
                  id="score-gradient"
                  x1="0%"
                  y1="0%"
                  x2="100%"
                  y2="100%"
                >
                  <stop offset="0%" stopColor={scoreTone.from} />
                  <stop offset="100%" stopColor={scoreTone.to} />
                </linearGradient>
              </defs>
              <circle
                cx="60"
                cy="60"
                r={radius}
                stroke="#eee9df"
                strokeWidth="10"
                fill="none"
              />
              <circle
                cx="60"
                cy="60"
                r={radius}
                stroke="url(#score-gradient)"
                className="transition-all duration-1000 ease-out"
                strokeWidth="10"
                fill="none"
                strokeLinecap="round"
                strokeDasharray={circumference}
                strokeDashoffset={strokeDashoffset}
              />
            </svg>

            <div
              style={{
                alignItems: "center",
                bottom: 0,
                display: "flex",
                flexDirection: "column",
                justifyContent: "center",
                left: 0,
                position: "absolute",
                right: 0,
                top: 0,
              }}
            >
              <span
                style={{
                  color: "#34312c",
                  fontSize: "min(13vmin, 60px)",
                  fontWeight: 800,
                  letterSpacing: "-0.06em",
                  lineHeight: 1,
                  marginBottom: 8,
                }}
              >
                {boundedScore}
              </span>
              <span
                style={{
                  color: "#9b958c",
                  fontSize: 14,
                  fontWeight: 800,
                  letterSpacing: "0.2em",
                  textTransform: "uppercase",
                }}
              >
                Score
              </span>
            </div>
          </div>

          <div
            style={{
              background: "rgba(236, 253, 245, 0.42)",
              border: `1px solid ${scoreTone.cardBorder}`,
              borderRadius: 20,
              boxShadow: "0 16px 34px -24px rgba(15, 23, 42, 0.28)",
              boxSizing: "border-box",
              display: "flex",
              gap: 20,
              marginTop: "auto",
              overflow: "hidden",
              padding: 20,
              position: "relative",
              width: "100%",
            }}
          >
            <div
              style={{
                background: scoreTone.cardGlow,
                borderRadius: 999,
                filter: "blur(28px)",
                height: 96,
                position: "absolute",
                right: -32,
                top: -32,
                width: 96,
              }}
            />

            <div
              style={{
                flexShrink: 0,
                marginTop: 4,
                position: "relative",
                zIndex: 1,
              }}
            >
              <div
                style={{
                  alignItems: "center",
                  backgroundImage: `linear-gradient(135deg, ${scoreTone.from}, ${scoreTone.to})`,
                  borderRadius: 14,
                  color: "#ffffff",
                  display: "flex",
                  justifyContent: "center",
                  padding: 12,
                }}
              >
                <ShieldCheck size={20} />
              </div>
            </div>
            <div style={{ minWidth: 0, position: "relative", zIndex: 1 }}>
              <h3
                style={{
                  color: "#34312c",
                  fontSize: 18,
                  fontWeight: 800,
                  lineHeight: 1.2,
                  margin: "0 0 8px",
                }}
              >
                {feedbackTitle}
              </h3>
              <p
                style={{
                  color: "#6f6a61",
                  fontSize: 15.5,
                  fontWeight: 650,
                  lineHeight: 1.55,
                  margin: 0,
                  maxHeight: 80,
                  overflow: "hidden",
                }}
              >
                {feedbackMessage}
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
