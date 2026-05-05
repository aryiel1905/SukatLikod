import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type ComponentType,
} from "react";
import { createPortal } from "react-dom";
import {
  Camera,
  VideoOff,
  Settings2,
  Activity,
  AlertCircle,
  CheckCircle2,
  ChevronRight,
  Maximize2,
  Bell,
  X,
  PanelRightClose,
  PanelRightOpen,
  Moon,
  Sun,
  Monitor,
  Volume2,
  BookOpen,
  ShieldCheck,
} from "lucide-react";
import {
  FilesetResolver,
  FaceLandmarker,
  type FaceLandmarkerResult,
  PoseLandmarker,
  type PoseLandmarkerResult,
} from "@mediapipe/tasks-vision";

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
type CameraDevice = {
  id: string;
  label: string;
};
type SpeechStatus = "unsupported" | "blocked" | "ready" | "loading";
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
const WINDOW = 12;
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
    title: "Start or stop monitoring",
    body: "Use this control to open the camera and begin live posture checks. Press it again when you want to stop the session.",
  },
  {
    target: "camera-stage",
    title: "Stay inside the camera frame",
    body: "This is where the camera feed and pose guide appear. Keep your head and shoulders visible so the app can track your posture reliably.",
  },
  {
    target: "posture-score",
    title: "Read your posture score",
    body: "The score gives a quick posture summary. The metric cards underneath show which part needs attention, such as head offset or shoulder level.",
  },
  {
    target: "session-log",
    title: "Follow the live feedback",
    body: "The session log records posture feedback while monitoring is active, so you can review what changed during the session.",
  },
  {
    target: "settings-panel",
    title: "Adjust the session setup",
    body: "Settings control the camera source, theme, voice feedback, tutorial replay, and floating status window.",
  },
];

type SideKind = "left" | "right";

function clamp(n: number, a: number, b: number) {
  return Math.max(a, Math.min(b, n));
}

function avg(arr: number[]) {
  if (arr.length === 0) return null;
  return arr.reduce((s, x) => s + x, 0) / arr.length;
}

function variance(arr: number[]) {
  if (arr.length < 2) return 0;
  const mean = avg(arr) ?? 0;
  const sq = arr.reduce((s, x) => s + (x - mean) * (x - mean), 0);
  return sq / arr.length;
}

function stabilityFromVariance(trunkVar: number) {
  // Lower variance means steadier posture over the sequence window.
  return Math.round(clamp(100 - trunkVar * 35, 0, 100));
}

function pushLimited(arr: number[], x: number) {
  arr.push(x);
  if (arr.length > WINDOW) arr.shift();
}

function findVisibleTourTarget(target: TutorialTarget): TutorialRect | null {
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
      return {
        top: rect.top,
        left: rect.left,
        right: rect.right,
        bottom: rect.bottom,
        width: rect.width,
        height: rect.height,
      };
    }
  }

  return null;
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

function metricQuality(value: number, threshold: number) {
  if (threshold <= 0) return 0;
  const ratio = value / threshold;
  return Math.round(clamp(100 - (ratio - 1) * 70, 0, 100));
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
  let color = "text-cyan-400";
  let bg = "bg-cyan-500/10 border-cyan-500/20";
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
    color = "text-emerald-400";
    bg = "bg-emerald-500/10 border-emerald-500/20";
    text = "Nice posture. Keep it steady.";
    audio = "That looks good. Keep it there.";
  }

  return { type, title, color, bg, text, audio };
}

export default function App() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const chatEndRef = useRef<HTMLDivElement | null>(null);

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
  const autoFloatingWindowRef = useRef(false);
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
  const [showSessionLog, setShowSessionLog] = useState(true);
  const [showTutorial, setShowTutorial] = useState(true);
  const [tutorialStepIndex, setTutorialStepIndex] = useState(0);
  const [tutorialTargetRect, setTutorialTargetRect] =
    useState<TutorialRect | null>(null);
  const [floatingWindowEnabled, setFloatingWindowEnabled] = useState(false);
  const [floatingWindowReady, setFloatingWindowReady] = useState(false);
  const [autoPipStatus, setAutoPipStatus] =
    useState<AutoPipStatus>("unsupported");
  const [isPageFocused, setIsPageFocused] = useState(() =>
    typeof document === "undefined" ? true : document.hasFocus() && !document.hidden,
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
  const overlayDetail = "detailed" as const;

  const modelPath = useMemo(() => "/models/pose_landmarker_lite.task", []);
  const faceModelPath = useMemo(
    () =>
      "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    [],
  );
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
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/wasm",
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
      if (!mlApiUrl || inferInFlightRef.current) return null;

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

        if (!res.ok) return null;
        const data = (await res.json()) as MlPrediction;
        return data;
      } catch {
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
      const upperBackwardLean = Math.max(
        0,
        chinLiftProxy - CHIN_LIFT_PROXY_NEUTRAL,
      ) * CHIN_LIFT_PROXY_TO_HEAD_LEAN_SCALE;
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

      const dBase = computeDecision(captureTier, effectiveSensitivity, baseline);
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
    floatingRootRef.current = null;
    setFloatingWindowReady(false);
    autoFloatingWindowRef.current = false;

    const pipWindow = floatingWindowRef.current;
    floatingWindowRef.current = null;

    if (pipWindow && !pipWindow.closed) {
      pipWindow.close();
    }
  }, []);

  const openFloatingWindow = useCallback(async () => {
    if (typeof window === "undefined") return;

    const pipApi = (
      window as Window & {
        documentPictureInPicture?: DocumentPictureInPictureApi;
      }
    ).documentPictureInPicture;

    if (!pipApi?.requestWindow) return;

    if (floatingWindowRef.current && !floatingWindowRef.current.closed) {
      floatingWindowRef.current.focus();
      setFloatingWindowReady(true);
      return;
    }

    const pipWindow = await pipApi.requestWindow({
      width: 456,
      height: 535,
    });

    floatingWindowRef.current = pipWindow;
    pipWindow.document.title = "SukatLikod Status";
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
      floatingWindowRef.current = null;
      floatingRootRef.current = null;
      setFloatingWindowReady(false);
      setFloatingWindowEnabled(false);
    });
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
    document.body.style.background = theme === "dark" ? "#0a0a0c" : "#eef2f7";
    if (floatingWindowRef.current?.document?.body) {
      floatingWindowRef.current.document.body.style.background = "transparent";
    }
  }, [theme]);

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
    if (floatingWindowEnabled) {
      void openFloatingWindow().catch((error) => {
        console.error("Floating window failed:", error);
        setFloatingWindowEnabled(false);
      });
      return;
    }

    if (!autoFloatingWindowRef.current) {
      closeFloatingWindow();
    }
  }, [closeFloatingWindow, floatingWindowEnabled, openFloatingWindow]);

  useEffect(() => closeFloatingWindow, [closeFloatingWindow]);

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
      console.warn("Unable to update camera activity for media session:", error);
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
        (
          window as Window & {
            documentPictureInPicture?: DocumentPictureInPictureApi;
          }
        ).documentPictureInPicture?.requestWindow
      );

    if (
      typeof window === "undefined" ||
      !("mediaSession" in navigator) ||
      !supportsFloatingWindow ||
      !isActive
    ) {
      return;
    }

    try {
      navigator.mediaSession.setActionHandler(AUTO_PIP_ACTION, () => {
        setFloatingWindowEnabled(true);
        void openFloatingWindow().catch((error) => {
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
  }, [isActive, openFloatingWindow]);

  useEffect(() => {
    const supportsFloatingWindow =
      typeof window !== "undefined" &&
      !!(
        (
          window as Window & {
            documentPictureInPicture?: DocumentPictureInPictureApi;
          }
        ).documentPictureInPicture?.requestWindow
      );

    if (typeof window === "undefined" || !supportsFloatingWindow || !isActive) {
      if (autoFloatingWindowRef.current && !floatingWindowEnabled) {
        closeFloatingWindow();
      }
      return;
    }

    const syncFloatingWindowWithFocus = () => {
      const pageFocused = document.hasFocus() && !document.hidden;

      if (pageFocused) {
        return;
      }

      if (floatingWindowEnabled || autoFloatingWindowRef.current) {
        return;
      }

      void openFloatingWindow()
        .then(() => {
          autoFloatingWindowRef.current = true;
        })
        .catch((error) => {
          console.error("Focus-based floating window failed:", error);
        });
    };

    syncFloatingWindowWithFocus();
    document.addEventListener("visibilitychange", syncFloatingWindowWithFocus);
    window.addEventListener("focus", syncFloatingWindowWithFocus);
    window.addEventListener("blur", syncFloatingWindowWithFocus);

    return () => {
      document.removeEventListener(
        "visibilitychange",
        syncFloatingWindowWithFocus,
      );
      window.removeEventListener("focus", syncFloatingWindowWithFocus);
      window.removeEventListener("blur", syncFloatingWindowWithFocus);
    };
  }, [closeFloatingWindow, floatingWindowEnabled, isActive, openFloatingWindow]);

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

  useEffect(() => {
    void ensureLandmarker().catch((error) => {
      console.error("Pose preload failed:", error);
    });
  }, [ensureLandmarker]);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.code !== "Space" || e.repeat) return;

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
  }, [isActive, pill, start, stop]);

  const currentTutorialStep = TUTORIAL_STEPS[tutorialStepIndex];

  useEffect(() => {
    if (!showTutorial) return;

    if (currentTutorialStep.target === "settings-panel") {
      setShowSettings(true);
    }

    if (currentTutorialStep.target === "session-log") {
      setShowSessionLog(true);
    }
  }, [currentTutorialStep.target, showTutorial]);

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

    measure();
    const deferredMeasure = window.setTimeout(measure, 230);

    window.addEventListener("resize", measure);
    window.addEventListener("scroll", measure, true);

    return () => {
      window.clearTimeout(deferredMeasure);
      window.cancelAnimationFrame(frame);
      window.removeEventListener("resize", measure);
      window.removeEventListener("scroll", measure, true);
    };
  }, [currentTutorialStep.target, showSettings, showSessionLog, showTutorial]);

  const closeTutorial = () => {
    setShowTutorial(false);
  };

  const openTutorial = () => {
    setTutorialStepIndex(0);
    setShowTutorial(true);
  };

  const goToNextTutorialStep = () => {
    if (tutorialStepIndex >= TUTORIAL_STEPS.length - 1) {
      closeTutorial();
      return;
    }

    setTutorialStepIndex((index) =>
      Math.min(index + 1, TUTORIAL_STEPS.length - 1),
    );
  };

  const goToPreviousTutorialStep = () => {
    setTutorialStepIndex((index) => Math.max(index - 1, 0));
  };

  const getScoreColor = (s: number) => {
    if (s > 80) return "text-emerald-400";
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
    ? "bg-[#0a0a0c] text-slate-100"
    : "bg-[#eef2f7] text-slate-900";
  const heroCardClass = isDarkTheme
    ? "bg-gradient-to-br from-white/[0.08] to-transparent border-white/10 hover:bg-white/[0.08]"
    : "bg-gradient-to-br from-white to-slate-50 border-slate-200 hover:bg-white";
  const stageClass = isDarkTheme
    ? "bg-[#08090c] border-white/5"
    : "bg-white border-slate-200";
  const stageGlassClass = isDarkTheme
    ? "bg-black/45 backdrop-blur-md border-white/10"
    : "bg-white/88 backdrop-blur-md border-slate-200";
  const settingsPanelClass = isDarkTheme
    ? "bg-black/30 backdrop-blur-md border-white/10"
    : "bg-white/92 backdrop-blur-md border-slate-200";
  const subtleTextClass = isDarkTheme ? "text-white/60" : "text-slate-500";
  const mutedTextClass = isDarkTheme ? "text-white/40" : "text-slate-500";
  const quietTextClass = isDarkTheme ? "text-white/70" : "text-slate-600";
  const iconButtonClass = isDarkTheme
    ? "border-white/30 bg-white/10 text-white/80 hover:bg-white hover:text-black"
    : "border-slate-200 bg-white text-slate-700 hover:bg-slate-100";
  const sessionLogIconButtonClass = isDarkTheme
    ? "border-white/10 bg-black/30 text-white/80 hover:bg-black/45 hover:text-white"
    : "border-slate-200 bg-white/75 text-slate-700 hover:bg-white hover:text-slate-900";
  const primaryButtonClass = isDarkTheme
    ? "bg-white text-black hover:bg-slate-200 shadow-lg"
    : "bg-slate-900 text-white hover:bg-slate-700 shadow-lg";
  const tutorialOverlayClass = isDarkTheme
    ? "bg-black/78"
    : "bg-slate-100/80";
  const themeVars: CSSProperties = {
    color: isDarkTheme ? "#e5edf7" : "#0f172a",
  };
  const floatingWindowSupported =
    typeof window !== "undefined" &&
    !!(
      window as Window & {
        documentPictureInPicture?: DocumentPictureInPictureApi;
      }
        ).documentPictureInPicture?.requestWindow;

  const metricsPaused = !isActive || pill === "detecting" || trackingHealth < 45;

  const autoPipStatusLabel =
    autoPipStatus === "supported"
      ? "Auto PiP supported"
      : autoPipStatus === "manual_only"
        ? "Manual only"
        : autoPipStatus === "blocked"
          ? "Auto PiP blocked"
          : "Unsupported";
  const autoPipStatusMessage =
    autoPipStatus === "supported"
      ? "This browser can auto-open the floating window during an active session when the tab or window moves out of focus."
      : autoPipStatus === "manual_only"
        ? "Manual floating window works here, but this browser context does not expose the full automatic PiP hooks."
        : autoPipStatus === "blocked"
          ? "The floating window works manually, but this browser appears to reject the automatic PiP action for this page."
          : "This browser does not support the floating document window here. Chromium-based browsers on HTTPS support it best.";
  const tutorialCardWidth =
    typeof window === "undefined" ? 360 : Math.min(360, window.innerWidth - 32);
  const tutorialHighlightStyle: CSSProperties | undefined = tutorialTargetRect
    ? {
        top: tutorialTargetRect.top - 10,
        left: tutorialTargetRect.left - 10,
        width: tutorialTargetRect.width + 20,
        height: tutorialTargetRect.height + 20,
      }
    : undefined;
  const tutorialCardStyle: CSSProperties =
    typeof window !== "undefined" && tutorialTargetRect
      ? (() => {
          const gap = 18;
          const cardHeight = 300;
          const viewportPadding = 16;
          const viewportWidth = window.innerWidth;
          const viewportHeight = window.innerHeight;
          const fitsRight =
            tutorialTargetRect.right + gap + tutorialCardWidth <=
            viewportWidth - viewportPadding;
          const fitsLeft =
            tutorialTargetRect.left - gap - tutorialCardWidth >= viewportPadding;
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
        })()
      : {
          width: tutorialCardWidth,
          left:
            typeof window === "undefined"
              ? 16
              : Math.max(16, (window.innerWidth - tutorialCardWidth) / 2),
          top:
            typeof window === "undefined"
              ? 16
              : Math.max(16, window.innerHeight / 2 - 150),
        };
  const visibleFeedbacks = feedbacks.slice(-5).reverse();

  return (
    <div
      className={`min-h-screen font-sans p-4 md:p-8 flex items-center justify-center transition-colors duration-300 ${shellClass}`}
      style={themeVars}
    >
      <div className="w-full flex flex-col gap-6 h-full lg:h-[85vh]">
        <div className="flex-1 flex min-h-0 gap-4 lg:gap-6">
          <div className="hidden lg:flex h-full min-h-0 w-60 xl:w-64 flex-shrink-0 flex-col gap-4">
            <div className="items-center text-center flex flex-col gap-2">
              <h1
                className={`text-5xl font-bold tracking-tight bg-clip-text text-transparent ${
                  isDarkTheme
                    ? "bg-gradient-to-r from-white to-white/60"
                    : "bg-gradient-to-r from-slate-900 to-slate-500"
                }`}
              >
                Uprightly
              </h1>
              <p
                className={`text-sm font-medium uppercase tracking-[0.2em] ${mutedTextClass}`}
              >
                AI Posture Assistant
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

            <div className="mt-auto flex flex-col gap-4">
              <div
                data-tour="posture-score"
                className={`backdrop-blur-md border rounded-2xl p-4 flex flex-col gap-1 transition-all relative overflow-hidden group ${heroCardClass}`}
              >
                <div
                  className={`flex items-center justify-between mb-1 z-10 ${subtleTextClass}`}
                >
                  <span className="text-xs font-medium uppercase tracking-wider">
                    Posture Score
                  </span>
                  {score > 70 ? (
                    <CheckCircle2 size={14} className="text-emerald-400" />
                  ) : (
                    <AlertCircle
                      size={14}
                      className={isDarkTheme ? "text-white" : "text-slate-600"}
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
                        isDarkTheme ? "text-white/10" : "text-slate-200"
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
          </div>

          <div
            className={`flex-1 min-w-0 flex min-h-0 transition-[gap] duration-200 ease-in-out ${showSettings ? "gap-6" : "gap-0"}`}
          >
            <div
              data-tour="camera-stage"
              className={`relative flex-1 rounded-[2rem] overflow-hidden border shadow-2xl group transition-colors duration-300 ${stageClass}`}
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
                  className={`absolute inset-0 flex flex-col items-center justify-center backdrop-blur-sm z-10 ${isDarkTheme ? "bg-black/45" : "bg-white/55"}`}
                >
                  <div
                    className={`w-20 h-20 rounded-full flex items-center justify-center mb-4 ${isDarkTheme ? "bg-white/5" : "bg-slate-100"}`}
                  >
                    <Camera
                      size={32}
                      className={
                        isDarkTheme ? "text-white/20" : "text-slate-400"
                      }
                    />
                  </div>
                  <p className={`font-medium ${mutedTextClass}`}>
                    Camera Feed Inactive
                  </p>
                </div>
              ) : null}

              <div
                className={`absolute top-4 left-4 right-4 z-20 lg:hidden rounded-2xl px-4 py-3 flex items-center justify-end gap-3 border ${stageGlassClass}`}
              >
                <div className="flex gap-3">
                  <button
                    onClick={() => setShowSettings((v) => !v)}
                    className={`flex items-center justify-center p-2.5 rounded-full transition-all border ${iconButtonClass}`}
                    title="Toggle Settings"
                  >
                    <Settings2 size={18} />
                  </button>
                  <button
                    onClick={isActive ? stop : () => void start()}
                    disabled={isLoading}
                    data-tour="start-session"
                    className={`lg:hidden flex items-center gap-2 px-5 py-2.5 rounded-full font-semibold transition-all ${isActive ? "bg-rose-500/20 text-rose-400 border border-rose-500/30 hover:bg-rose-500/30" : primaryButtonClass} ${isLoading ? "opacity-50 cursor-not-allowed" : ""}`}
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
                  className={`absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-48 h-64 border-2 border-dashed rounded-full ${isDarkTheme ? "border-white/20" : "border-slate-300"}`}
                />
              </div>

              <div className="absolute left-6 bottom-6 z-20">
                {isActive ? (
                  <>
                    <div className="text-sm font-bold uppercase tracking-wider">
                      Stability {stabilityScore}%
                    </div>
                    <div
                      className={`text-[11px] font-semibold uppercase tracking-wider ${quietTextClass}`}
                    >
                      Tracking {trackingHealth}%
                    </div>
                  </>
                ) : null}
              </div>

              <div
                data-tour="session-log"
                className={`absolute right-0 top-0 bottom-0 z-30 transition-all duration-300 ease-out ${showSessionLog ? "w-72 lg:w-80" : "w-16"}`}
              >
                {showSessionLog ? (
                  <div className="relative h-full">
                    <div
                      className="absolute inset-y-0 right-0 w-[38rem] pointer-events-none"
                      style={{
                        background:
                          isDarkTheme
                            ? "linear-gradient(to left, rgba(255,255,255,0.44) 0%, rgba(255,255,255,0.32) 16%, rgba(255,255,255,0.24) 30%, rgba(255,255,255,0.16) 46%, rgba(255,255,255,0.1) 62%, rgba(255,255,255,0.05) 78%, rgba(255,255,255,0.02) 90%, rgba(255,255,255,0) 100%)"
                            : "linear-gradient(to left, rgba(0,0,0,0.94) 0%, rgba(0,0,0,0.88) 16%, rgba(0,0,0,0.78) 30%, rgba(0,0,0,0.64) 46%, rgba(0,0,0,0.46) 62%, rgba(0,0,0,0.26) 78%, rgba(0,0,0,0.1) 90%, rgba(0,0,0,0) 100%)",
                      }}
                    />
                    <div
                      className={`absolute inset-y-0 right-0 w-full ${
                        isDarkTheme
                          ? "bg-gradient-to-l from-white/88 via-white/42 via-45% to-transparent"
                          : "bg-gradient-to-l from-black/78 via-black/42 via-45% to-transparent"
                      }`}
                    />
                    <div className="absolute top-4 right-4 left-12 z-10 pointer-events-none">
                      <div className="ml-auto flex w-full max-w-[22.5rem] items-center justify-between gap-2 pointer-events-auto">
                        <div
                          className={`flex flex-shrink-0 items-center gap-2 rounded-full border px-3 py-2 ${isDarkTheme ? "border-white/10 bg-black/30 text-white/80" : "border-slate-200 bg-white/75 text-slate-700"} backdrop-blur-md`}
                        >
                          <Bell size={14} />
                          <span className="whitespace-nowrap text-[11px] font-bold uppercase tracking-[0.22em]">
                            Session Log
                          </span>
                        </div>
                        <div className="flex items-center gap-2">
                          <button
                            onClick={() => setShowSettings((v) => !v)}
                            className={`flex items-center justify-center p-2.5 rounded-full transition-all border backdrop-blur-md ${sessionLogIconButtonClass}`}
                            title="Toggle Settings"
                          >
                            <Settings2 size={16} />
                          </button>
                          <button
                            onClick={() => setShowSessionLog(false)}
                            className={`flex items-center justify-center p-2.5 rounded-full transition-all border backdrop-blur-md ${sessionLogIconButtonClass}`}
                            title="Hide Session Log"
                            aria-label="Hide Session Log"
                          >
                            <PanelRightClose size={16} />
                          </button>
                        </div>
                      </div>
                    </div>

                    <div
                      className="absolute top-20 bottom-4 right-4 left-12 z-10 flex items-end justify-end pointer-events-none"
                      style={{
                        maskImage:
                          "linear-gradient(to top, black 0%, black 58%, rgba(0,0,0,0.78) 72%, rgba(0,0,0,0.28) 84%, transparent 100%)",
                        WebkitMaskImage:
                          "linear-gradient(to top, black 0%, black 58%, rgba(0,0,0,0.78) 72%, rgba(0,0,0,0.28) 84%, transparent 100%)",
                      }}
                    >
                      {!isActive && feedbacks.length === 0 ? (
                        <div
                          className={`max-w-[17rem] rounded-2xl border px-4 py-3 ${
                            isDarkTheme
                              ? "border-white/10 bg-[#05060a] text-white/78"
                              : "border-slate-300 bg-white text-slate-700 shadow-[0_18px_45px_-30px_rgba(15,23,42,0.24)]"
                          }`}
                        >
                          <div className="text-[11px] font-bold uppercase tracking-[0.22em] opacity-75">
                            Standby
                          </div>
                          <div className="mt-1 text-sm leading-relaxed">
                            Start a session to see live posture feedback here.
                          </div>
                        </div>
                      ) : (
                        <div className="w-full max-w-[22.5rem] flex flex-col-reverse gap-3">
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
                                className={`rounded-2xl border px-4 py-3 transition-all ${
                                  isDarkTheme
                                    ? "border-transparent bg-black/70"
                                    : "border-slate-300 bg-white shadow-[0_22px_45px_-30px_rgba(15,23,42,0.24)]"
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
                                        className="text-emerald-400 flex-shrink-0"
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
                                        className="text-sky-400 flex-shrink-0"
                                      />
                                    )}
                                    <span
                                      className={`text-[11px] font-extrabold uppercase tracking-[0.16em] ${
                                        isDarkTheme
                                          ? "text-white/92"
                                          : "text-slate-800"
                                      }`}
                                    >
                                      {f.title}
                                    </span>
                                  </div>
                                  <span
                                    className={`text-[10px] ${
                                      isDarkTheme
                                        ? "text-white/45"
                                        : "text-slate-500"
                                    }`}
                                  >
                                    {f.time}
                                  </span>
                                </div>
                                <p
                                  className={`text-[13px] leading-relaxed ${
                                    isDarkTheme
                                      ? "text-white/92"
                                      : "text-slate-700"
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

                  </div>
                ) : (
                  <div className="absolute top-4 right-4 z-10 flex flex-col items-center gap-3 pointer-events-auto">
                    <button
                      onClick={() => setShowSessionLog(true)}
                      className={`flex items-center justify-center p-2.5 rounded-full transition-all border backdrop-blur-md ${sessionLogIconButtonClass}`}
                      title="Show Session Log"
                      aria-label="Show Session Log"
                    >
                      <PanelRightOpen size={16} />
                    </button>
                    <button
                      onClick={() => setShowSettings((v) => !v)}
                      className={`flex items-center justify-center p-2.5 rounded-full transition-all border backdrop-blur-md ${sessionLogIconButtonClass}`}
                      title="Toggle Settings"
                    >
                      <Settings2 size={16} />
                    </button>
                  </div>
                )}
              </div>
            </div>

            <div
              className={`flex-shrink-0 overflow-hidden transition-[width,opacity] duration-200 ease-in-out ${showSettings ? "w-80 opacity-100" : "w-0 opacity-0"}`}
            >
              <div
                aria-hidden={!showSettings}
                data-tour="settings-panel"
                className={`w-80 h-full rounded-[2rem] p-6 flex flex-col overflow-y-auto transition-transform duration-200 ease-in-out border ${showSettings ? "translate-x-0 pointer-events-auto" : "translate-x-3 pointer-events-none"} ${settingsPanelClass}`}
              >
                <div className="flex items-center justify-between mb-8">
                  <div className="flex items-center gap-2">
                    <Settings2
                      size={18}
                      className={
                        isDarkTheme ? "text-white/60" : "text-slate-500"
                      }
                    />
                    <h3 className="font-bold text-sm uppercase tracking-wider">
                      Settings
                    </h3>
                  </div>
                  <button
                    onClick={() => setShowSettings(false)}
                    className={`w-9 h-9 rounded-lg transition-colors flex items-center justify-center ${isDarkTheme ? "text-white/70 hover:text-white hover:bg-white/10" : "text-slate-500 hover:text-slate-900 hover:bg-slate-100"}`}
                    title="Close settings"
                    aria-label="Close settings"
                  >
                    <X size={20} />
                  </button>
                </div>

                <div className="space-y-8">
                  <div className="space-y-3">
                    <div className="flex justify-between items-center px-1">
                      <label
                        className={`text-xs font-semibold ${subtleTextClass}`}
                      >
                        Camera Source
                      </label>
                      <span
                        className={`text-[10px] font-bold uppercase tracking-wider ${mutedTextClass}`}
                      >
                        {cameraDevices.length || 0} detected
                      </span>
                    </div>
                    <select
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
                      className={`w-full rounded-xl border px-3 py-2.5 text-sm font-semibold outline-none transition-colors ${isDarkTheme ? "border-white/15 bg-white/5 text-white hover:bg-white/10" : "border-slate-200 bg-white text-slate-900 hover:bg-slate-50"}`}
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
                              ? "bg-slate-900 text-white"
                              : "bg-white text-slate-900"
                          }
                        >
                          {camera.label}
                        </option>
                      ))}
                    </select>
                    <p
                      className={`text-[11px] leading-relaxed ${mutedTextClass}`}
                    >
                      If labels are blank, allow camera access first, then
                      reopen this panel or start a session.
                    </p>
                  </div>

                  <div className="space-y-3">
                    <div className="flex justify-between items-center px-1">
                      <label
                        className={`text-xs font-semibold ${subtleTextClass}`}
                      >
                        Theme
                      </label>
                      <span
                        className={`text-[10px] font-bold uppercase tracking-wider ${mutedTextClass}`}
                      >
                        {theme === "dark" ? "Dark Mode" : "Light Mode"}
                      </span>
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
                              ? isDarkTheme
                                ? "border-white/40 bg-white text-black"
                                : "border-slate-900 bg-slate-900 text-white"
                              : isDarkTheme
                                ? "border-white/15 bg-white/5 text-white/70 hover:bg-white/10 hover:text-white"
                                : "border-slate-200 bg-white text-slate-600 hover:bg-slate-50 hover:text-slate-900"
                          }`}
                        >
                          <Icon size={16} />
                          {label}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className="space-y-3">
                    <div className="flex justify-between items-center px-1">
                      <label
                        className={`text-xs font-semibold ${subtleTextClass}`}
                      >
                        Audio Feedback
                      </label>
                      <span
                        className={`text-[10px] font-bold uppercase tracking-wider ${mutedTextClass}`}
                      >
                        5s cooldown
                      </span>
                    </div>
                    <div className="grid grid-cols-2 gap-2">
                      {(["off", "voice"] as const).map((mode) => (
                        <button
                          key={mode}
                          onClick={() => setAudioMode(mode)}
                          className={`rounded-xl border px-3 py-2 text-sm font-semibold transition-colors ${
                            audioMode === mode
                              ? isDarkTheme
                                ? "border-white/40 bg-white text-black"
                                : "border-slate-900 bg-slate-900 text-white"
                              : isDarkTheme
                                ? "border-white/15 bg-white/5 text-white/70 hover:bg-white/10 hover:text-white"
                                : "border-slate-200 bg-white text-slate-600 hover:bg-slate-50 hover:text-slate-900"
                          }`}
                        >
                          {mode === "off" ? "Off" : "Voice"}
                        </button>
                      ))}
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
                      className={`w-full rounded-xl border text-sm font-semibold py-2.5 transition-colors flex items-center justify-center gap-2 ${isDarkTheme ? "border-white/15 bg-white/5 hover:bg-white/10 text-white/80 hover:text-white" : "border-slate-200 bg-white hover:bg-slate-50 text-slate-700 hover:text-slate-900"}`}
                    >
                      <Volume2 size={16} />
                      Test Voice
                    </button>
                    <p
                      className={`text-[11px] leading-relaxed ${mutedTextClass}`}
                    >
                      Voice prompts play only on stable posture changes and are
                      suppressed during weak tracking.
                    </p>
                  </div>

                  <div className="space-y-3">
                    <div className="flex justify-between items-center px-1">
                      <label
                        className={`text-xs font-semibold ${subtleTextClass}`}
                      >
                        Tutorial
                      </label>
                      <span
                        className={`text-[10px] font-bold uppercase tracking-wider ${mutedTextClass}`}
                      >
                        Reopen anytime
                      </span>
                    </div>
                    <button
                      onClick={openTutorial}
                      className={`w-full rounded-xl border text-sm font-semibold py-2.5 transition-colors flex items-center justify-center gap-2 ${isDarkTheme ? "border-white/15 bg-white/5 hover:bg-white/10 text-white/80 hover:text-white" : "border-slate-200 bg-white hover:bg-slate-50 text-slate-700 hover:text-slate-900"}`}
                    >
                      <BookOpen size={16} />
                      Show Tutorial
                    </button>
                  </div>

                  <div className="space-y-3">
                    <div className="flex justify-between items-center px-1">
                      <label
                        className={`text-xs font-semibold ${subtleTextClass}`}
                      >
                        Floating Window
                      </label>
                      <span
                        className={`text-[10px] font-bold uppercase tracking-wider ${mutedTextClass}`}
                      >
                        {floatingWindowReady ? "Open" : "Optional"}
                      </span>
                    </div>
                    <button
                      onClick={() =>
                        floatingWindowSupported &&
                        setFloatingWindowEnabled((value) => !value)
                      }
                      disabled={!floatingWindowSupported}
                      className={`w-full rounded-xl border text-sm font-semibold py-2.5 transition-colors flex items-center justify-center gap-2 ${
                        !floatingWindowSupported
                          ? "cursor-not-allowed border-slate-200 bg-slate-100 text-slate-400"
                          : floatingWindowEnabled
                            ? isDarkTheme
                              ? "border-white/40 bg-white text-black"
                              : "border-slate-900 bg-slate-900 text-white"
                            : isDarkTheme
                              ? "border-white/15 bg-white/5 hover:bg-white/10 text-white/80 hover:text-white"
                              : "border-slate-200 bg-white hover:bg-slate-50 text-slate-700 hover:text-slate-900"
                      }`}
                    >
                      <Monitor size={16} />
                      {floatingWindowEnabled
                        ? "Close Floating Window"
                        : "Open Floating Window"}
                    </button>
                    <p
                      className={`text-[11px] leading-relaxed ${mutedTextClass}`}
                    >
                      {floatingWindowSupported
                        ? "While a session is running, the floating window stays hidden until you switch away or minimize the browser, then it tries to open once and stays available for the rest of the session. You can still open or close it manually here."
                        : autoPipStatusMessage}
                    </p>
                    <div
                      className={`flex justify-between items-center px-1 text-[11px] ${mutedTextClass}`}
                    >
                      <span>Status: {autoPipStatusLabel}</span>
                      <span>{isActive ? "Session live" : "Session idle"}</span>
                    </div>
                    {floatingWindowSupported ? (
                      <p
                        className={`text-[11px] leading-relaxed ${mutedTextClass}`}
                      >
                        {autoPipStatusMessage}
                      </p>
                    ) : null}
                  </div>
                </div>

                <div className="mt-auto pt-6">
                  <div
                    className={`p-4 rounded-2xl border ${isDarkTheme ? "bg-white/5 border-white/5" : "bg-slate-50 border-slate-200"}`}
                  >
                    <p
                      className={`text-[10px] leading-relaxed italic ${mutedTextClass}`}
                    >
                      Settings now focus on the essentials: camera source, dark
                      or light mode, audio feedback, and voice testing.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-2 lg:hidden gap-4 flex-shrink-0">
          <div
            data-tour="posture-score"
            className={`backdrop-blur-md border rounded-2xl p-4 flex flex-col gap-1 transition-all relative overflow-hidden group ${heroCardClass}`}
          >
            <div
              className={`flex items-center justify-between z-10 ${subtleTextClass}`}
            >
              <span className="text-xs font-medium uppercase tracking-wider">
                Posture Score
              </span>
              {score > 70 ? (
                <CheckCircle2 size={14} className="text-emerald-400" />
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
                  className={isDarkTheme ? "text-white/10" : "text-slate-200"}
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
              compact={isActive && isPageFocused && !floatingWindowEnabled}
              isActive={isActive}
              pill={pill}
              score={score}
              feedback={feedback}
            />,
            floatingRootRef.current,
          )
        : null}

      {showTutorial ? (
        <div className="fixed inset-0 z-50 pointer-events-none">
          {tutorialHighlightStyle ? (
            <div
              className={`fixed rounded-[1.75rem] border-2 transition-all duration-200 ${isDarkTheme ? "border-sky-300" : "border-sky-600"}`}
              style={{
                ...tutorialHighlightStyle,
                boxShadow: `0 0 0 9999px ${isDarkTheme ? "rgba(2, 6, 23, 0.78)" : "rgba(241, 245, 249, 0.82)"}`,
              }}
            />
          ) : (
            <div
              className={`fixed inset-0 backdrop-blur-md ${tutorialOverlayClass}`}
            />
          )}

          <div
            className={`fixed pointer-events-auto rounded-2xl border p-5 shadow-2xl transition-all duration-200 ${isDarkTheme ? "border-white/10 bg-[#040507] text-white" : "border-slate-200 bg-white text-slate-900"}`}
            style={tutorialCardStyle}
          >
            <div className="flex items-start justify-between gap-4">
              <div className="flex items-center gap-2 text-[11px] font-bold uppercase tracking-[0.22em] text-sky-500">
                <BookOpen size={15} />
                Guided Tour
              </div>
              <button
                onClick={closeTutorial}
                className={`-mt-1 flex h-8 w-8 items-center justify-center rounded-full border transition-colors ${isDarkTheme ? "border-white/15 bg-white/5 text-white/70 hover:bg-white/10 hover:text-white" : "border-slate-200 bg-white text-slate-500 hover:bg-slate-100 hover:text-slate-900"}`}
                aria-label="Close tutorial"
                title="Close tutorial"
              >
                <X size={16} />
              </button>
            </div>

            <div
              className={`mt-3 text-[11px] font-bold uppercase tracking-[0.2em] ${mutedTextClass}`}
            >
              Step {tutorialStepIndex + 1} of {TUTORIAL_STEPS.length}
            </div>
            <h2 className="mt-2 text-xl font-black tracking-tight">
              {currentTutorialStep.title}
            </h2>
            <p className={`mt-3 text-sm leading-6 ${quietTextClass}`}>
              {currentTutorialStep.body}
            </p>

            <div className="mt-5 flex items-center justify-between gap-3">
              <div className="flex items-center gap-1.5">
                {TUTORIAL_STEPS.map((step, index) => (
                  <button
                    key={step.target}
                    onClick={() => setTutorialStepIndex(index)}
                    className={`h-2.5 rounded-full transition-all ${
                      index === tutorialStepIndex
                        ? isDarkTheme
                          ? "w-6 bg-sky-300"
                          : "w-6 bg-sky-600"
                        : isDarkTheme
                          ? "w-2.5 bg-white/20 hover:bg-white/35"
                          : "w-2.5 bg-slate-300 hover:bg-slate-400"
                    }`}
                    aria-label={`Go to tutorial step ${index + 1}`}
                  />
                ))}
              </div>

              <div className="flex items-center gap-2">
                <button
                  onClick={goToPreviousTutorialStep}
                  disabled={tutorialStepIndex === 0}
                  className={`rounded-xl border px-3 py-2 text-sm font-semibold transition-colors ${
                    tutorialStepIndex === 0
                      ? "cursor-not-allowed opacity-40"
                      : isDarkTheme
                        ? "border-white/15 bg-white/5 text-white/80 hover:bg-white/10 hover:text-white"
                        : "border-slate-200 bg-white text-slate-700 hover:bg-slate-50 hover:text-slate-900"
                  }`}
                >
                  Back
                </button>
                <button
                  onClick={goToNextTutorialStep}
                  className={`rounded-xl px-4 py-2 text-sm font-semibold transition-colors ${primaryButtonClass}`}
                >
                  {tutorialStepIndex === TUTORIAL_STEPS.length - 1
                    ? "Finish"
                    : "Next"}
                </button>
              </div>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}

function MetricCard({
  paused,
  theme,
  label,
  value,
  unit,
  icon: Icon,
  variant,
  rawValue,
  signedValue,
  threshold,
  progress: _progress,
  colorClass,
}: {
  paused: boolean;
  theme: ThemeMode;
  label: string;
  value: string;
  unit: string;
  icon: ComponentType<{ size?: number; className?: string }>;
  variant: "trunk" | "head" | "shoulder";
  rawValue: number;
  signedValue: number;
  threshold: number;
  progress: number;
  colorClass?: string;
}) {
  void variant;
  void rawValue;
  void signedValue;
  void threshold;
  void _progress;
  const isDarkTheme = theme === "dark";

  return (
    <div
      className={`backdrop-blur-md border rounded-2xl p-4 flex flex-col gap-1 transition-all relative overflow-hidden group ${
        isDarkTheme
          ? paused
            ? "bg-gradient-to-br from-white/10 to-transparent border-white/10"
            : "bg-gradient-to-br from-white/10 via-white/[0.045] to-transparent border-white/10 hover:bg-white/10"
          : paused
            ? "bg-gradient-to-br from-white to-slate-50 border-slate-200"
            : "bg-gradient-to-br from-white to-slate-50 border-slate-200 hover:bg-white"
      }`}
    >
      <div
        className={`flex items-center justify-between mb-1 z-10 ${
          paused
            ? isDarkTheme
              ? "text-white/35"
              : "text-slate-400"
            : isDarkTheme
              ? "text-white/50"
              : "text-slate-500"
        }`}
      >
        <span className="text-xs font-medium uppercase tracking-wider">
          {label}
        </span>
        <div className="flex items-center gap-2">
          <Icon size={14} />
        </div>
      </div>
      <div className="flex items-baseline gap-1 z-10 mt-0.5">
        <span
          className={`text-2xl font-bold tracking-tight ${
            paused
              ? isDarkTheme
                ? "text-white/55"
                : "text-slate-500"
              : colorClass || (isDarkTheme ? "text-white" : "text-slate-900")
          }`}
        >
          {value}
        </span>
        <span
          className={`text-xs ${
            paused
              ? isDarkTheme
                ? "text-white/28"
                : "text-slate-400"
              : isDarkTheme
                ? "text-white/40"
                : "text-slate-500"
          }`}
        >
          {unit}
        </span>
      </div>
    </div>
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
          from: "#34d399",
          to: "#059669",
          cardBorder: "#d1fae5",
          cardGlow: "#ecfdf5",
          dot: "#10b981",
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
              background: isActive ? scoreTone.dot : "#cbd5e1",
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
                color: "#0f172a",
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
                color: "#64748b",
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
              color: "#0f172a",
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
                background: isActive ? scoreTone.dot : "#cbd5e1",
                borderRadius: 999,
                display: "block",
                height: 7,
                width: 7,
              }}
            />
            <span
              style={{
                color: "#334155",
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
                stroke="#f1f5f9"
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
                  color: "#1e293b",
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
                  color: "#94a3b8",
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

            <div style={{ flexShrink: 0, marginTop: 4, position: "relative", zIndex: 1 }}>
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
                  color: "#1e293b",
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
                  color: "#64748b",
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
