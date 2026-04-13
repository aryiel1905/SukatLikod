import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ComponentType,
} from "react";
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
type ViewCalibration = "front";
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
type CameraDevice = {
  id: string;
  label: string;
};
type SpeechStatus = "unsupported" | "blocked" | "ready" | "loading";
type DebugMetrics = {
  chinCenterOffset: number;
  chinForwardLean: number;
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

const WINDOW = 30;
const EMA_ALPHA = 0.25;
const VIS_THRESHOLD = 0.35;
const DRAW_VIS_THRESHOLD = 0.12;
const HOLD_STILL_MS = 850;
const CALIBRATION_MS = 2500;
const PREDICTION_VOTE_WINDOW = 6;
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
const UPPER_FRONT_CALIBRATION_MS = 3600;
const UPPER_FRONT_TRACKING_MIN = 62;
const UPPER_FRONT_FRAME_MARGIN = 0.08;
const UPPER_FRONT_SCORE_CAP = 86;
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
  noseCenterOffset: 0,
  mouthLineTilt: 0,
  eyeOrEarTilt: 0,
  upperForwardLean: 0,
  upperShoulderTilt: 0,
};

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

  if (hipWidth >= FRONT_HIP_WIDTH_MIN && torsoLength >= FRONT_TORSO_LENGTH_MIN) {
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
  if (lower.includes("turn and face")) return "Turn a little and face the camera.";
  if (lower.includes("hold still")) return "Hold still for a moment.";
  if (lower.includes("level your shoulders") || dominant === "shoulder") {
    return "Relax and level your shoulders.";
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

  if (scoreValue < 60) {
    type = "critical";
    title = "Sit Straighter";
    color = "text-rose-400";
    bg = "bg-rose-500/10 border-rose-500/20";
    text = msg;
    audio = getNaturalAudioFromMessage(msg, dominant);
  } else if (dominant === "head" && h > headThreshold) {
    type = "warning";
    title = "Bring Your Head Back";
    color = "text-amber-400";
    bg = "bg-amber-500/10 border-amber-500/20";
    text = "Lift through the crown of your head and keep your chin gently tucked.";
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
  const lastInferTsRef = useRef<number>(0);
  const inferInFlightRef = useRef<boolean>(false);
  const lastAudioEventRef = useRef<{ key: string; at: number }>({
    key: "",
    at: 0,
  });
  const lastAnnouncedStateRef = useRef<"good" | "fix" | "idle">("idle");
  const baselineMetricsRef = useRef<Record<FrontCaptureTier, BaselineMetrics | null>>({
    full_front: null,
    upper_front: null,
  });
  const loadedModelPathRef = useRef<string | null>(null);
  const loadedFaceModelPathRef = useRef<string | null>(null);
  const landmarkerLoadPromiseRef = useRef<Promise<void> | null>(null);
  const holdStillStartRef = useRef<number>(0);
  const calibrationRef = useRef<{
    activeView: ViewCalibration | null;
    startedAt: number;
    done: Record<ViewCalibration, boolean>;
  }>({
    activeView: null,
    startedAt: 0,
    done: { front: false },
  });
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
  const [showDebugPanel, setShowDebugPanel] = useState(true);
  const [pill, setPill] = useState<Pill>("idle");

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
  const [debugMetrics, setDebugMetrics] =
    useState<DebugMetrics>(DEFAULT_DEBUG_METRICS);

  const [sensitivity, setSensitivity] =
    useState<Sensitivity>(DEFAULT_SENSITIVITY);
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
    holdStillStartRef.current = 0;
    lastAudioEventRef.current = { key: "", at: 0 };
    lastAnnouncedStateRef.current = "idle";
    baselineMetricsRef.current = {
      full_front: null,
      upper_front: null,
    };
    calibrationRef.current = {
      activeView: null,
      startedAt: 0,
      done: { front: false },
    };
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
      const tRatio = t / tThr;
      const hRatio = h / hThr;
      const sRatio = s / sThr;
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
        captureTier === "full_front"
          ? "Good posture."
          : "Looking good.";
      if (!ok) {
        if (captureTier === "full_front") {
          if (dominant === "trunk")
            msg = "Sit straighter.";
          else if (dominant === "head")
            msg = "Bring your head back a little.";
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
        msg =
          captureTier === "full_front"
            ? "Good posture."
            : "Looking good.";
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
                  ...(faceLandmarks ? [{ ...faceLandmarks[FACE_IDX.CHIN] }] : []),
                ]
              : [
                  IDX.NOSE,
                  IDX.L_EAR,
                  IDX.R_EAR,
                  IDX.L_SHOULDER,
                  IDX.R_SHOULDER,
                  IDX.L_HIP,
                  IDX.R_HIP,
                  ...(faceLandmarks ? [{ ...faceLandmarks[FACE_IDX.CHIN] }] : []),
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
                    ? [[IDX.NOSE, { ...faceLandmarks[FACE_IDX.CHIN] }] as [NodeRef, NodeRef]]
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
                    ? [[IDX.NOSE, { ...faceLandmarks[FACE_IDX.CHIN] }] as [NodeRef, NodeRef]]
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
        setFeedback(
          "Keep both shoulders in view.",
        );
        return;
      }
      const hipsReady =
        captureTier === "full_front" &&
        !!lh &&
        !!rh &&
        !!lhN &&
        !!rhN;

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
      );
      const upperShoulderTilt = Math.max(
        Math.abs(lsN.y - rsN.y) / shoulderWidth,
        mouthLineTilt,
      );
      setDebugMetrics({
        chinCenterOffset,
        chinForwardLean,
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
          : Math.max(
              normalizedDepthDelta(nose as Point3, midShoulder, shoulderWorldWidth),
              chinForwardLean,
            );
      const sSigned =
        captureTier === "full_front"
          ? shoulderTiltSignedM(ls as Point3, rs as Point3)
          : (lsN.y - rsN.y) / shoulderWidth;

      const now = performance.now();
      const currentView: ViewCalibration = "front";
      if (!calibrationRef.current.done[currentView]) {
        if (calibrationRef.current.activeView !== currentView) {
          calibrationRef.current.activeView = currentView;
          calibrationRef.current.startedAt = 0;
        }

        if (calibrationRef.current.startedAt === 0) {
          calibrationRef.current.startedAt = now;
        }
        const elapsed = now - calibrationRef.current.startedAt;
        const viewLabel = tierLabel;
        const calibrationDuration =
          captureTier === "upper_front"
            ? UPPER_FRONT_CALIBRATION_MS
            : CALIBRATION_MS;
        if (elapsed < calibrationDuration) {
          const pct = Math.round(
            clamp((elapsed / calibrationDuration) * 100, 0, 100),
          );
          setPill("detecting");
          setFeedback(
            captureTier === "upper_front"
              ? `Getting ready... ${pct}%. Sit tall and stay centered.`
              : `Calibrating ${viewLabel}... ${pct}%`,
          );
          return;
        }

        calibrationRef.current.done[currentView] = true;
        calibrationRef.current.startedAt = 0;
        baselineMetricsRef.current[captureTier] = {
          trunk: tDeg,
          head: hM,
          shoulder: sM,
        };
        setFeedback("Ready.");
        return;
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

      const d = computeDecision(captureTier, effectiveSensitivity, baseline);
      setMetrics({ trunkAngle: d.t ?? 0, headForward: d.h ?? 0, shoulderTilt: d.s ?? 0 });
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
          : d.score ?? 0;
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
      const stablePrompt =
        votedOk
          ? captureTier === "full_front"
            ? "Good posture."
            : "Looking good. Keep your head centered and shoulders level."
          : d.msg;
      if (votedOk) {
        speakFeedback("good", stablePresentation.audio, `good-${currentView}`);
      } else {
        speakFeedback("fix", stablePresentation.audio, stablePresentation.audio);
      }
      if (votedOk) {
        setFeedback(stablePrompt);
      }

      if (d.t != null && d.h != null) {
        const logMsg = votedOk
          ? stablePrompt
          : d.msg;
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
              `good-${currentView}-ml`,
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

  const start = useCallback(async (cameraId = selectedCameraId) => {
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
  }, [ensureLandmarker, loop, refreshCameraDevices, resetBuffers, selectedCameraId]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [feedbacks]);

  useEffect(() => stop, [stop]);

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

    window.speechSynthesis.addEventListener("voiceschanged", handleVoicesChanged);
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

  const getScoreColor = (s: number) => {
    if (s > 80) return "text-emerald-400";
    if (s > 60) return "text-amber-400";
    return "text-rose-400";
  };
  const isLoading = pill === "loading";
  const metricMeta =
    assessmentTier === "upper_front"
      ? {
          trunk: { label: "Head Offset", unit: "norm" },
          head: { label: "Forward Lean", unit: "norm" },
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

  return (
    <div className="min-h-screen bg-[#0a0a0c] text-slate-100 font-sans p-4 md:p-8 flex items-center justify-center">
      <div className="w-full flex flex-col gap-6 h-full lg:h-[85vh]">
        <div className="flex-1 flex min-h-0 gap-4 lg:gap-6">
          <div className="hidden lg:flex h-full min-h-0 w-60 xl:w-64 flex-shrink-0 flex-col gap-4">
            <div className="items-center text-center flex flex-col gap-2">
              <h1 className="text-5xl font-bold tracking-tight  bg-gradient-to-r from-white to-white/60 bg-clip-text text-transparent">
                SukatLikod
              </h1>
              <p className="text-sm text-white/40 font-medium uppercase tracking-[0.2em]">
                AI Posture Assistant
              </p>
            </div>

            <button
              onClick={isActive ? stop : () => void start()}
              disabled={isLoading}
              className={`w-full flex items-center justify-center gap-2 px-5 py-2.5 rounded-full font-semibold transition-all ${isActive ? "bg-rose-500/20 text-rose-400 border border-rose-500/30 hover:bg-rose-500/30" : "bg-white text-black hover:bg-slate-200 shadow-lg"} ${isLoading ? "opacity-50 cursor-not-allowed" : ""}`}
            >
              {isActive ? <VideoOff size={18} /> : <Camera size={18} />}
              {isActive ? "Stop" : "Start Session"}
            </button>

            <div className="mt-auto flex flex-col gap-4">
              <div className="bg-gradient-to-br from-white/10 to-transparent backdrop-blur-md border border-white/10 rounded-2xl p-4 flex flex-col gap-1 hover:bg-white/10 transition-all relative overflow-hidden group">
                <div className="flex items-center justify-between text-white/50 mb-1 z-10">
                  <span className="text-xs font-medium uppercase tracking-wider">
                    Posture Score
                  </span>
                  {score > 70 ? (
                    <CheckCircle2 size={14} className="text-emerald-400" />
                  ) : (
                    <AlertCircle size={14} className="text-white" />
                  )}
                </div>

                <div className="flex items-center justify-between mt-1 z-10">
                  <div className="flex items-baseline gap-1">
                    <span
                      className={`text-3xl font-black tracking-tight ${getScoreColor(score)}`}
                    >
                      {score}
                    </span>
                    <span className="text-xs text-white/40 font-medium">
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
                      className="text-white/10"
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
            <div className="relative flex-1 bg-slate-900 rounded-[2rem] overflow-hidden border border-white/5 shadow-2xl group">
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
                <div className="absolute inset-0 flex flex-col items-center justify-center bg-slate-900/50 backdrop-blur-sm z-10">
                  <div className="w-20 h-20 rounded-full bg-white/5 flex items-center justify-center mb-4">
                    <Camera size={32} className="text-white/20" />
                  </div>
                  <p className="text-white/40 font-medium">
                    Camera Feed Inactive
                  </p>
                </div>
              ) : null}

              <div className="absolute top-4 left-4 right-4 z-20 lg:hidden bg-black/45 backdrop-blur-md border border-white/10 rounded-2xl px-4 py-3 flex items-center justify-end gap-3">
                <div className="flex gap-3">
                  <button
                    onClick={() => setShowSettings((v) => !v)}
                    className="flex items-center justify-center p-2.5 rounded-full transition-all border border-transparent bg-transparent text-black hover:bg-white"
                    title="Toggle Calibration Settings"
                  >
                    <Settings2 size={18} />
                  </button>
                  <button
                    onClick={isActive ? stop : () => void start()}
                    disabled={isLoading}
                    className={`lg:hidden flex items-center gap-2 px-5 py-2.5 rounded-full font-semibold transition-all ${isActive ? "bg-rose-500/20 text-rose-400 border border-rose-500/30 hover:bg-rose-500/30" : "bg-white text-black hover:bg-slate-200 shadow-lg"} ${isLoading ? "opacity-50 cursor-not-allowed" : ""}`}
                  >
                    {isActive ? <VideoOff size={18} /> : <Camera size={18} />}
                    {isActive ? "Stop" : "Start Session"}
                  </button>
                </div>
              </div>

              <div
                className={`absolute inset-0 transition-opacity duration-1000 ${isActive ? "opacity-100" : "opacity-0"}`}
              >
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-48 h-64 border-2 border-white/20 border-dashed rounded-full" />
              </div>

              <div className="absolute left-6 bottom-6 z-20">
                {isActive ? (
                  <>
                    <div className="text-sm font-bold text-white uppercase tracking-wider">
                      Stability {stabilityScore}%
                    </div>
                    <div className="text-[11px] font-semibold text-white/75 uppercase tracking-wider">
                      Tracking {trackingHealth}%
                    </div>
                  </>
                ) : (
                  <>
                    <div className="text-sm font-bold text-white uppercase tracking-wider">
                      Session Standby
                    </div>
                    <div className="text-[11px] font-semibold text-white/60 uppercase tracking-wider">
                      Start a session to view live metrics
                    </div>
                  </>
                )}
              </div>

              <div
                className={`absolute right-4 top-4 bottom-4 z-30 shadow-2xl transition-all duration-200 ease-in-out ${
                  showSessionLog
                    ? "w-72 lg:w-80"
                    : "w-14"
                }`}
              >
                <div
                  className={`h-full bg-black/40 backdrop-blur-xl border border-white/10 rounded-2xl overflow-hidden ${
                    showSessionLog ? "flex flex-col" : "flex items-start justify-center"
                  }`}
                >
                  {showSessionLog ? (
                    <>
                      <div className="px-5 py-4 border-b border-white/10 bg-white/5 flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <Bell size={16} className="text-white/90" />
                          <h3 className="font-bold text-sm text-white uppercase tracking-wider">
                            Session Log
                          </h3>
                        </div>
                        <div className="flex items-center gap-2">
                          <button
                            onClick={() => setShowSessionLog(false)}
                            className="flex items-center justify-center p-2.5 rounded-full transition-all border border-white/30 bg-white/10 text-white/80 hover:bg-white hover:text-black"
                            title="Hide Session Log"
                            aria-label="Hide Session Log"
                          >
                            <PanelRightClose size={16} />
                          </button>
                          <button
                            onClick={() => setShowSettings((v) => !v)}
                            className="flex items-center justify-center p-2.5 rounded-full transition-all border border-white/30 bg-white/20 text-white/80 hover:bg-white  hover:text-black"
                            title="Toggle Calibration Settings"
                          >
                            <Settings2 size={16} />
                          </button>
                        </div>
                      </div>

                      <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-3 [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-track]:bg-transparent [&::-webkit-scrollbar-thumb]:bg-white/20 [&::-webkit-scrollbar-thumb]:rounded-full">
                        {!isActive && (
                          <div className="h-full flex flex-col items-center justify-center text-center opacity-70 space-y-3">
                            <Bell size={24} className="text-white/40" />
                            <span className="text-xs font-medium text-white/80">
                              Monitoring paused.
                              <br />
                              Start session to view feedback.
                            </span>
                          </div>
                        )}
                        {feedbacks.map((f) => (
                          <div
                            key={f.id}
                            className={`p-3 rounded-xl border ${f.bg} backdrop-blur-md transition-all`}
                          >
                            <div className="flex justify-between items-center mb-1">
                              <span className={`text-xs font-bold ${f.color}`}>
                                {f.title}
                              </span>
                              <span className="text-[10px] text-white/50">
                                {f.time}
                              </span>
                            </div>
                            <p className="text-[13px] text-white/90 leading-relaxed">
                              {f.text}
                            </p>
                          </div>
                        ))}
                        <div ref={chatEndRef} />
                      </div>

                      <div className="p-4 bg-white/5 border-t border-white/10">
                        <div className="bg-black/50 border border-white/10 rounded-xl px-3 py-2.5 flex items-center gap-2">
                          <div
                            className={`w-2 h-2 rounded-full ${isActive ? "bg-emerald-500 animate-pulse" : "bg-white/20"}`}
                          />
                          <span className="text-xs text-white/70">
                            {isActive ? feedback : "System standby"}
                          </span>
                        </div>
                      </div>
                    </>
                  ) : (
                    <div className="w-full h-full flex flex-col items-center justify-start py-4 gap-3">
                      <button
                        onClick={() => setShowSessionLog(true)}
                        className="flex items-center justify-center p-2.5 rounded-full transition-all border border-white/30 bg-white/20 text-white/80 hover:bg-white hover:text-black"
                        title="Show Session Log"
                        aria-label="Show Session Log"
                      >
                        <PanelRightOpen size={16} />
                      </button>
                      <div className="flex items-center justify-center">
                        <span
                          className="text-[10px] font-bold uppercase tracking-[0.25em] text-white/65 [writing-mode:vertical-rl] rotate-180"
                        >
                          Session Log
                        </span>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>

            <div
              className={`flex-shrink-0 overflow-hidden transition-[width,opacity] duration-200 ease-in-out ${showSettings ? "w-80 opacity-100" : "w-0 opacity-0"}`}
            >
              <div
                aria-hidden={!showSettings}
                className={`w-80 h-full bg-white/5 backdrop-blur-md border border-white/10 rounded-[2rem] p-6 flex flex-col overflow-y-auto transition-transform duration-200 ease-in-out ${showSettings ? "translate-x-0 pointer-events-auto" : "translate-x-3 pointer-events-none"}`}
              >
                <div className="flex items-center justify-between mb-8">
                  <div className="flex items-center gap-2">
                    <Settings2 size={18} className="text-white/60" />
                    <h3 className="font-bold text-sm uppercase tracking-wider">
                      Calibration
                    </h3>
                  </div>
                  <button
                    onClick={() => setShowSettings(false)}
                    className="w-9 h-9 rounded-lg  text-white/70 hover:text-white hover:bg-white/10 transition-colors flex items-center justify-center"
                    title="Close calibration"
                    aria-label="Close calibration"
                  >
                    <X size={20} />
                  </button>
                </div>

                <div className="space-y-8">
                  {(
                    [
                      {
                        key: "trunkAngle",
                        label: "Trunk Angle Threshold",
                        unit: "deg",
                        max: 45,
                        step: 1,
                      },
                      {
                        key: "headDistance",
                        label: "Head Distance Limit",
                        unit: "m",
                        max: 0.5,
                        step: 0.01,
                      },
                      {
                        key: "shoulderTilt",
                        label: "Shoulder Tilt Sensitivity",
                        unit: "m",
                        max: 0.2,
                        step: 0.01,
                      },
                    ] as const
                  ).map((setting) => (
                    <div key={setting.key} className="space-y-3">
                      <div className="flex justify-between items-center px-1">
                        <label className="text-xs font-semibold text-white/60">
                          {setting.label}
                        </label>
                        <span className="text-xs font-bold text-white tabular-nums">
                          {sensitivity[setting.key].toFixed(
                            setting.step < 1 ? 2 : 0,
                          )}{" "}
                          <span className="text-[10px] text-white/30 ml-0.5">
                            {setting.unit}
                          </span>
                        </span>
                      </div>
                      <div className="relative h-6 flex items-center">
                        <input
                          type="range"
                          min={setting.key === "trunkAngle" ? 5 : 0.01}
                          max={setting.max}
                          step={setting.step}
                          value={sensitivity[setting.key]}
                          onChange={(e) =>
                            setSensitivity((s) => ({
                              ...s,
                              [setting.key]: Number(e.target.value),
                            }))
                          }
                          className="w-full h-1 bg-white/10 rounded-full appearance-none cursor-pointer accent-white"
                        />
                      </div>
                    </div>
                  ))}
                </div>

                <div className="mt-8 space-y-3">
                  <div className="flex justify-between items-center px-1">
                    <label className="text-xs font-semibold text-white/60">
                      Camera Source
                    </label>
                    <span className="text-[10px] font-bold text-white/35 uppercase tracking-wider">
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
                    className="w-full rounded-xl border border-white/15 bg-white/5 px-3 py-2.5 text-sm font-semibold text-white outline-none transition-colors hover:bg-white/10"
                  >
                    {cameraDevices.length === 0 ? (
                      <option value="">No camera detected yet</option>
                    ) : null}
                    {cameraDevices.map((camera) => (
                      <option
                        key={camera.id}
                        value={camera.id}
                        className="bg-slate-900 text-white"
                      >
                        {camera.label}
                      </option>
                    ))}
                  </select>
                  <p className="text-[11px] leading-relaxed text-white/40">
                    If labels are blank, allow camera access first, then reopen
                    this panel or start a session.
                  </p>
                </div>

                <div className="mt-8 space-y-3">
                  <div className="flex justify-between items-center px-1">
                    <label className="text-xs font-semibold text-white/60">
                      Debug Tuning Panel
                    </label>
                    <button
                      type="button"
                      onClick={() => setShowDebugPanel((value) => !value)}
                      className={`rounded-full border px-3 py-1 text-[11px] font-semibold transition-colors ${
                        showDebugPanel
                          ? "border-white/40 bg-white text-black"
                          : "border-white/15 bg-white/5 text-white/70 hover:bg-white/10 hover:text-white"
                      }`}
                    >
                      {showDebugPanel ? "Visible" : "Hidden"}
                    </button>
                  </div>
                  <p className="text-[11px] leading-relaxed text-white/40">
                    Leave this on while capturing screenshots for neutral, mild
                    slouch, and clear slouch.
                  </p>
                </div>

                {showDebugPanel ? (
                  <div className="mt-8 space-y-3">
                    <div className="flex justify-between items-center px-1">
                      <label className="text-xs font-semibold text-white/60">
                        Live Tuning Values
                      </label>
                      <span className="text-[10px] font-bold text-white/35 uppercase tracking-wider">
                        {assessmentTier ?? "idle"}
                      </span>
                    </div>
                    <div className="grid grid-cols-2 gap-2">
                      {[
                        ["Chin Offset", debugMetrics.chinCenterOffset],
                        ["Chin Lean", debugMetrics.chinForwardLean],
                        ["Nose Offset", debugMetrics.noseCenterOffset],
                        ["Mouth Tilt", debugMetrics.mouthLineTilt],
                        ["Eye/Ear Tilt", debugMetrics.eyeOrEarTilt],
                        ["Upper Lean", debugMetrics.upperForwardLean],
                        ["Shoulder Level", debugMetrics.upperShoulderTilt],
                      ].map(([label, value]) => (
                        <div
                          key={label}
                          className="rounded-xl border border-white/10 bg-black/20 px-3 py-2"
                        >
                          <div className="text-[10px] font-semibold uppercase tracking-wider text-white/45">
                            {label}
                          </div>
                          <div className="mt-1 text-sm font-bold tabular-nums text-white">
                            {Number(value).toFixed(3)}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : null}

                <div className="mt-8 space-y-3">
                  <div className="flex justify-between items-center px-1">
                    <label className="text-xs font-semibold text-white/60">
                      Audio Feedback
                    </label>
                    <span className="text-[10px] font-bold text-white/35 uppercase tracking-wider">
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
                            ? "border-white/40 bg-white text-black"
                            : "border-white/15 bg-white/5 text-white/70 hover:bg-white/10 hover:text-white"
                        }`}
                      >
                        {mode === "off" ? "Off" : "Voice"}
                      </button>
                    ))}
                  </div>
                  <div className="flex justify-between items-center px-1 text-[11px] text-white/45">
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
                    className="w-full rounded-xl border border-white/15 bg-white/5 hover:bg-white/10 text-white/80 hover:text-white text-sm font-semibold py-2.5 transition-colors"
                  >
                    Test Voice
                  </button>
                  <p className="text-[11px] leading-relaxed text-white/40">
                    Voice prompts play only on stable posture changes and are
                    suppressed during calibration or weak tracking.
                  </p>
                </div>

                <div className="mt-6">
                  <button
                    onClick={() => setSensitivity(DEFAULT_SENSITIVITY)}
                    className="w-full rounded-xl border border-white/15 bg-white/5 hover:bg-white/10 text-white/80 hover:text-white text-sm font-semibold py-2.5 transition-colors"
                  >
                    Reset to Standard
                  </button>
                </div>

                <div className="mt-auto pt-6">
                  <div className="bg-white/5 border border-white/5 p-4 rounded-2xl">
                    <p className="text-[10px] leading-relaxed text-white/40 italic">
                      Note: Higher sensitivity values increase the threshold for
                      warnings. Adjust based on your ergonomic workstation
                      setup.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-2 lg:hidden gap-4 flex-shrink-0">
          <div className="bg-gradient-to-br from-white/10 to-transparent backdrop-blur-md border border-white/10 rounded-2xl p-4 flex flex-col gap-1 hover:bg-white/10 transition-all relative overflow-hidden group">
            <div className="flex items-center justify-between text-white/50 z-10">
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
                <span className="text-xs text-white/40 font-medium">/ 100</span>
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
                  className="text-white/10"
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
    </div>
  );
}

function MetricCard({
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

  return (
    <div className="bg-gradient-to-br from-white/10 to-transparent backdrop-blur-md border border-white/10 rounded-2xl p-4 flex flex-col gap-1 hover:bg-white/10 transition-all relative overflow-hidden group">
      <div className="flex items-center justify-between text-white/50 mb-1 z-10">
        <span className="text-xs font-medium uppercase tracking-wider">
          {label}
        </span>
        <Icon size={14} />
      </div>
      <div className="flex items-baseline gap-1 z-10">
        <span
          className={`text-2xl font-bold tracking-tight ${colorClass || "text-white"}`}
        >
          {value}
        </span>
        <span className="text-xs text-white/40">{unit}</span>
      </div>
    </div>
  );
}
