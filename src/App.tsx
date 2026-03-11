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
} from "lucide-react";
import {
  FilesetResolver,
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

type Pill = "idle" | "loading" | "detecting" | "good" | "fix" | "error";
type Point3 = { x: number; y: number; z: number };
type FeedbackType = "info" | "warning" | "critical" | "success";
type OrientationKind =
  | "front"
  | "back"
  | "side_left"
  | "side_right"
  | "unknown";
type ViewCalibration = "front" | "side" | "back";

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

type SilhouetteMetrics = {
  neckForwardContour: number;
  upperBackCurvature: number;
  torsoOutlineAngle: number;
  silhouetteStability: number;
};

type AudioMode = "off" | "voice";

type MlPrediction = {
  label: "proper" | "needs_correction" | string;
  confidence: number;
  probabilities: Record<string, number>;
  feedback: string;
};

const WINDOW = 30;
const EMA_ALPHA = 0.25;
const VIS_THRESHOLD = 0.35;
const DRAW_VIS_THRESHOLD = 0.12;
const HOLD_STILL_MS = 1400;
const CALIBRATION_MS = 2500;
const PREDICTION_VOTE_WINDOW = 10;
const AUDIO_COOLDOWN_MS = 9000;
const DEFAULT_SENSITIVITY: Sensitivity = {
  trunkAngle: 18,
  headDistance: 0.08,
  shoulderTilt: 0.05,
};
const DEFAULT_SILHOUETTE_METRICS: SilhouetteMetrics = {
  neckForwardContour: 0,
  upperBackCurvature: 0,
  torsoOutlineAngle: 0,
  silhouetteStability: 0,
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

function headForwardSideNorm(nose: Point3, midShoulder: Point3) {
  return Math.abs((nose?.x ?? 0) - (midShoulder?.x ?? 0));
}

function headForwardSideSignedNorm(nose: Point3, midShoulder: Point3) {
  return (nose?.x ?? 0) - (midShoulder?.x ?? 0);
}

function shoulderTiltM(ls: Point3, rs: Point3) {
  return Math.abs((ls?.y ?? 0) - (rs?.y ?? 0));
}

function shoulderTiltSignedM(ls: Point3, rs: Point3) {
  return (ls?.y ?? 0) - (rs?.y ?? 0);
}

function planarDistance(a: Point3, b: Point3) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function torsoOutlineAngleDeg(shoulder: Point3, hip: Point3) {
  return Math.abs((Math.atan2(shoulder.x - hip.x, hip.y - shoulder.y) * 180) / Math.PI);
}

function neckForwardContourNorm(nose: Point3, shoulder: Point3, hip: Point3) {
  const torsoLen = Math.max(planarDistance(shoulder, hip), 1e-3);
  return Math.abs(nose.x - shoulder.x) / torsoLen;
}

function pointLineDistanceNorm(point: Point3, lineStart: Point3, lineEnd: Point3) {
  const dx = lineEnd.x - lineStart.x;
  const dy = lineEnd.y - lineStart.y;
  const denom = Math.max(Math.hypot(dx, dy), 1e-3);
  const areaTwice = Math.abs(
    dx * (lineStart.y - point.y) - (lineStart.x - point.x) * dy,
  );
  return areaTwice / denom / denom;
}

function upperBackCurvatureNorm(ear: Point3, shoulder: Point3, hip: Point3) {
  return pointLineDistanceNorm(shoulder, ear, hip);
}

function silhouetteStabilityScore(
  contourValues: number[],
  curvatureValues: number[],
  outlineValues: number[],
) {
  const contourVar = variance(contourValues) / 0.0025;
  const curvatureVar = variance(curvatureValues) / 0.0025;
  const outlineVar = variance(outlineValues) / 64;
  return clamp(1 - (contourVar + curvatureVar + outlineVar) / 3, 0, 1);
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
  if (!lsW || !rsW || !lsN || !rsN) return { kind: "unknown", label: "Unknown" };

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

export default function App() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const chatEndRef = useRef<HTMLDivElement | null>(null);

  const poseRef = useRef<PoseLandmarker | null>(null);
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
  const loadedModelPathRef = useRef<string | null>(null);
  const landmarkerLoadPromiseRef = useRef<Promise<void> | null>(null);
  const holdStillStartRef = useRef<number>(0);
  const calibrationRef = useRef<{
    activeView: ViewCalibration | null;
    startedAt: number;
    done: Record<ViewCalibration, boolean>;
  }>({
    activeView: null,
    startedAt: 0,
    done: { front: false, side: false, back: false },
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
  const [silhouetteMetrics, setSilhouetteMetrics] = useState<SilhouetteMetrics>(
    DEFAULT_SILHOUETTE_METRICS,
  );
  const [audioMode, setAudioMode] = useState<AudioMode>("voice");

  const [sensitivity, setSensitivity] = useState<Sensitivity>(
    DEFAULT_SENSITIVITY,
  );
  const [stabilityScore, setStabilityScore] = useState(0);
  const [trackingHealth, setTrackingHealth] = useState(0);
  const overlayDetail: "detailed" = "detailed";

  const modelPath = useMemo(() => "/models/pose_landmarker_lite.task", []);
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
    calibrationRef.current = {
      activeView: null,
      startedAt: 0,
      done: { front: false, side: false, back: false },
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
    setSilhouetteMetrics(DEFAULT_SILHOUETTE_METRICS);
    setStabilityScore(0);
    setTrackingHealth(0);
    resetBuffers();
  }, [resetBuffers]);

  const ensureLandmarker = useCallback(async () => {
    if (poseRef.current && loadedModelPathRef.current === modelPath) return;
    if (landmarkerLoadPromiseRef.current) return landmarkerLoadPromiseRef.current;
    poseRef.current = null;
    loadedModelPathRef.current = null;

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
    })();

    try {
      await landmarkerLoadPromiseRef.current;
    } catch (error) {
      poseRef.current = null;
      loadedModelPathRef.current = null;
      throw error;
    } finally {
      landmarkerLoadPromiseRef.current = null;
    }
  }, [modelPath]);

  const computeDecision = useCallback((thresholds?: Sensitivity) => {
    const t = avg(buffersRef.current.trunk);
    const h = avg(buffersRef.current.head);
    const s = avg(buffersRef.current.shoulder);

    if (t == null || h == null || s == null) {
      return {
        ok: false,
        score: null as number | null,
        msg: "Hold still...",
        t,
        h,
        s,
      };
    }

    const tThr = thresholds?.trunkAngle ?? sensitivity.trunkAngle;
    const hThr = thresholds?.headDistance ?? sensitivity.headDistance;
    const sThr = thresholds?.shoulderTilt ?? sensitivity.shoulderTilt;
    const tRatio = t / tThr;
    const hRatio = h / hThr;
    const sRatio = s / sThr;
    const worst = Math.max(tRatio, hRatio, sRatio);

    const nextScore = Math.round(clamp(100 - (worst - 1) * 45, 0, 100));
    const ok = worst <= 1;

    let msg = "Good posture - keep it.";
    if (!ok) {
      if (worst === tRatio) msg = "Straighten your back (reduce trunk lean).";
      else if (worst === hRatio)
        msg = "Bring your head back (avoid head-forward).";
      else msg = "Level your shoulders.";
    }

    return { ok, score: nextScore, msg, t, h, s };
  }, [
    sensitivity.headDistance,
    sensitivity.shoulderTilt,
    sensitivity.trunkAngle,
  ]);

  const pushFeedback = useCallback(
    (
      scoreValue: number,
      msg: string,
      t: number,
      h: number,
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

      let type: FeedbackType = "info";
      let title = "Status Stable";
      let text = `Trunk angle stable at ${t.toFixed(1)}deg.`;
      let color = "text-cyan-400";
      let bg = "bg-cyan-500/10 border-cyan-500/20";

      if (scoreValue < 60) {
        type = "critical";
        title = "Posture Alert";
        color = "text-rose-400";
        bg = "bg-rose-500/10 border-rose-500/20";
        text = msg;
      } else if (h > headThreshold) {
        type = "warning";
        title = "Head Forward Warning";
        color = "text-amber-400";
        bg = "bg-amber-500/10 border-amber-500/20";
        text = `Head distance exceeded threshold (${h.toFixed(2)}m).`;
      } else if (scoreValue > 85) {
        type = "success";
        title = "Excellent Form";
        color = "text-emerald-400";
        bg = "bg-emerald-500/10 border-emerald-500/20";
        text = "Optimal posture maintained. Great job!";
      }

      setFeedbacks((prev) =>
        [...prev, { id: now, type, title, color, bg, text, time }].slice(-50),
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
      if (typeof window === "undefined" || !("speechSynthesis" in window)) return;

      const now = Date.now();
      const last = lastAudioEventRef.current;
      const stateChanged = lastAnnouncedStateRef.current !== nextState;
      if (!stateChanged && last.key === eventKey && now - last.at < AUDIO_COOLDOWN_MS) {
        return;
      }

      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(message);
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
    (result: PoseLandmarkerResult) => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      ctx.save();
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const landmarks = result.landmarks?.[0];
      const world = result.worldLandmarks?.[0];
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
                ]
              : [
                  IDX.NOSE,
                  IDX.L_EAR,
                  IDX.R_EAR,
                  IDX.L_SHOULDER,
                  IDX.R_SHOULDER,
                  IDX.L_HIP,
                  IDX.R_HIP,
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
                ]
              : [
                  [IDX.NOSE, IDX.L_EAR],
                  [IDX.NOSE, IDX.R_EAR],
                  [IDX.L_SHOULDER, IDX.R_SHOULDER],
                  [IDX.L_SHOULDER, IDX.L_HIP],
                  [IDX.R_SHOULDER, IDX.R_HIP],
                  [IDX.L_HIP, IDX.R_HIP],
                ];
        } else if (orient.kind === "side_left" || orient.kind === "side_right") {
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
              : [IDX.NOSE, sidePrimary.ear, sidePrimary.shoulder, sidePrimary.hip];
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
    (result: PoseLandmarkerResult) => {
      const world = result.worldLandmarks?.[0];
      const norm = result.landmarks?.[0];
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
        !lh ||
        !rh ||
        !noseN ||
        !lsN ||
        !rsN ||
        !lhN ||
        !rhN ||
        !lEarN ||
        !rEarN ||
        !leN ||
        !reN
      ) {
        return;
      }

      const health = Math.round(
        avgVisibility([noseN, lsN, rsN, lhN, rhN, lEarN, rEarN, leN, reN]) * 100,
      );
      setTrackingHealth(health);

      if (!visOk(noseN) || !visOk(lsN) || !visOk(rsN) || health < 45) {
        setPill("detecting");
        setFeedback("Low landmark confidence. Improve lighting and hold still.");
        return;
      }

      const orient = detectOrientation(world as Point3[], norm);
      const isSide =
        orient.kind === "side_left" || orient.kind === "side_right";
      const isBack = orient.kind === "back";
      const dominantSide = dominantSideFromNorm(norm);
      const sidePrimaryNorm =
        dominantSide === "left"
          ? {
              shoulder: lsN as Point3,
              hip: lhN as Point3,
              ear: lEarN as Point3,
              eye: leN as Point3,
            }
          : {
              shoulder: rsN as Point3,
              hip: rhN as Point3,
              ear: rEarN as Point3,
              eye: reN as Point3,
            };

      if (orient.kind === "unknown") {
        holdStillStartRef.current = 0;
        lastSmoothedRef.current = null;
        setPill("detecting");
        setScore(0);
        setMetrics({ trunkAngle: 0, headForward: 0, shoulderTilt: 0 });
        setSignedMetrics({ trunkAngle: 0, headForward: 0, shoulderTilt: 0 });
        setSilhouetteMetrics(DEFAULT_SILHOUETTE_METRICS);
        setStabilityScore(0);
        setFeedback("Adjust your position so shoulders and hips are clearly visible.");
        return;
      }

      const frontVisibleCount = [
        visOk(noseN),
        visOk(lsN),
        visOk(rsN),
        visOk(lhN),
        visOk(rhN),
        visOk(lEarN),
        visOk(rEarN),
      ].filter(Boolean).length;
      if (!isSide && frontVisibleCount < 5) {
        setPill("detecting");
        setFeedback("Partial front view detected. Keep head, shoulders, and hips visible.");
        return;
      }

      const midShoulder: Point3 = {
        x: (ls.x + rs.x) / 2,
        y: (ls.y + rs.y) / 2,
        z: (ls.z + rs.z) / 2,
      };
      const midHip: Point3 = {
        x: (lh.x + rh.x) / 2,
        y: (lh.y + rh.y) / 2,
        z: (lh.z + rh.z) / 2,
      };
      const midShoulderNorm: Point3 = {
        x: (lsN.x + rsN.x) / 2,
        y: (lsN.y + rsN.y) / 2,
        z: (lsN.z + rsN.z) / 2,
      };

      const tRaw = trunkAngleDeg(midShoulder, midHip);
      let hRaw = headForwardM(nose as Point3, midShoulder);
      if (isSide) {
        const sideVisibleCount = [
          visOk(noseN),
          visOk(sidePrimaryNorm.eye as { x: number; y: number; z: number; visibility?: number }),
          visOk(sidePrimaryNorm.ear as { x: number; y: number; z: number; visibility?: number }),
          visOk(sidePrimaryNorm.shoulder as { x: number; y: number; z: number; visibility?: number }),
          visOk(sidePrimaryNorm.hip as { x: number; y: number; z: number; visibility?: number }),
        ].filter(Boolean).length;
        if (sideVisibleCount < 4) {
          setPill("detecting");
          setFeedback("Partial side view detected. Keep head, shoulder, and hip visible.");
          return;
        }
        const sideHip = sidePrimaryNorm.hip;
        if (!visOk(sideHip as { x: number; y: number; z: number; visibility?: number })) {
          setPill("detecting");
          setFeedback("Keep shoulder and hip visible for side-view analysis.");
          return;
        }
        const neckNorm: Point3 = {
          x: (lsN.x + rsN.x) / 2,
          y: (lsN.y + rsN.y) / 2,
          z: 0,
        };
        const torsoLen = Math.hypot(
          sidePrimaryNorm.shoulder.x - sideHip.x,
          sidePrimaryNorm.shoulder.y - sideHip.y,
        );
        hRaw = headForwardSideNorm(noseN as Point3, neckNorm) / Math.max(torsoLen, 1e-3);
      } else if (isBack) {
        // Back view has no reliable forward-head estimate from face landmarks.
        hRaw = 0;
      }
      const sRaw = shoulderTiltM(ls as Point3, rs as Point3);
      const contourRaw = isSide
        ? neckForwardContourNorm(
            noseN as Point3,
            sidePrimaryNorm.shoulder,
            sidePrimaryNorm.hip,
          )
        : 0;
      const curvatureRaw = isSide
        ? upperBackCurvatureNorm(
            sidePrimaryNorm.ear,
            sidePrimaryNorm.shoulder,
            sidePrimaryNorm.hip,
          )
        : 0;
      const outlineRaw = isSide
        ? torsoOutlineAngleDeg(sidePrimaryNorm.shoulder, sidePrimaryNorm.hip)
        : 0;

      let tDeg = ema(emaRef.current.trunk, tRaw);
      let hM = ema(emaRef.current.head, hRaw);
      let sM = ema(emaRef.current.shoulder, sRaw);
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
      const tSigned = trunkAngleSignedDeg(midShoulder, midHip);
      const hSigned = isSide
        ? headForwardSideSignedNorm(noseN as Point3, midShoulderNorm)
        : headForwardSignedM(nose as Point3, midShoulder);
      const sSigned = shoulderTiltSignedM(ls as Point3, rs as Point3);

      const now = performance.now();
      const currentView: ViewCalibration = isSide
        ? "side"
        : isBack
          ? "back"
          : "front";
      if (!calibrationRef.current.done[currentView]) {
        if (calibrationRef.current.activeView !== currentView) {
          calibrationRef.current.activeView = currentView;
          calibrationRef.current.startedAt = 0;
        }

        if (calibrationRef.current.startedAt === 0) {
          calibrationRef.current.startedAt = now;
        }
        const elapsed = now - calibrationRef.current.startedAt;
        const viewLabel =
          currentView === "front"
            ? "Front view"
            : currentView === "side"
              ? "Side view"
              : "Back view";
        if (elapsed < CALIBRATION_MS) {
          const pct = Math.round(clamp((elapsed / CALIBRATION_MS) * 100, 0, 100));
          setPill("detecting");
          setFeedback(`Calibrating ${viewLabel}... ${pct}%`);
          return;
        }

        calibrationRef.current.done[currentView] = true;
        calibrationRef.current.startedAt = 0;
        setFeedback(`${viewLabel} calibrated. Monitoring posture.`);
        return;
      }

      pushLimited(buffersRef.current.trunk, tDeg);
      pushLimited(buffersRef.current.head, hM);
      pushLimited(buffersRef.current.shoulder, sM);
      pushLimited(buffersRef.current.contour, contour);
      pushLimited(buffersRef.current.curvature, curvature);
      pushLimited(buffersRef.current.outline, outline);
      const trunkVar = variance(buffersRef.current.trunk);
      const silhouetteStability = isSide
        ? silhouetteStabilityScore(
            buffersRef.current.contour,
            buffersRef.current.curvature,
            buffersRef.current.outline,
          )
        : 0;
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

      const effectiveSensitivity: Sensitivity = isSide
        ? {
            trunkAngle: sensitivity.trunkAngle * 0.82,
            headDistance: sensitivity.headDistance * 1.2,
            shoulderTilt: sensitivity.shoulderTilt * 1.1,
          }
        : isBack
          ? {
              trunkAngle: sensitivity.trunkAngle,
              headDistance: 999, // ignored in back-view local decision
              shoulderTilt: sensitivity.shoulderTilt * 0.9,
            }
          : sensitivity;

      const d = computeDecision(effectiveSensitivity);
      setMetrics({ trunkAngle: tDeg, headForward: hM, shoulderTilt: sM });
      setSignedMetrics({
        trunkAngle: tSigned,
        headForward: hSigned,
        shoulderTilt: sSigned,
      });
      setSilhouetteMetrics({
        neckForwardContour: isSide ? contour : 0,
        upperBackCurvature: isSide ? curvature : 0,
        torsoOutlineAngle: isSide ? outline : 0,
        silhouetteStability,
      });

      if (!holdReady) {
        setPill("detecting");
        setFeedback("Hold still for stable reading...");
        return;
      }

      const nextScore = d.score ?? 0;
      const votedOk = applyPredictionVote(d.ok);
      setScore(nextScore);
      setFeedback(d.msg);
      setPill(votedOk ? "good" : "fix");
      const stablePrompt =
        isSide && votedOk
          ? "Side view stable. Keep head aligned with your torso."
          : isBack && votedOk
            ? "Back view stable. Keep shoulders level and back upright."
            : d.msg;
      if (votedOk) {
        speakFeedback("good", "Good posture.", "good");
      } else {
        speakFeedback("fix", stablePrompt, stablePrompt);
      }
      if (isSide && votedOk) {
        setFeedback("Side view stable. Keep head aligned with your torso.");
      } else if (isBack && votedOk) {
        setFeedback("Back view stable. Keep shoulders level and back upright.");
      }

      if (d.t != null && d.h != null) {
        const logMsg =
          isSide && votedOk
            ? "Side view stable. Keep head aligned with your torso."
            : isBack && votedOk
              ? "Back view stable. Keep shoulders level and back upright."
            : d.msg;
        pushFeedback(nextScore, logMsg, d.t, d.h, effectiveSensitivity.headDistance);
      }

      if (!isBack && d.t != null && d.h != null && d.s != null) {
        const dT = d.t;
        const dH = d.h;
        const dS = d.s;
        void inferMl({
          trunk_angle: dT,
          head_forward: dH,
          shoulder_tilt: dS,
          trunk_variance: trunkVar,
          neck_forward_contour: isSide ? contour : 0,
          upper_back_curvature: isSide ? curvature : 0,
          torso_outline_angle: isSide ? outline : 0,
          silhouette_stability: silhouetteStability,
        }).then((pred) => {
          if (!pred) return;

          const mlOk = pred.label === "proper";
          const votedMlOk = applyPredictionVote(mlOk);
          const mlScore = Math.round(clamp(pred.confidence * 100, 0, 100));
          const mlMsg = pred.feedback || (mlOk ? "Good posture - keep it." : "Needs correction.");
          const finalOk = d.ok ? votedMlOk : false;
          const finalScore = d.ok ? Math.min(nextScore, mlScore) : nextScore;
          const finalMsg = d.ok ? mlMsg : d.msg;

          setScore(finalScore);
          setFeedback(finalMsg);
          setPill(finalOk ? "good" : "fix");
          if (finalOk) {
            speakFeedback("good", "Good posture.", "good");
          } else {
            speakFeedback("fix", finalMsg, finalMsg);
          }
          pushFeedback(finalScore, finalMsg, dT, dH);
        });
      }
    },
    [applyPredictionVote, computeDecision, inferMl, pushFeedback, speakFeedback],
  );

  const loop = useCallback(
    function tick(): void {
      const pose = poseRef.current;
      const video = videoRef.current;
      if (!pose || !video) return;

      if (video.currentTime !== lastVideoTimeRef.current) {
        const now = performance.now();
        pose.detectForVideo(video, now, (result) => {
          draw(result);
          process(result);
        });
        lastVideoTimeRef.current = video.currentTime;
      }

      rafRef.current = requestAnimationFrame(tick);
    },
    [draw, process],
  );

  const start = useCallback(async () => {
    try {
      setPill("loading");
      setFeedback("Loading pose model...");
      await ensureLandmarker();

      setFeedback("Requesting webcam...");
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
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
      setIsActive(true);
      setPill("detecting");
      setFeedback("Stand centered. Keep shoulders + hips visible.");
      rafRef.current = requestAnimationFrame(loop);
    } catch (error) {
      console.error(error);
      setIsActive(false);
      setPill("error");
      setFeedback("Failed to start. Check camera permission and reload.");
    }
  }, [ensureLandmarker, loop, resetBuffers]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [feedbacks]);

  useEffect(() => stop, [stop]);

  useEffect(() => {
    if (audioMode !== "off") return;
    if (typeof window !== "undefined" && "speechSynthesis" in window) {
      window.speechSynthesis.cancel();
    }
  }, [audioMode]);

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
              onClick={isActive ? stop : start}
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
                label="Trunk Angle"
                value={metrics.trunkAngle.toFixed(1)}
                unit="deg"
                icon={Activity}
                variant="trunk"
                rawValue={metrics.trunkAngle}
                signedValue={signedMetrics.trunkAngle}
                threshold={sensitivity.trunkAngle}
                progress={metricQuality(
                  metrics.trunkAngle,
                  sensitivity.trunkAngle,
                )}
              />
              <MetricCard
                label="Head Forward"
                value={metrics.headForward.toFixed(2)}
                unit="m"
                icon={ChevronRight}
                variant="head"
                rawValue={metrics.headForward}
                signedValue={signedMetrics.headForward}
                threshold={sensitivity.headDistance}
                progress={metricQuality(
                  metrics.headForward,
                  sensitivity.headDistance,
                )}
              />
              <MetricCard
                label="Shoulder Tilt"
                value={metrics.shoulderTilt.toFixed(2)}
                unit="m"
                icon={Maximize2}
                variant="shoulder"
                rawValue={metrics.shoulderTilt}
                signedValue={signedMetrics.shoulderTilt}
                threshold={sensitivity.shoulderTilt}
                progress={metricQuality(
                  metrics.shoulderTilt,
                  sensitivity.shoulderTilt,
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
                    onClick={isActive ? stop : start}
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
                <div className="text-sm font-bold text-white uppercase tracking-wider">
                  Stability {stabilityScore}%
                </div>
                <div className="text-[11px] font-semibold text-white/75 uppercase tracking-wider">
                  Tracking {trackingHealth}%
                </div>
                <div className="text-[11px] font-semibold text-white/55 uppercase tracking-wider">
                  Silhouette {Math.round(silhouetteMetrics.silhouetteStability * 100)}%
                </div>
              </div>

              <div className="absolute right-4 top-4 bottom-4 w-72 lg:w-80 bg-black/40 backdrop-blur-xl border border-white/10 rounded-2xl flex flex-col overflow-hidden z-30 shadow-2xl">
                <div className="px-5 py-4 border-b border-white/10 bg-white/5 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Bell size={16} className="text-white/90" />
                    <h3 className="font-bold text-sm text-white uppercase tracking-wider">
                      Session Log
                    </h3>
                  </div>
                  <div className="flex items-center gap-2">
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
                      Audio Feedback
                    </label>
                    <span className="text-[10px] font-bold text-white/35 uppercase tracking-wider">
                      9s cooldown
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
                  <p className="text-[11px] leading-relaxed text-white/40">
                    Voice prompts play only on stable posture changes and are suppressed during calibration or weak tracking.
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
            label="Trunk Angle"
            value={metrics.trunkAngle.toFixed(1)}
            unit="deg"
            icon={Activity}
            variant="trunk"
            rawValue={metrics.trunkAngle}
            signedValue={signedMetrics.trunkAngle}
            threshold={sensitivity.trunkAngle}
            progress={metricQuality(metrics.trunkAngle, sensitivity.trunkAngle)}
          />
          <MetricCard
            label="Head Forward"
            value={metrics.headForward.toFixed(2)}
            unit="m"
            icon={ChevronRight}
            variant="head"
            rawValue={metrics.headForward}
            signedValue={signedMetrics.headForward}
            threshold={sensitivity.headDistance}
            progress={metricQuality(
              metrics.headForward,
              sensitivity.headDistance,
            )}
          />
          <MetricCard
            label="Shoulder Tilt"
            value={metrics.shoulderTilt.toFixed(2)}
            unit="m"
            icon={Maximize2}
            variant="shoulder"
            rawValue={metrics.shoulderTilt}
            signedValue={signedMetrics.shoulderTilt}
            threshold={sensitivity.shoulderTilt}
            progress={metricQuality(
              metrics.shoulderTilt,
              sensitivity.shoulderTilt,
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
