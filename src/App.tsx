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
  DrawingUtils,
  FilesetResolver,
  PoseLandmarker,
  type PoseLandmarkerResult,
} from "@mediapipe/tasks-vision";

const IDX = {
  NOSE: 0,
  L_SHOULDER: 11,
  R_SHOULDER: 12,
  L_HIP: 23,
  R_HIP: 24,
} as const;

type Pill = "idle" | "loading" | "detecting" | "good" | "fix" | "error";
type Point3 = { x: number; y: number; z: number };
type FeedbackType = "info" | "warning" | "critical" | "success";

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

const WINDOW = 30;

function clamp(n: number, a: number, b: number) {
  return Math.max(a, Math.min(b, n));
}

function avg(arr: number[]) {
  if (arr.length === 0) return null;
  return arr.reduce((s, x) => s + x, 0) / arr.length;
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

function metricQuality(value: number, threshold: number) {
  if (threshold <= 0) return 0;
  const ratio = value / threshold;
  return Math.round(clamp(100 - (ratio - 1) * 70, 0, 100));
}

export default function App() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const chatEndRef = useRef<HTMLDivElement | null>(null);

  const poseRef = useRef<PoseLandmarker | null>(null);
  const drawRef = useRef<DrawingUtils | null>(null);
  const rafRef = useRef<number | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const lastVideoTimeRef = useRef<number>(-1);
  const lastFeedbackRef = useRef<string>("");

  const buffersRef = useRef<{
    trunk: number[];
    head: number[];
    shoulder: number[];
  }>({
    trunk: [],
    head: [],
    shoulder: [],
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

  const [sensitivity, setSensitivity] = useState<Sensitivity>({
    trunkAngle: 18,
    headDistance: 0.08,
    shoulderTilt: 0.05,
  });

  const modelPath = useMemo(() => "/models/pose_landmarker_lite.task", []);

  const resetBuffers = useCallback(() => {
    buffersRef.current.trunk = [];
    buffersRef.current.head = [];
    buffersRef.current.shoulder = [];
    lastVideoTimeRef.current = -1;
    lastFeedbackRef.current = "";
  }, []);

  const stop = useCallback(() => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }

    if (videoRef.current) videoRef.current.srcObject = null;

    const c = canvasRef.current;
    if (c) c.getContext("2d")?.clearRect(0, 0, c.width, c.height);

    setIsActive(false);
    setPill("idle");
    setFeedback("Press Start Session to begin.");
    setFeedbacks([]);
    setScore(0);
    setMetrics({ trunkAngle: 0, headForward: 0, shoulderTilt: 0 });
    setSignedMetrics({ trunkAngle: 0, headForward: 0, shoulderTilt: 0 });
    resetBuffers();
  }, [resetBuffers]);

  const ensureLandmarker = useCallback(async () => {
    if (poseRef.current) return;

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
  }, [modelPath]);

  const computeDecision = useCallback(() => {
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

    const tRatio = t / sensitivity.trunkAngle;
    const hRatio = h / sensitivity.headDistance;
    const sRatio = s / sensitivity.shoulderTilt;
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
    (scoreValue: number, msg: string, t: number, h: number) => {
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
      } else if (h > sensitivity.headDistance) {
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

  const draw = useCallback((result: PoseLandmarkerResult) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    if (!drawRef.current) drawRef.current = new DrawingUtils(ctx);

    ctx.save();
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const landmarks = result.landmarks?.[0];
    if (landmarks) {
      drawRef.current.drawConnectors(
        landmarks,
        PoseLandmarker.POSE_CONNECTIONS,
        {
          color: "#22d3ee",
          lineWidth: 2,
        },
      );
      drawRef.current.drawLandmarks(landmarks, {
        radius: 3,
        color: "#e2e8f0",
        lineWidth: 1,
      });
    }

    ctx.restore();
  }, []);

  const process = useCallback(
    (result: PoseLandmarkerResult) => {
      const world = result.worldLandmarks?.[0];
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
      if (!nose || !ls || !rs || !lh || !rh) return;

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

      const tDeg = trunkAngleDeg(midShoulder, midHip);
      const hM = headForwardM(nose as Point3, midShoulder);
      const sM = shoulderTiltM(ls as Point3, rs as Point3);
      const tSigned = trunkAngleSignedDeg(midShoulder, midHip);
      const hSigned = headForwardSignedM(nose as Point3, midShoulder);
      const sSigned = shoulderTiltSignedM(ls as Point3, rs as Point3);

      pushLimited(buffersRef.current.trunk, tDeg);
      pushLimited(buffersRef.current.head, hM);
      pushLimited(buffersRef.current.shoulder, sM);

      const d = computeDecision();
      setMetrics({ trunkAngle: tDeg, headForward: hM, shoulderTilt: sM });
      setSignedMetrics({
        trunkAngle: tSigned,
        headForward: hSigned,
        shoulderTilt: sSigned,
      });

      const nextScore = d.score ?? 0;
      setScore(nextScore);
      setFeedback(d.msg);
      setPill(d.ok ? "good" : "fix");

      if (d.t != null && d.h != null) {
        pushFeedback(nextScore, d.msg, d.t, d.h);
      }
    },
    [computeDecision, pushFeedback],
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
          width: { ideal: 1280 },
          height: { ideal: 720 },
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

              <div className="absolute right-4 top-4 bottom-4 w-72 lg:w-80 bg-black/40 backdrop-blur-xl border border-white/10 rounded-2xl flex flex-col overflow-hidden z-30 shadow-2xl">
                <div className="px-5 py-4 border-b border-white/10 bg-white/5 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Bell size={16} className="text-white/90" />
                    <h3 className="font-bold text-sm text-white uppercase tracking-wider">
                      Session Log
                    </h3>
                  </div>
                  <button
                    onClick={() => setShowSettings((v) => !v)}
                    className="flex items-center justify-center p-2.5 rounded-full transition-all border border-white/30 bg-white/20 text-white/80 hover:bg-white  hover:text-black"
                    title="Toggle Calibration Settings"
                  >
                    <Settings2 size={16} />
                  </button>
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

                <div className="mt-auto pt-8">
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
