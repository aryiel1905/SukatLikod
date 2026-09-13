import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useGSAP } from "@gsap/react";
import { gsap } from "gsap";
import {
  ArrowLeft,
  Camera,
  ChevronDown,
  Download,
  Image as ImageIcon,
  Pause,
  Play,
  Trash2,
  X,
} from "lucide-react";
import type {
  FaceLandmarker,
  FaceLandmarkerResult,
  NormalizedLandmark,
  PoseLandmarker,
  PoseLandmarkerResult,
} from "@mediapipe/tasks-vision";
import { createZip, dataUrlToBytes } from "./captureArchive";

gsap.registerPlugin(useGSAP);

type CaptureStatus = "idle" | "loading" | "live" | "error";
type LandmarkSource = "pose" | "face";

type DisplayLandmark = {
  key: string;
  source: LandmarkSource;
  index: number;
  shortName: string;
  name: string;
  x: number;
  y: number;
  z: number;
  visibility: number | null;
  presence: number | null;
  pixelX: number;
  pixelY: number;
};

type LabCapture = {
  id: string;
  capturedAt: string;
  imageDataUrl: string;
  imageName: string;
  width: number;
  height: number;
  landmarks: DisplayLandmark[];
};

const POSE_NODES = [
  { index: 0, shortName: "N", name: "Nose" },
  { index: 2, shortName: "LE", name: "Left eye" },
  { index: 5, shortName: "RE", name: "Right eye" },
  { index: 7, shortName: "LA", name: "Left ear" },
  { index: 8, shortName: "RA", name: "Right ear" },
  { index: 11, shortName: "LS", name: "Left shoulder" },
  { index: 12, shortName: "RS", name: "Right shoulder" },
] as const;

const FACE_NODES = [{ index: 152, shortName: "C", name: "Chin" }] as const;

const POSE_LINKS: Array<[number, number]> = [
  [7, 2],
  [2, 0],
  [0, 5],
  [5, 8],
  [11, 12],
];

const MODEL_PATH = "/models/pose_landmarker_lite.task";
const FACE_MODEL_PATH = "/models/face_landmarker.task";
const VISIBILITY_THRESHOLD = 0.12;

function pointIsVisible(point: NormalizedLandmark | undefined) {
  return Boolean(
    point &&
      point.x >= 0 &&
      point.x <= 1 &&
      point.y >= 0 &&
      point.y <= 1 &&
      (point.visibility == null || point.visibility >= VISIBILITY_THRESHOLD),
  );
}

function roundedLabel(
  context: CanvasRenderingContext2D,
  text: string,
  x: number,
  y: number,
) {
  context.font = "600 12px Outfit, sans-serif";
  const width = Math.ceil(context.measureText(text).width) + 14;
  const height = 24;
  const left = Math.min(Math.max(6, x + 9), context.canvas.width - width - 6);
  const top = Math.min(Math.max(6, y - 29), context.canvas.height - height - 6);
  context.fillStyle = "rgba(12, 16, 18, 0.82)";
  context.beginPath();
  context.roundRect(left, top, width, height, 7);
  context.fill();
  context.fillStyle = "#f8fafc";
  context.fillText(text, left + 7, top + 16);
}

function formatCoordinate(value: number) {
  return value.toFixed(4);
}

function getTheme() {
  if (typeof window === "undefined") return "dark";
  return window.localStorage.getItem("sukatlikod-theme") === "light"
    ? "light"
    : "dark";
}

export default function CaptureLab() {
  const shellRef = useRef<HTMLElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const poseRef = useRef<PoseLandmarker | null>(null);
  const faceRef = useRef<FaceLandmarker | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const lastVideoTimeRef = useRef(-1);
  const lastUiLandmarkUpdateRef = useRef(0);
  const latestLandmarksRef = useRef<DisplayLandmark[]>([]);
  const captureRef = useRef<() => void>(() => undefined);

  const [theme] = useState(getTheme);
  const [status, setStatus] = useState<CaptureStatus>("idle");
  const [statusMessage, setStatusMessage] = useState(
    "Start the camera to inspect landmark coordinates.",
  );
  const [liveLandmarks, setLiveLandmarks] = useState<DisplayLandmark[]>([]);
  const [captures, setCaptures] = useState<LabCapture[]>([]);
  const [selectedCaptureId, setSelectedCaptureId] = useState<string | null>(null);
  const [showCoordinates, setShowCoordinates] = useState(true);
  const [inspectorOpen, setInspectorOpen] = useState(true);

  const isDark = theme === "dark";
  const selectedCapture = useMemo(
    () => captures.find((capture) => capture.id === selectedCaptureId) ?? null,
    [captures, selectedCaptureId],
  );
  const inspectedLandmarks = selectedCapture?.landmarks ?? liveLandmarks;

  useGSAP(
    () => {
      gsap.fromTo(
        "[data-lab-reveal]",
        { opacity: 0, y: 18, scale: 0.985 },
        {
          opacity: 1,
          y: 0,
          scale: 1,
          duration: 0.7,
          stagger: 0.07,
          ease: "power3.out",
        },
      );
    },
    { scope: shellRef },
  );

  useGSAP(
    () => {
      if (captures.length === 0) return;
      gsap.fromTo(
        "[data-capture-thumbnail]:first-child",
        { opacity: 0, y: 14, scale: 0.9 },
        { opacity: 1, y: 0, scale: 1, duration: 0.45, ease: "back.out(1.5)" },
      );
    },
    { scope: shellRef, dependencies: [captures.length] },
  );

  const stopCamera = useCallback(() => {
    if (animationFrameRef.current != null) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    overlayRef.current
      ?.getContext("2d")
      ?.clearRect(0, 0, overlayRef.current.width, overlayRef.current.height);
    latestLandmarksRef.current = [];
    setLiveLandmarks([]);
    setStatus("idle");
    setStatusMessage("Camera paused. Your captures are still available below.");
  }, []);

  const drawOverlay = useCallback(
    (
      result: PoseLandmarkerResult,
      faceResult: FaceLandmarkerResult | undefined,
      width: number,
      height: number,
    ) => {
      const canvas = overlayRef.current;
      const context = canvas?.getContext("2d");
      if (!canvas || !context) return;

      const pose = result.landmarks?.[0];
      const face = faceResult?.faceLandmarks?.[0];
      context.clearRect(0, 0, width, height);
      if (!pose) {
        if (latestLandmarksRef.current.length > 0) {
          latestLandmarksRef.current = [];
          setLiveLandmarks([]);
        }
        return;
      }

      const landmarks: DisplayLandmark[] = [];
      POSE_NODES.forEach((node) => {
        const point = pose[node.index];
        if (!point) return;
        landmarks.push({
          key: `pose-${node.index}`,
          source: "pose",
          index: node.index,
          shortName: node.shortName,
          name: node.name,
          x: point.x,
          y: point.y,
          z: point.z,
          visibility: point.visibility ?? null,
          presence: null,
          pixelX: (1 - point.x) * width,
          pixelY: point.y * height,
        });
      });
      FACE_NODES.forEach((node) => {
        const point = face?.[node.index];
        if (!point) return;
        landmarks.push({
          key: `face-${node.index}`,
          source: "face",
          index: node.index,
          shortName: node.shortName,
          name: node.name,
          x: point.x,
          y: point.y,
          z: point.z,
          visibility: point.visibility ?? null,
          presence: null,
          pixelX: (1 - point.x) * width,
          pixelY: point.y * height,
        });
      });

      context.save();
      context.lineCap = "round";
      context.lineJoin = "round";
      context.strokeStyle = "#22d3ee";
      context.lineWidth = Math.max(2, width / 480);

      POSE_LINKS.forEach(([fromIndex, toIndex]) => {
        const from = pose[fromIndex];
        const to = pose[toIndex];
        if (!pointIsVisible(from) || !pointIsVisible(to)) return;
        context.beginPath();
        context.moveTo(from.x * width, from.y * height);
        context.lineTo(to.x * width, to.y * height);
        context.stroke();
      });

      const nose = pose[0];
      const chin = face?.[152];
      if (pointIsVisible(nose) && chin) {
        context.beginPath();
        context.moveTo(nose.x * width, nose.y * height);
        context.lineTo(chin.x * width, chin.y * height);
        context.stroke();
      }

      const leftEar = pose[7];
      const rightEar = pose[8];
      if (pointIsVisible(leftEar) && pointIsVisible(rightEar) && chin) {
        const centerX = ((leftEar.x + rightEar.x) / 2) * width;
        const centerY = ((nose.y + chin.y) / 2) * height;
        const radiusX = Math.abs(leftEar.x - rightEar.x) * width * 0.62;
        const radiusY = Math.abs(chin.y - Math.min(leftEar.y, rightEar.y)) * height * 0.72;
        context.save();
        context.setLineDash([5, 6]);
        context.strokeStyle = "rgba(248,250,252,.48)";
        context.lineWidth = 1.25;
        context.beginPath();
        context.ellipse(centerX, centerY, radiusX, radiusY, 0, 0, Math.PI * 2);
        context.stroke();
        context.restore();
      }

      landmarks.forEach((landmark) => {
        if (
          landmark.x < 0 ||
          landmark.x > 1 ||
          landmark.y < 0 ||
          landmark.y > 1 ||
          (landmark.visibility != null && landmark.visibility < VISIBILITY_THRESHOLD)
        ) {
          return;
        }
        context.beginPath();
        context.arc(
          landmark.x * width,
          landmark.y * height,
          Math.max(4, width / 210),
          0,
          Math.PI * 2,
        );
        context.fillStyle = "#f8fafc";
        context.fill();
      });
      context.restore();

      latestLandmarksRef.current = landmarks;
      const now = performance.now();
      if (now - lastUiLandmarkUpdateRef.current >= 100) {
        lastUiLandmarkUpdateRef.current = now;
        setLiveLandmarks(landmarks);
      }
    },
    [],
  );

  const runLoop = useCallback(() => {
    const tick = () => {
      const pose = poseRef.current;
      const video = videoRef.current;
      if (!pose || !video || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
        animationFrameRef.current = requestAnimationFrame(tick);
        return;
      }

      if (video.currentTime !== lastVideoTimeRef.current) {
        const timestamp = performance.now();
        pose.detectForVideo(video, timestamp, (result) => {
          const faceResult = faceRef.current?.detectForVideo(video, timestamp);
          drawOverlay(result, faceResult, video.videoWidth, video.videoHeight);
        });
        lastVideoTimeRef.current = video.currentTime;
      }
      animationFrameRef.current = requestAnimationFrame(tick);
    };
    animationFrameRef.current = requestAnimationFrame(tick);
  }, [drawOverlay]);

  const startCamera = useCallback(async () => {
    try {
      setStatus("loading");
      setStatusMessage("Loading the landmark models…");
      if (!poseRef.current || !faceRef.current) {
        const { FaceLandmarker, FilesetResolver, PoseLandmarker } = await import(
          "@mediapipe/tasks-vision"
        );
        const vision = await FilesetResolver.forVisionTasks("/mediapipe/wasm");
        poseRef.current = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: MODEL_PATH, delegate: "GPU" },
          runningMode: "VIDEO",
          numPoses: 1,
          minPoseDetectionConfidence: 0.5,
          minPosePresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });
        faceRef.current = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: FACE_MODEL_PATH, delegate: "GPU" },
          runningMode: "VIDEO",
          numFaces: 1,
          minFaceDetectionConfidence: 0.5,
          minFacePresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
          outputFaceBlendshapes: false,
          outputFacialTransformationMatrixes: false,
        });
      }

      setStatusMessage("Requesting camera permission…");
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: "user" },
          width: { ideal: 1280 },
          height: { ideal: 720 },
          frameRate: { ideal: 30, max: 30 },
        },
        audio: false,
      });
      streamRef.current = stream;
      const video = videoRef.current;
      const canvas = overlayRef.current;
      if (!video || !canvas) return;
      video.srcObject = stream;
      await new Promise<void>((resolve) => {
        video.onloadedmetadata = () => resolve();
      });
      await video.play();
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      lastVideoTimeRef.current = -1;
      lastUiLandmarkUpdateRef.current = 0;
      setStatus("live");
      setStatusMessage("Camera live. Press Space when you are ready.");
      runLoop();
    } catch (error) {
      console.error(error);
      setStatus("error");
      setStatusMessage("Camera could not start. Check permission and try again.");
    }
  }, [runLoop]);

  const captureFrame = useCallback(() => {
    const video = videoRef.current;
    const overlay = overlayRef.current;
    const landmarks = latestLandmarksRef.current;
    if (status !== "live" || !video || !overlay || landmarks.length === 0) {
      setStatusMessage("Keep your face and shoulders visible before capturing.");
      return;
    }

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext("2d");
    if (!context) return;

    context.save();
    context.translate(canvas.width, 0);
    context.scale(-1, 1);
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    context.drawImage(overlay, 0, 0, canvas.width, canvas.height);
    context.restore();

    if (showCoordinates) {
      landmarks.forEach((landmark) => {
        if (
          landmark.x < 0 ||
          landmark.x > 1 ||
          landmark.y < 0 ||
          landmark.y > 1
        ) {
          return;
        }
        roundedLabel(
          context,
          `${landmark.shortName} ${formatCoordinate(landmark.x)}, ${formatCoordinate(landmark.y)}, ${formatCoordinate(landmark.z)}`,
          landmark.pixelX,
          landmark.pixelY,
        );
      });
    }

    const capturedAt = new Date().toISOString();
    const id = crypto.randomUUID();
    const imageName = `uprightly-capture-${capturedAt.replace(/[:.]/g, "-")}.jpg`;
    const capture: LabCapture = {
      id,
      capturedAt,
      imageDataUrl: canvas.toDataURL("image/jpeg", 0.92),
      imageName,
      width: canvas.width,
      height: canvas.height,
      landmarks: landmarks.map((landmark) => ({ ...landmark })),
    };
    setCaptures((current) => [capture, ...current]);
    setSelectedCaptureId(id);
    setStatusMessage("Capture saved locally. Press Space to take another.");
  }, [showCoordinates, status]);

  captureRef.current = captureFrame;

  const deleteCapture = useCallback((id: string) => {
    setCaptures((current) => current.filter((capture) => capture.id !== id));
    setSelectedCaptureId((current) => (current === id ? null : current));
  }, []);

  const downloadCaptures = useCallback(() => {
    if (captures.length === 0) return;
    const manifest = {
      format: "uprightly-capture-lab",
      version: 1,
      exportedAt: new Date().toISOString(),
      coordinateSystem: {
        normalized: "MediaPipe coordinates from the unmirrored camera frame",
        pixel: "Coordinates in the mirrored image included in this archive",
      },
      captureCount: captures.length,
      captures: captures.map((capture) => ({
        id: capture.id,
        capturedAt: capture.capturedAt,
        imageName: capture.imageName,
        image: `images/${capture.imageName}`,
        width: capture.width,
        height: capture.height,
        landmarks: capture.landmarks,
      })),
    };
    const encoder = new TextEncoder();
    const archive = createZip([
      {
        name: "captures.json",
        data: encoder.encode(JSON.stringify(manifest, null, 2)),
      },
      ...captures.map((capture) => ({
        name: `images/${capture.imageName}`,
        data: dataUrlToBytes(capture.imageDataUrl),
      })),
    ]);
    const url = URL.createObjectURL(archive);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `uprightly-captures-${new Date().toISOString().slice(0, 10)}.zip`;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1_000);
  }, [captures]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (
        event.code !== "Space" ||
        event.repeat ||
        target?.closest("button, a, input, select, textarea, [contenteditable='true']")
      ) {
        return;
      }
      event.preventDefault();
      captureRef.current();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  useEffect(() => {
    const warnBeforeLeaving = (event: BeforeUnloadEvent) => {
      if (captures.length === 0) return;
      event.preventDefault();
    };
    window.addEventListener("beforeunload", warnBeforeLeaving);
    return () => window.removeEventListener("beforeunload", warnBeforeLeaving);
  }, [captures.length]);

  useEffect(
    () => () => {
      if (animationFrameRef.current != null) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      streamRef.current?.getTracks().forEach((track) => track.stop());
      poseRef.current?.close();
      faceRef.current?.close();
    },
    [],
  );

  return (
    <main
      ref={shellRef}
      className={`min-h-dvh w-full max-w-full overflow-x-hidden ${
        isDark ? "bg-[#10100f] text-[#f4f0e8]" : "bg-[#f2efe7] text-[#1c1b19]"
      }`}
    >
      <section className="flex min-h-dvh items-center justify-center px-5 text-center md:hidden">
        <div className="max-w-sm">
          <Camera className="mx-auto opacity-45" size={34} aria-hidden="true" />
          <h1 className="mt-5 text-2xl font-bold">Open Capture Lab on a computer</h1>
          <p className="mt-3 text-sm leading-6 opacity-60">
            The landmark inspector needs a laptop or desktop camera and a wider screen.
          </p>
          <a href="/" className="mt-6 inline-flex min-h-11 items-center rounded-xl border border-current/20 px-5 text-sm font-semibold">
            Back to Uprightly
          </a>
        </div>
      </section>

      <div className="mx-auto hidden min-h-dvh w-full max-w-[1600px] px-6 py-6 md:block xl:px-10 xl:py-8">
        <header data-lab-reveal className="flex items-center justify-between gap-6">
          <div className="flex min-w-0 items-center gap-4">
            <a
              href="/"
              aria-label="Back to Uprightly"
              className={`flex h-11 w-11 shrink-0 items-center justify-center rounded-xl border transition-transform hover:-translate-x-0.5 ${
                isDark ? "border-white/12 bg-white/5" : "border-[#0A3A72]/15 bg-white/70"
              }`}
            >
              <ArrowLeft size={19} aria-hidden="true" />
            </a>
            <div className="min-w-0">
              <h1 className="max-w-2xl truncate text-2xl font-bold tracking-[-0.035em] sm:text-3xl">
                Uprightly Capture Lab
              </h1>
              <p className="mt-1 text-xs opacity-55 sm:text-sm">
                Inspect and capture landmark values. No posture classification.
              </p>
            </div>
          </div>
          <div className="flex shrink-0 items-center gap-3">
            <span className={`hidden text-xs lg:block ${captures.length > 0 ? "opacity-70" : "opacity-45"}`}>
              {captures.length > 0 ? `${captures.length} unsaved capture${captures.length === 1 ? "" : "s"}` : "Nothing saved yet"}
            </span>
            <button
              type="button"
              onClick={downloadCaptures}
              disabled={captures.length === 0}
              className={`flex min-h-11 items-center gap-2 rounded-xl px-4 text-sm font-semibold transition-[transform,opacity] hover:-translate-y-0.5 disabled:cursor-not-allowed disabled:opacity-35 ${
                isDark ? "bg-[#f4f0e8] text-[#151513]" : "bg-[#0A3A72] text-white"
              }`}
            >
              <Download size={17} aria-hidden="true" />
              Download ZIP
            </button>
          </div>
        </header>

        <div className="mt-6 grid grid-flow-dense grid-cols-12 gap-5">
          <section
            data-lab-reveal
            aria-label="Live camera and capture controls"
            className={`col-span-12 overflow-hidden rounded-[1.75rem] border lg:col-span-8 ${
              isDark ? "border-white/10 bg-[#171715]" : "border-[#0A3A72]/12 bg-white/80"
            }`}
          >
            <div className="relative aspect-video overflow-hidden bg-[#090a0a]">
              <video ref={videoRef} autoPlay playsInline muted className="absolute inset-0 h-full w-full -scale-x-100 object-cover" />
              <canvas ref={overlayRef} className="absolute inset-0 h-full w-full -scale-x-100 object-cover" />

              {status === "live" && showCoordinates
                ? liveLandmarks.map((landmark) => (
                    <span
                      key={landmark.key}
                      className="pointer-events-none absolute z-10 -translate-y-[calc(100%+8px)] rounded-md bg-black/80 px-1.5 py-1 font-mono text-[9px] font-semibold leading-none text-white shadow-lg"
                      style={{ left: `${(1 - landmark.x) * 100}%`, top: `${landmark.y * 100}%` }}
                    >
                      {landmark.shortName} {formatCoordinate(landmark.x)}, {formatCoordinate(landmark.y)}, {formatCoordinate(landmark.z)}
                    </span>
                  ))
                : null}

              {status !== "live" ? (
                <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/58 px-6 text-center text-white backdrop-blur-sm">
                  <div className="flex h-16 w-16 items-center justify-center rounded-full border border-white/12 bg-white/8">
                    {status === "loading" ? <span className="h-6 w-6 animate-spin rounded-full border-2 border-white/25 border-t-white" /> : <Camera size={27} className="text-white/60" />}
                  </div>
                  <p className="mt-4 max-w-md text-sm leading-6 text-white/65">{statusMessage}</p>
                  {status !== "loading" ? (
                    <button type="button" onClick={() => void startCamera()} className="mt-5 flex min-h-11 items-center gap-2 rounded-xl bg-white px-5 text-sm font-semibold text-black transition-transform hover:-translate-y-0.5">
                      <Play size={16} fill="currentColor" aria-hidden="true" />
                      {status === "error" ? "Try again" : "Start camera"}
                    </button>
                  ) : null}
                </div>
              ) : null}

              <div className="absolute left-4 top-4 flex items-center gap-2 rounded-xl bg-black/70 px-3 py-2 text-xs text-white backdrop-blur-xl">
                <span className={`h-2 w-2 rounded-full ${status === "live" ? "bg-emerald-400" : "bg-white/35"}`} />
                {status === "live" ? `${liveLandmarks.length} nodes detected` : status === "loading" ? "Loading" : "Camera off"}
              </div>

              {status === "live" ? (
                <button type="button" onClick={stopCamera} className="absolute right-4 top-4 flex min-h-10 items-center gap-2 rounded-xl border border-white/12 bg-black/70 px-3 text-xs font-semibold text-white backdrop-blur-xl hover:bg-black/85">
                  <Pause size={14} fill="currentColor" aria-hidden="true" />
                  Pause
                </button>
              ) : null}
            </div>

            <div className="flex flex-wrap items-center justify-between gap-4 p-4 sm:p-5">
              <div>
                <p className="text-sm font-semibold">{statusMessage}</p>
                <p className="mt-1 text-xs opacity-50">Coordinates use the unmirrored MediaPipe frame; pixel values match the saved mirrored image.</p>
              </div>
              <div className="flex items-center gap-3">
                <label className="flex cursor-pointer items-center gap-2 text-xs font-semibold opacity-70">
                  <input type="checkbox" checked={showCoordinates} onChange={(event) => setShowCoordinates(event.target.checked)} className="h-4 w-4 accent-cyan-400" />
                  Values on image
                </label>
                <button
                  type="button"
                  onClick={captureFrame}
                  disabled={status !== "live" || liveLandmarks.length === 0}
                  className={`flex min-h-12 items-center gap-3 rounded-xl px-5 text-sm font-semibold transition-[transform,opacity] hover:-translate-y-0.5 disabled:cursor-not-allowed disabled:opacity-35 ${
                    isDark ? "bg-[#f4f0e8] text-[#151513]" : "bg-[#0A3A72] text-white"
                  }`}
                >
                  <Camera size={18} aria-hidden="true" />
                  Capture
                  <kbd className="rounded-md border border-current/20 px-1.5 py-0.5 font-sans text-[10px] opacity-70">Space</kbd>
                </button>
              </div>
            </div>
          </section>

          <aside data-lab-reveal className={`col-span-12 flex min-h-0 flex-col rounded-[1.75rem] border lg:col-span-4 ${isDark ? "border-white/10 bg-[#171715]" : "border-[#0A3A72]/12 bg-white/80"}`}>
            <button type="button" onClick={() => setInspectorOpen((value) => !value)} aria-expanded={inspectorOpen} className="flex w-full items-center justify-between gap-4 p-5 text-left">
              <div>
                <h2 className="text-base font-bold">{selectedCapture ? "Captured node values" : "Live node values"}</h2>
                <p className="mt-1 text-xs opacity-50">Normalized position and mirrored image pixels</p>
              </div>
              <ChevronDown size={18} className={`transition-transform ${inspectorOpen ? "rotate-180" : ""}`} />
            </button>
            {inspectorOpen ? (
              <div className="max-h-[min(58vh,35rem)] overflow-y-auto border-t border-current/10 px-3 pb-3">
                {inspectedLandmarks.length > 0 ? inspectedLandmarks.map((landmark) => (
                  <div key={landmark.key} className="grid grid-cols-[1fr_auto] gap-3 border-b border-current/8 px-2 py-3 last:border-0">
                    <div className="min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="flex h-6 min-w-6 items-center justify-center rounded-md bg-cyan-400/12 px-1.5 font-mono text-[10px] font-bold text-cyan-500">{landmark.shortName}</span>
                        <p className="truncate text-xs font-semibold">{landmark.name}</p>
                      </div>
                      <p className="mt-1.5 font-mono text-[10px] opacity-50">{landmark.source} · node {landmark.index}</p>
                    </div>
                    <dl className="grid grid-cols-[auto_auto] gap-x-2 text-right font-mono text-[10px] leading-5">
                      <dt className="opacity-40">x</dt><dd>{formatCoordinate(landmark.x)}</dd>
                      <dt className="opacity-40">y</dt><dd>{formatCoordinate(landmark.y)}</dd>
                      <dt className="opacity-40">z</dt><dd>{formatCoordinate(landmark.z)}</dd>
                      <dt className="opacity-40">px</dt><dd>{Math.round(landmark.pixelX)}, {Math.round(landmark.pixelY)}</dd>
                      <dt className="opacity-40">vis</dt><dd>{landmark.visibility == null ? "—" : landmark.visibility.toFixed(3)}</dd>
                    </dl>
                  </div>
                )) : (
                  <p className="px-3 py-10 text-center text-xs leading-5 opacity-45">Node values will appear when your face and shoulders are visible.</p>
                )}
              </div>
            ) : null}
          </aside>

          <section data-lab-reveal className={`col-span-12 rounded-[1.75rem] border p-5 ${isDark ? "border-white/10 bg-[#171715]" : "border-[#0A3A72]/12 bg-white/80"}`}>
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                <div className={`flex h-10 w-10 items-center justify-center rounded-xl ${isDark ? "bg-white/6" : "bg-[#0A3A72]/7"}`}><ImageIcon size={18} aria-hidden="true" /></div>
                <div>
                  <h2 className="text-base font-bold">Captured photos</h2>
                  <p className="text-xs opacity-50">{captures.length} photo{captures.length === 1 ? "" : "s"} ready to download</p>
                </div>
              </div>
              {captures.length > 0 ? (
                <button type="button" onClick={() => { setCaptures([]); setSelectedCaptureId(null); }} className="flex min-h-10 items-center gap-2 rounded-xl border border-current/12 px-3.5 text-xs font-semibold opacity-65 hover:opacity-100">
                  <Trash2 size={14} aria-hidden="true" /> Clear all
                </button>
              ) : null}
            </div>

            {captures.length === 0 ? (
              <div className={`mt-5 flex min-h-36 items-center justify-center rounded-2xl border border-dashed text-center ${isDark ? "border-white/10 bg-black/10" : "border-[#0A3A72]/12 bg-[#eef4fa]/55"}`}>
                <p className="max-w-sm px-6 text-xs leading-5 opacity-45">Your captures will appear here before anything is downloaded. They stay only in this browser tab.</p>
              </div>
            ) : (
              <div className="mt-5 flex gap-3 overflow-x-auto pb-2">
                {captures.map((capture, index) => (
                  <article key={capture.id} data-capture-thumbnail className={`group relative w-52 shrink-0 overflow-hidden rounded-2xl border ${selectedCaptureId === capture.id ? "border-cyan-400" : "border-current/10"}`}>
                    <button type="button" onClick={() => setSelectedCaptureId(capture.id)} className="block w-full overflow-hidden text-left">
                      <img src={capture.imageDataUrl} alt={`Capture ${captures.length - index}`} className="aspect-video w-full object-cover transition-transform duration-700 ease-out group-hover:scale-105" />
                      <div className={`flex items-center justify-between gap-2 px-3 py-2.5 ${isDark ? "bg-[#11110f]" : "bg-white"}`}>
                        <span className="text-xs font-semibold">Capture {captures.length - index}</span>
                        <time className="text-[10px] opacity-45">{new Date(capture.capturedAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}</time>
                      </div>
                    </button>
                    <button type="button" onClick={() => deleteCapture(capture.id)} aria-label={`Delete capture ${captures.length - index}`} className="absolute right-2 top-2 flex h-8 w-8 items-center justify-center rounded-lg bg-black/75 text-white opacity-0 backdrop-blur-md transition-opacity group-hover:opacity-100 focus:opacity-100">
                      <X size={14} aria-hidden="true" />
                    </button>
                  </article>
                ))}
              </div>
            )}
          </section>
        </div>

        <footer className="flex flex-wrap items-center justify-between gap-3 px-1 pb-2 pt-5 text-[11px] opacity-45">
          <p>Captures are kept in memory and are not uploaded.</p>
          <p>Closing this tab clears anything you have not downloaded.</p>
        </footer>
      </div>
    </main>
  );
}
