import { useRef } from "react";
import { useGSAP } from "@gsap/react";
import { gsap } from "gsap";
import {
  Activity,
  AlertCircle,
  AlertTriangle,
  CheckCircle2,
  LoaderCircle,
} from "lucide-react";
import type { PostureStateKind } from "../features/posture/postureState";

type PostureStateCardProps = {
  state: PostureStateKind;
  message: string;
  theme: "dark" | "light";
};

const STATE_META = {
  inactive: {
    label: "Start a session",
    Icon: Activity,
  },
  analyzing: {
    label: "Analyzing posture",
    Icon: LoaderCircle,
  },
  neutral: {
    label: "Neutral Posture",
    Icon: CheckCircle2,
  },
  mild: {
    label: "Mild Asymmetry",
    Icon: AlertCircle,
  },
  severe: {
    label: "Severe Misalignment",
    Icon: AlertTriangle,
  },
  unavailable: {
    label: "Camera unavailable",
    Icon: AlertCircle,
  },
} as const;

const DARK_STATE_CLASSES: Record<PostureStateKind, string> = {
  inactive: "border-white/10 bg-gradient-to-br from-white/[0.08] to-transparent",
  analyzing: "border-white/12 bg-gradient-to-br from-white/[0.09] to-transparent",
  neutral:
    "border-emerald-400/35 bg-gradient-to-br from-emerald-400/[0.16] via-emerald-400/[0.06] to-transparent",
  mild:
    "border-amber-400/35 bg-gradient-to-br from-amber-400/[0.16] via-amber-400/[0.06] to-transparent",
  severe:
    "border-rose-400/40 bg-gradient-to-br from-rose-400/[0.18] via-rose-400/[0.07] to-transparent",
  unavailable:
    "border-rose-400/30 bg-gradient-to-br from-rose-400/[0.12] to-transparent",
};

const LIGHT_STATE_CLASSES: Record<PostureStateKind, string> = {
  inactive: "border-stone-200 bg-gradient-to-br from-white to-stone-50",
  analyzing: "border-stone-200 bg-gradient-to-br from-white to-stone-50",
  neutral:
    "border-emerald-600/25 bg-gradient-to-br from-emerald-50 via-white to-white",
  mild:
    "border-amber-600/25 bg-gradient-to-br from-amber-50 via-white to-white",
  severe:
    "border-rose-600/25 bg-gradient-to-br from-rose-50 via-white to-white",
  unavailable:
    "border-rose-600/20 bg-gradient-to-br from-rose-50 via-white to-white",
};

const ACCENT_CLASSES: Record<PostureStateKind, string> = {
  inactive: "bg-white/30 text-white/65",
  analyzing: "bg-white/35 text-white/80",
  neutral: "bg-emerald-400 text-emerald-950",
  mild: "bg-amber-400 text-amber-950",
  severe: "bg-rose-400 text-rose-950",
  unavailable: "bg-rose-400 text-rose-950",
};

const LIGHT_ACCENT_CLASSES: Record<PostureStateKind, string> = {
  inactive: "bg-stone-200 text-stone-600",
  analyzing: "bg-stone-200 text-stone-700",
  neutral: "bg-emerald-600 text-white",
  mild: "bg-amber-500 text-stone-950",
  severe: "bg-rose-600 text-white",
  unavailable: "bg-rose-600 text-white",
};

export function PostureStateCard({
  state,
  message,
  theme,
}: PostureStateCardProps) {
  const cardRef = useRef<HTMLElement | null>(null);
  const meta = STATE_META[state];
  const Icon = meta.Icon;
  const isDarkTheme = theme === "dark";

  useGSAP(
    () => {
      const reduceMotion = window.matchMedia(
        "(prefers-reduced-motion: reduce)",
      ).matches;
      const content = cardRef.current?.querySelector("[data-posture-state-content]");
      const accent = cardRef.current?.querySelector("[data-posture-state-accent]");
      if (!content || !accent) return;

      gsap.killTweensOf([content, accent]);
      gsap.fromTo(
        content,
        { autoAlpha: 0, y: reduceMotion ? 0 : 10 },
        {
          autoAlpha: 1,
          y: 0,
          duration: reduceMotion ? 0 : 0.42,
          ease: "power3.out",
        },
      );
      gsap.fromTo(
        accent,
        { scaleY: reduceMotion ? 1 : 0.35 },
        {
          scaleY: 1,
          duration: reduceMotion ? 0 : 0.5,
          ease: "back.out(1.7)",
        },
      );
    },
    { scope: cardRef, dependencies: [state] },
  );

  return (
    <article
      ref={cardRef}
      data-testid="posture-state-card"
      data-posture-state={state}
      aria-live="polite"
      aria-atomic="true"
      className={`group relative h-48 flex-shrink-0 overflow-hidden rounded-2xl border p-5 backdrop-blur-md transition-colors duration-300 ${
        isDarkTheme ? DARK_STATE_CLASSES[state] : LIGHT_STATE_CLASSES[state]
      }`}
    >
      <span
        data-posture-state-accent
        className={`absolute bottom-5 left-0 top-5 w-1 origin-center rounded-r-full ${
          isDarkTheme ? ACCENT_CLASSES[state] : LIGHT_ACCENT_CLASSES[state]
        }`}
        aria-hidden="true"
      />

      <div data-posture-state-content className="flex h-full flex-col pl-2">
        <div className="flex items-center justify-between gap-4">
          <p
            className={`text-xs font-semibold tracking-wide ${
              isDarkTheme ? "text-white/60" : "text-stone-500"
            }`}
          >
            Current posture
          </p>
          <span
            className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-xl ${
              isDarkTheme ? ACCENT_CLASSES[state] : LIGHT_ACCENT_CLASSES[state]
            }`}
            aria-hidden="true"
          >
            <Icon
              size={19}
              className={
                state === "analyzing" ? "animate-spin motion-reduce:animate-none" : ""
              }
            />
          </span>
        </div>

        <p
          className={`mt-3 max-w-[15rem] text-[clamp(1.25rem,1.6vw,1.55rem)] font-bold leading-[1.05] tracking-[-0.035em] ${
            isDarkTheme ? "text-[#f4f0e8]" : "text-[#1c1b19]"
          }`}
        >
          {meta.label}
        </p>
        <p
          className={`mt-1.5 text-sm leading-5 ${
            isDarkTheme ? "text-white/65" : "text-stone-600"
          }`}
        >
          {message}
        </p>
      </div>
    </article>
  );
}
