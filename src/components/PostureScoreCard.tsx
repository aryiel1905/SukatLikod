import { useId, useRef } from "react";
import { useGSAP } from "@gsap/react";
import { gsap } from "gsap";

type PostureScoreCardProps = {
  score: number;
  paused: boolean;
  theme: "dark" | "light";
};

const RING_RADIUS = 43;
const RING_CIRCUMFERENCE = 2 * Math.PI * RING_RADIUS;
const TICK_COUNT = 40;

function getScorePresentation(score: number, paused: boolean) {
  if (paused && score <= 0) {
    return {
      label: "Waiting for a stable reading",
      tone: "neutral" as const,
    };
  }
  if (score > 80) {
    return {
      label: "Strong alignment",
      tone: "positive" as const,
    };
  }
  if (score > 60) {
    return {
      label: "A small adjustment may help",
      tone: "caution" as const,
    };
  }
  return {
    label: "Your posture needs attention",
    tone: "critical" as const,
  };
}

const DARK_TONES = {
  neutral: {
    text: "text-white/45",
    glow: "bg-white/[0.05]",
    ringStart: "#737373",
    ringEnd: "#d6d3d1",
    center: "rgba(255,255,255,0.045)",
  },
  positive: {
    text: "text-[#c5d6bf]",
    glow: "bg-[#91a889]/20",
    ringStart: "#6f8967",
    ringEnd: "#c5d6bf",
    center: "rgba(145,168,137,0.10)",
  },
  caution: {
    text: "text-amber-200",
    glow: "bg-amber-400/[0.15]",
    ringStart: "#d18a17",
    ringEnd: "#fde68a",
    center: "rgba(251,191,36,0.09)",
  },
  critical: {
    text: "text-rose-200",
    glow: "bg-rose-400/[0.15]",
    ringStart: "#be4562",
    ringEnd: "#fecdd3",
    center: "rgba(251,113,133,0.09)",
  },
} as const;

const LIGHT_TONES = {
  neutral: {
    text: "text-stone-400",
    glow: "bg-stone-200/55",
    ringStart: "#a8a29e",
    ringEnd: "#57534e",
    center: "rgba(120,113,108,0.06)",
  },
  positive: {
    text: "text-[#526b4b]",
    glow: "bg-[#91a889]/25",
    ringStart: "#91a889",
    ringEnd: "#40573b",
    center: "rgba(145,168,137,0.12)",
  },
  caution: {
    text: "text-amber-700",
    glow: "bg-amber-200/50",
    ringStart: "#fbbf24",
    ringEnd: "#a16207",
    center: "rgba(251,191,36,0.11)",
  },
  critical: {
    text: "text-rose-700",
    glow: "bg-rose-200/50",
    ringStart: "#fb7185",
    ringEnd: "#9f1239",
    center: "rgba(251,113,133,0.10)",
  },
} as const;

export function PostureScoreCard({
  score,
  paused,
  theme,
}: PostureScoreCardProps) {
  const cardRef = useRef<HTMLElement | null>(null);
  const scoreGroupRef = useRef<HTMLDivElement | null>(null);
  const scoreNumberRef = useRef<HTMLSpanElement | null>(null);
  const progressRef = useRef<SVGCircleElement | null>(null);
  const animatedScoreRef = useRef({ value: score });
  const gradientId = `posture-score-${useId().replace(/:/g, "")}`;
  const presentation = getScorePresentation(score, paused);
  const isDarkTheme = theme === "dark";
  const tones = isDarkTheme ? DARK_TONES : LIGHT_TONES;
  const tone = tones[presentation.tone];
  const visibleScore = Math.max(0, Math.min(100, score));
  const ringOffset = RING_CIRCUMFERENCE * (1 - visibleScore / 100);
  const activeTicks = Math.round((visibleScore / 100) * TICK_COUNT);

  useGSAP(
    () => {
      const reduceMotion = window.matchMedia(
        "(prefers-reduced-motion: reduce)",
      ).matches;

      gsap.killTweensOf(animatedScoreRef.current);

      if (scoreNumberRef.current) {
        if (reduceMotion) {
          animatedScoreRef.current.value = visibleScore;
          scoreNumberRef.current.textContent = String(visibleScore);
        } else {
          gsap.to(animatedScoreRef.current, {
            value: visibleScore,
            duration: 0.48,
            ease: "power3.out",
            overwrite: true,
            onUpdate: () => {
              if (scoreNumberRef.current) {
                scoreNumberRef.current.textContent = String(
                  Math.round(animatedScoreRef.current.value),
                );
              }
            },
          });
        }
      }

      if (scoreGroupRef.current && !reduceMotion) {
        gsap.fromTo(
          scoreGroupRef.current,
          { scale: 0.96 },
          { scale: 1, duration: 0.42, ease: "power3.out", overwrite: true },
        );
      }

      if (progressRef.current) {
        gsap.to(progressRef.current, {
          attr: { strokeDashoffset: ringOffset },
          duration: reduceMotion ? 0 : 0.65,
          ease: "power3.out",
          overwrite: true,
        });
      }
    },
    { scope: cardRef, dependencies: [visibleScore, paused] },
  );

  return (
    <article
      ref={cardRef}
      data-tour="posture-score"
      data-testid="posture-score-card"
      data-score-tone={presentation.tone}
      aria-label={`Posture score ${visibleScore} out of 100. ${presentation.label}.`}
      className={`group relative h-48 flex-shrink-0 overflow-hidden rounded-2xl border backdrop-blur-md transition-colors duration-300 ${
        isDarkTheme
          ? "border-white/10 bg-[#1a1917]"
          : "border-stone-200 bg-[#fcfbf8]"
      }`}
    >
      <span
        className={`pointer-events-none absolute -right-10 -top-12 h-44 w-44 rounded-full blur-3xl transition-colors duration-500 ${tone.glow}`}
        aria-hidden="true"
      />
      <span
        className={`pointer-events-none absolute inset-x-5 top-0 h-px ${
          isDarkTheme
            ? "bg-gradient-to-r from-transparent via-white/20 to-transparent"
            : "bg-gradient-to-r from-transparent via-stone-300 to-transparent"
        }`}
        aria-hidden="true"
      />

      <div className="absolute left-1/2 top-1/2 z-10 flex h-[9.5rem] w-[9.5rem] -translate-x-1/2 -translate-y-1/2 items-center justify-center">
          <span
            className={`absolute inset-5 rounded-full blur-2xl transition-colors duration-500 ${tone.glow}`}
            aria-hidden="true"
          />
          <svg
            viewBox="0 0 120 120"
            className="absolute inset-0 h-full w-full"
            aria-hidden="true"
          >
            <defs>
              <linearGradient
                id={gradientId}
                x1="18"
                y1="18"
                x2="102"
                y2="102"
                gradientUnits="userSpaceOnUse"
              >
                <stop offset="0" stopColor={tone.ringStart} />
                <stop offset="1" stopColor={tone.ringEnd} />
              </linearGradient>
              <radialGradient id={`${gradientId}-center`}>
                <stop offset="0" stopColor={tone.center} />
                <stop offset="1" stopColor="transparent" />
              </radialGradient>
            </defs>

            {Array.from({ length: TICK_COUNT }, (_, index) => (
              <line
                key={index}
                x1="60"
                y1="3"
                x2="60"
                y2="7"
                transform={`rotate(${index * (360 / TICK_COUNT)} 60 60)`}
                stroke={
                  index < activeTicks
                    ? tone.ringEnd
                    : isDarkTheme
                      ? "rgba(255,255,255,0.13)"
                      : "rgba(41,37,36,0.16)"
                }
                strokeWidth="1.25"
                strokeLinecap="round"
                opacity={index < activeTicks ? 0.78 : 1}
              />
            ))}

            <circle
              cx="60"
              cy="60"
              r="35"
              fill={`url(#${gradientId}-center)`}
              stroke={isDarkTheme ? "rgba(255,255,255,0.08)" : "rgba(41,37,36,0.08)"}
              strokeWidth="1"
            />
            <circle
              cx="60"
              cy="60"
              r={RING_RADIUS}
              fill="none"
              stroke="currentColor"
              strokeWidth="6"
              className={isDarkTheme ? "text-white/[0.09]" : "text-stone-200/80"}
            />
            <circle
              ref={progressRef}
              cx="60"
              cy="60"
              r={RING_RADIUS}
              fill="none"
              stroke={`url(#${gradientId})`}
              strokeWidth="6"
              strokeLinecap="round"
              strokeDasharray={RING_CIRCUMFERENCE}
              strokeDashoffset={ringOffset}
              transform="rotate(-90 60 60)"
            />
          </svg>

          <div
            ref={scoreGroupRef}
            className="relative flex flex-col items-center justify-center"
          >
            <span
              className={`mb-2 text-[9px] font-semibold leading-none tracking-[0.04em] ${
                isDarkTheme ? "text-white/60" : "text-stone-500"
              }`}
            >
              Posture Score
            </span>
            <span
              ref={scoreNumberRef}
              className={`text-[2.7rem] font-black leading-[0.82] tracking-[-0.065em] ${tone.text}`}
            >
              {visibleScore}
            </span>
            <span
              className={`mt-2 text-[9px] font-semibold tracking-[0.08em] ${
                isDarkTheme ? "text-white/45" : "text-stone-400"
              }`}
            >
              /100
            </span>
          </div>
      </div>

      <span className="sr-only" aria-live="polite">
        {presentation.label}
      </span>
    </article>
  );
}
