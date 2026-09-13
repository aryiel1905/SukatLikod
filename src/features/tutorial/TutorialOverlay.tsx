import { useEffect, useRef } from "react";
import { useGSAP } from "@gsap/react";
import { gsap } from "gsap";
import { Check, ChevronLeft, ChevronRight, X } from "lucide-react";

export type TutorialVisualKind =
  | "baseline"
  | "forward"
  | "shoulders"
  | "framing"
  | "neutral"
  | "floating";

export type TutorialOverlayStep = {
  id: TutorialVisualKind;
  title: string;
  instruction: string;
  expected: string;
};

type TutorialOverlayProps = {
  step: TutorialOverlayStep;
  stepIndex: number;
  stepCount: number;
  detected: boolean;
  floatingWindowSupported: boolean;
  isDarkTheme: boolean;
  onPrevious: () => void;
  onNext: () => void;
  onClose: () => void;
};

function BrowserTutorialVisual({
  kind,
  detected,
}: {
  kind: TutorialVisualKind;
  detected: boolean;
}) {
  const rootRef = useRef<HTMLDivElement | null>(null);
  const personRef = useRef<SVGGElement | null>(null);
  const headRef = useRef<SVGGElement | null>(null);
  const shouldersRef = useRef<SVGGElement | null>(null);
  const directionRef = useRef<SVGGElement | null>(null);
  const frameRef = useRef<SVGRectElement | null>(null);
  const browserRef = useRef<SVGGElement | null>(null);
  const workTabRef = useRef<SVGGElement | null>(null);
  const floatingRef = useRef<SVGGElement | null>(null);

  useGSAP(
    () => {
      const reduceMotion = window.matchMedia(
        "(prefers-reduced-motion: reduce)",
      ).matches;
      const person = personRef.current;
      const head = headRef.current;
      const shoulders = shouldersRef.current;
      const direction = directionRef.current;
      const frame = frameRef.current;
      const browser = browserRef.current;
      const workTab = workTabRef.current;
      const floating = floatingRef.current;
      if (
        !person ||
        !head ||
        !shoulders ||
        !direction ||
        !frame ||
        !browser ||
        !workTab ||
        !floating
      ) {
        return;
      }

      gsap.killTweensOf([
        person,
        head,
        shoulders,
        direction,
        frame,
        browser,
        workTab,
        floating,
      ]);
      gsap.set([person, head, shoulders, direction, frame, browser, workTab, floating], {
        clearProps: "all",
      });
      gsap.set(rootRef.current, { autoAlpha: 1 });

      const repeat = reduceMotion || detected ? 0 : -1;
      const personTimeline = gsap.timeline({ repeat, repeatDelay: 0.45 });

      if (kind === "floating") {
        gsap.set(person, { autoAlpha: 0 });
        gsap.set(browser, { autoAlpha: 1 });
        gsap.set(workTab, { y: detected ? 0 : 8, opacity: detected ? 1 : 0.45 });
        gsap.set(floating, {
          autoAlpha: detected ? 1 : 0,
          scale: detected ? 1 : 0.82,
          transformOrigin: "center center",
        });
        if (!reduceMotion && !detected) {
          personTimeline
            .to(workTab, { y: 0, opacity: 1, duration: 0.65, ease: "power2.out" })
            .to(floating, {
              autoAlpha: 1,
              scale: 1,
              duration: 0.55,
              ease: "back.out(1.6)",
            })
            .to(floating, { y: -5, duration: 0.7, ease: "sine.inOut" })
            .to(floating, { y: 0, duration: 0.7, ease: "sine.inOut" });
        }
      } else {
        gsap.set(browser, { autoAlpha: 0 });
        gsap.set(person, {
          autoAlpha: 1,
          x: 0,
          y: 0,
          scale: 1,
          rotation: 0,
          transformOrigin: "center bottom",
        });
        gsap.set(head, { x: 0, y: 0, scale: 1, transformOrigin: "center center" });
        gsap.set(shoulders, { rotation: 0, svgOrigin: "360 300" });
        gsap.set(direction, { autoAlpha: 0 });
        gsap.set(frame, { opacity: 0.35 });

        if (kind === "baseline") {
          if (!reduceMotion && !detected) {
            personTimeline
              .fromTo(
                person,
                { y: 12, opacity: 0.55, scale: 0.97 },
                { y: 0, opacity: 1, scale: 1, duration: 0.9, ease: "power3.out" },
              )
              .to(frame, { opacity: 0.72, duration: 0.55, ease: "sine.inOut" })
              .to(frame, { opacity: 0.35, duration: 0.55, ease: "sine.inOut" });
          }
        } else if (kind === "forward") {
          gsap.set(direction, { autoAlpha: 1 });
          if (detected || reduceMotion) {
            gsap.set(person, { scale: 1.09, y: 8 });
            gsap.set(head, { scale: 1.05, y: 5 });
          } else {
            personTimeline
              .to(person, { scale: 1.09, y: 8, duration: 1, ease: "power2.inOut" })
              .to(head, { scale: 1.05, y: 5, duration: 1, ease: "power2.inOut" }, "<")
              .to(direction, { x: 12, duration: 0.55, ease: "sine.inOut" }, "<")
              .to(person, { scale: 1, y: 0, duration: 0.9, ease: "power2.inOut" })
              .to(head, { scale: 1, y: 0, duration: 0.9, ease: "power2.inOut" }, "<")
              .to(direction, { x: 0, duration: 0.55, ease: "sine.inOut" }, "<");
          }
        } else if (kind === "shoulders") {
          if (detected || reduceMotion) {
            gsap.set(shoulders, { rotation: -7 });
          } else {
            personTimeline
              .to(shoulders, { rotation: -7, duration: 0.8, ease: "power2.inOut" })
              .to(shoulders, { rotation: 7, duration: 1.1, ease: "sine.inOut" })
              .to(shoulders, { rotation: 0, duration: 0.8, ease: "power2.inOut" });
          }
        } else if (kind === "framing") {
          if (detected || reduceMotion) {
            gsap.set(person, { x: -62 });
            gsap.set(frame, { opacity: 0.8 });
          } else {
            personTimeline
              .to(frame, { opacity: 0.8, duration: 0.4 })
              .to(person, { x: -62, duration: 1.05, ease: "power2.inOut" })
              .to(person, { x: 0, duration: 1.05, ease: "power2.inOut", delay: 0.35 })
              .to(frame, { opacity: 0.35, duration: 0.4 }, "<");
          }
        } else if (kind === "neutral") {
          if (!reduceMotion && !detected) {
            personTimeline
              .fromTo(
                person,
                { x: -22, rotation: -4, scale: 1.06 },
                { x: 0, rotation: 0, scale: 1, duration: 1.1, ease: "power3.out" },
              )
              .to(frame, { opacity: 0.72, duration: 0.5, ease: "sine.inOut" })
              .to(frame, { opacity: 0.35, duration: 0.5, ease: "sine.inOut" });
          }
        }
      }

      gsap.fromTo(
        rootRef.current,
        { opacity: 0, scale: 0.94 },
        { opacity: 1, scale: 1, duration: reduceMotion ? 0 : 0.55, ease: "power3.out" },
      );

      return () => personTimeline.kill();
    },
    { scope: rootRef, dependencies: [kind, detected] },
  );

  return (
    <div
      ref={rootRef}
      data-tutorial-visual={kind}
      className="relative flex h-full min-h-[12rem] w-full items-center justify-center overflow-hidden"
      aria-label={`Animated guide for ${kind}`}
    >
      <div className="pointer-events-none absolute inset-[8%] rounded-full bg-white/[0.035] blur-2xl" />
      <svg
        viewBox="0 0 520 420"
        role="img"
        aria-hidden="true"
        className="relative h-full max-h-[27rem] w-full max-w-[34rem] overflow-visible"
      >
        <defs>
          <linearGradient id="tutorial-guide-line" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stopColor="#ffffff" />
            <stop offset="1" stopColor="#b9c6ca" />
          </linearGradient>
          <filter id="tutorial-guide-glow" x="-40%" y="-40%" width="180%" height="180%">
            <feGaussianBlur stdDeviation="3.5" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        <g ref={personRef}>
          <rect
            ref={frameRef}
            x="112"
            y="34"
            width="296"
            height="350"
            rx="144"
            fill="none"
            stroke="rgba(255,255,255,.55)"
            strokeWidth="2"
            strokeDasharray="8 10"
          />
          <path
            d="M126 384 C132 310 166 272 215 258 L305 258 C354 272 388 310 394 384"
            fill="rgba(255,255,255,.055)"
            stroke="url(#tutorial-guide-line)"
            strokeWidth="8"
            strokeLinecap="round"
          />
          <g ref={shouldersRef}>
            <path
              d="M160 300 Q260 257 360 300"
              fill="none"
              stroke="url(#tutorial-guide-line)"
              strokeWidth="9"
              strokeLinecap="round"
              filter="url(#tutorial-guide-glow)"
            />
            <circle cx="160" cy="300" r="8" fill="#f8fafc" />
            <circle cx="360" cy="300" r="8" fill="#f8fafc" />
          </g>
          <path
            d="M235 252 L235 226 M285 252 L285 226"
            stroke="url(#tutorial-guide-line)"
            strokeWidth="7"
            strokeLinecap="round"
          />
          <g ref={headRef}>
            <ellipse
              cx="260"
              cy="160"
              rx="76"
              ry="91"
              fill="rgba(255,255,255,.07)"
              stroke="url(#tutorial-guide-line)"
              strokeWidth="8"
              filter="url(#tutorial-guide-glow)"
            />
            <path d="M222 151 H241 M279 151 H298" stroke="#f8fafc" strokeWidth="7" strokeLinecap="round" />
            <path d="M260 158 V190" stroke="#f8fafc" strokeWidth="6" strokeLinecap="round" />
            <path d="M237 207 Q260 222 283 207" fill="none" stroke="#f8fafc" strokeWidth="6" strokeLinecap="round" />
          </g>
          <path d="M260 69 V252" stroke="rgba(255,255,255,.2)" strokeWidth="2" strokeDasharray="5 8" />
          <g ref={directionRef}>
            <path d="M385 148 H438" stroke="#f8fafc" strokeWidth="5" strokeLinecap="round" />
            <path d="M421 132 L439 148 L421 164" fill="none" stroke="#f8fafc" strokeWidth="5" strokeLinecap="round" strokeLinejoin="round" />
          </g>
        </g>

        <g ref={browserRef}>
          <rect x="68" y="70" width="384" height="270" rx="24" fill="rgba(13,14,14,.9)" stroke="rgba(255,255,255,.7)" strokeWidth="4" />
          <path d="M68 126 H452" stroke="rgba(255,255,255,.24)" strokeWidth="3" />
          <circle cx="96" cy="98" r="6" fill="rgba(255,255,255,.32)" />
          <circle cx="116" cy="98" r="6" fill="rgba(255,255,255,.32)" />
          <g opacity=".48">
            <rect x="148" y="84" width="112" height="29" rx="9" fill="rgba(255,255,255,.12)" />
            <text x="166" y="104" fill="#f8fafc" fontSize="13" fontFamily="Outfit, sans-serif">Uprightly</text>
          </g>
          <g ref={workTabRef}>
            <rect x="266" y="84" width="122" height="29" rx="9" fill="rgba(255,255,255,.9)" />
            <text x="285" y="104" fill="#171715" fontSize="13" fontWeight="600" fontFamily="Outfit, sans-serif">Your work</text>
            <rect x="103" y="158" width="220" height="18" rx="7" fill="rgba(255,255,255,.18)" />
            <rect x="103" y="190" width="280" height="12" rx="6" fill="rgba(255,255,255,.1)" />
            <rect x="103" y="216" width="247" height="12" rx="6" fill="rgba(255,255,255,.1)" />
            <rect x="103" y="256" width="112" height="48" rx="12" fill="rgba(255,255,255,.12)" />
          </g>
          <g ref={floatingRef} filter="url(#tutorial-guide-glow)">
            <rect x="318" y="224" width="166" height="108" rx="20" fill="#f4f0e8" />
            <circle cx="346" cy="252" r="9" fill="#718f73" />
            <text x="365" y="257" fill="#171715" fontSize="14" fontWeight="700" fontFamily="Outfit, sans-serif">Posture visible</text>
            <rect x="338" y="278" width="116" height="9" rx="4.5" fill="rgba(23,22,18,.18)" />
            <rect x="338" y="297" width="88" height="9" rx="4.5" fill="rgba(23,22,18,.1)" />
          </g>
        </g>
      </svg>
    </div>
  );
}

const TUTORIAL_ARTWORK: Record<
  Exclude<TutorialVisualKind, "floating">,
  { src: string; alt: string }
> = {
  baseline: {
    src: "/tutorial/uprightly-step-1.png",
    alt: "Person sitting upright and centered inside the camera frame",
  },
  forward: {
    src: "/tutorial/uprightly-step-2.png",
    alt: "Person gently leaning forward from an upright position",
  },
  shoulders: {
    src: "/tutorial/uprightly-step-3.png",
    alt: "Person raising one shoulder while remaining centered",
  },
  framing: {
    src: "/tutorial/uprightly-step-4.png",
    alt: "Person moving partly outside the camera frame",
  },
  neutral: {
    src: "/tutorial/uprightly-step-5.png",
    alt: "Person returning to a centered upright posture",
  },
};

function TutorialVisual({
  kind,
  detected,
}: {
  kind: TutorialVisualKind;
  detected: boolean;
}) {
  const artworkRef = useRef<HTMLDivElement | null>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);

  useGSAP(
    () => {
      const image = imageRef.current;
      if (!image || kind === "floating") return;

      const reduceMotion = window.matchMedia(
        "(prefers-reduced-motion: reduce)",
      ).matches;
      gsap.killTweensOf(image);
      gsap.set(image, { clearProps: "all" });
      gsap.fromTo(
        image,
        { autoAlpha: 0, scale: reduceMotion ? 1 : 0.94, y: reduceMotion ? 0 : 10 },
        {
          autoAlpha: 1,
          scale: 1,
          y: 0,
          duration: reduceMotion ? 0 : 0.55,
          ease: "power3.out",
        },
      );

      if (reduceMotion || detected) return;
      const movement =
        kind === "forward"
          ? { scale: 1.025, y: 4 }
          : kind === "shoulders"
            ? { rotation: -0.8, y: -2 }
            : kind === "framing"
              ? { x: 6 }
              : { y: -3 };
      const loop = gsap.to(image, {
        ...movement,
        duration: 1.25,
        delay: 0.65,
        ease: "sine.inOut",
        repeat: -1,
        yoyo: true,
      });
      return () => loop.kill();
    },
    { scope: artworkRef, dependencies: [kind, detected] },
  );

  if (kind === "floating") {
    return <BrowserTutorialVisual kind={kind} detected={detected} />;
  }

  const artwork = TUTORIAL_ARTWORK[kind];
  return (
    <div
      ref={artworkRef}
      data-tutorial-visual={kind}
      className="relative flex h-full min-h-[12rem] w-full items-center justify-center overflow-hidden"
    >
      <div className="pointer-events-none absolute inset-[12%] rounded-full bg-white/[0.04] blur-3xl" />
      <img
        ref={imageRef}
        src={artwork.src}
        alt={artwork.alt}
        draggable={false}
        className="relative h-full max-h-[31rem] w-full max-w-[42rem] select-none object-contain drop-shadow-[0_18px_42px_rgba(0,0,0,0.32)]"
      />
    </div>
  );
}

export function TutorialOverlay({
  step,
  stepIndex,
  stepCount,
  detected,
  floatingWindowSupported,
  isDarkTheme,
  onPrevious,
  onNext,
  onClose,
}: TutorialOverlayProps) {
  const overlayRef = useRef<HTMLElement | null>(null);
  const statusRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    overlayRef.current?.focus({ preventScroll: true });
  }, []);

  useGSAP(
    () => {
      const reduceMotion = window.matchMedia(
        "(prefers-reduced-motion: reduce)",
      ).matches;
      const copy = overlayRef.current?.querySelector("[data-tutorial-copy]");
      if (!copy) return;

      gsap.fromTo(
        copy,
        { autoAlpha: 0, y: reduceMotion ? 0 : 14 },
        {
          autoAlpha: 1,
          y: 0,
          duration: reduceMotion ? 0 : 0.42,
          ease: "power3.out",
        },
      );
    },
    { scope: overlayRef, dependencies: [stepIndex] },
  );

  useGSAP(
    () => {
      const status = statusRef.current;
      if (!status) return;
      gsap.killTweensOf(status);
      gsap.set(status, { clearProps: "all" });
      if (!detected) return;

      const reduceMotion = window.matchMedia(
        "(prefers-reduced-motion: reduce)",
      ).matches;
      gsap.fromTo(
        status,
        { scale: reduceMotion ? 1 : 0.96, autoAlpha: reduceMotion ? 1 : 0.65 },
        {
          scale: 1,
          autoAlpha: 1,
          duration: reduceMotion ? 0 : 0.42,
          ease: "back.out(1.55)",
        },
      );
    },
    { scope: overlayRef, dependencies: [detected, stepIndex] },
  );

  return (
    <aside
      ref={overlayRef}
      role="dialog"
      aria-modal="true"
      aria-label="Tutorial"
      aria-describedby="tutorial-instruction tutorial-result"
      tabIndex={-1}
      data-testid="tutorial-overlay"
      className="absolute inset-0 z-30 flex overflow-hidden rounded-[inherit] bg-[linear-gradient(120deg,rgba(8,9,9,.62),rgba(13,14,14,.88)_56%,rgba(13,14,14,.97))] text-[#f4f0e8] shadow-2xl outline-none backdrop-blur-[1.5px]"
    >
      <div className="flex min-h-0 w-full flex-col p-5 sm:p-6 lg:p-7">
        <header className="flex flex-none items-start gap-5">
          <div className="min-w-0 flex-1">
            <div className="flex items-baseline gap-3">
              <p className="text-xs font-bold uppercase tracking-[0.18em] text-white/80">Tutorial</p>
              <p className="text-xs font-medium text-white/60">Step {stepIndex + 1} of {stepCount}</p>
            </div>
            <div className="mt-4 flex gap-2" aria-label="Tutorial progress">
              {Array.from({ length: stepCount }, (_, index) => (
                <span
                  key={index}
                  className={`h-1 flex-1 rounded-full transition-colors duration-300 ${
                    index <= stepIndex ? "bg-[#f4f0e8]" : "bg-white/16"
                  }`}
                />
              ))}
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="End tutorial"
            className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl border border-white/14 bg-black/15 text-white/65 transition-[background-color,color,transform] hover:-translate-y-0.5 hover:bg-white/10 hover:text-white"
          >
            <X size={19} aria-hidden="true" />
          </button>
        </header>

        <div className="mt-4 grid min-h-0 flex-1 grid-flow-dense grid-cols-1 overflow-hidden xl:grid-cols-12">
          <section className="relative col-span-1 min-h-[12rem] overflow-hidden xl:col-span-7" aria-label="Movement demonstration">
            <TutorialVisual kind={step.id} detected={detected} />
          </section>

          <section className="col-span-1 flex min-h-0 flex-col border-t border-white/12 px-1 pt-5 xl:col-span-5 xl:border-l xl:border-t-0 xl:pl-8 xl:pt-2">
            <div key={step.id} data-tutorial-copy className="min-h-0 overflow-y-auto pr-2">
              <h2 className="max-w-2xl text-[clamp(1.55rem,2.25vw,2.5rem)] font-bold leading-[1.02] tracking-[-0.045em]">
                {step.title}
              </h2>

              <p id="tutorial-instruction" className="mt-5 max-w-xl text-[clamp(.95rem,1.25vw,1.1rem)] leading-7 text-white/80">
                {step.instruction}
              </p>

              <div
                ref={statusRef}
                role="status"
                aria-live="polite"
                data-tutorial-detected={detected ? "true" : "false"}
                className={`mt-6 flex min-h-[4.5rem] items-center gap-3 rounded-2xl border px-4 py-3 transition-[background-color,border-color,box-shadow] duration-300 ${
                  detected
                    ? "border-[#4ade80]/55 bg-[#123b2a]/90 shadow-[0_12px_34px_-18px_rgba(74,222,128,0.75)]"
                    : "border-white/12 bg-white/[0.045]"
                }`}
              >
                <span
                  className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border ${
                    detected
                      ? "border-[#86efac]/55 bg-[#22c55e] text-[#092116]"
                      : "border-white/12 bg-white/[0.045] text-white/55"
                  }`}
                  aria-hidden="true"
                >
                  {detected ? (
                    <Check size={21} strokeWidth={3.2} />
                  ) : (
                    <span className="h-2.5 w-2.5 rounded-full bg-current" />
                  )}
                </span>
                <span className="min-w-0">
                  <span
                    className={`block text-xs font-bold uppercase tracking-[0.13em] ${
                      detected ? "text-[#bbf7d0]" : "text-white/70"
                    }`}
                  >
                    {detected ? "Detected" : "Waiting for movement"}
                  </span>
                  <span
                    className={`mt-1 block text-sm font-medium ${
                      detected ? "text-[#dcfce7]" : "text-white/58"
                    }`}
                  >
                    {detected
                      ? "Movement recognized. You can continue when ready."
                      : "Follow the illustrated posture and hold briefly."}
                  </span>
                </span>
              </div>

              <div className="mt-6 border-l-2 border-white/28 pl-4">
                <p className="text-[10px] font-bold uppercase tracking-[0.15em] text-white/65">Look for</p>
                <p id="tutorial-result" className="mt-2 text-sm font-medium leading-6 text-white/85">{step.expected}</p>
              </div>

              {step.id === "floating" && !floatingWindowSupported ? (
                <p className="mt-5 rounded-xl border border-white/10 bg-black/[0.18] px-4 py-3 text-xs leading-5 text-white/58">
                  Automatic floating status is unavailable in this browser. You can still use the manual floating-window control in Settings when supported.
                </p>
              ) : null}
            </div>

            <div className="mt-auto flex-none pt-6">
              <div className="grid grid-cols-2 gap-3">
                <button
                  type="button"
                  onClick={onPrevious}
                  disabled={stepIndex === 0}
                  className="flex min-h-12 items-center justify-center gap-2 rounded-xl border border-white/18 px-4 text-sm font-semibold text-white/78 transition-[background-color,color,transform] hover:-translate-y-0.5 hover:bg-white/[0.08] hover:text-white disabled:cursor-not-allowed disabled:opacity-30 disabled:hover:translate-y-0"
                >
                  <ChevronLeft size={17} aria-hidden="true" />
                  Previous
                </button>
                <button
                  type="button"
                  onClick={onNext}
                  className={`flex min-h-12 items-center justify-center gap-2 rounded-xl px-4 text-sm font-semibold transition-transform hover:-translate-y-0.5 ${
                    isDarkTheme
                      ? "bg-[#f4f0e8] text-[#171612]"
                      : "bg-[#0A3A72] text-white"
                  }`}
                >
                  {stepIndex === stepCount - 1 ? "Finish Tutorial" : "Next"}
                  {stepIndex < stepCount - 1 ? <ChevronRight size={17} aria-hidden="true" /> : null}
                </button>
              </div>
              <p className="mt-3 text-center text-[10px] font-medium tracking-[0.04em] text-white/60">
                Space or → next · ← previous · Esc close
              </p>
            </div>
          </section>
        </div>
      </div>
    </aside>
  );
}
