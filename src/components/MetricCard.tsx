import type { ComponentType } from "react";

type ThemeMode = "dark" | "light";

type MetricCardProps = {
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
};

export function MetricCard({
  paused,
  theme,
  label,
  value,
  unit,
  icon: Icon,
  colorClass,
}: MetricCardProps) {
  const isDarkTheme = theme === "dark";

  return (
    <article
      className={`group relative flex min-h-24 flex-col gap-1 overflow-hidden rounded-2xl border p-4 backdrop-blur-md transition duration-300 ease-out motion-reduce:transition-none ${
        isDarkTheme
          ? paused
            ? "border-white/10 bg-gradient-to-br from-white/10 to-transparent"
            : "border-white/10 bg-gradient-to-br from-white/10 via-white/[0.045] to-transparent hover:-translate-y-0.5 hover:bg-white/10"
          : paused
            ? "border-stone-200 bg-gradient-to-br from-white to-stone-50"
            : "border-stone-200 bg-gradient-to-br from-white to-stone-50 hover:-translate-y-0.5 hover:bg-white"
      }`}
    >
      <div
        className={`z-10 mb-1 flex items-center justify-between ${
          paused
            ? isDarkTheme
              ? "text-white/45"
              : "text-stone-400"
            : isDarkTheme
              ? "text-white/60"
              : "text-stone-500"
        }`}
      >
        <span className="text-xs font-medium tracking-wide">
          {label}
        </span>
        <Icon size={14} aria-hidden="true" />
      </div>
      <div className="z-10 mt-0.5 flex items-baseline gap-1">
        <span
          className={`text-2xl font-bold tracking-tight ${
            paused
              ? isDarkTheme
                ? "text-white/65"
                : "text-stone-500"
              : colorClass || (isDarkTheme ? "text-white" : "text-stone-900")
          }`}
        >
          {value}
        </span>
        <span
          className={`text-xs ${
            paused
              ? isDarkTheme
                ? "text-white/40"
                : "text-stone-400"
              : isDarkTheme
                ? "text-white/50"
                : "text-stone-500"
          }`}
        >
          {unit}
        </span>
      </div>
    </article>
  );
}
