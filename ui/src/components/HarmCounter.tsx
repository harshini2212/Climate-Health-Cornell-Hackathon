import { useEffect, useRef, useState } from "react";
import type { BaselineResult } from "../lib/types";

const NAMES: Record<string, string> = {
  leeward: "Leeward",
  rank_by_age: "Oldest first",
  rank_by_chronic: "Most conditions first",
  random: "Random",
};

/**
 * Tween a number toward its target over ~600 ms with an ease-out curve. A timer
 * fallback lands on the target even when the tab is hidden and animation frames pause,
 * so the number on screen is never stale.
 */
function useTween(target: number, ms = 600): number {
  const [shown, setShown] = useState(target);
  const from = useRef(target);
  useEffect(() => {
    const start = performance.now();
    const a = from.current;
    let raf = 0;
    const step = (t: number) => {
      const p = Math.min(1, (t - start) / ms);
      const eased = 1 - Math.pow(1 - p, 3);
      setShown(a + (target - a) * eased);
      if (p < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    const settle = setTimeout(() => {
      cancelAnimationFrame(raf);
      from.current = target;
      setShown(target);
    }, ms + 50);
    return () => {
      cancelAnimationFrame(raf);
      clearTimeout(settle);
      from.current = target;
    };
  }, [target, ms]);
  return shown;
}

/**
 * The hero number (expected harm averted for today's list) and the baseline bars.
 * The bars are four nominal strategies, so they share one hue; Leeward is set apart by
 * weight and position, not by color.
 */
export function HarmCounter({ total, baselines }: { total: number; baselines: BaselineResult[] }) {
  const shown = useTween(total);
  const max = Math.max(total, ...baselines.map((b) => b.total_eha), 1e-9);
  const rows = [
    { name: "leeward", total_eha: total },
    ...baselines.filter((b) => b.name !== "leeward"),
  ];
  return (
    <div className="hero">
      <div className="value" aria-live="polite">{shown.toFixed(1)}</div>
      <div className="label">expected harm averted today, severity-weighted need-days</div>
      {baselines.length > 0 && (
        <div className="bars" style={{ marginTop: 12 }}>
          {rows.map((b) => (
            <div key={b.name} className={`bar-row${b.name === "leeward" ? " lead" : ""}`}>
              <span className="name">{NAMES[b.name] ?? b.name}</span>
              <div className="bar-track">
                <div className="bar-fill" style={{ width: `${(100 * b.total_eha) / max}%` }} />
              </div>
              <span className="num">{b.total_eha.toFixed(1)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
