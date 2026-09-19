import { useEffect, useRef, useState } from "react";

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

/** The hero number: expected harm averted for today's list, animated between values. */
export function HarmCounter({ total }: { total: number }) {
  const shown = useTween(total);
  return <span aria-live="polite">{shown.toFixed(1)}</span>;
}
