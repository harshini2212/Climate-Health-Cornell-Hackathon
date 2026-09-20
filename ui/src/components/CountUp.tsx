import { useEffect, useRef, useState } from "react";

/**
 * A number that counts up to its target over ~900 ms with an ease-out curve. A timer
 * fallback lands on the target even when the tab is hidden and animation frames pause,
 * so the number on screen is never stale.
 */
export function useCountUp(target: number, ms = 900): number {
  const [shown, setShown] = useState(target);
  const from = useRef(0);
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

export function CountUp({ value, digits = 0 }: { value: number; digits?: number }) {
  const shown = useCountUp(value);
  return <span aria-live="polite">{digits ? shown.toFixed(digits) : Math.round(shown).toLocaleString("en-US")}</span>;
}
