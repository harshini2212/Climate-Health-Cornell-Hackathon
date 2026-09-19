import { useEffect, useState } from "react";

interface Props {
  value: number;
  min?: number;
  max?: number;
  /** Fires on release (pointer up, key up, or a change event), never on every tick. */
  onCommit: (calls: number) => void;
  disabled?: boolean;
}

/**
 * The demo's most persuasive three seconds: drag from 40 to 20 and the list re-ranks,
 * drag to 80 and the Find-out tier fills. Refetches only when the thumb is released.
 */
export function CapacitySlider({ value, min = 10, max = 100, onCommit, disabled }: Props) {
  const [local, setLocal] = useState(value);
  useEffect(() => setLocal(value), [value]);

  const commit = () => {
    if (local !== value) onCommit(local);
  };

  return (
    <div className="slider">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <span>Calls the team can make today</span>
        <span className="value">{local}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={1}
        value={local}
        disabled={disabled}
        aria-label="Daily call capacity"
        onChange={(e) => setLocal(Number(e.target.value))}
        onPointerUp={commit}
        onKeyUp={commit}
        onBlur={commit}
      />
      <div className="ticks">
        <span>{min}</span>
        <span>40</span>
        <span>{max}</span>
      </div>
    </div>
  );
}
