import { TIER_LABEL, TIER_STEP } from "../lib/colors";

/** Tier is ordinal, so the dot walks one blue ramp; the label always says the word. */
export function TierBadge({ tier }: { tier: string }) {
  return (
    <span className="tier" title={tier}>
      <span className="dot" style={{ background: TIER_STEP[tier] ?? "#c3c2b7" }} />
      {TIER_LABEL[tier] ?? tier}
    </span>
  );
}
