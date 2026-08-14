/**
 * Formats a 0..1 fraction as a percentage string, stepping down to more
 * decimal places for small values so a real-but-tiny probability (e.g. a
 * long tail token at 0.06%) doesn't collapse to a misleading "0.0%".
 */
export function formatPercent(fraction: number): string {
  const pct = fraction * 100;
  const abs = Math.abs(pct);
  if (abs === 0) return "0%";
  if (abs >= 0.1) return `${pct.toFixed(1)}%`;
  if (abs >= 0.01) return `${pct.toFixed(2)}%`;
  if (abs >= 0.001) return `${pct.toFixed(3)}%`;
  if (abs >= 0.0001) return `${pct.toFixed(4)}%`;
  return `${pct.toPrecision(1)}%`;
}
