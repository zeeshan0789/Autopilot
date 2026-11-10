export const int = (n) => (n == null ? "n/a" : Math.round(n).toLocaleString("en-US"));

export const pct = (ratio, digits = 1) => (ratio == null ? "n/a" : `${(ratio * 100).toFixed(digits)}%`);

export const ms = (n) => (n == null ? "n/a" : `${Math.round(n).toLocaleString("en-US")} ms`);

// Costs here are tiny, so they need more digits than normal money
export function usd(n) {
  if (n == null) return "n/a";
  if (n === 0) return "$0";
  if (Math.abs(n) < 0.01) return `$${n.toFixed(6)}`;
  return `$${n.toFixed(4)}`;
}

export function shortTime(value, bucket = "hour") {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  const day = date.toLocaleDateString("en-GB", { day: "2-digit", month: "short" });
  if (bucket === "day") return day;
  return `${day} ${date.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" })}`;
}