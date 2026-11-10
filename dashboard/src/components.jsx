export const RANGES = [
  { label: "Last hour", hours: 1 },
  { label: "Last 24 hours", hours: 24 },
  { label: "Last 7 days", hours: 168 },
  { label: "Last 30 days", hours: 720 },
];

export const inputClass =
  "w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500";

export function Card({ title, subtitle, actions, children }) {
  return (
    <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
      {(title || actions) && (
        <div className="mb-4 flex items-start justify-between gap-4">
          <div>
            {title && <h2 className="text-sm font-semibold text-slate-900">{title}</h2>}
            {subtitle && <p className="mt-0.5 text-xs text-slate-500">{subtitle}</p>}
          </div>
          {actions}
        </div>
      )}
      {children}
    </section>
  );
}

export function Stat({ label, value, hint }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="text-xs font-medium uppercase tracking-wide text-slate-500">{label}</div>
      <div className="mt-1 text-2xl font-semibold text-slate-900">{value}</div>
      {hint && <div className="mt-1 text-xs text-slate-500">{hint}</div>}
    </div>
  );
}

export function ErrorBox({ error }) {
  if (!error) return null;
  return (
    <div role="alert" className="rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
      {error.message ?? String(error)}
    </div>
  );
}

export function Loading({ label = "Loading..." }) {
  return <p className="text-sm text-slate-500">{label}</p>;
}

export function RangeSelect({ hours, onChange }) {
  return (
    <select
      value={hours}
      onChange={(event) => onChange(Number(event.target.value))}
      className="rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-sm"
    >
      {RANGES.map((range) => (
        <option key={range.hours} value={range.hours}>
          {range.label}
        </option>
      ))}
    </select>
  );
}

const BUTTONS = {
  primary: "bg-indigo-600 text-white hover:bg-indigo-500",
  secondary: "border border-slate-300 bg-white text-slate-700 hover:bg-slate-50",
  danger: "bg-rose-600 text-white hover:bg-rose-500",
};

export function Button({ variant = "primary", className = "", ...props }) {
  return (
    <button
      {...props}
      className={`rounded-lg px-3 py-1.5 text-sm font-medium disabled:cursor-not-allowed disabled:opacity-50 ${BUTTONS[variant]} ${className}`}
    />
  );
}

const TONES = {
  green: "bg-emerald-50 text-emerald-700",
  red: "bg-rose-50 text-rose-700",
  amber: "bg-amber-50 text-amber-700",
  indigo: "bg-indigo-50 text-indigo-700",
  slate: "bg-slate-100 text-slate-600",
};

export function Badge({ tone = "slate", children }) {
  return <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${TONES[tone]}`}>{children}</span>;
}

export const cacheTone = (status) =>
  ({ hit: "green", miss: "slate", blocked: "amber", bypass: "indigo" })[status] ?? "slate";