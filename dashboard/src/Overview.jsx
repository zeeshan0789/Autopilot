import { useState } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Card, ErrorBox, Loading, RangeSelect, Stat } from "./components.jsx";
import { int, ms, pct, shortTime, usd } from "./format.js";
import TraceTable from "./TraceTable.jsx";
import { useFetch } from "./useFetch.js";

const BREAKDOWNS = [
  { value: "route", label: "Routing rule" },
  { value: "provider", label: "Provider" },
  { value: "model", label: "Served model" },
  { value: "cache", label: "Cache result" },
  { value: "status", label: "Status" },
];

function Chart({ children, height = 260 }) {
  return (
    <div style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        {children}
      </ResponsiveContainer>
    </div>
  );
}

export default function Overview() {
  const [hours, setHours] = useState(24);
  const [by, setBy] = useState("route");
  const bucket = hours <= 48 ? "hour" : "day";

  const overview = useFetch("/api/stats/overview", { hours });
  const series = useFetch("/api/stats/timeseries", { hours, bucket });
  const breakdown = useFetch("/api/stats/breakdown", { hours, by });
  const recent = useFetch("/api/traces", { limit: 8 });

  const o = overview.data;
  const points = (series.data?.points ?? []).map((p) => ({ ...p, label: shortTime(p.bucket, bucket) }));

  return (
    <>
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-bold text-slate-900">Overview</h1>
        <RangeSelect hours={hours} onChange={setHours} />
      </div>

      <ErrorBox error={overview.error} />
      {overview.loading && !o && <Loading />}

      {o && (
        <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
          <Stat label="Requests" value={int(o.requests)} hint={`${int(o.fallbacks)} used a fallback`} />
          <Stat label="Error rate" value={pct(o.error_rate)} hint={`${int(o.errors)} errors`} />
          <Stat
            label="Cache hit rate"
            value={pct(o.cache.hit_rate)}
            hint={`${int(o.cache.hits)} of ${int(o.cache.consulted)} looked up, ${int(o.cache.blocked)} blocked by the guard, ${int(o.cache.bypass)} bypassed`}
          />
          <Stat
            label="Latency (provider calls)"
            value={ms(o.latency_ms.avg)}
            hint={`p95 ${ms(o.latency_ms.p95)}, cache hits ${ms(o.latency_ms.avg_cache_hit)}`}
          />
          <Stat label="Estimated cost spent" value={usd(o.cost.spent_usd)} hint="at list prices, not real spend" />
          <Stat
            label="Estimated cost saved"
            value={usd(o.cost.saved_usd)}
            hint={`${pct(o.cost.saving_pct)} of what the same traffic would have cost`}
          />
          <Stat label="Tokens saved" value={int(o.tokens.saved)} hint={`${int(o.tokens.used)} tokens used`} />
        </div>
      )}

      <Card title="Requests over time" subtitle={`One point per ${bucket}`}>
        <ErrorBox error={series.error} />
        {points.length === 0 && !series.loading ? (
          <p className="text-sm text-slate-500">No traffic in this period.</p>
        ) : (
          <Chart>
            <AreaChart data={points}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="label" tick={{ fontSize: 11 }} />
              <YAxis allowDecimals={false} tick={{ fontSize: 11 }} />
              <Tooltip />
              <Legend />
              <Area type="monotone" dataKey="requests" name="Requests" stroke="#4f46e5" fill="#c7d2fe" />
              <Area type="monotone" dataKey="cache_hits" name="Cache hits" stroke="#059669" fill="#a7f3d0" />
              <Area type="monotone" dataKey="errors" name="Errors" stroke="#e11d48" fill="#fecdd3" />
            </AreaChart>
          </Chart>
        )}
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="Estimated cost" subtitle="At published list prices. The real spend is $0 (free tiers and a local model).">
          {points.length === 0 ? (
            <p className="text-sm text-slate-500">No traffic in this period.</p>
          ) : (
            <Chart>
              <LineChart data={points}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="label" tick={{ fontSize: 11 }} />
                <YAxis tickFormatter={usd} tick={{ fontSize: 11 }} width={80} />
                <Tooltip formatter={(value) => usd(value)} />
                <Legend />
                <Line type="monotone" dataKey="cost_spent_usd" name="Spent" stroke="#4f46e5" dot={false} />
                <Line type="monotone" dataKey="cost_saved_usd" name="Saved by the cache" stroke="#059669" dot={false} />
              </LineChart>
            </Chart>
          )}
        </Card>

        <Card
          title="Requests by"
          actions={
            <select
              value={by}
              onChange={(event) => setBy(event.target.value)}
              className="rounded-lg border border-slate-300 bg-white px-2 py-1 text-xs"
            >
              {BREAKDOWNS.map((b) => (
                <option key={b.value} value={b.value}>
                  {b.label}
                </option>
              ))}
            </select>
          }
        >
          <ErrorBox error={breakdown.error} />
          {(breakdown.data?.items ?? []).length === 0 && !breakdown.loading ? (
            <p className="text-sm text-slate-500">No traffic in this period.</p>
          ) : (
            <Chart>
              <BarChart data={breakdown.data?.items ?? []} layout="vertical" margin={{ left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis type="number" allowDecimals={false} tick={{ fontSize: 11 }} />
                <YAxis type="category" dataKey="name" width={150} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="requests" name="Requests" fill="#4f46e5" />
              </BarChart>
            </Chart>
          )}
        </Card>
      </div>

      <Card title="Latest requests">
        <ErrorBox error={recent.error} />
        <TraceTable traces={recent.data?.traces ?? []} />
      </Card>
    </>
  );
}