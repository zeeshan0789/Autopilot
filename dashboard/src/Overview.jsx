import { useState } from "react";
import { Link } from "react-router-dom";
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
import { GATEWAY_URL } from "./api.js";
import { useAuth } from "./auth.jsx";
import { Card, ErrorBox, Loading, RangeSelect, Stat } from "./components.jsx";
import { Snippet } from "./CopyButton.jsx";
import { int, ms, pct, shortTime, usd } from "./format.js";
import { powershellSnippet } from "./snippets.js";
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

function GetStarted() {
  const { user } = useAuth();
  const keys = useFetch("/api/keys");
  const hasKey = (keys.data?.keys ?? []).some((key) => key.active);

  return (
    <Card title="Get started" subtitle="Three steps and your first request shows up on this page.">
      <ol className="space-y-5 text-sm text-slate-700">
        <li>
          <div className="font-medium text-slate-900">1. Create an API key {hasKey && <span className="text-emerald-600">(you have one)</span>}</div>
          <p className="mt-1 text-slate-600">
            Keys are made on the{" "}
            <Link to="/keys" className="text-indigo-600 hover:underline">
              API keys page
            </Link>
            . A key is shown once, copy it somewhere safe.
          </p>
        </li>
        <li>
          <div className="font-medium text-slate-900">2. Send a request to the gateway</div>
          <p className="mb-2 mt-1 text-slate-600">Replace ap-YOUR-KEY with your key. The model name auto lets the gateway choose a model.</p>
          <Snippet text={powershellSnippet(GATEWAY_URL)} />
        </li>
        <li>
          <div className="font-medium text-slate-900">3. Come back and refresh</div>
          <p className="mt-1 text-slate-600">The request appears in the charts and in the latest requests below.</p>
        </li>
      </ol>
      {user.role !== "admin" && (
        <p className="mt-5 text-xs text-slate-500">
          Your account can send {int(user.daily_request_limit)} requests a day (all keys together) and ask for answers of up to{" "}
          {int(user.max_tokens_cap)} tokens.
        </p>
      )}
    </Card>
  );
}

function RecentRequests() {
  const recent = useFetch("/api/traces", { limit: 8 });
  return (
    <Card title="Latest requests">
      <ErrorBox error={recent.error} />
      <TraceTable traces={recent.data?.traces ?? []} />
    </Card>
  );
}

// scope "mine": the numbers of the logged in user. scope "platform": everybody (the admin pages only).
export default function Overview({ base = "/api/stats", title = "Overview", scope = "mine" }) {
  const [hours, setHours] = useState(24);
  const [by, setBy] = useState("route");
  const bucket = hours <= 48 ? "hour" : "day";

  const overview = useFetch(`${base}/overview`, { hours });
  const series = useFetch(`${base}/timeseries`, { hours, bucket });
  const breakdown = useFetch(`${base}/breakdown`, { hours, by });

  const o = overview.data;
  const points = (series.data?.points ?? []).map((p) => ({ ...p, label: shortTime(p.bucket, bucket) }));

  return (
    <>
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-bold text-slate-900">{title}</h1>
        <RangeSelect hours={hours} onChange={setHours} />
      </div>

      <ErrorBox error={overview.error} />
      {overview.loading && !o && <Loading />}

      {o && o.requests === 0 && scope === "mine" && <GetStarted />}

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

      {scope === "mine" && <RecentRequests />}
    </>
  );
}