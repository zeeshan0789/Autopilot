import { Badge, cacheTone } from "./components.jsx";
import { int, ms, usd } from "./format.js";

function models(t) {
  const routed = t.routed_model ?? t.requested_model;
  let text = t.requested_model;
  if (routed !== t.requested_model) text += ` \u2192 ${routed}`;
  if (t.served_model && t.served_model !== routed) text += ` \u2192 ${t.served_model}`;
  return text;
}

export default function TraceTable({ traces }) {
  if (traces.length === 0) return <p className="text-sm text-slate-500">No traces yet.</p>;
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-xs">
        <thead className="text-slate-500">
          <tr>
            {["Time", "Model (asked, routed, served)", "Provider", "Route", "Cache", "Status", "Tokens", "Cost", "Latency", "Prompt"].map(
              (heading) => (
                <th key={heading} className="whitespace-nowrap px-2 py-2 font-medium">
                  {heading}
                </th>
              ),
            )}
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-100">
          {traces.map((t) => (
            <tr key={t.request_id}>
              <td className="whitespace-nowrap px-2 py-2">{new Date(t.created_at).toLocaleString()}</td>
              <td className="px-2 py-2">{models(t)}</td>
              <td className="px-2 py-2">{t.provider ?? "none"}</td>
              <td className="px-2 py-2">{t.policy_rule ?? "n/a"}</td>
              <td className="px-2 py-2">
                {t.cache_status ? (
                  <span title={t.cache_reason ?? ""}>
                    <Badge tone={cacheTone(t.cache_status)}>{t.cache_status}</Badge>
                  </span>
                ) : (
                  "n/a"
                )}
              </td>
              <td className="px-2 py-2">
                <Badge tone={t.status === "success" ? "green" : "red"}>{t.http_status}</Badge>
                {t.fallback_used && <span className="ml-1 text-amber-600">fallback</span>}
              </td>
              <td className="px-2 py-2">{int(t.total_tokens)}</td>
              <td className="px-2 py-2">{t.estimated_cost_usd == null ? "n/a" : usd(Number(t.estimated_cost_usd))}</td>
              <td className="whitespace-nowrap px-2 py-2">{ms(t.latency_ms)}</td>
              <td className="px-2 py-2">{t.prompt_name ? `${t.prompt_name}:${t.prompt_version}` : ""}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}