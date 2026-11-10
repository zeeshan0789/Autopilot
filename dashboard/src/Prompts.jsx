import { useState } from "react";
import { api } from "./api.js";
import { useAuth } from "./auth.jsx";
import { Badge, Button, Card, ErrorBox, Loading, RangeSelect, inputClass } from "./components.jsx";
import { int, ms, pct, usd } from "./format.js";
import { useFetch } from "./useFetch.js";

function CreatePrompt({ onCreated }) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [error, setError] = useState(null);

  async function submit(event) {
    event.preventDefault();
    setError(null);
    try {
      await api("/api/prompts", { method: "POST", body: { name, description: description || undefined } });
      setName("");
      setDescription("");
      onCreated();
    } catch (e) {
      setError(e);
    }
  }

  return (
    <Card title="New prompt" subtitle="A prompt is a name with versions. Clients ask for it with autopilot.prompt in the request.">
      <form onSubmit={submit} className="flex flex-wrap items-end gap-3">
        <label className="text-xs text-slate-600">
          Name (lowercase, digits, underscores)
          <input className={`${inputClass} mt-1 w-56`} value={name} onChange={(e) => setName(e.target.value)} placeholder="support" required />
        </label>
        <label className="min-w-48 grow text-xs text-slate-600">
          Description
          <input className={`${inputClass} mt-1`} value={description} onChange={(e) => setDescription(e.target.value)} />
        </label>
        <Button type="submit">Create</Button>
      </form>
      <div className="mt-3">
        <ErrorBox error={error} />
      </div>
    </Card>
  );
}

const EMPTY_VERSION = { system_prompt: "", user_template: "", weight: "0", notes: "" };

function AddVersion({ name, onAdded }) {
  const [form, setForm] = useState(EMPTY_VERSION);
  const [error, setError] = useState(null);
  const set = (key) => (event) => setForm({ ...form, [key]: event.target.value });

  async function submit(event) {
    event.preventDefault();
    setError(null);
    try {
      await api(`/api/prompts/${encodeURIComponent(name)}/versions`, {
        method: "POST",
        body: {
          system_prompt: form.system_prompt,
          user_template: form.user_template || undefined,
          weight: Number(form.weight),
          notes: form.notes || undefined,
        },
      });
      setForm(EMPTY_VERSION);
      onAdded();
    } catch (e) {
      setError(e);
    }
  }

  return (
    <form onSubmit={submit} className="mt-4 space-y-3 rounded-lg bg-slate-50 p-4">
      <p className="text-xs text-slate-500">
        A version cannot be edited later (A/B numbers would stop meaning anything). A change is a new version. Use
        {" {{name}} "}placeholders, and {"{{input}}"} in the user template for the caller's last message.
      </p>
      <label className="block text-xs text-slate-600">
        System prompt
        <textarea className={`${inputClass} mt-1`} rows={3} value={form.system_prompt} onChange={set("system_prompt")} required />
      </label>
      <label className="block text-xs text-slate-600">
        User template (optional)
        <textarea className={`${inputClass} mt-1`} rows={2} value={form.user_template} onChange={set("user_template")} />
      </label>
      <div className="flex flex-wrap items-end gap-3">
        <label className="text-xs text-slate-600">
          Weight (share of traffic)
          <input className={`${inputClass} mt-1 w-32`} type="number" min="0" value={form.weight} onChange={set("weight")} />
        </label>
        <label className="min-w-48 grow text-xs text-slate-600">
          Notes
          <input className={`${inputClass} mt-1`} value={form.notes} onChange={set("notes")} />
        </label>
        <Button type="submit">Add version</Button>
      </div>
      <ErrorBox error={error} />
    </form>
  );
}

function PromptCard({ prompt, isAdmin, onChanged }) {
  const [weights, setWeights] = useState(() =>
    Object.fromEntries(prompt.versions.map((v) => [v.version, String(v.weight)])),
  );
  const [adding, setAdding] = useState(false);
  const [error, setError] = useState(null);
  const [saved, setSaved] = useState(false);

  const total = prompt.versions.reduce((sum, v) => sum + (Number(weights[v.version]) || 0), 0);

  async function saveWeights() {
    setError(null);
    setSaved(false);
    try {
      const body = Object.fromEntries(Object.entries(weights).map(([version, weight]) => [version, Number(weight)]));
      await api(`/api/prompts/${encodeURIComponent(prompt.name)}/weights`, { method: "PUT", body: { weights: body } });
      setSaved(true);
      onChanged();
    } catch (e) {
      setError(e);
    }
  }

  return (
    <Card
      title={prompt.name}
      subtitle={prompt.description || "No description"}
      actions={
        isAdmin && (
          <Button variant="secondary" onClick={() => setAdding(!adding)}>
            {adding ? "Close" : "Add version"}
          </Button>
        )
      }
    >
      {prompt.versions.length === 0 ? (
        <p className="text-sm text-slate-500">No versions yet.</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="text-slate-500">
              <tr>
                <th className="px-2 py-2 font-medium">Version</th>
                <th className="px-2 py-2 font-medium">Weight</th>
                <th className="px-2 py-2 font-medium">Share</th>
                <th className="px-2 py-2 font-medium">System prompt</th>
                <th className="px-2 py-2 font-medium">Notes</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {prompt.versions.map((v) => {
                const weight = Number(weights[v.version]) || 0;
                return (
                  <tr key={v.version}>
                    <td className="px-2 py-2 font-medium">v{v.version}</td>
                    <td className="px-2 py-2">
                      {isAdmin ? (
                        <input
                          type="number"
                          min="0"
                          className="w-20 rounded border border-slate-300 px-2 py-1"
                          value={weights[v.version]}
                          onChange={(e) => setWeights({ ...weights, [v.version]: e.target.value })}
                        />
                      ) : (
                        v.weight
                      )}
                    </td>
                    <td className="px-2 py-2">
                      {total > 0 && weight > 0 ? `${Math.round((weight / total) * 100)}%` : <Badge>not served</Badge>}
                    </td>
                    <td className="max-w-md truncate px-2 py-2" title={v.system_prompt}>
                      {v.system_prompt}
                    </td>
                    <td className="px-2 py-2">{v.notes ?? ""}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {prompt.versions.length > 0 && total === 0 && (
        <p className="mt-3 text-xs text-amber-700">All weights are 0: requests for this prompt get an error until one version has a weight.</p>
      )}

      {isAdmin && prompt.versions.length > 0 && (
        <div className="mt-3 flex items-center gap-3">
          <Button onClick={saveWeights}>Save weights</Button>
          {saved && <span className="text-xs text-emerald-700">Saved. The gateway sees it within 10 seconds.</span>}
        </div>
      )}
      <div className="mt-3">
        <ErrorBox error={error} />
      </div>

      {isAdmin && adding && (
        <AddVersion
          name={prompt.name}
          onAdded={() => {
            setAdding(false);
            onChanged();
          }}
        />
      )}
    </Card>
  );
}

export default function Prompts() {
  const { isAdmin } = useAuth();
  const [hours, setHours] = useState(24);
  const prompts = useFetch("/api/prompts");
  const stats = useFetch("/api/stats/prompts", { hours });

  return (
    <>
      <h1 className="text-xl font-bold text-slate-900">Prompts and A/B</h1>

      {isAdmin && <CreatePrompt onCreated={prompts.reload} />}

      <ErrorBox error={prompts.error} />
      {prompts.loading && !prompts.data && <Loading />}
      {(prompts.data?.prompts ?? []).map((prompt) => (
        <PromptCard
          key={`${prompt.name}:${prompt.versions.length}`}
          prompt={prompt}
          isAdmin={isAdmin}
          onChanged={prompts.reload}
        />
      ))}
      {prompts.data && prompts.data.prompts.length === 0 && (
        <p className="text-sm text-slate-500">No prompts yet.</p>
      )}

      <Card
        title="A/B comparison from real traffic"
        subtitle="Latency and tokens count provider calls only (not cache hits). Answer quality is not measured yet (phase 4), and small samples differ by chance."
        actions={<RangeSelect hours={hours} onChange={setHours} />}
      >
        <ErrorBox error={stats.error} />
        {(stats.data?.versions ?? []).length === 0 ? (
          <p className="text-sm text-slate-500">No prompt traffic in this period.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="text-slate-500">
                <tr>
                  {["Prompt", "Version", "Requests", "Errors", "Cache hit rate", "Avg latency", "Avg tokens", "Estimated cost"].map((h) => (
                    <th key={h} className="px-2 py-2 font-medium">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {stats.data.versions.map((v) => (
                  <tr key={`${v.prompt}:${v.version}`}>
                    <td className="px-2 py-2 font-medium">{v.prompt}</td>
                    <td className="px-2 py-2">v{v.version}</td>
                    <td className="px-2 py-2">{int(v.requests)}</td>
                    <td className="px-2 py-2">{int(v.errors)}</td>
                    <td className="px-2 py-2">{pct(v.cache_hit_rate)}</td>
                    <td className="px-2 py-2">{ms(v.avg_latency_ms)}</td>
                    <td className="px-2 py-2">{int(v.avg_tokens)}</td>
                    <td className="px-2 py-2">{usd(v.cost_spent_usd)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </>
  );
}