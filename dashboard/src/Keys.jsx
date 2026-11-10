import { useState } from "react";
import { api } from "./api.js";
import { Badge, Button, Card, ErrorBox, Loading, inputClass } from "./components.jsx";
import { useFetch } from "./useFetch.js";

function NewKeyBanner({ created, onDismiss }) {
  const [copied, setCopied] = useState(false);

  async function copy() {
    try {
      await navigator.clipboard.writeText(created.key);
      setCopied(true);
    } catch {
      setCopied(false);
    }
  }

  return (
    <div className="rounded-xl border border-amber-300 bg-amber-50 p-4">
      <p className="text-sm font-semibold text-amber-900">Key "{created.name}" created. Copy it now, it is shown only once.</p>
      <p className="mt-1 text-xs text-amber-800">Only its hash is stored. If you lose it, create a new key.</p>
      <div className="mt-3 flex flex-wrap items-center gap-3">
        <code className="break-all rounded bg-white px-3 py-2 text-sm">{created.key}</code>
        <Button variant="secondary" onClick={copy}>
          {copied ? "Copied" : "Copy"}
        </Button>
        <Button variant="secondary" onClick={onDismiss}>
          I have saved it
        </Button>
      </div>
    </div>
  );
}

export default function Keys() {
  const keys = useFetch("/api/keys");
  const [name, setName] = useState("");
  const [limit, setLimit] = useState("60");
  const [created, setCreated] = useState(null);
  const [error, setError] = useState(null);

  async function create(event) {
    event.preventDefault();
    setError(null);
    try {
      const result = await api("/api/keys", { method: "POST", body: { name, rate_limit_per_min: Number(limit) } });
      setCreated(result);
      setName("");
      keys.reload();
    } catch (e) {
      setError(e);
    }
  }

  async function revoke(key) {
    if (!window.confirm(`Revoke "${key.name}"? Apps using it will get 401 within about 10 seconds.`)) return;
    setError(null);
    try {
      await api(`/api/keys/${key.id}/revoke`, { method: "POST" });
      keys.reload();
    } catch (e) {
      setError(e);
    }
  }

  async function changeLimit(key) {
    const answer = window.prompt("New limit (requests per minute)", String(key.rate_limit_per_min));
    if (answer === null) return;
    setError(null);
    try {
      await api(`/api/keys/${key.id}`, { method: "PATCH", body: { rate_limit_per_min: Number(answer) } });
      keys.reload();
    } catch (e) {
      setError(e);
    }
  }

  return (
    <>
      <h1 className="text-xl font-bold text-slate-900">API keys</h1>

      {created && <NewKeyBanner created={created} onDismiss={() => setCreated(null)} />}

      <Card title="New key" subtitle="Keys are for the gateway (port 8000). The gateway checks them in Postgres and remembers answers for 10 seconds.">
        <form onSubmit={create} className="flex flex-wrap items-end gap-3">
          <label className="text-xs text-slate-600">
            Name
            <input className={`${inputClass} mt-1 w-56`} value={name} onChange={(e) => setName(e.target.value)} placeholder="my app" required />
          </label>
          <label className="text-xs text-slate-600">
            Requests per minute
            <input className={`${inputClass} mt-1 w-40`} type="number" min="1" value={limit} onChange={(e) => setLimit(e.target.value)} required />
          </label>
          <Button type="submit">Create key</Button>
        </form>
      </Card>

      <ErrorBox error={error ?? keys.error} />
      {keys.loading && !keys.data && <Loading />}

      {keys.data && (
        <Card title="Keys">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="text-slate-500">
                <tr>
                  {["Name", "Id", "Limit per minute", "Status", "Created", "Actions"].map((h) => (
                    <th key={h} className="px-2 py-2 font-medium">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {keys.data.keys.map((key) => (
                  <tr key={key.id}>
                    <td className="px-2 py-2 font-medium">{key.name}</td>
                    <td className="px-2 py-2 font-mono">{key.key_id}</td>
                    <td className="px-2 py-2">{key.rate_limit_per_min}</td>
                    <td className="px-2 py-2">
                      {key.active ? <Badge tone="green">active</Badge> : <Badge tone="red">revoked</Badge>}
                    </td>
                    <td className="px-2 py-2">{new Date(key.created_at).toLocaleString()}</td>
                    <td className="space-x-2 px-2 py-2">
                      {key.active && (
                        <>
                          <Button variant="secondary" onClick={() => changeLimit(key)}>
                            Change limit
                          </Button>
                          <Button variant="danger" onClick={() => revoke(key)}>
                            Revoke
                          </Button>
                        </>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </>
  );
}