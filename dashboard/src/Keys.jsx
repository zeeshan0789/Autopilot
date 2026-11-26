import { useState } from "react";
import { api, GATEWAY_URL } from "./api.js";
import { useAuth } from "./auth.jsx";
import { Badge, Button, Card, ErrorBox, Loading, inputClass } from "./components.jsx";
import { CopyButton, Snippet } from "./CopyButton.jsx";
import { int } from "./format.js";
import { powershellSnippet } from "./snippets.js";
import { useFetch } from "./useFetch.js";

const MEMBER_MAX_PER_MINUTE = 120; // the web API enforces the same limit

function NewKeyBanner({ created, onDismiss }) {
  return (
    <div className="space-y-3 rounded-xl border border-amber-300 bg-amber-50 p-4">
      <div>
        <p className="text-sm font-semibold text-amber-900">Key "{created.name}" created. Copy it now, it is shown only once.</p>
        <p className="mt-1 text-xs text-amber-800">Only its hash is stored. If you lose it, create a new key. Never put a key in a web page or in a public repository.</p>
      </div>
      <div className="flex flex-wrap items-center gap-3">
        <code className="break-all rounded bg-white px-3 py-2 text-sm">{created.key}</code>
        <CopyButton text={created.key} />
        <Button variant="secondary" onClick={onDismiss}>
          I have saved it
        </Button>
      </div>
      <div>
        <p className="mb-2 text-xs text-amber-900">Use it like this (put your key where it says ap-YOUR-KEY):</p>
        <Snippet text={powershellSnippet(GATEWAY_URL)} />
      </div>
    </div>
  );
}

export default function Keys() {
  const { user, isAdmin } = useAuth();
  const keys = useFetch("/api/keys");
  const [name, setName] = useState("");
  const [limit, setLimit] = useState("60");
  const [created, setCreated] = useState(null);
  const [error, setError] = useState(null);
  const maxPerMinute = isAdmin ? 100000 : MEMBER_MAX_PER_MINUTE;

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
    const answer = window.prompt(`New limit (requests per minute, up to ${maxPerMinute})`, String(key.rate_limit_per_min));
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

      <Card
        title="New key"
        subtitle={
          isAdmin
            ? "Keys are for the gateway (port 8000). The gateway checks them in Postgres and remembers answers for 10 seconds."
            : `A key can send up to ${MEMBER_MAX_PER_MINUTE} requests a minute. Your account can send ${int(user.daily_request_limit)} requests a day with all keys together.`
        }
      >
        <form onSubmit={create} className="flex flex-wrap items-end gap-3">
          <label className="text-xs text-slate-600">
            Name
            <input className={`${inputClass} mt-1 w-56`} value={name} onChange={(e) => setName(e.target.value)} placeholder="my app" required />
          </label>
          <label className="text-xs text-slate-600">
            Requests per minute
            <input className={`${inputClass} mt-1 w-40`} type="number" min="1" max={maxPerMinute} value={limit} onChange={(e) => setLimit(e.target.value)} required />
          </label>
          <Button type="submit">Create key</Button>
        </form>
      </Card>

      <ErrorBox error={error ?? keys.error} />
      {keys.loading && !keys.data && <Loading />}

      {keys.data && (
        <Card title="Your keys">
          {keys.data.keys.length === 0 ? (
            <p className="text-sm text-slate-500">You have no keys yet. Create one above.</p>
          ) : (
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
                      <td className="px-2 py-2">{key.active ? <Badge tone="green">active</Badge> : <Badge tone="red">revoked</Badge>}</td>
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
          )}
        </Card>
      )}
    </>
  );
}