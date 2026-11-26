import { useState } from "react";
import { api } from "../api.js";
import { Badge, Button, Card, ErrorBox, Loading } from "../components.jsx";
import { int } from "../format.js";
import { useFetch } from "../useFetch.js";

export default function Users() {
  const users = useFetch("/api/admin/users");
  const [error, setError] = useState(null);

  async function change(target, changes) {
    setError(null);
    try {
      await api(`/api/admin/users/${target.id}`, { method: "PATCH", body: changes });
      users.reload();
    } catch (e) {
      setError(e);
    }
  }

  function toggle(target) {
    const next = target.status === "active" ? "suspended" : "active";
    if (next === "suspended" && !window.confirm(`Suspend ${target.email}? Their login and all their keys stop working at once.`)) return;
    change(target, { status: next });
  }

  function setLimit(target, field, label) {
    const answer = window.prompt(`${label} for ${target.email}`, String(target[field]));
    if (answer === null) return;
    const value = Number(answer);
    if (!Number.isInteger(value) || value < 0) {
      setError(new Error("Enter a whole number."));
      return;
    }
    change(target, { [field]: value });
  }

  return (
    <>
      <h1 className="text-xl font-bold text-slate-900">Users</h1>
      <p className="text-sm text-slate-600">
        People who signed up are members. The role of an account cannot be changed here: administrators are created with the command line.
      </p>
      <ErrorBox error={error ?? users.error} />
      {users.loading && !users.data && <Loading />}

      {users.data && (
        <Card>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="text-slate-500">
                <tr>
                  {["Email", "Role", "Status", "Requests a day", "Max tokens", "Keys", "Prompts", "Requests (24 h)", "Actions"].map((h) => (
                    <th key={h} className="whitespace-nowrap px-2 py-2 font-medium">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {users.data.users.map((u) => (
                  <tr key={u.id}>
                    <td className="px-2 py-2 font-medium">{u.email}</td>
                    <td className="px-2 py-2">{u.role === "admin" ? <Badge tone="indigo">admin</Badge> : <Badge>member</Badge>}</td>
                    <td className="px-2 py-2">{u.status === "active" ? <Badge tone="green">active</Badge> : <Badge tone="red">suspended</Badge>}</td>
                    <td className="px-2 py-2">{u.role === "admin" ? "no limit" : int(u.daily_request_limit)}</td>
                    <td className="px-2 py-2">{u.role === "admin" ? "no limit" : int(u.max_tokens_cap)}</td>
                    <td className="px-2 py-2">{u.active_keys}</td>
                    <td className="px-2 py-2">{u.prompts}</td>
                    <td className="px-2 py-2">{int(u.requests_24h)}</td>
                    <td className="space-x-2 whitespace-nowrap px-2 py-2">
                      {u.role === "member" ? (
                        <>
                          <Button variant={u.status === "active" ? "danger" : "secondary"} onClick={() => toggle(u)}>
                            {u.status === "active" ? "Suspend" : "Activate"}
                          </Button>
                          <Button variant="secondary" onClick={() => setLimit(u, "daily_request_limit", "Requests a day")}>
                            Daily limit
                          </Button>
                          <Button variant="secondary" onClick={() => setLimit(u, "max_tokens_cap", "Max tokens per answer")}>
                            Max tokens
                          </Button>
                        </>
                      ) : (
                        <span className="text-slate-400">command line</span>
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