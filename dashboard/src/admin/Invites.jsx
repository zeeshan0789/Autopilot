import { useState } from "react";
import { api } from "../api.js";
import { Badge, Button, Card, ErrorBox, Loading, inputClass } from "../components.jsx";
import { CopyButton } from "../CopyButton.jsx";
import { useFetch } from "../useFetch.js";

function NewInviteBanner({ created, onDismiss }) {
  const link = `${window.location.origin}/signup?invite=${created.code}`;
  return (
    <div className="space-y-3 rounded-xl border border-amber-300 bg-amber-50 p-4">
      <div>
        <p className="text-sm font-semibold text-amber-900">Invite created. Copy it now, the code is shown only once.</p>
        <p className="mt-1 text-xs text-amber-800">It works for one signup. Only its hash is stored.</p>
      </div>
      <div className="flex flex-wrap items-center gap-3">
        <code className="break-all rounded bg-white px-3 py-2 text-sm">{created.code}</code>
        <CopyButton text={created.code} label="Copy code" />
      </div>
      <div className="flex flex-wrap items-center gap-3">
        <code className="break-all rounded bg-white px-3 py-2 text-xs">{link}</code>
        <CopyButton text={link} label="Copy link" />
        <Button variant="secondary" onClick={onDismiss}>
          I have saved it
        </Button>
      </div>
    </div>
  );
}

export default function Invites() {
  const invites = useFetch("/api/admin/invites");
  const [note, setNote] = useState("");
  const [created, setCreated] = useState(null);
  const [error, setError] = useState(null);

  async function create(event) {
    event.preventDefault();
    setError(null);
    try {
      const result = await api("/api/admin/invites", { method: "POST", body: note.trim() ? { note: note.trim() } : {} });
      setCreated(result);
      setNote("");
      invites.reload();
    } catch (e) {
      setError(e);
    }
  }

  async function remove(invite) {
    if (!window.confirm("Delete this invite? Its code will stop working.")) return;
    setError(null);
    try {
      await api(`/api/admin/invites/${invite.id}`, { method: "DELETE" });
      invites.reload();
    } catch (e) {
      setError(e);
    }
  }

  return (
    <>
      <h1 className="text-xl font-bold text-slate-900">Invites</h1>
      {created && <NewInviteBanner created={created} onDismiss={() => setCreated(null)} />}

      <Card title="New invite" subtitle="Each code lets one person sign up. Signups from invite codes use up the free model quota of the platform, so invite people you know.">
        <form onSubmit={create} className="flex flex-wrap items-end gap-3">
          <label className="min-w-64 grow text-xs text-slate-600">
            Note (who is it for?)
            <input className={`${inputClass} mt-1`} value={note} onChange={(e) => setNote(e.target.value)} maxLength={200} placeholder="for Sara" />
          </label>
          <Button type="submit">Create invite</Button>
        </form>
      </Card>

      <ErrorBox error={error ?? invites.error} />
      {invites.loading && !invites.data && <Loading />}

      {invites.data && (
        <Card title="All invites">
          {invites.data.invites.length === 0 ? (
            <p className="text-sm text-slate-500">No invites yet.</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="text-slate-500">
                  <tr>
                    {["Note", "Created", "Status", "Actions"].map((h) => (
                      <th key={h} className="px-2 py-2 font-medium">
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {invites.data.invites.map((invite) => (
                    <tr key={invite.id}>
                      <td className="px-2 py-2">{invite.note ?? ""}</td>
                      <td className="px-2 py-2">{new Date(invite.created_at).toLocaleString()}</td>
                      <td className="px-2 py-2">
                        {invite.used_at ? <Badge tone="slate">used by {invite.used_by_email ?? "a deleted account"}</Badge> : <Badge tone="green">unused</Badge>}
                      </td>
                      <td className="px-2 py-2">
                        {!invite.used_at && (
                          <Button variant="danger" onClick={() => remove(invite)}>
                            Delete
                          </Button>
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