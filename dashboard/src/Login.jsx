import { useState } from "react";
import { Navigate, useNavigate } from "react-router-dom";
import { useAuth } from "./auth.jsx";
import { Button, ErrorBox, inputClass } from "./components.jsx";

export default function Login() {
  const { user, login } = useAuth();
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  if (user) return <Navigate to="/" replace />;

  async function submit(event) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await login(email, password);
      navigate("/", { replace: true });
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center p-6">
      <form onSubmit={submit} className="w-full max-w-sm space-y-4 rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
        <div>
          <h1 className="text-xl font-bold text-slate-900">Autopilot</h1>
          <p className="text-sm text-slate-500">Sign in to the dashboard</p>
        </div>
        <label className="block text-xs text-slate-600">
          Email
          <input className={`${inputClass} mt-1`} type="email" value={email} onChange={(e) => setEmail(e.target.value)} required autoFocus />
        </label>
        <label className="block text-xs text-slate-600">
          Password
          <input className={`${inputClass} mt-1`} type="password" value={password} onChange={(e) => setPassword(e.target.value)} required />
        </label>
        <ErrorBox error={error} />
        <Button type="submit" disabled={busy} className="w-full">
          {busy ? "Signing in..." : "Sign in"}
        </Button>
      </form>
    </div>
  );
}