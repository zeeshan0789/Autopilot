import { useEffect, useState } from "react";
import { Link, Navigate, useNavigate, useSearchParams } from "react-router-dom";
import { api } from "./api.js";
import { useAuth } from "./auth.jsx";
import { Button, ErrorBox, Loading, inputClass } from "./components.jsx";
import { passwordProblems, passwordStrength } from "./password.js";

const BAR_COLORS = ["", "bg-rose-400", "bg-amber-400", "bg-lime-500", "bg-emerald-500"];

export default function Signup() {
  const { user, signup } = useAuth();
  const navigate = useNavigate();
  const [params] = useSearchParams();
  const [mode, setMode] = useState(null);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [invite, setInvite] = useState(params.get("invite") ?? "");
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api("/api/auth/config")
      .then((config) => setMode(config.signup_mode))
      .catch((e) => setError(e));
  }, []);

  if (user) return <Navigate to="/" replace />;

  const strength = passwordStrength(password);

  async function submit(event) {
    event.preventDefault();
    const problems = passwordProblems(password);
    if (problems.length > 0) {
      setError(new Error(`Password: ${problems.join(", ")}`));
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await signup(email, password, mode === "invite" ? invite : undefined);
      navigate("/", { replace: true });
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center p-6">
      <div className="w-full max-w-sm space-y-4 rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
        <div>
          <h1 className="text-xl font-bold text-slate-900">Create your account</h1>
          <p className="text-sm text-slate-500">You get your own keys, prompts and numbers.</p>
        </div>

        {mode === null && !error && <Loading />}
        {mode === "closed" && (
          <p className="rounded-lg bg-slate-100 px-3 py-2 text-sm text-slate-700">Signup is closed at the moment.</p>
        )}

        {(mode === "invite" || mode === "open") && (
          <form onSubmit={submit} className="space-y-4">
            {mode === "invite" && (
              <>
                <label className="block text-xs text-slate-600">
                  Invite code
                  <input className={`${inputClass} mt-1`} value={invite} onChange={(e) => setInvite(e.target.value)} placeholder="inv-..." required autoFocus />
                </label>
                <p className="text-xs text-slate-500">Signup is by invitation at the moment. Ask the owner of this site for a code.</p>
              </>
            )}
            <label className="block text-xs text-slate-600">
              Email
              <input className={`${inputClass} mt-1`} type="email" value={email} onChange={(e) => setEmail(e.target.value)} required />
            </label>
            <label className="block text-xs text-slate-600">
              Password
              <input className={`${inputClass} mt-1`} type="password" value={password} onChange={(e) => setPassword(e.target.value)} required />
              <div className="mt-2 flex gap-1" aria-hidden="true">
                {[1, 2, 3, 4].map((level) => (
                  <div key={level} className={`h-1.5 flex-1 rounded ${strength.score >= level ? BAR_COLORS[strength.score] : "bg-slate-200"}`} />
                ))}
              </div>
              <span className="mt-1 block text-xs text-slate-500">
                {password ? strength.label : "At least 10 characters. A few words make a good password."}
              </span>
            </label>
            <ErrorBox error={error} />
            <Button type="submit" disabled={busy} className="w-full">
              {busy ? "Creating the account..." : "Create account"}
            </Button>
          </form>
        )}

        {mode !== "invite" && mode !== "open" && <ErrorBox error={error} />}

        <p className="text-center text-xs text-slate-500">
          Already have an account?{" "}
          <Link to="/login" className="text-indigo-600 hover:underline">
            Sign in
          </Link>
        </p>
      </div>
    </div>
  );
}