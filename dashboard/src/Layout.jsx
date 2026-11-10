import { NavLink, Outlet } from "react-router-dom";
import { useAuth } from "./auth.jsx";

const link = ({ isActive }) =>
  `block rounded-lg px-3 py-2 text-sm font-medium ${
    isActive ? "bg-indigo-50 text-indigo-700" : "text-slate-600 hover:bg-slate-100"
  }`;

export default function Layout() {
  const { user, logout, isAdmin } = useAuth();
  return (
    <div className="flex min-h-screen">
      <aside className="w-56 shrink-0 border-r border-slate-200 bg-white p-4">
        <div className="mb-6 px-3 text-lg font-bold text-slate-900">Autopilot</div>
        <nav className="space-y-1">
          <NavLink to="/" end className={link}>
            Overview
          </NavLink>
          <NavLink to="/traces" className={link}>
            Traces
          </NavLink>
          <NavLink to="/prompts" className={link}>
            Prompts and A/B
          </NavLink>
          {isAdmin && (
            <NavLink to="/keys" className={link}>
              API keys
            </NavLink>
          )}
        </nav>
        <div className="mt-8 border-t border-slate-200 px-3 pt-4 text-xs text-slate-500">
          <div className="truncate font-medium text-slate-700">{user.email}</div>
          <div>{user.role}</div>
          <button onClick={logout} className="mt-2 text-indigo-600 hover:underline">
            Log out
          </button>
        </div>
      </aside>
      <main className="min-w-0 flex-1 space-y-6 p-6">
        <Outlet />
      </main>
    </div>
  );
}