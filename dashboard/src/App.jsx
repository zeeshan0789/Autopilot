import { Suspense, lazy } from "react";
import { Link, Navigate, Outlet, Route, Routes } from "react-router-dom";
import { useAuth } from "./auth.jsx";
import { Loading } from "./components.jsx";
import Keys from "./Keys.jsx";
import Layout from "./Layout.jsx";
import Login from "./Login.jsx";
import Overview from "./Overview.jsx";
import Prompts from "./Prompts.jsx";
import Signup from "./Signup.jsx";
import Traces from "./Traces.jsx";

// The admin pages are separate chunks: the browser of a member never downloads them, because
// nothing below renders them for a member. This only saves a download. The server decides who may
// see admin data (a member gets 403 on every /api/admin path).
const AdminUsers = lazy(() => import("./admin/Users.jsx"));
const AdminInvites = lazy(() => import("./admin/Invites.jsx"));
const AdminPlatform = lazy(() => import("./admin/Platform.jsx"));

function RequireAuth({ children }) {
  const { user, loading } = useAuth();
  if (loading) {
    return (
      <div className="p-6">
        <Loading />
      </div>
    );
  }
  if (!user) return <Navigate to="/login" replace />;
  return children;
}

function NotAllowed() {
  return (
    <div className="max-w-md space-y-2">
      <h1 className="text-xl font-bold text-slate-900">This area is for administrators</h1>
      <p className="text-sm text-slate-600">Your account does not have access to it.</p>
      <Link to="/" className="text-sm text-indigo-600 hover:underline">
        Back to the overview
      </Link>
    </div>
  );
}

function AdminArea() {
  const { isAdmin } = useAuth();
  if (!isAdmin) return <NotAllowed />;
  return (
    <Suspense fallback={<Loading />}>
      <Outlet />
    </Suspense>
  );
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route path="/signup" element={<Signup />} />
      <Route
        element={
          <RequireAuth>
            <Layout />
          </RequireAuth>
        }
      >
        <Route index element={<Overview />} />
        <Route path="traces" element={<Traces />} />
        <Route path="prompts" element={<Prompts />} />
        <Route path="keys" element={<Keys />} />
        <Route path="admin" element={<AdminArea />}>
          <Route index element={<Navigate to="/admin/platform" replace />} />
          <Route path="platform" element={<AdminPlatform />} />
          <Route path="users" element={<AdminUsers />} />
          <Route path="invites" element={<AdminInvites />} />
          <Route path="*" element={<Navigate to="/admin/platform" replace />} />
        </Route>
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}