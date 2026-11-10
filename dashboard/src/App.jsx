import { Navigate, Route, Routes } from "react-router-dom";
import { useAuth } from "./auth.jsx";
import { Loading } from "./components.jsx";
import Keys from "./Keys.jsx";
import Layout from "./Layout.jsx";
import Login from "./Login.jsx";
import Overview from "./Overview.jsx";
import Prompts from "./Prompts.jsx";
import Traces from "./Traces.jsx";

function RequireAuth({ children }) {
  const { user, loading } = useAuth();
  if (loading) return <div className="p-6"><Loading /></div>;
  if (!user) return <Navigate to="/login" replace />;
  return children;
}

function RequireAdmin({ children }) {
  const { isAdmin } = useAuth();
  if (!isAdmin) return <p className="text-sm text-slate-600">This page is for admins only.</p>;
  return children;
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
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
        <Route
          path="keys"
          element={
            <RequireAdmin>
              <Keys />
            </RequireAdmin>
          }
        />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}