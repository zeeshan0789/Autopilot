import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { api, getToken, setToken, setUnauthorizedHandler } from "./api.js";

const AuthContext = createContext(null);
export const useAuth = () => useContext(AuthContext);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(Boolean(getToken()));

  const logout = useCallback(() => {
    setToken(null);
    setUser(null);
  }, []);

  useEffect(() => {
    setUnauthorizedHandler(logout);
    if (!getToken()) return;
    api("/api/me")
      .then(setUser)
      .catch(() => logout())
      .finally(() => setLoading(false));
  }, [logout]);

  // The account (with its role and limits) always comes from the server, never from the token
  const startSession = useCallback(async (token) => {
    setToken(token);
    try {
      setUser(await api("/api/me"));
    } catch (error) {
      setToken(null);
      throw error;
    }
  }, []);

  const login = useCallback(
    async (email, password) => {
      const result = await api("/api/auth/login", { method: "POST", body: { email, password } });
      await startSession(result.token);
    },
    [startSession],
  );

  const signup = useCallback(
    async (email, password, inviteCode) => {
      const body = { email, password };
      if (inviteCode) body.invite_code = inviteCode;
      const result = await api("/api/auth/signup", { method: "POST", body });
      await startSession(result.token);
    },
    [startSession],
  );

  const value = useMemo(
    () => ({ user, loading, login, signup, logout, isAdmin: user?.role === "admin" }),
    [user, loading, login, signup, logout],
  );
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}