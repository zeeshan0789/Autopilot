import { createApp } from "../src/app.js";
import { signToken } from "../src/auth.js";
import { createRateLimiter } from "../src/rateLimit.js";

export const config = {
  port: 0,
  databaseUrl: "unused",
  jwtSecret: "x".repeat(40),
  jwtExpiresIn: "1h",
  dashboardOrigin: "http://localhost:5173",
};

// A database that answers from a list of [regex, result] pairs and records every query
export function fakeDb(routes = []) {
  const calls = [];
  return {
    calls,
    query: async (text, params = []) => {
      calls.push({ text, params });
      for (const [matcher, result] of routes) {
        if (matcher.test(text)) {
          if (result instanceof Error) throw result;
          return typeof result === "function" ? result(params) : result;
        }
      }
      return { rows: [], rowCount: 0 };
    },
  };
}

export function makeApp(routes = [], limiter = createRateLimiter({ windowMs: 60000, max: 1000 })) {
  const db = fakeDb(routes);
  return { app: createApp({ db, config, loginLimiter: limiter }), db };
}

export function tokenFor(role = "admin") {
  return signToken({ id: 1, email: `${role}@example.com`, role }, config);
}

export const auth = (role = "admin") => ({ Authorization: `Bearer ${tokenFor(role)}` });