import { createApp } from "../src/app.js";
import { signToken } from "../src/auth.js";
import { createRateLimiter } from "../src/rateLimit.js";

export const config = {
  port: 0,
  host: "127.0.0.1",
  databaseUrl: "unused",
  jwtSecret: "x".repeat(40),
  jwtExpiresIn: "1h",
  dashboardOrigin: "http://localhost:5173",
  signupMode: "invite",
  trustProxyHops: 0,
  maxKeysPerUser: 3,
  maxPromptsPerUser: 3,
  maxVersionsPerPrompt: 50,
  maxMemberRatePerMin: 120,
};

export const ADMIN = 1;
export const ALICE = 2;
export const BOB = 3;

export const makeUsers = () => ({
  1: { id: 1, email: "admin@example.com", role: "admin", status: "active", daily_request_limit: 1000000, max_tokens_cap: 100000 },
  2: { id: 2, email: "alice@example.com", role: "member", status: "active", daily_request_limit: 200, max_tokens_cap: 1024 },
  3: { id: 3, email: "bob@example.com", role: "member", status: "active", daily_request_limit: 200, max_tokens_cap: 1024 },
});

const USER_LOOKUP = /select id, email, role, status, daily_request_limit, max_tokens_cap from users where id = \$1/;

// A database that answers the user lookup from a table of users and everything else from a list of
// [regex, result] pairs. It records every query, and a transaction just runs its statements in order.
export function fakeDb(routes = [], users = makeUsers()) {
  const calls = [];
  const db = {
    calls,
    users,
    query: async (text, params = []) => {
      calls.push({ text, params });
      if (USER_LOOKUP.test(text)) {
        const user = users[params[0]];
        return { rows: user ? [user] : [], rowCount: user ? 1 : 0 };
      }
      for (const [matcher, result] of routes) {
        if (matcher.test(text)) {
          if (result instanceof Error) throw result;
          return typeof result === "function" ? result(params) : result;
        }
      }
      return { rows: [], rowCount: 0 };
    },
    transaction: async (work) => work({ query: (text, params) => db.query(text, params) }),
  };
  return db;
}

export function makeApp(routes = [], options = {}) {
  const users = options.users ?? makeUsers();
  const db = fakeDb(routes, users);
  const appConfig = { ...config, ...(options.config ?? {}) };
  const generous = () => createRateLimiter({ windowMs: 60000, max: 1000 });
  const app = createApp({
    db,
    config: appConfig,
    loginLimiter: options.loginLimiter ?? generous(),
    signupLimiter: options.signupLimiter ?? generous(),
    ...(options.accountFailures ? { accountFailures: options.accountFailures } : {}),
  });
  return { app, db, users };
}

export const tokenFor = (userId) => signToken({ id: userId }, config);
export const auth = (userId = ALICE) => ({ Authorization: `Bearer ${tokenFor(userId)}` });
export const callsMatching = (db, regex) => db.calls.filter((c) => regex.test(c.text));