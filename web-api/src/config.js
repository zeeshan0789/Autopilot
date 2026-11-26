import "dotenv/config";
import { DEFAULT_DATABASE_URL } from "./db.js";

const SIGNUP_MODES = ["invite", "open", "closed"];

function wholeNumber(env, name, fallback, min, max) {
  const value = Number(env[name] ?? fallback);
  if (!Number.isInteger(value) || value < min || value > max) {
    throw new Error(`${name} must be a whole number from ${min} to ${max}`);
  }
  return value;
}

export function loadConfig(env = process.env) {
  const jwtSecret = env.JWT_SECRET;
  if (!jwtSecret || jwtSecret.length < 32) {
    throw new Error("JWT_SECRET must be set and at least 32 characters long (see .env.example)");
  }

  // invite: only people with an invite code from an admin can sign up. open: anyone. closed: nobody.
  const signupMode = env.SIGNUP_MODE ?? "open";
  if (!SIGNUP_MODES.includes(signupMode)) {
    throw new Error(`SIGNUP_MODE must be one of: ${SIGNUP_MODES.join(", ")}`);
  }

  return {
    port: Number(env.PORT ?? 3001),
    host: env.HOST ?? "127.0.0.1",
    databaseUrl: env.DATABASE_URL ?? DEFAULT_DATABASE_URL,
    jwtSecret,
    jwtExpiresIn: env.JWT_EXPIRES_IN ?? "8h",
    dashboardOrigin: env.DASHBOARD_ORIGIN ?? "http://localhost:5173",
    signupMode,
    // How many reverse proxies stand in front of this server (0 when none). It decides which address
    // counts as the address of the visitor, and so how the login and signup limits work.
    trustProxyHops: wholeNumber(env, "TRUST_PROXY_HOPS", 0, 0, 5),
    // The models behind the platform are a free quota: the number of accounts is limited, and a new
    // account starts with a small daily limit (an admin can raise it)
    maxUsers: wholeNumber(env, "MAX_USERS", 200, 1, 100000),
    newUserDailyLimit: wholeNumber(env, "NEW_USER_DAILY_LIMIT", 50, 1, 100000),
    newUserMaxTokens: wholeNumber(env, "NEW_USER_MAX_TOKENS", 1024, 1, 100000),
    maxKeysPerUser: 10,
    maxPromptsPerUser: 20,
    maxVersionsPerPrompt: 50,
    maxMemberRatePerMin: 120,
  };
}