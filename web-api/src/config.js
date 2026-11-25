import "dotenv/config";
import { DEFAULT_DATABASE_URL } from "./db.js";

const SIGNUP_MODES = ["invite", "open", "closed"];

export function loadConfig(env = process.env) {
  const jwtSecret = env.JWT_SECRET;
  if (!jwtSecret || jwtSecret.length < 32) {
    throw new Error("JWT_SECRET must be set and at least 32 characters long (see .env.example)");
  }

  // invite: only people with an invite code from an admin can sign up. open: anyone. closed: nobody.
  const signupMode = env.SIGNUP_MODE ?? "invite";
  if (!SIGNUP_MODES.includes(signupMode)) {
    throw new Error(`SIGNUP_MODE must be one of: ${SIGNUP_MODES.join(", ")}`);
  }

  // How many reverse proxies stand in front of this server (0 when none). It decides which address
  // counts as the address of the visitor, and so how the login limits work.
  const trustProxyHops = Number(env.TRUST_PROXY_HOPS ?? 0);
  if (!Number.isInteger(trustProxyHops) || trustProxyHops < 0 || trustProxyHops > 5) {
    throw new Error("TRUST_PROXY_HOPS must be a whole number from 0 to 5");
  }

  return {
    port: Number(env.PORT ?? 3001),
    host: env.HOST ?? "127.0.0.1",
    databaseUrl: env.DATABASE_URL ?? DEFAULT_DATABASE_URL,
    jwtSecret,
    jwtExpiresIn: env.JWT_EXPIRES_IN ?? "8h",
    dashboardOrigin: env.DASHBOARD_ORIGIN ?? "http://localhost:5173",
    signupMode,
    trustProxyHops,
    maxKeysPerUser: 10,
    maxPromptsPerUser: 20,
    maxVersionsPerPrompt: 50,
    maxMemberRatePerMin: 120,
  };
}