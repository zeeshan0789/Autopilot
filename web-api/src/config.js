import "dotenv/config";
import { DEFAULT_DATABASE_URL } from "./db.js";

export function loadConfig(env = process.env) {
  const jwtSecret = env.JWT_SECRET;
  if (!jwtSecret || jwtSecret.length < 32) {
    throw new Error("JWT_SECRET must be set and at least 32 characters long (see .env.example)");
  }
  return {
    port: Number(env.PORT ?? 3001),
    host: env.HOST ?? "127.0.0.1",
    databaseUrl: env.DATABASE_URL ?? DEFAULT_DATABASE_URL,
    jwtSecret,
    jwtExpiresIn: env.JWT_EXPIRES_IN ?? "8h",
    dashboardOrigin: env.DASHBOARD_ORIGIN ?? "http://localhost:5173",
  };
}