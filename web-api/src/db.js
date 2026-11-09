import pg from "pg";

// 127.0.0.1, not localhost: on Windows localhost can try IPv6 first and add a delay
export const DEFAULT_DATABASE_URL =
  "postgres://autopilot:autopilot_dev@127.0.0.1:5432/autopilot";

export function createDb(databaseUrl = DEFAULT_DATABASE_URL) {
  const pool = new pg.Pool({
    connectionString: databaseUrl,
    max: 10,
    connectionTimeoutMillis: 3000,
  });
  return {
    query: (text, params) => pool.query(text, params),
    close: () => pool.end(),
  };
}