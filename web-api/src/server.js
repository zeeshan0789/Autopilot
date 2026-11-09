import { createApp } from "./app.js";
import { loadConfig } from "./config.js";
import { createDb } from "./db.js";
import { createRateLimiter } from "./rateLimit.js";

const config = loadConfig();
const db = createDb(config.databaseUrl);
const loginLimiter = createRateLimiter({ windowMs: 60_000, max: 10 });
const app = createApp({ db, config, loginLimiter });

// Local only: the dashboard talks to this server from the same machine
const server = app.listen(config.port, "127.0.0.1", () => {
  console.log(`web-api listening on http://127.0.0.1:${config.port}`);
});

async function shutdown() {
  server.close();
  await db.close();
  process.exit(0);
}
process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);