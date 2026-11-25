import { createApp } from "./app.js";
import { loadConfig } from "./config.js";
import { createDb } from "./db.js";
import { createRateLimiter } from "./rateLimit.js";

const config = loadConfig();
const db = createDb(config.databaseUrl);
const loginLimiter = createRateLimiter({ windowMs: 60_000, max: 10 });
const signupLimiter = createRateLimiter({ windowMs: 3_600_000, max: 5 });
const app = createApp({ db, config, loginLimiter, signupLimiter });

const server = app.listen(config.port, config.host, () => {
  console.log(`web-api listening on http://${config.host}:${config.port} (signup: ${config.signupMode})`);
});

async function shutdown() {
  server.close();
  await db.close();
  process.exit(0);
}
process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);