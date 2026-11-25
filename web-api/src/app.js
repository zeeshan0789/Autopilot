import cors from "cors";
import express from "express";
import helmet from "helmet";
import { authenticate, requireAdmin } from "./auth.js";
import { errorHandler, wrap } from "./http.js";
import { createFailureTracker } from "./rateLimit.js";
import { adminRoutes } from "./routes/admin.js";
import { authRoutes } from "./routes/auth.js";
import { keysRoutes } from "./routes/keys.js";
import { promptsRoutes } from "./routes/prompts.js";
import { statsRoutes } from "./routes/stats.js";
import { tracesRoutes } from "./routes/traces.js";

export function createApp({
  db,
  config,
  loginLimiter,
  signupLimiter,
  // 10 wrong passwords for one account in 15 minutes block that account for the rest of the window
  accountFailures = createFailureTracker({ windowMs: 15 * 60 * 1000, max: 10 }),
}) {
  const app = express();
  app.disable("x-powered-by");
  if (config.trustProxyHops > 0) app.set("trust proxy", config.trustProxyHops);
  app.use(helmet());
  app.use(cors({ origin: config.dashboardOrigin }));
  app.use(express.json({ limit: "100kb" }));

  app.get(
    "/api/health",
    wrap(async (req, res) => {
      await db.query("select 1");
      res.json({ status: "ok" });
    }),
  );

  app.use("/api/auth", authRoutes({ db, config, loginLimiter, signupLimiter, accountFailures }));

  // Everything below needs a valid login. The user is read from the database on every request.
  const api = express.Router();
  api.use(authenticate({ db, config }));
  api.get("/me", (req, res) => res.json(req.user));
  api.use("/keys", keysRoutes({ db, config }));
  api.use("/prompts", promptsRoutes({ db, config }));
  api.use("/stats", statsRoutes({ db }));
  api.use("/traces", tracesRoutes({ db }));
  // The guard is on the mount point: whatever path follows /api/admin, a member gets 403
  api.use("/admin", requireAdmin, adminRoutes({ db }));
  app.use("/api", api);

  app.use((req, res) => res.status(404).json({ error: "not found" }));
  app.use(errorHandler);
  return app;
}