import cors from "cors";
import express from "express";
import helmet from "helmet";
import { authenticate, requireRole } from "./auth.js";
import { errorHandler, wrap } from "./http.js";
import { authRoutes } from "./routes/auth.js";
import { keysRoutes } from "./routes/keys.js";
import { promptsRoutes } from "./routes/prompts.js";
import { statsRoutes } from "./routes/stats.js";
import { tracesRoutes } from "./routes/traces.js";

export function createApp({ db, config, loginLimiter }) {
  const app = express();
  app.disable("x-powered-by");
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

  app.use("/api/auth", authRoutes({ db, config, loginLimiter }));

  // Everything below needs a valid login token
  const api = express.Router();
  api.use(authenticate(config));
  api.get("/me", (req, res) => res.json(req.user));
  api.use("/keys", requireRole("admin"), keysRoutes({ db }));
  api.use("/prompts", promptsRoutes({ db }));
  api.use("/stats", statsRoutes({ db }));
  api.use("/traces", tracesRoutes({ db }));
  app.use("/api", api);

  app.use((req, res) => res.status(404).json({ error: "not found" }));
  app.use(errorHandler);
  return app;
}