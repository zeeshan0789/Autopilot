import { Router } from "express";
import { parse, wrap } from "../http.js";
import { tracesQuerySchema } from "../validation.js";

export function tracesRoutes({ db }) {
  const router = Router();

  router.get(
    "/",
    wrap(async (req, res) => {
      const { limit, offset, status } = parse(tracesQuerySchema, req.query);
      const params = [];
      let where = "";
      if (status) {
        params.push(status);
        where = `where status = $${params.length}`;
      }
      params.push(limit, offset);
      const { rows } = await db.query(
        `select created_at, request_id, requested_model, routed_model, served_model, provider,
                policy_rule, cache_status, cache_reason, status, http_status, fallback_used,
                total_tokens, estimated_cost_usd, latency_ms, prompt_name, prompt_version
         from traces ${where}
         order by created_at desc
         limit $${params.length - 1} offset $${params.length}`,
        params,
      );
      res.json({ traces: rows, limit, offset });
    }),
  );

  return router;
}