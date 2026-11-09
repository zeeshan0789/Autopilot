import { Router } from "express";
import { parse, wrap } from "../http.js";
import { bucketSchema, dimensionSchema, hoursSchema } from "../validation.js";

// Provider calls are the requests that were not answered from the cache
const PROVIDER_CALL = "status = 'success' and cache_status is distinct from 'hit'";

const OVERVIEW_SQL = `
select
  count(*) as requests,
  count(*) filter (where status = 'error') as errors,
  count(*) filter (where cache_status = 'hit') as cache_hits,
  count(*) filter (where cache_status in ('hit', 'miss', 'blocked')) as cache_consulted,
  count(*) filter (where cache_status = 'blocked') as cache_blocked,
  count(*) filter (where cache_status = 'bypass') as cache_bypass,
  count(*) filter (where fallback_used) as fallbacks,
  coalesce(sum(total_tokens), 0) as tokens_used,
  coalesce(sum(saved_tokens), 0) as tokens_saved,
  coalesce(sum(estimated_cost_usd), 0) as cost_spent_usd,
  coalesce(sum(saved_cost_usd), 0) as cost_saved_usd,
  round(avg(latency_ms) filter (where ${PROVIDER_CALL})) as avg_latency_ms,
  round((percentile_cont(0.95) within group (order by latency_ms)
         filter (where ${PROVIDER_CALL}))::numeric) as p95_latency_ms,
  round(avg(latency_ms) filter (where cache_status = 'hit')) as avg_hit_latency_ms
from traces
where created_at >= now() - make_interval(hours => $1)`;

const TIMESERIES_SQL = `
select date_trunc($2, created_at) as bucket,
       count(*) as requests,
       count(*) filter (where status = 'error') as errors,
       count(*) filter (where cache_status = 'hit') as cache_hits,
       coalesce(sum(estimated_cost_usd), 0) as cost_spent_usd,
       coalesce(sum(saved_cost_usd), 0) as cost_saved_usd
from traces
where created_at >= now() - make_interval(hours => $1)
group by 1 order by 1`;

const PROMPTS_SQL = `
select prompt_name, prompt_version,
       count(*) as requests,
       count(*) filter (where status = 'error') as errors,
       count(*) filter (where cache_status = 'hit') as cache_hits,
       count(*) filter (where cache_status in ('hit', 'miss', 'blocked')) as cache_consulted,
       round(avg(latency_ms) filter (where ${PROVIDER_CALL})) as avg_latency_ms,
       round(avg(total_tokens) filter (where ${PROVIDER_CALL})) as avg_tokens,
       coalesce(sum(estimated_cost_usd), 0) as cost_spent_usd
from traces
where prompt_name is not null and created_at >= now() - make_interval(hours => $1)
group by prompt_name, prompt_version
order by prompt_name, prompt_version`;

// Only these expressions can be grouped by (never user supplied SQL)
const DIMENSIONS = {
  provider: "coalesce(provider, 'none')",
  model: "coalesce(served_model, 'none')",
  cache: "coalesce(cache_status, 'n/a')",
  route: "coalesce(policy_rule, 'n/a')",
  status: "status",
};

const num = (value) => (value === null || value === undefined ? null : Number(value));
const ratio = (part, whole) => (whole > 0 ? part / whole : null);

export function statsRoutes({ db }) {
  const router = Router();

  router.get(
    "/overview",
    wrap(async (req, res) => {
      const hours = parse(hoursSchema, req.query.hours);
      const { rows } = await db.query(OVERVIEW_SQL, [hours]);
      const r = rows[0];
      const requests = Number(r.requests);
      const errors = Number(r.errors);
      const hits = Number(r.cache_hits);
      const consulted = Number(r.cache_consulted);
      const spent = Number(r.cost_spent_usd);
      const saved = Number(r.cost_saved_usd);

      res.json({
        hours,
        requests,
        errors,
        error_rate: ratio(errors, requests),
        fallbacks: Number(r.fallbacks),
        cache: {
          hits,
          consulted,
          blocked: Number(r.cache_blocked),
          bypass: Number(r.cache_bypass),
          hit_rate: ratio(hits, consulted),
        },
        tokens: { used: Number(r.tokens_used), saved: Number(r.tokens_saved) },
        // Estimated at list prices, not real spend
        cost: { spent_usd: spent, saved_usd: saved, saving_pct: ratio(saved, spent + saved) },
        latency_ms: {
          avg: num(r.avg_latency_ms),
          p95: num(r.p95_latency_ms),
          avg_cache_hit: num(r.avg_hit_latency_ms),
        },
      });
    }),
  );

  router.get(
    "/timeseries",
    wrap(async (req, res) => {
      const hours = parse(hoursSchema, req.query.hours);
      const bucket = parse(bucketSchema, req.query.bucket);
      const { rows } = await db.query(TIMESERIES_SQL, [hours, bucket]);
      res.json({
        hours,
        bucket,
        points: rows.map((r) => ({
          bucket: r.bucket,
          requests: Number(r.requests),
          errors: Number(r.errors),
          cache_hits: Number(r.cache_hits),
          cost_spent_usd: Number(r.cost_spent_usd),
          cost_saved_usd: Number(r.cost_saved_usd),
        })),
      });
    }),
  );

  router.get(
    "/breakdown",
    wrap(async (req, res) => {
      const hours = parse(hoursSchema, req.query.hours);
      const by = parse(dimensionSchema, req.query.by);
      const expression = DIMENSIONS[by];
      const { rows } = await db.query(
        `select ${expression} as name, count(*) as requests
         from traces where created_at >= now() - make_interval(hours => $1)
         group by 1 order by 2 desc`,
        [hours],
      );
      res.json({ hours, by, items: rows.map((r) => ({ name: r.name, requests: Number(r.requests) })) });
    }),
  );

  router.get(
    "/prompts",
    wrap(async (req, res) => {
      const hours = parse(hoursSchema, req.query.hours);
      const { rows } = await db.query(PROMPTS_SQL, [hours]);
      res.json({
        hours,
        versions: rows.map((r) => ({
          prompt: r.prompt_name,
          version: r.prompt_version,
          requests: Number(r.requests),
          errors: Number(r.errors),
          cache_hit_rate: ratio(Number(r.cache_hits), Number(r.cache_consulted)),
          avg_latency_ms: num(r.avg_latency_ms),
          avg_tokens: num(r.avg_tokens),
          cost_spent_usd: Number(r.cost_spent_usd),
        })),
      });
    }),
  );

  return router;
}