import { Router } from "express";
import { requireRole } from "../auth.js";
import { HttpError, parse, wrap } from "../http.js";
import { createPromptSchema, createVersionSchema, weightsSchema } from "../validation.js";

export function promptsRoutes({ db }) {
  const router = Router();
  const admin = requireRole("admin");

  router.get(
    "/",
    wrap(async (req, res) => {
      const { rows } = await db.query(
        `select p.id, p.name, p.description, p.created_at,
                v.version, v.system_prompt, v.user_template, v.weight, v.notes,
                v.created_at as version_created_at
         from prompts p
         left join prompt_versions v on v.prompt_id = p.id
         order by p.name, v.version`,
      );
      const byName = new Map();
      for (const r of rows) {
        if (!byName.has(r.name)) {
          byName.set(r.name, {
            id: r.id,
            name: r.name,
            description: r.description,
            created_at: r.created_at,
            versions: [],
          });
        }
        if (r.version !== null) {
          byName.get(r.name).versions.push({
            version: r.version,
            system_prompt: r.system_prompt,
            user_template: r.user_template,
            weight: r.weight,
            notes: r.notes,
            created_at: r.version_created_at,
          });
        }
      }
      res.json({ prompts: [...byName.values()] });
    }),
  );

  router.post(
    "/",
    admin,
    wrap(async (req, res) => {
      const { name, description } = parse(createPromptSchema, req.body);
      const { rows } = await db.query(
        "insert into prompts (name, description) values ($1, $2) returning id, name, description, created_at",
        [name, description ?? null],
      );
      res.status(201).json(rows[0]);
    }),
  );

  // The text of a version is never edited: a change is a new version
  router.post(
    "/:name/versions",
    admin,
    wrap(async (req, res) => {
      const body = parse(createVersionSchema, req.body);
      const { rows } = await db.query(
        `insert into prompt_versions (prompt_id, version, system_prompt, user_template, weight, notes)
         select p.id,
                coalesce((select max(version) from prompt_versions where prompt_id = p.id), 0) + 1,
                $2, $3, $4, $5
         from prompts p where p.name = $1
         returning version, system_prompt, user_template, weight, notes, created_at`,
        [req.params.name, body.system_prompt, body.user_template ?? null, body.weight, body.notes ?? null],
      );
      if (rows.length === 0) throw new HttpError(404, "prompt not found");
      res.status(201).json(rows[0]);
    }),
  );

  // Versions that are not listed get weight 0
  router.put(
    "/:name/weights",
    admin,
    wrap(async (req, res) => {
      const { weights } = parse(weightsSchema, req.body);
      const existing = await db.query(
        `select v.version from prompts p
         join prompt_versions v on v.prompt_id = p.id where p.name = $1`,
        [req.params.name],
      );
      if (existing.rows.length === 0) {
        const prompt = await db.query("select 1 from prompts where name = $1", [req.params.name]);
        throw new HttpError(404, prompt.rows.length ? "this prompt has no versions yet" : "prompt not found");
      }
      const known = new Set(existing.rows.map((r) => String(r.version)));
      for (const version of Object.keys(weights)) {
        if (!known.has(version)) throw new HttpError(400, `version ${version} does not exist`);
      }
      const { rows } = await db.query(
        `update prompt_versions v
         set weight = coalesce(($2::jsonb ->> v.version::text)::int, 0)
         from prompts p
         where v.prompt_id = p.id and p.name = $1
         returning v.version, v.weight`,
        [req.params.name, JSON.stringify(weights)],
      );
      res.json({ weights: rows.sort((a, b) => a.version - b.version) });
    }),
  );

  return router;
}