import { Router } from "express";
import { HttpError, parse, wrap } from "../http.js";
import { createPromptSchema, createVersionSchema, weightsSchema } from "../validation.js";

// A prompt belongs to the user who created it, and every query is limited to the logged in user.
export function promptsRoutes({ db, config }) {
  const router = Router();

  router.get(
    "/",
    wrap(async (req, res) => {
      const { rows } = await db.query(
        `select p.id, p.name, p.description, p.created_at,
                v.version, v.system_prompt, v.user_template, v.weight, v.notes,
                v.created_at as version_created_at
         from prompts p
         left join prompt_versions v on v.prompt_id = p.id
         where p.user_id = $1
         order by p.name, v.version`,
        [req.user.id],
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
    wrap(async (req, res) => {
      const { name, description } = parse(createPromptSchema, req.body);
      const count = await db.query("select count(*)::int as n from prompts where user_id = $1", [req.user.id]);
      if (Number(count.rows[0]?.n ?? 0) >= config.maxPromptsPerUser) {
        throw new HttpError(409, `you already have ${config.maxPromptsPerUser} prompts`);
      }
      const { rows } = await db.query(
        "insert into prompts (user_id, name, description) values ($1, $2, $3) returning id, name, description, created_at",
        [req.user.id, name, description ?? null],
      );
      res.status(201).json(rows[0]);
    }),
  );

  // The text of a version is never edited: a change is a new version
  router.post(
    "/:name/versions",
    wrap(async (req, res) => {
      const body = parse(createVersionSchema, req.body);
      const { rows } = await db.query(
        `insert into prompt_versions (prompt_id, version, system_prompt, user_template, weight, notes)
         select p.id,
                coalesce((select max(version) from prompt_versions where prompt_id = p.id), 0) + 1,
                $3, $4, $5, $6
         from prompts p
         where p.name = $1 and p.user_id = $2
           and (select count(*) from prompt_versions where prompt_id = p.id) < $7
         returning version, system_prompt, user_template, weight, notes, created_at`,
        [
          req.params.name,
          req.user.id,
          body.system_prompt,
          body.user_template ?? null,
          body.weight,
          body.notes ?? null,
          config.maxVersionsPerPrompt,
        ],
      );
      if (rows.length === 0) {
        const exists = await db.query("select 1 from prompts where name = $1 and user_id = $2", [
          req.params.name,
          req.user.id,
        ]);
        if (exists.rows.length > 0) {
          throw new HttpError(409, `a prompt can have at most ${config.maxVersionsPerPrompt} versions`);
        }
        throw new HttpError(404, "prompt not found");
      }
      res.status(201).json(rows[0]);
    }),
  );

  // Versions that are not listed get weight 0
  router.put(
    "/:name/weights",
    wrap(async (req, res) => {
      const { weights } = parse(weightsSchema, req.body);
      const existing = await db.query(
        `select v.version from prompts p
         join prompt_versions v on v.prompt_id = p.id
         where p.name = $1 and p.user_id = $2`,
        [req.params.name, req.user.id],
      );
      if (existing.rows.length === 0) {
        const prompt = await db.query("select 1 from prompts where name = $1 and user_id = $2", [
          req.params.name,
          req.user.id,
        ]);
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
         where v.prompt_id = p.id and p.name = $1 and p.user_id = $3
         returning v.version, v.weight`,
        [req.params.name, JSON.stringify(weights), req.user.id],
      );
      res.json({ weights: rows.sort((a, b) => a.version - b.version) });
    }),
  );

  return router;
}