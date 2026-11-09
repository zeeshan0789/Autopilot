import { Router } from "express";
import { HttpError, parse, parseId, wrap } from "../http.js";
import { generateApiKey, hashApiKey, keyIdOf } from "../keys.js";
import { createKeySchema, updateKeySchema } from "../validation.js";

// The hash is never selected: it must not leave the database
const PUBLIC_COLUMNS = "id, name, key_id, rate_limit_per_min, active, created_at, revoked_at";

export function keysRoutes({ db }) {
  const router = Router();

  router.get(
    "/",
    wrap(async (req, res) => {
      const { rows } = await db.query(`select ${PUBLIC_COLUMNS} from api_keys order by id desc`);
      res.json({ keys: rows });
    }),
  );

  router.post(
    "/",
    wrap(async (req, res) => {
      const { name, rate_limit_per_min } = parse(createKeySchema, req.body);
      const key = generateApiKey();
      const hash = hashApiKey(key);
      const { rows } = await db.query(
        `insert into api_keys (key_hash, key_id, name, rate_limit_per_min)
         values ($1, $2, $3, $4) returning ${PUBLIC_COLUMNS}`,
        [hash, keyIdOf(hash), name, rate_limit_per_min],
      );
      // The full key is shown only now. Only its hash is stored.
      res.status(201).json({ ...rows[0], key });
    }),
  );

  router.patch(
    "/:id",
    wrap(async (req, res) => {
      const id = parseId(req.params.id);
      const { rate_limit_per_min } = parse(updateKeySchema, req.body);
      const { rows } = await db.query(
        `update api_keys set rate_limit_per_min = $2
         where id = $1 and active returning ${PUBLIC_COLUMNS}`,
        [id, rate_limit_per_min],
      );
      if (rows.length === 0) throw new HttpError(404, "active key not found");
      res.json(rows[0]);
    }),
  );

  router.post(
    "/:id/revoke",
    wrap(async (req, res) => {
      const id = parseId(req.params.id);
      const { rows } = await db.query(
        `update api_keys set active = false, revoked_at = now()
         where id = $1 and active returning ${PUBLIC_COLUMNS}`,
        [id],
      );
      if (rows.length === 0) throw new HttpError(404, "active key not found");
      res.json(rows[0]);
    }),
  );

  return router;
}