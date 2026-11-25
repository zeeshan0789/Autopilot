import { Router } from "express";
import { HttpError, parse, parseId, wrap } from "../http.js";
import { generateApiKey, hashApiKey, keyIdOf } from "../keys.js";
import { createKeySchema, updateKeySchema } from "../validation.js";

// The hash is never selected: it must not leave the database
const PUBLIC_COLUMNS = "id, name, key_id, rate_limit_per_min, active, created_at, revoked_at";

// Every query below is limited to the keys of the logged in user: another user's key id gives a 404
export function keysRoutes({ db, config }) {
  const router = Router();

  const checkRate = (req, rate) => {
    if (req.user.role !== "admin" && rate > config.maxMemberRatePerMin) {
      throw new HttpError(400, `the limit per minute can be at most ${config.maxMemberRatePerMin}`);
    }
  };

  router.get(
    "/",
    wrap(async (req, res) => {
      const { rows } = await db.query(
        `select ${PUBLIC_COLUMNS} from api_keys where user_id = $1 order by id desc`,
        [req.user.id],
      );
      res.json({ keys: rows });
    }),
  );

  router.post(
    "/",
    wrap(async (req, res) => {
      const { name, rate_limit_per_min } = parse(createKeySchema, req.body);
      checkRate(req, rate_limit_per_min);

      const active = await db.query(
        "select count(*)::int as n from api_keys where user_id = $1 and active",
        [req.user.id],
      );
      if (Number(active.rows[0]?.n ?? 0) >= config.maxKeysPerUser) {
        throw new HttpError(409, `you already have ${config.maxKeysPerUser} active keys, revoke one first`);
      }

      const key = generateApiKey();
      const hash = hashApiKey(key);
      const { rows } = await db.query(
        `insert into api_keys (key_hash, key_id, name, rate_limit_per_min, user_id)
         values ($1, $2, $3, $4, $5) returning ${PUBLIC_COLUMNS}`,
        [hash, keyIdOf(hash), name, rate_limit_per_min, req.user.id],
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
      checkRate(req, rate_limit_per_min);
      const { rows } = await db.query(
        `update api_keys set rate_limit_per_min = $3
         where id = $1 and user_id = $2 and active returning ${PUBLIC_COLUMNS}`,
        [id, req.user.id, rate_limit_per_min],
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
         where id = $1 and user_id = $2 and active returning ${PUBLIC_COLUMNS}`,
        [id, req.user.id],
      );
      if (rows.length === 0) throw new HttpError(404, "active key not found");
      res.json(rows[0]);
    }),
  );

  return router;
}