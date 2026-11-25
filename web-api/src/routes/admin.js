import { Router } from "express";
import { HttpError, parse, parseId, wrap } from "../http.js";
import { generateInviteCode, hashInviteCode } from "../keys.js";
import { createInviteSchema, updateUserSchema } from "../validation.js";
import { statsRoutes } from "./stats.js";

// Mounted behind requireAdmin (the role comes from the database), so every route here is for admins only.
// The role of an account cannot be changed here: admins are created with the command line.
export function adminRoutes({ db }) {
  const router = Router();

  router.get(
    "/users",
    wrap(async (req, res) => {
      const { rows } = await db.query(
        `select u.id, u.email, u.role, u.status, u.daily_request_limit, u.max_tokens_cap, u.created_at,
                (select count(*) from api_keys k where k.user_id = u.id and k.active)::int as active_keys,
                (select count(*) from prompts p where p.user_id = u.id)::int as prompts,
                (select count(*) from traces t where t.user_id = u.id
                   and t.created_at >= now() - interval '24 hours')::int as requests_24h
         from users u order by u.id`,
      );
      res.json({ users: rows });
    }),
  );

  router.patch(
    "/users/:id",
    wrap(async (req, res) => {
      const id = parseId(req.params.id);
      const changes = parse(updateUserSchema, req.body);
      if (id === req.user.id) throw new HttpError(400, "you cannot change your own account here");

      // The column names come from the schema (a fixed list), the values are parameters
      const params = [id];
      const sets = Object.entries(changes).map(([column, value]) => {
        params.push(value);
        return `${column} = $${params.length}`;
      });
      const { rows } = await db.query(
        `update users set ${sets.join(", ")} where id = $1 and role = 'member'
         returning id, email, role, status, daily_request_limit, max_tokens_cap`,
        params,
      );
      if (rows.length === 0) throw new HttpError(404, "member not found");
      res.json(rows[0]);
    }),
  );

  router.get(
    "/invites",
    wrap(async (req, res) => {
      const { rows } = await db.query(
        `select i.id, i.note, i.created_at, i.used_at, u.email as used_by_email
         from invites i left join users u on u.id = i.used_by
         order by i.id desc limit 200`,
      );
      res.json({ invites: rows });
    }),
  );

  router.post(
    "/invites",
    wrap(async (req, res) => {
      const { note } = parse(createInviteSchema, req.body);
      const code = generateInviteCode();
      const { rows } = await db.query(
        "insert into invites (code_hash, note, created_by) values ($1, $2, $3) returning id, note, created_at",
        [hashInviteCode(code), note ?? null, req.user.id],
      );
      // The code is shown only now. Only its hash is stored.
      res.status(201).json({ ...rows[0], code });
    }),
  );

  router.delete(
    "/invites/:id",
    wrap(async (req, res) => {
      const id = parseId(req.params.id);
      const { rows } = await db.query("delete from invites where id = $1 and used_at is null returning id", [id]);
      if (rows.length === 0) throw new HttpError(404, "unused invite not found");
      res.json({ deleted: id });
    }),
  );

  // The numbers of the whole platform
  router.use("/stats", statsRoutes({ db, platform: true }));

  return router;
}