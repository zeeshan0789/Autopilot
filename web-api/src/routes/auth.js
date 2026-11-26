import { Router } from "express";
import { DUMMY_HASH, checkPassword, hashPassword, signToken } from "../auth.js";
import { HttpError, parse, wrap } from "../http.js";
import { hashInviteCode } from "../keys.js";
import { loginSchema, signupSchema } from "../validation.js";

const publicUser = (user) => ({ id: user.id, email: user.email, role: user.role });

export function authRoutes({ db, config, loginLimiter, signupLimiter, accountFailures }) {
  const router = Router();

  // The dashboard asks this to know whether to show a signup page and an invite code field
  router.get("/config", (req, res) => res.json({ signup_mode: config.signupMode }));

  router.post(
    "/login",
    loginLimiter,
    wrap(async (req, res) => {
      const { email, password } = parse(loginSchema, req.body);

      if (accountFailures.blocked(email)) {
        res.set("Retry-After", String(accountFailures.retryAfterSeconds(email)));
        throw new HttpError(429, "too many failed attempts for this account, try again later");
      }

      const { rows } = await db.query(
        "select id, email, password_hash, role, status from users where email = $1",
        [email],
      );
      const user = rows[0];
      const ok = await checkPassword(password, user ? user.password_hash : DUMMY_HASH);
      if (!user || !ok) {
        accountFailures.fail(email);
        throw new HttpError(401, "wrong email or password");
      }
      accountFailures.clear(email);
      if (user.status !== "active") throw new HttpError(403, "this account is suspended");

      res.json({ token: signToken(user, config), user: publicUser(user) });
    }),
  );

  router.post(
    "/signup",
    signupLimiter,
    wrap(async (req, res) => {
      if (config.signupMode === "closed") throw new HttpError(403, "signup is closed");
      const { email, password, invite_code } = parse(signupSchema, req.body);
      if (config.signupMode === "invite" && !invite_code) {
        throw new HttpError(403, "an invite code is needed to sign up");
      }

      const passwordHash = await hashPassword(password); // slow work stays outside the transaction
      const user = await db.transaction(async (tx) => {
        // The number of accounts is limited (checked first, so a full platform does not use up an invite).
        // Two signups at the very same moment can pass one account over the limit: that is accepted.
        const total = await tx.query("select count(*)::int as n from users");
        if (Number(total.rows[0]?.n ?? 0) >= config.maxUsers) {
          throw new HttpError(403, "registrations are full for now, try again later");
        }

        let inviteId = null;
        if (config.signupMode === "invite") {
          const claimed = await tx.query(
            "update invites set used_at = now() where code_hash = $1 and used_at is null returning id",
            [hashInviteCode(invite_code)],
          );
          if (claimed.rows.length === 0) throw new HttpError(403, "this invite code is not valid");
          inviteId = claimed.rows[0].id;
        }

        // The role is never taken from the request: everyone who signs up is a member, with the
        // starting limits of a new account
        const created = await tx.query(
          `insert into users (email, password_hash, role, daily_request_limit, max_tokens_cap)
           values ($1, $2, 'member', $3, $4) returning id, email, role, status`,
          [email, passwordHash, config.newUserDailyLimit, config.newUserMaxTokens],
        );
        if (inviteId !== null) {
          await tx.query("update invites set used_by = $1 where id = $2", [created.rows[0].id, inviteId]);
        }
        return created.rows[0];
      });

      res.status(201).json({ token: signToken(user, config), user: publicUser(user) });
    }),
  );

  return router;
}