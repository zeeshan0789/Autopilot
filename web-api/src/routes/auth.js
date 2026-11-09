import { Router } from "express";
import { DUMMY_HASH, checkPassword, signToken } from "../auth.js";
import { HttpError, parse, wrap } from "../http.js";
import { loginSchema } from "../validation.js";

export function authRoutes({ db, config, loginLimiter }) {
  const router = Router();

  router.post(
    "/login",
    loginLimiter,
    wrap(async (req, res) => {
      const { email, password } = parse(loginSchema, req.body);
      const { rows } = await db.query(
        "select id, email, password_hash, role from users where email = $1",
        [email],
      );
      const user = rows[0];
      const ok = await checkPassword(password, user ? user.password_hash : DUMMY_HASH);
      if (!user || !ok) throw new HttpError(401, "wrong email or password");

      res.json({
        token: signToken(user, config),
        user: { id: user.id, email: user.email, role: user.role },
      });
    }),
  );

  return router;
}