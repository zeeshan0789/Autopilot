import bcrypt from "bcryptjs";
import jwt from "jsonwebtoken";

export const hashPassword = (password) => bcrypt.hash(password, 10);
export const checkPassword = (password, hash) => bcrypt.compare(password, hash);

// Used when the email is unknown, so an unknown email and a wrong password take the same time
export const DUMMY_HASH = bcrypt.hashSync("not-a-real-password", 10);

// The token says only who the user is. The role and the status are read from the database on every
// request, so a changed role or a suspension applies at once, and a role written into a token by
// someone else means nothing.
export function signToken(user, config) {
  return jwt.sign({ sub: String(user.id) }, config.jwtSecret, {
    algorithm: "HS256",
    expiresIn: config.jwtExpiresIn,
  });
}

const LOOKUP =
  "select id, email, role, status, daily_request_limit, max_tokens_cap from users where id = $1";

export function authenticate({ db, config }) {
  return async (req, res, next) => {
    try {
      const [scheme, token] = (req.headers.authorization ?? "").split(" ");
      if (scheme !== "Bearer" || !token) {
        return res.status(401).json({ error: "login required" });
      }
      let payload;
      try {
        payload = jwt.verify(token, config.jwtSecret, { algorithms: ["HS256"] });
      } catch {
        return res.status(401).json({ error: "invalid or expired token" });
      }
      const id = Number(payload.sub);
      if (!Number.isInteger(id) || id < 1) {
        return res.status(401).json({ error: "invalid or expired token" });
      }

      const { rows } = await db.query(LOOKUP, [id]);
      const user = rows[0];
      if (!user) return res.status(401).json({ error: "invalid or expired token" });
      if (user.status !== "active") {
        return res.status(403).json({ error: "this account is suspended" });
      }
      req.user = {
        id: user.id,
        email: user.email,
        role: user.role,
        daily_request_limit: user.daily_request_limit,
        max_tokens_cap: user.max_tokens_cap,
      };
      return next();
    } catch (error) {
      return next(error);
    }
  };
}

// Runs after authenticate, so req.user.role comes from the database
export function requireAdmin(req, res, next) {
  return req.user?.role === "admin"
    ? next()
    : res.status(403).json({ error: "not allowed" });
}