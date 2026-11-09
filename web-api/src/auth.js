import bcrypt from "bcryptjs";
import jwt from "jsonwebtoken";

export const hashPassword = (password) => bcrypt.hash(password, 10);
export const checkPassword = (password, hash) => bcrypt.compare(password, hash);

// Used when the email is unknown, so an unknown email and a wrong password take the same time
export const DUMMY_HASH = bcrypt.hashSync("not-a-real-password", 10);

export function signToken(user, config) {
  return jwt.sign(
    { sub: String(user.id), email: user.email, role: user.role },
    config.jwtSecret,
    { algorithm: "HS256", expiresIn: config.jwtExpiresIn },
  );
}

export function authenticate(config) {
  return (req, res, next) => {
    const [scheme, token] = (req.headers.authorization ?? "").split(" ");
    if (scheme !== "Bearer" || !token) {
      return res.status(401).json({ error: "login required" });
    }
    try {
      const payload = jwt.verify(token, config.jwtSecret, { algorithms: ["HS256"] });
      req.user = { id: Number(payload.sub), email: payload.email, role: payload.role };
      return next();
    } catch {
      return res.status(401).json({ error: "invalid or expired token" });
    }
  };
}

export function requireRole(...roles) {
  return (req, res, next) =>
    roles.includes(req.user?.role)
      ? next()
      : res.status(403).json({ error: "not allowed for your role" });
}