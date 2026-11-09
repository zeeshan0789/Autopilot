// A small fixed window limiter, kept in memory (enough for one local server)
export function createRateLimiter({ windowMs, max, now = Date.now }) {
  const hits = new Map();
  return (req, res, next) => {
    const key = req.ip;
    const t = now();
    const entry = hits.get(key);

    if (!entry || t - entry.start >= windowMs) {
      if (hits.size > 10000) hits.clear();
      hits.set(key, { start: t, count: 1 });
      return next();
    }

    entry.count += 1;
    if (entry.count > max) {
      res.set("Retry-After", String(Math.ceil((entry.start + windowMs - t) / 1000)));
      return res.status(429).json({ error: "too many attempts, try again later" });
    }
    return next();
  };
}