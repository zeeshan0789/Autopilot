// A small fixed window limiter, kept in memory (enough for one server)
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

// Counts failures per key (for example per email), whatever address they come from. After `max`
// failures inside the window the key is blocked until the window ends. A success clears it.
export function createFailureTracker({ windowMs, max, now = Date.now }) {
  const entries = new Map();

  const live = (key) => {
    const entry = entries.get(key);
    if (!entry) return null;
    if (now() - entry.start >= windowMs) {
      entries.delete(key);
      return null;
    }
    return entry;
  };

  return {
    blocked(key) {
      const entry = live(key);
      return entry !== null && entry.count >= max;
    },
    retryAfterSeconds(key) {
      const entry = live(key);
      return entry ? Math.max(1, Math.ceil((entry.start + windowMs - now()) / 1000)) : 0;
    },
    fail(key) {
      const entry = live(key);
      if (entry) {
        entry.count += 1;
        return;
      }
      if (entries.size > 10000) entries.clear();
      entries.set(key, { start: now(), count: 1 });
    },
    clear(key) {
      entries.delete(key);
    },
  };
}