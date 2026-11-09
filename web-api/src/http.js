export class HttpError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

// Express 4 does not catch rejected promises by itself
export const wrap = (handler) => (req, res, next) =>
  Promise.resolve(handler(req, res, next)).catch(next);

export function parse(schema, data) {
  const result = schema.safeParse(data);
  if (!result.success) {
    const message = result.error.issues
      .map((issue) => `${issue.path.join(".") || "input"}: ${issue.message}`)
      .join("; ");
    throw new HttpError(400, message);
  }
  return result.data;
}

export function parseId(value) {
  const id = Number(value);
  if (!Number.isInteger(id) || id < 1) {
    throw new HttpError(400, "id must be a positive integer");
  }
  return id;
}

// eslint-disable-next-line no-unused-vars
export function errorHandler(err, req, res, next) {
  if (res.headersSent) return next(err);
  if (err instanceof HttpError) return res.status(err.status).json({ error: err.message });
  if (err?.code === "23505") return res.status(409).json({ error: "already exists" });
  if (err?.status >= 400 && err?.status < 500) {
    return res.status(err.status).json({ error: "bad request" });
  }
  console.error(err);
  return res.status(500).json({ error: "internal error" });
}