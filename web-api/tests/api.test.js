import jwt from "jsonwebtoken";
import request from "supertest";
import { describe, expect, it } from "vitest";
import { hashPassword } from "../src/auth.js";
import { hashApiKey } from "../src/keys.js";
import { createRateLimiter } from "../src/rateLimit.js";
import { auth, config, makeApp } from "./helpers.js";

describe("login", () => {
  it("returns a token that opens /api/me", async () => {
    const hash = await hashPassword("correct-password");
    const { app, db } = makeApp([
      [/from users/, { rows: [{ id: 7, email: "a@example.com", password_hash: hash, role: "admin" }] }],
    ]);
    const login = await request(app)
      .post("/api/auth/login")
      .send({ email: "A@Example.com", password: "correct-password" });
    expect(login.status).toBe(200);
    expect(db.calls[0].params[0]).toBe("a@example.com"); // the email is lowercased
    expect(login.body.user).toEqual({ id: 7, email: "a@example.com", role: "admin" });

    const me = await request(app).get("/api/me").set("Authorization", `Bearer ${login.body.token}`);
    expect(me.body).toEqual({ id: 7, email: "a@example.com", role: "admin" });
  });

  it("rejects a wrong password and an unknown email with the same answer", async () => {
    const hash = await hashPassword("correct-password");
    const known = makeApp([[/from users/, { rows: [{ id: 1, email: "a@example.com", password_hash: hash, role: "admin" }] }]]);
    const wrong = await request(known.app).post("/api/auth/login").send({ email: "a@example.com", password: "nope" });
    const unknown = await request(makeApp().app).post("/api/auth/login").send({ email: "b@example.com", password: "nope" });
    expect(wrong.status).toBe(401);
    expect(unknown.status).toBe(401);
    expect(wrong.body).toEqual(unknown.body);
  });

  it("rejects a malformed body", async () => {
    const res = await request(makeApp().app).post("/api/auth/login").send({ email: "not-an-email" });
    expect(res.status).toBe(400);
  });

  it("limits login attempts", async () => {
    const { app } = makeApp([], createRateLimiter({ windowMs: 60000, max: 2 }));
    const attempt = () => request(app).post("/api/auth/login").send({ email: "a@example.com", password: "x" });
    expect((await attempt()).status).toBe(401);
    expect((await attempt()).status).toBe(401);
    const third = await attempt();
    expect(third.status).toBe(429);
    expect(third.headers["retry-after"]).toBeDefined();
  });

  it("answers 400 to invalid JSON", async () => {
    const res = await request(makeApp().app)
      .post("/api/auth/login")
      .set("Content-Type", "application/json")
      .send("{bad json");
    expect(res.status).toBe(400);
  });
});

describe("tokens and roles", () => {
  it("needs a token", async () => {
    expect((await request(makeApp().app).get("/api/me")).status).toBe(401);
  });

  it("rejects garbage, expired and wrongly signed tokens", async () => {
    const { app } = makeApp();
    const claims = { sub: "1", email: "a@example.com", role: "admin" };
    const expired = jwt.sign(claims, config.jwtSecret, { expiresIn: -10 });
    const forged = jwt.sign(claims, "another-secret-another-secret-another-secret");
    for (const token of ["garbage", expired, forged]) {
      const res = await request(app).get("/api/me").set("Authorization", `Bearer ${token}`);
      expect(res.status).toBe(401);
    }
  });

  it("keeps the keys area for admins", async () => {
    const { app } = makeApp();
    expect((await request(app).get("/api/keys").set(auth("viewer"))).status).toBe(403);
    expect((await request(app).post("/api/keys").set(auth("viewer")).send({ name: "x" })).status).toBe(403);
  });

  it("lets a viewer read stats but not change prompts", async () => {
    const { app } = makeApp();
    expect((await request(app).get("/api/prompts").set(auth("viewer"))).status).toBe(200);
    expect((await request(app).post("/api/prompts").set(auth("viewer")).send({ name: "qa" })).status).toBe(403);
  });
});

describe("api keys", () => {
  const insertRow = (params) => ({
    rows: [{ id: 1, name: params[2], key_id: params[1], rate_limit_per_min: params[3], active: true, created_at: "now", revoked_at: null }],
    rowCount: 1,
  });

  it("creates a key, shows it once and stores only its hash", async () => {
    const { app, db } = makeApp([[/insert into api_keys/, insertRow]]);
    const res = await request(app).post("/api/keys").set(auth()).send({ name: "demo", rate_limit_per_min: 30 });
    expect(res.status).toBe(201);
    expect(res.body.key).toMatch(/^ap-/);
    expect(res.body.rate_limit_per_min).toBe(30);
    expect(res.body.key_hash).toBeUndefined();

    const params = db.calls[0].params;
    expect(params[0]).toBe(hashApiKey(res.body.key));
    expect(params[1]).toBe(hashApiKey(res.body.key).slice(0, 8));
    expect(JSON.stringify(params)).not.toContain(res.body.key);
  });

  it("defaults the rate limit and validates input", async () => {
    const { app, db } = makeApp([[/insert into api_keys/, insertRow]]);
    await request(app).post("/api/keys").set(auth()).send({ name: "demo" });
    expect(db.calls[0].params[3]).toBe(60);
    expect((await request(app).post("/api/keys").set(auth()).send({ name: "" })).status).toBe(400);
    expect((await request(app).post("/api/keys").set(auth()).send({ name: "x", rate_limit_per_min: 0 })).status).toBe(400);
  });

  it("never selects the hash when listing", async () => {
    const { app, db } = makeApp();
    await request(app).get("/api/keys").set(auth());
    expect(db.calls[0].text).not.toMatch(/key_hash/);
  });

  it("answers 404 when the key to revoke does not exist or is already revoked", async () => {
    const { app } = makeApp();
    expect((await request(app).post("/api/keys/5/revoke").set(auth())).status).toBe(404);
    expect((await request(app).post("/api/keys/abc/revoke").set(auth())).status).toBe(400);
  });
});

describe("prompts", () => {
  it("validates names and maps a duplicate to 409", async () => {
    const duplicate = Object.assign(new Error("duplicate"), { code: "23505" });
    const { app } = makeApp([[/insert into prompts/, duplicate]]);
    expect((await request(app).post("/api/prompts").set(auth()).send({ name: "Bad Name" })).status).toBe(400);
    expect((await request(app).post("/api/prompts").set(auth()).send({ name: "support" })).status).toBe(409);
  });

  it("answers 404 for a version of a missing prompt", async () => {
    const { app } = makeApp();
    const res = await request(app).post("/api/prompts/missing/versions").set(auth()).send({ system_prompt: "hi" });
    expect(res.status).toBe(404);
  });

  it("refuses weights for a version that does not exist", async () => {
    const { app } = makeApp([[/from prompts p\s+join prompt_versions/, { rows: [{ version: 1 }] }]]);
    const res = await request(app).put("/api/prompts/qa/weights").set(auth()).send({ weights: { 2: 50 } });
    expect(res.status).toBe(400);
  });
});

describe("stats", () => {
  const overview = {
    requests: "10", errors: "1", cache_hits: "2", cache_consulted: "4", cache_blocked: "1",
    cache_bypass: "2", fallbacks: "0", tokens_used: "100", tokens_saved: "50",
    cost_spent_usd: "0.0006", cost_saved_usd: "0.00015",
    avg_latency_ms: "900", p95_latency_ms: "1200", avg_hit_latency_ms: "20",
  };

  it("computes the rates", async () => {
    const { app } = makeApp([[/from traces/, { rows: [overview] }]]);
    const res = await request(app).get("/api/stats/overview?hours=24").set(auth("viewer"));
    expect(res.status).toBe(200);
    expect(res.body.error_rate).toBeCloseTo(0.1);
    expect(res.body.cache.hit_rate).toBeCloseTo(0.5);
    expect(res.body.cost.saving_pct).toBeCloseTo(0.2);
    expect(res.body.latency_ms).toEqual({ avg: 900, p95: 1200, avg_cache_hit: 20 });
  });

  it("has no rates when there is no traffic", async () => {
    const empty = { ...overview, requests: "0", errors: "0", cache_hits: "0", cache_consulted: "0", cost_spent_usd: "0", cost_saved_usd: "0" };
    const { app } = makeApp([[/from traces/, { rows: [empty] }]]);
    const res = await request(app).get("/api/stats/overview").set(auth());
    expect(res.body.error_rate).toBeNull();
    expect(res.body.cache.hit_rate).toBeNull();
    expect(res.body.cost.saving_pct).toBeNull();
  });

  it("validates the query", async () => {
    const { app } = makeApp();
    expect((await request(app).get("/api/stats/overview?hours=0").set(auth())).status).toBe(400);
    expect((await request(app).get("/api/stats/breakdown?by=evil").set(auth())).status).toBe(400);
    expect((await request(app).get("/api/stats/timeseries?bucket=week").set(auth())).status).toBe(400);
  });

  it("groups only by a whitelisted expression", async () => {
    const { app, db } = makeApp();
    const res = await request(app).get("/api/stats/breakdown?by=provider").set(auth());
    expect(res.status).toBe(200);
    expect(db.calls[0].text).toContain("coalesce(provider, 'none')");
  });
});

describe("health", () => {
  it("checks the database", async () => {
    const res = await request(makeApp().app).get("/api/health");
    expect(res.body).toEqual({ status: "ok" });
  });
});