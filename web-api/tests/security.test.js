import jwt from "jsonwebtoken";
import request from "supertest";
import { describe, expect, it } from "vitest";
import { hashPassword } from "../src/auth.js";
import { createFailureTracker, createRateLimiter } from "../src/rateLimit.js";
import { ADMIN, ALICE, BOB, auth, callsMatching, config, makeApp, tokenFor } from "./helpers.js";

const bearer = (token) => ({ Authorization: `Bearer ${token}` });
const b64 = (value) => Buffer.from(JSON.stringify(value)).toString("base64url");

describe("a member cannot reach the admin area", () => {
  const paths = [
    ["get", "/api/admin"],
    ["get", "/api/admin/users"],
    ["patch", "/api/admin/users/3"],
    ["post", "/api/admin/invites"],
    ["delete", "/api/admin/invites/1"],
    ["get", "/api/admin/stats/overview"],
    ["post", "/api/admin/users/3/promote"],
    ["put", "/api/admin/anything/at/all"],
    ["get", "/api/admin/"],
  ];

  it("gets 403 on every admin path and method, even for paths that do not exist", async () => {
    const { app } = makeApp();
    for (const [method, path] of paths) {
      const res = await request(app)[method](path).set(auth(ALICE)).send({});
      expect(res.status, `${method} ${path}`).toBe(403);
    }
  });

  it("gets 401 on every admin path without a login", async () => {
    const { app } = makeApp();
    for (const [method, path] of paths) {
      const res = await request(app)[method](path).send({});
      expect(res.status, `${method} ${path}`).toBe(401);
    }
  });

  it("an admin is let in", async () => {
    const { app } = makeApp();
    expect((await request(app).get("/api/admin/users").set(auth(ADMIN))).status).toBe(200);
  });

  it("an admin url without /api is just not found (the admin area has no page of its own on the server)", async () => {
    const { app } = makeApp();
    expect((await request(app).get("/admin").set(auth(ALICE))).status).toBe(404);
  });
});

describe("the role comes from the database, not from the token", () => {
  it("ignores a role that was written into a token", async () => {
    const { app } = makeApp();
    const forged = jwt.sign({ sub: String(ALICE), role: "admin" }, config.jwtSecret);
    expect((await request(app).get("/api/admin/users").set(bearer(forged))).status).toBe(403);
    const me = await request(app).get("/api/me").set(bearer(forged));
    expect(me.body.role).toBe("member");
  });

  it("applies a demotion at once", async () => {
    const { app, users } = makeApp();
    const header = auth(ADMIN);
    expect((await request(app).get("/api/admin/users").set(header)).status).toBe(200);
    users[ADMIN].role = "member";
    expect((await request(app).get("/api/admin/users").set(header)).status).toBe(403);
  });

  it("applies a suspension at once, to the token that was issued before it", async () => {
    const { app, users } = makeApp();
    const header = auth(ALICE);
    expect((await request(app).get("/api/me").set(header)).status).toBe(200);
    users[ALICE].status = "suspended";
    expect((await request(app).get("/api/me").set(header)).status).toBe(403);
  });

  it("a deleted user's token stops working", async () => {
    const { app, users } = makeApp();
    const header = auth(BOB);
    delete users[BOB];
    expect((await request(app).get("/api/me").set(header)).status).toBe(401);
  });
});

describe("tokens that are not real", () => {
  it("rejects garbage, an expired token, a wrong secret and an unsigned token", async () => {
    const { app } = makeApp();
    const claims = { sub: String(ADMIN) };
    const unsigned = `${b64({ alg: "none", typ: "JWT" })}.${b64(claims)}.`;
    const tokens = [
      "garbage",
      jwt.sign(claims, config.jwtSecret, { expiresIn: -10 }),
      jwt.sign(claims, "another-secret-another-secret-another-secret"),
      unsigned,
      jwt.sign({ sub: "0" }, config.jwtSecret),
      jwt.sign({ sub: "1; drop table users" }, config.jwtSecret),
      jwt.sign({}, config.jwtSecret),
    ];
    for (const token of tokens) {
      const res = await request(app).get("/api/me").set(bearer(token));
      expect(res.status, token.slice(0, 30)).toBe(401);
    }
  });

  it("needs a token on every protected route", async () => {
    const { app } = makeApp();
    for (const path of ["/api/me", "/api/keys", "/api/prompts", "/api/stats/overview", "/api/traces"]) {
      expect((await request(app).get(path)).status, path).toBe(401);
    }
  });
});

describe("signing up as an admin", () => {
  it("never creates anything but a member, whatever the request says", async () => {
    const { app, db } = makeApp([
      [/update invites set used_at/, { rows: [{ id: 1 }], rowCount: 1 }],
      [/insert into users/, { rows: [{ id: 9, email: "x@example.com", role: "member", status: "active" }], rowCount: 1 }],
    ]);
    const attempts = [
      { role: "admin" },
      { role: "ADMIN" },
      { is_admin: true },
      { admin: true },
      { status: "active", daily_request_limit: 1000000 },
    ];
    for (const extra of attempts) {
      const res = await request(app)
        .post("/api/auth/signup")
        .send({ email: "x@example.com", password: "a-long-password", invite_code: "inv-1", ...extra });
      expect(res.status, JSON.stringify(extra)).toBe(400);
    }
    expect(callsMatching(db, /insert into users/)).toHaveLength(0);

    const ok = await request(app).post("/api/auth/signup").send({ email: "x@example.com", password: "a-long-password", invite_code: "inv-1" });
    expect(ok.body.user.role).toBe("member");
    expect(callsMatching(db, /insert into users/)[0].text).toContain("'member'");
  });

  it("cannot be done by changing an existing account either", async () => {
    const { app } = makeApp();
    for (const [method, path, body] of [
      ["patch", `/api/admin/users/${BOB}`, { role: "admin" }],
      ["put", `/api/users/${BOB}`, { role: "admin" }],
      ["post", "/api/me", { role: "admin" }],
    ]) {
      const res = await request(app)[method](path).set(auth(ALICE)).send(body);
      expect([403, 404], `${method} ${path}`).toContain(res.status);
    }
  });

  it("limits how many accounts one address can create", async () => {
    const { app } = makeApp([], { signupLimiter: createRateLimiter({ windowMs: 60000, max: 2 }) });
    const attempt = () => request(app).post("/api/auth/signup").send({ email: "a@example.com", password: "a-long-password", invite_code: "inv-x" });
    expect((await attempt()).status).toBe(403);
    expect((await attempt()).status).toBe(403);
    expect((await attempt()).status).toBe(429);
  });
});

describe("swapping the number in the url", () => {
  it("every key and prompt query carries the id of the logged in user", async () => {
    const { app, db } = makeApp();
    await request(app).patch("/api/keys/1").set(auth(BOB)).send({ rate_limit_per_min: 5 });
    await request(app).post("/api/keys/1/revoke").set(auth(BOB));
    await request(app).put("/api/prompts/qa/weights").set(auth(BOB)).send({ weights: { 1: 1 } });
    await request(app).post("/api/prompts/qa/versions").set(auth(BOB)).send({ system_prompt: "x" });
    for (const call of db.calls.filter((c) => /api_keys|prompts|prompt_versions/.test(c.text))) {
      expect(call.params, call.text.slice(0, 40)).toContain(BOB);
      expect(call.text).toMatch(/user_id = \$\d/);
    }
  });

  it("a user id in the url or in the query string is not taken for the owner", async () => {
    const { app, db } = makeApp();
    await request(app).get(`/api/keys?user_id=${ADMIN}`).set(auth(ALICE));
    await request(app).get(`/api/traces?user_id=${ADMIN}`).set(auth(ALICE));
    await request(app).get(`/api/stats/overview?user_id=${ADMIN}`).set(auth(ALICE));
    for (const call of db.calls.filter((c) => /api_keys|traces/.test(c.text))) {
      expect(call.params).not.toContain(ADMIN);
    }
  });
});

describe("code in a username, a key name or a prompt name", () => {
  const payloads = ["'; drop table users; --", "' or 1=1 --", "<script>alert(1)</script>", "${7*7}", "{{constructor}}", "a\nb"];

  it("a key name is only ever a parameter, never part of the SQL", async () => {
    const { app, db } = makeApp([
      [/select count\(\*\)::int as n from api_keys/, { rows: [{ n: 0 }] }],
      [/insert into api_keys/, (params) => ({ rows: [{ id: 1, name: params[2] }], rowCount: 1 })],
    ]);
    for (const name of payloads) {
      const res = await request(app).post("/api/keys").set(auth(ALICE)).send({ name });
      expect(res.status, name).toBe(201);
    }
    for (const call of callsMatching(db, /insert into api_keys/)) {
      expect(call.text).not.toMatch(/drop table|or 1=1|script/i);
    }
    expect(callsMatching(db, /insert into api_keys/).map((c) => c.params[2])).toEqual(payloads.map((p) => p.trim()));
  });

  it("a prompt name that is not plain is refused before it reaches the database", async () => {
    const { app, db } = makeApp();
    for (const name of payloads) {
      expect((await request(app).post("/api/prompts").set(auth(ALICE)).send({ name })).status, name).toBe(400);
    }
    expect(callsMatching(db, /insert into prompts/)).toHaveLength(0);
  });

  it("an email with code in it is not an email", async () => {
    const { app, db } = makeApp();
    for (const email of ["' or 1=1 --", "a@b.com'; drop table users; --", "<script>@x.com"]) {
      expect((await request(app).post("/api/auth/login").send({ email, password: "x" })).status, email).toBe(400);
    }
    expect(callsMatching(db, /from users where email/)).toHaveLength(0);
  });

  it("the column that a stats query groups by can only be one of a fixed list", async () => {
    const { app } = makeApp();
    for (const by of ["status; drop table traces", "1", "provider,user_id", "(select 1)"]) {
      expect((await request(app).get(`/api/stats/breakdown?by=${encodeURIComponent(by)}`).set(auth(ALICE))).status, by).toBe(400);
    }
  });
});

describe("guessing the login", () => {
  const wrongLogin = (app, email = "alice@example.com") => request(app).post("/api/auth/login").send({ email, password: "wrong-password" });

  it("blocks an account after repeated failures, even from other addresses and even with the right password", async () => {
    const hash = await hashPassword("right-password");
    const row = { id: ALICE, email: "alice@example.com", password_hash: hash, role: "member", status: "active" };
    const { app } = makeApp([[/from users where email/, { rows: [row] }]], {
      accountFailures: createFailureTracker({ windowMs: 60000, max: 3 }),
    });
    for (let i = 0; i < 3; i += 1) expect((await wrongLogin(app)).status).toBe(401);
    const blocked = await wrongLogin(app);
    expect(blocked.status).toBe(429);
    expect(blocked.headers["retry-after"]).toBeDefined();
    const right = await request(app).post("/api/auth/login").send({ email: "alice@example.com", password: "right-password" });
    expect(right.status).toBe(429);
  });

  it("does not block an account for failures of another account, and a success clears the count", async () => {
    const hash = await hashPassword("right-password");
    const row = (email) => ({ id: ALICE, email, password_hash: hash, role: "member", status: "active" });
    const { app } = makeApp([[/from users where email/, (params) => ({ rows: [row(params[0])] })]], {
      accountFailures: createFailureTracker({ windowMs: 60000, max: 3 }),
    });
    await wrongLogin(app, "a@example.com");
    await wrongLogin(app, "a@example.com");
    await request(app).post("/api/auth/login").send({ email: "a@example.com", password: "right-password" });
    await wrongLogin(app, "a@example.com");
    await wrongLogin(app, "a@example.com");
    expect((await wrongLogin(app, "a@example.com")).status).toBe(401); // the count started again
    expect((await wrongLogin(app, "b@example.com")).status).toBe(401);
  });

  it("limits the attempts per address too", async () => {
    const { app } = makeApp([], { loginLimiter: createRateLimiter({ windowMs: 60000, max: 2 }) });
    expect((await wrongLogin(app)).status).toBe(401);
    expect((await wrongLogin(app)).status).toBe(401);
    expect((await wrongLogin(app)).status).toBe(429);
  });

  it("the failure tracker forgets a failure after its window", () => {
    let now = 1000;
    const tracker = createFailureTracker({ windowMs: 100, max: 2, now: () => now });
    tracker.fail("k");
    tracker.fail("k");
    expect(tracker.blocked("k")).toBe(true);
    expect(tracker.retryAfterSeconds("k")).toBe(1);
    now += 101;
    expect(tracker.blocked("k")).toBe(false);
    tracker.fail("k");
    tracker.clear("k");
    expect(tracker.blocked("k")).toBe(false);
  });

  it("counts the real address of a visitor behind a proxy only when told how many proxies there are", () => {
    expect(makeApp([], { config: { trustProxyHops: 0 } }).app.get("trust proxy")).toBe(false);
    expect(makeApp([], { config: { trustProxyHops: 1 } }).app.get("trust proxy")).toBe(1);
  });
});

describe("changing what the browser should not decide", () => {
  it("fields for an owner, a status or a limit are refused in every body", async () => {
    const { app } = makeApp();
    const header = auth(ALICE);
    expect((await request(app).post("/api/keys").set(header).send({ name: "x", user_id: ADMIN })).status).toBe(400);
    expect((await request(app).post("/api/keys").set(header).send({ name: "x", active: true })).status).toBe(400);
    expect((await request(app).patch("/api/keys/1").set(header).send({ rate_limit_per_min: 5, user_id: ADMIN })).status).toBe(400);
    expect((await request(app).post("/api/prompts").set(header).send({ name: "qa", user_id: ADMIN })).status).toBe(400);
    expect((await request(app).post("/api/prompts/qa/versions").set(header).send({ system_prompt: "x", prompt_id: 1 })).status).toBe(400);
  });

  it("a request that is too large is refused", async () => {
    const { app } = makeApp();
    const res = await request(app).post("/api/auth/login").send({ email: "a@example.com", password: "x".repeat(200_000) });
    expect(res.status).toBe(413);
  });
});

describe("what the server tells the browser", () => {
  it("never sends a password hash or an api key hash", async () => {
    const hash = await hashPassword("right-password");
    const { app } = makeApp([[/from users where email/, { rows: [{ id: ALICE, email: "a@example.com", password_hash: hash, role: "member", status: "active" }] }]]);
    const login = await request(app).post("/api/auth/login").send({ email: "a@example.com", password: "right-password" });
    expect(JSON.stringify(login.body)).not.toContain(hash);
    const me = await request(app).get("/api/me").set(auth(ALICE));
    expect(JSON.stringify(me.body)).not.toMatch(/hash/);
  });

  it("only allows the dashboard origin and sends security headers", async () => {
    const { app } = makeApp();
    const res = await request(app).get("/api/health").set("Origin", "https://evil.example");
    expect(res.headers["access-control-allow-origin"]).toBe("http://localhost:5173");
    expect(res.headers["x-powered-by"]).toBeUndefined();
    expect(res.headers["x-content-type-options"]).toBe("nosniff");
  });

  it("answers an unexpected error without details", async () => {
    const { app, db } = makeApp();
    const failing = db.query;
    db.query = async (text, params) => {
      if (/select count/.test(text)) throw new Error("connection to 10.0.0.5 refused, password=secret");
      return failing(text, params);
    };
    const res = await request(app).post("/api/keys").set(auth(ALICE)).send({ name: "x" });
    expect(res.status).toBe(500);
    expect(JSON.stringify(res.body)).not.toMatch(/10\.0\.0\.5|secret/);
  });
});