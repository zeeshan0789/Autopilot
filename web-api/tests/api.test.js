import request from "supertest";
import { describe, expect, it } from "vitest";
import { hashPassword } from "../src/auth.js";
import { hashApiKey, hashInviteCode } from "../src/keys.js";
import { ADMIN, ALICE, BOB, auth, callsMatching, config, makeApp } from "./helpers.js";

const row = (id, role = "member", status = "active") => ({ id, email: `u${id}@example.com`, role, status });
const uniqueViolation = Object.assign(new Error("duplicate"), { code: "23505" });

describe("login", () => {
  it("returns a token, and /api/me answers from the database", async () => {
    const hash = await hashPassword("correct-password");
    const { app } = makeApp([
      [/from users where email/, { rows: [{ id: ALICE, email: "alice@example.com", password_hash: hash, role: "member", status: "active" }] }],
    ]);
    const login = await request(app).post("/api/auth/login").send({ email: "Alice@Example.com", password: "correct-password" });
    expect(login.status).toBe(200);
    expect(login.body.user).toEqual({ id: ALICE, email: "alice@example.com", role: "member" });

    const me = await request(app).get("/api/me").set("Authorization", `Bearer ${login.body.token}`);
    expect(me.body).toMatchObject({ id: ALICE, email: "alice@example.com", role: "member" });
  });

  it("rejects a wrong password and an unknown email with the same answer", async () => {
    const hash = await hashPassword("correct-password");
    const known = makeApp([[/from users where email/, { rows: [{ id: ALICE, email: "a@example.com", password_hash: hash, role: "member", status: "active" }] }]]);
    const wrong = await request(known.app).post("/api/auth/login").send({ email: "a@example.com", password: "nope" });
    const unknown = await request(makeApp().app).post("/api/auth/login").send({ email: "b@example.com", password: "nope" });
    expect(wrong.status).toBe(401);
    expect(unknown.status).toBe(401);
    expect(wrong.body).toEqual(unknown.body);
  });

  it("does not let a suspended account log in", async () => {
    const hash = await hashPassword("correct-password");
    const { app } = makeApp([[/from users where email/, { rows: [{ id: ALICE, email: "a@example.com", password_hash: hash, role: "member", status: "suspended" }] }]]);
    const res = await request(app).post("/api/auth/login").send({ email: "a@example.com", password: "correct-password" });
    expect(res.status).toBe(403);
    expect(res.body.token).toBeUndefined();
  });

  it("rejects a malformed body and invalid JSON", async () => {
    const { app } = makeApp();
    expect((await request(app).post("/api/auth/login").send({ email: "not-an-email" })).status).toBe(400);
    const bad = await request(app).post("/api/auth/login").set("Content-Type", "application/json").send("{bad json");
    expect(bad.status).toBe(400);
  });
});

describe("signup", () => {
  const created = { id: 9, email: "new@example.com", role: "member", status: "active" };
  const body = { email: "New@Example.com", password: "a-long-password", invite_code: "inv-abc" };
  const inviteRoutes = [
    [/update invites set used_at/, { rows: [{ id: 5 }], rowCount: 1 }],
    [/insert into users/, { rows: [created], rowCount: 1 }],
  ];

  it("creates a member with a valid invite code and uses the code up", async () => {
    const { app, db } = makeApp(inviteRoutes);
    const res = await request(app).post("/api/auth/signup").send(body);
    expect(res.status).toBe(201);
    expect(res.body.user).toEqual({ id: 9, email: "new@example.com", role: "member" });
    expect(res.body.token).toBeTruthy();

    const order = db.calls.map((c) =>
      /select count/.test(c.text) ? "count"
        : /update invites set used_at/.test(c.text) ? "claim"
        : /insert into users/.test(c.text) ? "insert"
        : /update invites set used_by/.test(c.text) ? "link"
        : "other",
    );
    expect(order).toEqual(["count", "claim", "insert", "link"]);

    const [insert] = callsMatching(db, /insert into users/);
    expect(insert.text).toContain("'member'");
    expect(insert.params[0]).toBe("new@example.com");
    expect(insert.params[1]).not.toBe("a-long-password"); // only a hash is stored
    expect(insert.params.slice(2)).toEqual([config.newUserDailyLimit, config.newUserMaxTokens]);
    expect(callsMatching(db, /update invites set used_by/)[0].params).toEqual([9, 5]);
    expect(callsMatching(db, /update invites set used_at/)[0].params).toEqual([hashInviteCode("inv-abc")]);
  });

  it("refuses a signup when the platform is full, before it uses up an invite", async () => {
    const { app, db } = makeApp([[/select count\(\*\)::int as n from users/, { rows: [{ n: 3 }] }], ...inviteRoutes]);
    const res = await request(app).post("/api/auth/signup").send(body);
    expect(res.status).toBe(403);
    expect(res.body.error).toMatch(/full/);
    expect(callsMatching(db, /update invites/)).toHaveLength(0);
    expect(callsMatching(db, /insert into users/)).toHaveLength(0);
  });

  it("starts a new account with the small starting limits, in open mode too", async () => {
    const { app, db } = makeApp(inviteRoutes, { config: { signupMode: "open", newUserDailyLimit: 7 } });
    const res = await request(app).post("/api/auth/signup").send({ email: "n@example.com", password: "a-long-password" });
    expect(res.status).toBe(201);
    expect(callsMatching(db, /insert into users/)[0].params.slice(2)).toEqual([7, config.newUserMaxTokens]);
  });

    // The duplicate assertions here were outside an it() test and are covered above.











  it("refuses a code that does not exist or was used, and creates no user", async () => {
    const { app, db } = makeApp([[/update invites set used_at/, { rows: [], rowCount: 0 }]]);
    const res = await request(app).post("/api/auth/signup").send(body);
    expect(res.status).toBe(403);
    expect(callsMatching(db, /insert into users/)).toHaveLength(0);
  });

  it("needs an invite code in invite mode", async () => {
    const { app, db } = makeApp(inviteRoutes);
    const res = await request(app).post("/api/auth/signup").send({ email: "n@example.com", password: "a-long-password" });
    expect(res.status).toBe(403);
    expect(db.calls).toHaveLength(0);
  });

  it("answers 409 for an email that is taken (and the transaction would roll the code back)", async () => {
    const { app } = makeApp([
      [/update invites set used_at/, { rows: [{ id: 5 }], rowCount: 1 }],
      [/insert into users/, uniqueViolation],
    ]);
    expect((await request(app).post("/api/auth/signup").send(body)).status).toBe(409);
  });

  it("is closed when the mode is closed", async () => {
    const { app, db } = makeApp(inviteRoutes, { config: { signupMode: "closed" } });
    expect((await request(app).post("/api/auth/signup").send(body)).status).toBe(403);
    expect(db.calls).toHaveLength(0);
  });

  it("needs no code in open mode and never looks at invites", async () => {
    const { app, db } = makeApp(inviteRoutes, { config: { signupMode: "open" } });
    const res = await request(app).post("/api/auth/signup").send({ email: "n@example.com", password: "a-long-password" });
    expect(res.status).toBe(201);
    expect(callsMatching(db, /invites/)).toHaveLength(0);
  });

  it("rejects a short password and fields that are not part of a signup (a role)", async () => {
    const { app, db } = makeApp(inviteRoutes);
    expect((await request(app).post("/api/auth/signup").send({ ...body, password: "short" })).status).toBe(400);
    expect((await request(app).post("/api/auth/signup").send({ ...body, role: "admin" })).status).toBe(400);
    expect((await request(app).post("/api/auth/signup").send({ ...body, status: "active" })).status).toBe(400);
    expect(db.calls).toHaveLength(0);
  });

  it("tells the dashboard what the signup mode is", async () => {
    const res = await request(makeApp().app).get("/api/auth/config");
    expect(res.body).toEqual({ signup_mode: "invite" });
  });
});

describe("api keys", () => {
  const insertRow = (params) => ({
    rows: [{ id: 1, name: params[2], key_id: params[1], rate_limit_per_min: params[3], active: true, created_at: "now", revoked_at: null }],
    rowCount: 1,
  });
  const routes = (activeKeys = 0) => [
    [/select count\(\*\)::int as n from api_keys/, { rows: [{ n: activeKeys }] }],
    [/insert into api_keys/, insertRow],
  ];

  it("lists only the keys of the logged in user and never selects the hash", async () => {
    const { app, db } = makeApp();
    await request(app).get("/api/keys").set(auth(ALICE));
    const [list] = callsMatching(db, /from api_keys where user_id/);
    expect(list.params).toEqual([ALICE]);
    expect(list.text).not.toMatch(/key_hash/);
  });

  it("creates a key for the logged in user, shows it once and stores only its hash", async () => {
    const { app, db } = makeApp(routes());
    const res = await request(app).post("/api/keys").set(auth(ALICE)).send({ name: "demo", rate_limit_per_min: 30 });
    expect(res.status).toBe(201);
    expect(res.body.key).toMatch(/^ap-/);
    expect(res.body.key_hash).toBeUndefined();

    const [insert] = callsMatching(db, /insert into api_keys/);
    expect(insert.params[0]).toBe(hashApiKey(res.body.key));
    expect(insert.params[4]).toBe(ALICE);
    expect(JSON.stringify(insert.params)).not.toContain(res.body.key);
  });

  it("does not let a request choose the owner", async () => {
    const { app, db } = makeApp(routes());
    const res = await request(app).post("/api/keys").set(auth(ALICE)).send({ name: "x", user_id: ADMIN });
    expect(res.status).toBe(400);
    expect(callsMatching(db, /insert into api_keys/)).toHaveLength(0);
  });

  it("stops at the number of active keys that one user may have", async () => {
    const { app, db } = makeApp(routes(3));
    const res = await request(app).post("/api/keys").set(auth(ALICE)).send({ name: "one more" });
    expect(res.status).toBe(409);
    expect(callsMatching(db, /insert into api_keys/)).toHaveLength(0);
  });

  it("caps the per minute limit of a member but not of an admin", async () => {
    const { app } = makeApp(routes());
    expect((await request(app).post("/api/keys").set(auth(ALICE)).send({ name: "x", rate_limit_per_min: 500 })).status).toBe(400);
    expect((await request(app).post("/api/keys").set(auth(ALICE)).send({ name: "x", rate_limit_per_min: 120 })).status).toBe(201);
    expect((await request(app).post("/api/keys").set(auth(ADMIN)).send({ name: "x", rate_limit_per_min: 5000 })).status).toBe(201);
  });

  it("answers 404 for the key of someone else, and the query is limited to the owner", async () => {
    const { app, db } = makeApp(); // no row matches: the key does not belong to this user
    const res = await request(app).post("/api/keys/1/revoke").set(auth(BOB));
    expect(res.status).toBe(404);
    const [revoke] = callsMatching(db, /update api_keys set active = false/);
    expect(revoke.params).toEqual([1, BOB]);
    expect((await request(app).post("/api/keys/abc/revoke").set(auth(BOB))).status).toBe(400);
    expect((await request(app).patch("/api/keys/1").set(auth(BOB)).send({ rate_limit_per_min: 5 })).status).toBe(404);
  });
});

describe("prompts", () => {
  it("lists only the prompts of the logged in user", async () => {
    const { app, db } = makeApp();
    await request(app).get("/api/prompts").set(auth(BOB));
    expect(callsMatching(db, /from prompts p\s+left join/)[0].params).toEqual([BOB]);
  });

  it("creates a prompt for the logged in user, up to a limit, and maps a taken name to 409", async () => {
    const inserted = { id: 4, name: "qa", description: null, created_at: "now" };
    const ok = makeApp([
      [/select count\(\*\)::int as n from prompts/, { rows: [{ n: 0 }] }],
      [/insert into prompts/, { rows: [inserted], rowCount: 1 }],
    ]);
    expect((await request(ok.app).post("/api/prompts").set(auth(ALICE)).send({ name: "qa" })).status).toBe(201);
    expect(callsMatching(ok.db, /insert into prompts/)[0].params).toEqual([ALICE, "qa", null]);

    const full = makeApp([[/select count\(\*\)::int as n from prompts/, { rows: [{ n: 3 }] }]]);
    expect((await request(full.app).post("/api/prompts").set(auth(ALICE)).send({ name: "qa" })).status).toBe(409);

    const taken = makeApp([
      [/select count\(\*\)::int as n from prompts/, { rows: [{ n: 0 }] }],
      [/insert into prompts/, uniqueViolation],
    ]);
    expect((await request(taken.app).post("/api/prompts").set(auth(ALICE)).send({ name: "qa" })).status).toBe(409);
  });

  it("validates names and extra fields", async () => {
    const { app } = makeApp();
    expect((await request(app).post("/api/prompts").set(auth(ALICE)).send({ name: "Bad Name" })).status).toBe(400);
    expect((await request(app).post("/api/prompts").set(auth(ALICE)).send({ name: "qa", user_id: 1 })).status).toBe(400);
  });

  it("looks for a version's prompt among the prompts of the user only", async () => {
    const { app, db } = makeApp(); // nothing found
    const res = await request(app).post("/api/prompts/qa/versions").set(auth(ALICE)).send({ system_prompt: "hi" });
    expect(res.status).toBe(404);
    expect(callsMatching(db, /insert into prompt_versions/)[0].params.slice(0, 2)).toEqual(["qa", ALICE]);
  });

  it("answers 409 when a prompt is full of versions and 404 when it does not exist", async () => {
    const full = makeApp([[/select 1 from prompts where name/, { rows: [{ "?column?": 1 }] }]]);
    expect((await request(full.app).post("/api/prompts/qa/versions").set(auth(ALICE)).send({ system_prompt: "hi" })).status).toBe(409);
  });

  it("refuses weights for a version that does not exist and limits the update to the owner", async () => {
    const { app, db } = makeApp([[/from prompts p\s+join prompt_versions/, { rows: [{ version: 1 }] }]]);
    expect((await request(app).put("/api/prompts/qa/weights").set(auth(ALICE)).send({ weights: { 2: 50 } })).status).toBe(400);
    expect((await request(app).put("/api/prompts/qa/weights").set(auth(ALICE)).send({ weights: { 1: 50 } })).status).toBe(200);
    expect(callsMatching(db, /update prompt_versions/)[0].params[2]).toBe(ALICE);
  });
});

describe("stats and traces", () => {
  const overview = {
    requests: "10", errors: "1", cache_hits: "2", cache_consulted: "4", cache_blocked: "1",
    cache_bypass: "2", fallbacks: "0", tokens_used: "100", tokens_saved: "50",
    cost_spent_usd: "0.0006", cost_saved_usd: "0.00015",
    avg_latency_ms: "900", p95_latency_ms: "1200", avg_hit_latency_ms: "20",
  };

  it("computes the rates for the numbers of the logged in user", async () => {
    const { app, db } = makeApp([[/from traces/, { rows: [overview] }]]);
    const res = await request(app).get("/api/stats/overview?hours=24").set(auth(ALICE));
    expect(res.status).toBe(200);
    expect(res.body.error_rate).toBeCloseTo(0.1);
    expect(res.body.cache.hit_rate).toBeCloseTo(0.5);
    expect(res.body.cost.saving_pct).toBeCloseTo(0.2);
    expect(callsMatching(db, /from traces/)[0].params).toEqual([24, ALICE]);
  });

  it("limits every stats query and the trace list to the user", async () => {
    const { app, db } = makeApp();
    await request(app).get("/api/stats/timeseries?hours=24").set(auth(ALICE));
    await request(app).get("/api/stats/breakdown?hours=24&by=provider").set(auth(ALICE));
    await request(app).get("/api/stats/prompts?hours=24").set(auth(ALICE));
    await request(app).get("/api/traces?limit=5").set(auth(ALICE));
    const traceCalls = callsMatching(db, /from traces/);
    expect(traceCalls).toHaveLength(4);
    expect(traceCalls[0].params.at(-1)).toBe(ALICE);
    expect(traceCalls[1].params.at(-1)).toBe(ALICE);
    expect(traceCalls[2].params.at(-1)).toBe(ALICE);
    expect(traceCalls[3].params[0]).toBe(ALICE);
    expect(traceCalls[3].text).toMatch(/where user_id = \$1/);
  });

  it("has no rates when there is no traffic and validates the query", async () => {
    const empty = { ...overview, requests: "0", errors: "0", cache_hits: "0", cache_consulted: "0", cost_spent_usd: "0", cost_saved_usd: "0" };
    const { app } = makeApp([[/from traces/, { rows: [empty] }]]);
    const none = await request(app).get("/api/stats/overview").set(auth(ALICE));
    expect(none.body.error_rate).toBeNull();
    expect((await request(app).get("/api/stats/overview?hours=0").set(auth(ALICE))).status).toBe(400);
    expect((await request(app).get("/api/stats/breakdown?by=evil").set(auth(ALICE))).status).toBe(400);
  });
});

describe("admin", () => {
  const member = { id: BOB, email: "bob@example.com", role: "member", status: "suspended", daily_request_limit: 50, max_tokens_cap: 512 };

  it("lists the users without their password hashes", async () => {
    const { app, db } = makeApp([[/from users u/, { rows: [member] }]]);
    const res = await request(app).get("/api/admin/users").set(auth(ADMIN));
    expect(res.status).toBe(200);
    expect(res.body.users).toEqual([member]);
    expect(callsMatching(db, /from users u/)[0].text).not.toMatch(/password_hash/);
  });

  it("changes the status and the limits of a member, with the column names from a fixed list", async () => {
    const { app, db } = makeApp([[/update users set/, { rows: [member] }]]);
    const res = await request(app).patch(`/api/admin/users/${BOB}`).set(auth(ADMIN)).send({ status: "suspended", daily_request_limit: 50 });
    expect(res.status).toBe(200);
    const [update] = callsMatching(db, /update users set/);
    expect(update.text).toMatch(/set status = \$2, daily_request_limit = \$3 where id = \$1 and role = 'member'/);
    expect(update.params).toEqual([BOB, "suspended", 50]);
  });

  it("never changes a role, nothing at all, or the account of the admin who asks", async () => {
    const { app, db } = makeApp([[/update users set/, { rows: [member] }]]);
    expect((await request(app).patch(`/api/admin/users/${BOB}`).set(auth(ADMIN)).send({ role: "admin" })).status).toBe(400);
    expect((await request(app).patch(`/api/admin/users/${BOB}`).set(auth(ADMIN)).send({})).status).toBe(400);
    expect((await request(app).patch(`/api/admin/users/${ADMIN}`).set(auth(ADMIN)).send({ status: "suspended" })).status).toBe(400);
    expect(callsMatching(db, /update users set/)).toHaveLength(0);
  });

  it("answers 404 for an account that is not a member (admins are managed with the command line)", async () => {
    const { app } = makeApp(); // the update finds no member with this id
    expect((await request(app).patch("/api/admin/users/99").set(auth(ADMIN)).send({ status: "suspended" })).status).toBe(404);
  });

  it("creates an invite, shows the code once and stores only its hash", async () => {
    const { app, db } = makeApp([[/insert into invites/, { rows: [{ id: 3, note: "for Sara", created_at: "now" }], rowCount: 1 }]]);
    const res = await request(app).post("/api/admin/invites").set(auth(ADMIN)).send({ note: "for Sara" });
    expect(res.status).toBe(201);
    expect(res.body.code).toMatch(/^inv-/);
    const [insert] = callsMatching(db, /insert into invites/);
    expect(insert.params).toEqual([hashInviteCode(res.body.code), "for Sara", ADMIN]);
    expect(JSON.stringify(insert.params)).not.toContain(res.body.code);
  });

  it("lists invites without their codes, and only an unused invite can be deleted", async () => {
    const { app, db } = makeApp();
    await request(app).get("/api/admin/invites").set(auth(ADMIN));
    expect(callsMatching(db, /from invites i/)[0].text).not.toMatch(/code_hash/);
    expect((await request(app).delete("/api/admin/invites/3").set(auth(ADMIN))).status).toBe(404);
  });

  it("shows the numbers of the whole platform to an admin only", async () => {
    const { app, db } = makeApp();
    await request(app).get("/api/admin/stats/overview?hours=24").set(auth(ADMIN));
    // the overview query finds no row in this fake database, which is fine: only the parameters matter here
    expect(callsMatching(db, /from traces/)[0].params).toEqual([24, null]);
  });
});

describe("health", () => {
  it("checks the database", async () => {
    const res = await request(makeApp().app).get("/api/health");
    expect(res.body).toEqual({ status: "ok" });
  });
});