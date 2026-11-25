// Usage: npm run create-user -- <email> <password> [member|admin]
// This is the only way to create an admin: there is no signup route for it.
import "dotenv/config";
import { hashPassword } from "../auth.js";
import { createDb } from "../db.js";

const [emailArg, password, role = "member"] = process.argv.slice(2);

if (!emailArg || !emailArg.includes("@") || !password || password.length < 10) {
  console.error("Usage: npm run create-user -- <email> <password of 10+ characters> [member|admin]");
  process.exit(1);
}
if (!["admin", "member"].includes(role)) {
  console.error("The role must be member or admin");
  process.exit(1);
}

const db = createDb(process.env.DATABASE_URL);
try {
  const limits = role === "admin" ? [1_000_000, 100_000] : [200, 1024];
  await db.query(
    "insert into users (email, password_hash, role, daily_request_limit, max_tokens_cap) values ($1, $2, $3, $4, $5)",
    [emailArg.toLowerCase(), await hashPassword(password), role, limits[0], limits[1]],
  );
  console.log(`user ${emailArg.toLowerCase()} created with role ${role}`);
} catch (e) {
  if (e.code === "23505") console.error("a user with this email already exists");
  else throw e;
} finally {
  await db.close();
}