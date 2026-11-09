// Usage: npm run create-user -- <email> <password> [admin|viewer]
import "dotenv/config";
import { hashPassword } from "../auth.js";
import { createDb } from "../db.js";

const [emailArg, password, role = "admin"] = process.argv.slice(2);

if (!emailArg || !emailArg.includes("@") || !password || password.length < 10) {
  console.error("Usage: npm run create-user -- <email> <password of 10+ characters> [admin|viewer]");
  process.exit(1);
}
if (!["admin", "viewer"].includes(role)) {
  console.error("The role must be admin or viewer");
  process.exit(1);
}

const db = createDb(process.env.DATABASE_URL);
try {
  await db.query("insert into users (email, password_hash, role) values ($1, $2, $3)", [
    emailArg.toLowerCase(),
    await hashPassword(password),
    role,
  ]);
  console.log(`user ${emailArg.toLowerCase()} created with role ${role}`);
} catch (e) {
  if (e.code === "23505") console.error("a user with this email already exists");
  else throw e;
} finally {
  await db.close();
}