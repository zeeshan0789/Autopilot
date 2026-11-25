// Creates an invite code for a signup. The code is shown once, only its hash is stored.
// Usage: npm run create-invite -- ["a note about who it is for"]
import "dotenv/config";
import { createDb } from "../db.js";
import { generateInviteCode, hashInviteCode } from "../keys.js";

const [note = null] = process.argv.slice(2);
const db = createDb(process.env.DATABASE_URL);
try {
  const code = generateInviteCode();
  await db.query("insert into invites (code_hash, note) values ($1, $2)", [hashInviteCode(code), note]);
  console.log(`invite code (shown once): ${code}`);
} finally {
  await db.close();
}