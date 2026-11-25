// Gives the keys, prompts and traces that were created before accounts existed to one user.
// Usage: npm run claim-legacy -- <email of the owner>
import "dotenv/config";
import { createDb } from "../db.js";

const [emailArg] = process.argv.slice(2);
if (!emailArg) {
  console.error("Usage: npm run claim-legacy -- <email of the owner>");
  process.exit(1);
}

const db = createDb(process.env.DATABASE_URL);
try {
  const found = await db.query("select id, email from users where email = $1", [emailArg.toLowerCase()]);
  const owner = found.rows[0];
  if (!owner) {
    console.error("no user with this email");
    process.exitCode = 1;
  } else {
    const counts = {};
    for (const table of ["api_keys", "prompts", "traces"]) {
      const result = await db.query(`update ${table} set user_id = $1 where user_id is null`, [owner.id]);
      counts[table] = result.rowCount;
    }
    console.log(`assigned to ${owner.email}: ${counts.api_keys} keys, ${counts.prompts} prompts, ${counts.traces} traces`);
  }
} finally {
  await db.close();
}