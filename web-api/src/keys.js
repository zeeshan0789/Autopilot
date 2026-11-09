import { createHash, randomBytes } from "node:crypto";

// Same format and same hash as the Python gateway: "ap-" plus 43 url safe characters,
// and the SHA 256 hex digest of the whole key.
export function generateApiKey() {
  return "ap-" + randomBytes(32).toString("base64url");
}

export function hashApiKey(key) {
  return createHash("sha256").update(key, "utf8").digest("hex");
}

export function keyIdOf(hash) {
  return hash.slice(0, 8);
}