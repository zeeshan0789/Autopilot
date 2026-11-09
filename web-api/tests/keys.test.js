import { describe, expect, it } from "vitest";
import { generateApiKey, hashApiKey, keyIdOf } from "../src/keys.js";

describe("api keys", () => {
  it("hashes exactly like the Python gateway (known SHA 256 vector)", () => {
    expect(hashApiKey("abc")).toBe("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  });

  it("generates keys in the gateway format", () => {
    const key = generateApiKey();
    expect(key).toMatch(/^ap-[A-Za-z0-9_-]{43}$/);
    expect(generateApiKey()).not.toBe(key);
  });

  it("uses the first 8 characters of the hash as the short id", () => {
    const hash = hashApiKey("abc");
    expect(keyIdOf(hash)).toBe("ba7816bf");
  });
});