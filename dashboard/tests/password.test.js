import { describe, expect, it } from "vitest";
import { passwordProblems, passwordStrength } from "../src/password.js";

describe("password hints", () => {
  it("asks for 10 characters", () => {
    expect(passwordProblems("short")).toEqual(["use at least 10 characters"]);
    expect(passwordProblems("a-long-enough-one")).toEqual([]);
  });

  it("does not accept one repeated character", () => {
    expect(passwordProblems("aaaaaaaaaaaa")).toEqual(["do not repeat one character"]);
    expect(passwordProblems("aaaaa")).toEqual(["use at least 10 characters", "do not repeat one character"]);
  });

  it("rates the strength", () => {
    expect(passwordStrength("")).toEqual({ score: 0, label: "" });
    expect(passwordStrength("short")).toEqual({ score: 0, label: "too short" });
    expect(passwordStrength("abcdefghij")).toEqual({ score: 1, label: "weak" });
    expect(passwordStrength("abcdefghij1")).toEqual({ score: 2, label: "fair" });
    expect(passwordStrength("correct horse battery staple")).toEqual({ score: 3, label: "good" });
    expect(passwordStrength("correct horse battery staple 42")).toEqual({ score: 4, label: "strong" });
  });
});