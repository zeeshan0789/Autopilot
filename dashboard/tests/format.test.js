import { describe, expect, it } from "vitest";
import { int, ms, pct, usd } from "../src/format.js";

describe("format", () => {
  it("shows n/a for missing values", () => {
    expect(int(null)).toBe("n/a");
    expect(pct(undefined)).toBe("n/a");
    expect(ms(null)).toBe("n/a");
    expect(usd(null)).toBe("n/a");
  });

  it("formats numbers", () => {
    expect(int(1234.4)).toBe("1,234");
    expect(pct(0.5)).toBe("50.0%");
    expect(pct(0.1234, 0)).toBe("12%");
    expect(ms(21.4)).toBe("21 ms");
  });

  it("shows tiny costs with enough digits", () => {
    expect(usd(0)).toBe("$0");
    expect(usd(0.00028)).toBe("$0.000280");
    expect(usd(0.1234)).toBe("$0.1234");
  });
});