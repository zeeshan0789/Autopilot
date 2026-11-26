import { describe, expect, it } from "vitest";
import { curlSnippet, powershellSnippet } from "../src/snippets.js";

describe("examples", () => {
  it("use the gateway address and a placeholder key, never a real one", () => {
    for (const text of [powershellSnippet("http://gw:8000"), curlSnippet("http://gw:8000")]) {
      expect(text).toContain("http://gw:8000/v1/chat/completions");
      expect(text).toContain("ap-YOUR-KEY");
      expect(text).toContain('"auto"');
    }
  });

  it("can show another placeholder", () => {
    expect(powershellSnippet("http://gw:8000", "ap-OTHER")).toContain('$key = "ap-OTHER"');
    expect(curlSnippet("http://gw:8000", "ap-OTHER")).toContain("Bearer ap-OTHER");
  });
});