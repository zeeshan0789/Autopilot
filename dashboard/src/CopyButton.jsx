import { useState } from "react";
import { Button } from "./components.jsx";

export function CopyButton({ text, label = "Copy" }) {
  const [copied, setCopied] = useState(false);

  async function copy() {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      setCopied(false);
    }
  }

  return (
    <Button variant="secondary" onClick={copy}>
      {copied ? "Copied" : label}
    </Button>
  );
}

export function Snippet({ text }) {
  return (
    <div className="relative">
      <pre className="overflow-x-auto rounded-lg bg-slate-900 p-3 pr-20 text-xs text-slate-100">{text}</pre>
      <div className="absolute right-2 top-2">
        <CopyButton text={text} />
      </div>
    </div>
  );
}