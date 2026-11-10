import { useState } from "react";
import { Button, Card, ErrorBox, Loading } from "./components.jsx";
import TraceTable from "./TraceTable.jsx";
import { useFetch } from "./useFetch.js";

const LIMIT = 25;

export default function Traces() {
  const [status, setStatus] = useState("");
  const [offset, setOffset] = useState(0);
  const { data, error, loading } = useFetch("/api/traces", { limit: LIMIT, offset, status });
  const traces = data?.traces ?? [];

  function changeStatus(value) {
    setStatus(value);
    setOffset(0);
  }

  return (
    <>
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-bold text-slate-900">Traces</h1>
        <select
          value={status}
          onChange={(event) => changeStatus(event.target.value)}
          className="rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-sm"
        >
          <option value="">All requests</option>
          <option value="success">Successful</option>
          <option value="error">Errors</option>
        </select>
      </div>

      <Card subtitle="Prompts are not stored in traces, only the numbers about each request.">
        <ErrorBox error={error} />
        {loading && !data ? <Loading /> : <TraceTable traces={traces} />}
        <div className="mt-4 flex items-center gap-2">
          <Button variant="secondary" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - LIMIT))}>
            Newer
          </Button>
          <Button variant="secondary" disabled={traces.length < LIMIT} onClick={() => setOffset(offset + LIMIT)}>
            Older
          </Button>
          <span className="text-xs text-slate-500">
            Showing {traces.length === 0 ? 0 : offset + 1} to {offset + traces.length}
          </span>
        </div>
      </Card>
    </>
  );
}