import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api.js";

// Loads a path whenever the path or the query changes. An older answer never replaces a newer one.
export function useFetch(path, query) {
  const [state, setState] = useState({ data: null, error: null, loading: true });
  const latest = useRef(0);
  const key = JSON.stringify([path, query]);

  const load = useCallback(async () => {
    const id = ++latest.current;
    setState((s) => ({ ...s, loading: true, error: null }));
    try {
      const data = await api(path, { query });
      if (id === latest.current) setState({ data, error: null, loading: false });
    } catch (error) {
      if (id === latest.current) setState({ data: null, error, loading: false });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);

  useEffect(() => {
    load();
  }, [load]);

  return { ...state, reload: load };
}