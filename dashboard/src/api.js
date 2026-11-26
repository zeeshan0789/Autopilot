// The web API (Node.js). It is separate from the gateway on port 8000.
const API_URL = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:3001";
// Where applications send their requests. Only shown in the examples: the dashboard itself never calls it.
export const GATEWAY_URL = import.meta.env.VITE_GATEWAY_URL ?? "http://127.0.0.1:8000";
const TOKEN_KEY = "autopilot_token";

// sessionStorage: the login ends when the tab is closed
export const getToken = () => sessionStorage.getItem(TOKEN_KEY);
export const setToken = (token) =>
  token ? sessionStorage.setItem(TOKEN_KEY, token) : sessionStorage.removeItem(TOKEN_KEY);

let onUnauthorized = () => {};
export const setUnauthorizedHandler = (handler) => {
  onUnauthorized = handler;
};

export class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

export async function api(path, { method = "GET", body, query } = {}) {
  const url = new URL(path, API_URL);
  for (const [key, value] of Object.entries(query ?? {})) {
    if (value !== undefined && value !== null && value !== "") url.searchParams.set(key, value);
  }

  const headers = {};
  if (body !== undefined) headers["Content-Type"] = "application/json";
  const token = getToken();
  if (token) headers.Authorization = `Bearer ${token}`;

  let response;
  try {
    response = await fetch(url, {
      method,
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    throw new ApiError(0, `Cannot reach the web API at ${API_URL}. Is it running?`);
  }

  const data = await response.json().catch(() => ({}));
  const message = data.error ?? `request failed (${response.status})`;
  // An expired login, or an account that was suspended while the page was open: back to the login page
  if (token && (response.status === 401 || (response.status === 403 && /suspended/i.test(message)))) {
    onUnauthorized();
  }
  if (!response.ok) throw new ApiError(response.status, message);
  return data;
}