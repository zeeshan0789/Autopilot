# Autopilot

An OpenAI compatible LLM gateway prototype. It sits between an application and its model providers and adds reliability (retries, fallback), access control (API keys, rate limiting) and observability (a trace for every request, with estimated cost).

Everything runs locally for free: local models through Ollama and free API tiers. No billing, no credit card.

**Status:** Phase 1 (gateway MVP) is complete. Caching, routing, evaluation and a dashboard are planned and not built yet.

## What it does today

- OpenAI compatible `POST /v1/chat/completions` (non streaming)
- Providers are defined in config, with no per provider if/else in code: local Ollama, Groq, and a mock provider that can simulate errors, 429s and timeouts
- Timeouts, retries with exponential backoff and jitter, and an ordered fallback chain per model
- API key authentication (keys are stored only as SHA 256 hashes)
- Redis rate limiting per key (fixed window) with `429` and `Retry-After`
- A PostgreSQL trace for every request that reaches the handler: requested and served model, provider, failed attempts, tokens, estimated cost, latency, status
- Automated tests: unit tests plus integration tests against real Redis and PostgreSQL

## Architecture

```
Client
  │  Authorization: Bearer ap-...
  ▼
FastAPI gateway
  1. API key check          → 401 if missing or invalid
  2. Redis rate limit       → 429 if over the limit
  3. Model → fallback chain (config/providers.yaml)
  4. Provider call: timeout, retry with backoff
       Groq → Ollama (local) → Mock
  5. Estimated cost (config/pricing.yaml)
  6. Trace row → PostgreSQL (background task, never blocks the response)
  │
  ▼
OpenAI format response + X-Request-ID and X-Autopilot-* headers
```

## Quick start (Windows PowerShell)

Prerequisites: Python 3.11, Docker Desktop, Ollama, and a free Groq API key.

```
ollama pull qwen2.5:0.5b

docker compose up -d

cd ai-gateway
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Put your Groq key in `ai-gateway/.env`, then:

```
python -m scripts.init_db
python -m scripts.create_key dev 60
uvicorn app.main:app --reload
```

`create_key` prints the key once. Only its hash is stored in `ai-gateway/data/api_keys.json` (git ignored).

Call the gateway:

```
$key = "ap-your-key"
$body = @{ model = "openai/gpt-oss-20b"; messages = @(@{ role = "user"; content = "Say hello in one short sentence." }) } | ConvertTo-Json -Depth 5
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/chat/completions -ContentType "application/json" -Headers @{ Authorization = "Bearer $key" } -Body $body
```

Only send public or synthetic data: free tier providers may use prompts to improve their products.

## Models and fallback

Configured in `ai-gateway/config/providers.yaml`:

| Requested model | Provider | Fallback order |
|---|---|---|
| `openai/gpt-oss-120b` | Groq | `openai/gpt-oss-20b`, then `qwen2.5:0.5b` |
| `openai/gpt-oss-20b` | Groq | `qwen2.5:0.5b` |
| `qwen2.5:0.5b` | Ollama (local) | none |
| `mock`, `mock-down`, `mock-429`, `mock-dead` | Mock | used for testing failures |

Response headers: `X-Request-ID`, `X-Autopilot-Provider`, `X-Autopilot-Model`, `X-Autopilot-Fallback`, `X-RateLimit-Limit`, `X-RateLimit-Remaining`, and `Retry-After` on a 429.

## Inspect traces

```
docker exec autopilot-postgres psql -U autopilot -d autopilot -c "select created_at, requested_model, served_model, provider, fallback_used, status, total_tokens, estimated_cost_usd, latency_ms from traces order by created_at desc limit 10;"
```

## Tests

```
cd ai-gateway
pytest -v                      # everything (integration tests skip if Docker is down)
pytest -v -m "not integration" # fast, no Docker needed
pytest -v -m integration       # real Redis and PostgreSQL
```

## Design decisions

- **Rate limiter fails open.** If Redis is down, requests pass and a warning is logged. Redis going down should not take the gateway down. The gateway pings Redis at startup so this is visible immediately.
- **Tracing never blocks a request.** Traces are written in a background task and database errors are swallowed and logged.
- **Cost uses `Decimal`, and the model that actually answered.** If a fallback served the request, the fallback model's rate is used. A model with no price gets `NULL` cost, not zero.
- **Use `127.0.0.1`, not `localhost`.** On Windows `localhost` can try IPv6 first and add a one second delay. Tests enforce this.
- **Python `None` is stored as SQL `NULL`** (`JSONB(none_as_null=True)`), and an integration test checks it at the SQL level.

## Limitations

- **Estimated cost is not real spend.** Actual spend is $0 (free tiers and a local model). Cost is calculated from published list prices in `config/pricing.yaml` (source and check date recorded there). The local model rate is a notional placeholder, not a market price. Some third party sites list different prices for the same model, so verify rates before relying on them.
- Free tier limits and models change without notice. No limits are hardcoded; they live in config.
- The rate limiter is a fixed window: around a minute boundary a client can send up to twice the limit.
- Traces can be lost if the server stops right after a request or the database is down. Requests rejected by auth (401), validation (422) or the gateway rate limit are not traced yet.
- API keys are stored in a local JSON file (hashed). Moving them to PostgreSQL is planned.
- No streaming yet. Mock provider token counts are word counts.
- Tables are created with `create_all`. Migrations will be added when the schema changes.
- The development database password is for local use only.

## Roadmap

1. Policy layer and a semantic cache with safety checks
2. Prompt versioning, A/B traffic splitting, and a dashboard
3. Evaluation, judge calibration and regression detection
4. ML router and failure forensics
5. RAG and Text to SQL demos, a benchmark with real measurements, CI