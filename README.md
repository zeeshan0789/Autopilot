# Autopilot

An OpenAI compatible LLM gateway prototype with a web dashboard. The gateway sits between an application and its model providers and adds reliability (retries, fallback), access control (API keys, rate limiting), cost awareness (a safe semantic cache and a routing policy), prompt versioning with A/B traffic splits, and a trace for every request. A Node.js web API and a React dashboard show the numbers and manage keys and prompts.

Everything runs locally for free: local models through Ollama and free API tiers. No billing, no credit card.

**Status:** phases 1 to 3 are complete (gateway, routing policy and semantic cache, prompt versioning and the dashboard). Evaluation, an ML router and failure forensics are planned and not built yet.

## What it does today

**Gateway (FastAPI, Python)**

- OpenAI compatible `POST /v1/chat/completions` (non streaming)
- Providers are defined in config: local Ollama, Groq, and a mock provider that can simulate errors, 429s and timeouts
- Timeouts, retries with exponential backoff and jitter, an ordered fallback chain per model
- API keys stored as SHA 256 hashes in PostgreSQL, Redis rate limiting per key
- Routing policy: `model: "auto"` lets declarative rules choose between a simple and a complex model
- Safe semantic cache: similar prompts reuse an answer only if a guard confirms that numbers, years, names, negations and style words match
- Managed prompts: named prompts with versions and a weighted A/B split (stable per `user`), filled with `{{variables}}`
- A PostgreSQL trace for every request: requested, routed and served model, provider, cache result, prompt version, tokens, estimated cost, estimated savings, latency

**Web API (Node.js, Express)**

- Dashboard login (JWT, bcrypt) with two roles: `admin` can do everything, `viewer` can only look
- Create and revoke gateway API keys (the key is shown once, only its hash is stored), change limits
- Create prompts, add versions (a version is never edited, a change is a new version), change weights
- Stats: overview, time series, breakdowns by route, provider, model or cache result, A/B numbers per prompt version, recent traces

**Dashboard (React, Vite, Tailwind, Recharts)**

- Login, overview with charts, traces, prompts and A/B, API keys (admin only)

## Architecture

```
 Browser ──► Dashboard (React, nginx, :5173) ──► Web API (Node.js, :3001) ──┐
                                                                            ▼
 Client app ──► Gateway (FastAPI, :8000) ──► Groq, Ollama (local) ──► PostgreSQL
                    │                                                  ▲
                    ├── Redis (rate limits)                            │
                    └── writes one trace row per request ──────────────┘
```

Request path inside the gateway:

```
  1. API key check            401 if missing or invalid (503 if the key store is down)
  2. Redis rate limit         429 if over the limit
  3. Managed prompt           pick a version (A/B), fill the template
  4. Routing policy           model "auto" → simple or complex model (config/policy.yaml)
  5. Semantic cache lookup    hit → answer returned, no provider call
  6. Provider call            timeout, retry with backoff, fallback chain
  7. Store the answer         never fallback answers, empty answers or refusals
  8. Estimated cost, then a trace row (background task, never blocks the response)
```

The Python gateway owns the database schema (Alembic migrations). The Node.js web API reads the traces and manages `api_keys`, `users` and the prompt tables. The request path never goes through Node.js.

## Quick start (Windows PowerShell)

Prerequisites: Python 3.11, Docker Desktop, Ollama, a free Groq API key (Node.js is only needed to develop the web API or dashboard outside Docker).

```
ollama pull qwen2.5:0.5b

cd ai-gateway
copy .env.example .env             # put your Groq key in .env
cd ..\web-api
copy .env.example .env             # set JWT_SECRET (see the comment inside)
cd ..
docker compose up -d --build       # Redis, PostgreSQL, web API, dashboard

cd ai-gateway
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
alembic upgrade head
python -m scripts.create_key dev 60
uvicorn app.main:app --reload
```

Create a dashboard user and open http://localhost:5173 (use `localhost`, the web API only allows that origin):

```
docker compose exec web-api node src/scripts/create-user.js admin@example.com "a long password" admin
```

The first gateway start downloads the embedding model (about 90 MB) from Hugging Face. After that you can add `HF_HUB_OFFLINE=1` to `ai-gateway/.env` to start without the network. Only send public or synthetic data: free tier providers may use prompts to improve their products.

```
$key = "ap-your-key"
$body = @{ model = "auto"; messages = @(@{ role = "user"; content = "What is the capital of France?" }) } | ConvertTo-Json -Depth 5
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/chat/completions -ContentType "application/json" -Headers @{ Authorization = "Bearer $key" } -Body $body
```

Ask the same question twice and the second answer comes from the cache (`X-Autopilot-Cache: hit`).

Managed prompts (A/B): create a prompt and versions in the dashboard (or `python -m scripts.prompts_admin`), then add `"autopilot": {"prompt": "support"}` to a request. Add `"user": "some id"` to keep one person on one version.

## Response headers

`X-Request-ID`, `X-Autopilot-Cache` (hit, miss, blocked, bypass), `X-Autopilot-Provider` (`cache` on a hit), `X-Autopilot-Model`, `X-Autopilot-Fallback`, `X-Autopilot-Route` and `X-Autopilot-Routed-Model` (only for `auto`), `X-Autopilot-Prompt` (`name:version`), `X-RateLimit-Limit`, `X-RateLimit-Remaining`, and `Retry-After` on a 429.

## Routing policy

Rules live in `ai-gateway/config/policy.yaml`: the first matching rule wins, otherwise the default tier is used. Conditions are length, history length, number of questions, requested output size, keywords and code detection. The shipped rules send code, analysis style keywords, long prompts, long conversations, multi part questions and large outputs to `openai/gpt-oss-120b`, and everything else to `openai/gpt-oss-20b`. These rules are heuristics with untuned thresholds. Nothing here shows that they improve answer quality: that needs the evaluation planned for phase 4. The list price gap between the two models is small, so today the main purpose is sending hard prompts to the bigger model and building the mechanism a learned router will later use.

## Semantic cache

- **Scope:** entries are only compared within the same API key, resolved model, system prompt, temperature bucket and prompt version, so A/B variants never share answers.
- **Embeddings:** `all-MiniLM-L6-v2` (local, 384 dimensions, input truncated at 256 tokens, so prompts longer than 800 characters skip the cache). Nearest entries are found with FAISS.
- **Hit rule:** similarity at or above the threshold (`config/cache.yaml`) and a passing guard.
- **Guard:** numbers (in order, digits and number words), years, names (capitalized words, compared in any case), negations and polarity words (not, never, illegal, false...) and style words (briefly, detailed, simple...) must match.
- **Never cached or never looked up:** time sensitive prompts, high temperature, `max_tokens`, streaming, fallback answers, empty answers and short refusals.
- **Fails open:** if embedding or lookup breaks, the request goes to the provider.

### Measuring the cache

```
python -m scripts.tune_cache                 # threshold sweep on labeled prompt pairs
python -m scripts.cache_report --hours 24    # hit rate, tokens and estimated cost saved, latency
```

`benchmarks/cache_pairs.jsonl` holds 99 hand labeled pairs (36 whose answer can be reused, 63 where it cannot). A hit on a pair that cannot be reused is a false hit, a wrong answer. Result on that sample at the shipped threshold of 0.88:

| | hits | false hits | reusable pairs served |
|---|---|---|---|
| similarity threshold only | 54 | 21 (38.9%) | 28 of 36 |
| threshold plus guard | 31 | 0 (0.0%, 95% interval 0% to 11%) | 31 of 36 (86.1%) |

Pairs with a changed year, negation or style word are as similar to the original as real paraphrases (median similarity 0.90 to 0.95 against 0.94 for paraphrases), so no threshold alone can separate them. How to read the numbers: the guard rules were developed while looking at these same pairs, so the figures are optimistic and must be confirmed on new pairs; 99 pairs written by one person is a small sample, hence the wide interval; and false hits cannot be measured from traffic because traces do not store prompts.

## Security notes

- Gateway API keys and dashboard passwords are stored as hashes (SHA 256 for keys, bcrypt for passwords). A new key is shown once.
- A revoked key can keep working for up to 10 seconds (the gateway remembers key checks for that long). If PostgreSQL is down, a recently seen key keeps working and an unknown key gets 503; nothing is let in by default.
- The web API needs `JWT_SECRET` (32+ characters), limits login attempts, answers wrong email and wrong password the same way, and only allows the dashboard origin. The dashboard keeps its token in `sessionStorage`. Hiding buttons from viewers is cosmetic, the web API enforces the roles.
- Everything listens on `127.0.0.1` only. There is no HTTPS: this is a local prototype, not a deployment.
- The development database password in `docker-compose.yml` is for local use only.

## Design decisions

- **A wrong cached answer is worse than a miss.** The guard is strict and the threshold sits above the lowest safe value on purpose.
- **The cache and tracing never break a request.** Errors are caught, logged and the request continues.
- **The rate limiter fails open.** If Redis is down, requests pass and a warning is logged. The gateway pings Redis at startup.
- **A prompt version is never edited.** A change is a new version, otherwise past A/B numbers would stop meaning anything.
- **Cost uses `Decimal` and the model that actually answered.** On a cache hit the trace records zero tokens and cost for the request, and `saved_tokens` and `saved_cost_usd` for what the cache avoided.
- **Use `127.0.0.1`, not `localhost`, for services.** On Windows `localhost` can try IPv6 first and add a one second delay. Tests enforce this.
- **Database changes go through Alembic.** A test fails if the models and the migrations disagree.

## Tests

```
cd ai-gateway
pytest -v -m "not slow"        # unit and integration tests (integration needs docker compose up -d)
pytest -v -m slow              # tests that load the real embedding model
cd ..\web-api ; npm test       # web API tests (fake database)
cd ..\dashboard ; npm test     # dashboard formatting tests
```

## Limitations

- **Estimated cost is not real spend.** Actual spend is $0 (free tiers and a local model). Cost is calculated from published list prices in `ai-gateway/config/pricing.yaml`. The local model rate is a notional placeholder. Some third party sites list different prices for the same model, so verify before relying on them.
- Free tier limits and models change without notice. No limits are hardcoded.
- A/B numbers cover latency, tokens, cost, errors and cache hits. Answer quality is not measured yet.
- The cache is in memory: a restart empties it. Two identical requests at the same moment both miss and both store.
- The guard, the refusal filter and the routing keywords are English only and heuristic. Known blind spots: names written in lowercase everywhere, aliases ("US" against "United States" is blocked, a lost hit), short lists of polarity and style words.
- The cache threshold was tuned on 99 hand labeled pairs written by one person, and the guard was developed on the same pairs. The routing rules have not been evaluated.
- The rate limiter uses a fixed window (a client can send up to twice the limit around a minute boundary).
- Traces can be lost if the server stops right after a request or the database is down. Requests rejected by auth (401), validation (422) or the gateway rate limit are not traced.
- The gateway does not run in Docker yet (it needs Ollama and the embedding model). No streaming.
- The web API and the dashboard have no HTTPS and are meant for a local machine.

## Roadmap

1. Evaluation: eval dataset, rule checks, an LLM judge calibrated on hand labels, regression detection
2. ML router and failure forensics
3. RAG and Text to SQL demos, a benchmark with real measurements, gateway in Docker, CI, demo video