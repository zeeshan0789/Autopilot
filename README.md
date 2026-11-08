# Autopilot

An OpenAI compatible LLM gateway prototype. It sits between an application and its model providers and adds reliability (retries, fallback), access control (API keys, rate limiting), cost awareness (a safe semantic cache and a routing policy) and observability (a trace for every request, with estimated cost).

Everything runs locally for free: local models through Ollama and free API tiers. No billing, no credit card.

**Status:** Phase 1 (gateway) and Phase 2 (routing policy and semantic cache) are complete. Prompt versioning, evaluation, an ML router and a dashboard are planned and not built yet.

## What it does today

- OpenAI compatible `POST /v1/chat/completions` (non streaming)
- Providers are defined in config, with no per provider if/else in code: local Ollama, Groq, and a mock provider that can simulate errors, 429s and timeouts
- Timeouts, retries with exponential backoff and jitter, and an ordered fallback chain per model
- API key authentication (keys are stored only as SHA 256 hashes) and Redis rate limiting per key
- **Routing policy:** send `model: "auto"` and declarative rules in `config/policy.yaml` choose between a simple and a complex model. Any other model name is used exactly as requested
- **Safe semantic cache:** answers are reused for similar prompts, but only if a guard confirms that numbers, years, names, negations and style words match
- A PostgreSQL trace for every request that reaches the handler: requested, routed and served model, provider, cache status and similarity, tokens, estimated cost, estimated savings, latency
- Tools to measure the cache: a threshold tuner on labeled prompt pairs and a report from real traffic
- Automated tests (unit and integration) and database migrations with Alembic

## Architecture

```
Client
  │  Authorization: Bearer ap-...
  ▼
FastAPI gateway
  1. API key check            → 401 if missing or invalid
  2. Redis rate limit         → 429 if over the limit
  3. Routing policy           model "auto" → simple or complex model (config/policy.yaml)
  4. Semantic cache lookup    hit → answer returned, no provider call
       embed prompt → closest entries → similarity threshold → safety guard
  5. Provider call            timeout, retry with backoff, fallback chain
       Groq → Ollama (local) → Mock
  6. Store the answer         never fallback answers, empty answers or refusals
  7. Estimated cost           (config/pricing.yaml)
  8. Trace row → PostgreSQL   background task, never blocks the response
  │
  ▼
OpenAI format response + X-Request-ID and X-Autopilot-* headers
```

## Quick start (Windows PowerShell)

Prerequisites: Python 3.11, Docker Desktop, Ollama, a free Groq API key.

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
alembic upgrade head
python -m scripts.create_key dev 60
uvicorn app.main:app --reload
```

The first start downloads the embedding model (about 90 MB) from Hugging Face. After that you can add `HF_HUB_OFFLINE=1` to `.env` to start without the network.

`create_key` prints the key once. Only its hash is stored in `ai-gateway/data/api_keys.json` (git ignored). Only send public or synthetic data: free tier providers may use prompts to improve their products.

```
$key = "ap-your-key"
$body = @{ model = "auto"; messages = @(@{ role = "user"; content = "What is the capital of France?" }) } | ConvertTo-Json -Depth 5
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/chat/completions -ContentType "application/json" -Headers @{ Authorization = "Bearer $key" } -Body $body
```

Ask the same question twice and the second answer comes from the cache (`X-Autopilot-Cache: hit`).

## Response headers

`X-Request-ID`, `X-Autopilot-Cache` (hit, miss, blocked, bypass), `X-Autopilot-Provider` (`cache` on a hit), `X-Autopilot-Model`, `X-Autopilot-Fallback`, `X-Autopilot-Route` and `X-Autopilot-Routed-Model` (only for `auto`), `X-RateLimit-Limit`, `X-RateLimit-Remaining`, and `Retry-After` on a 429.

## Routing policy

Rules live in `ai-gateway/config/policy.yaml`: the first matching rule wins, otherwise the default tier is used. Conditions are length, history length, number of questions, requested output size, keywords and code detection. The shipped rules send code, analysis style keywords, long prompts, long conversations, multi part questions and large outputs to `openai/gpt-oss-120b`, and everything else to `openai/gpt-oss-20b`.

These rules are heuristics and their thresholds are untuned guesses. Nothing here shows that they improve answer quality. That needs the evaluation planned for phase 4. The list price gap between the two models is small (see `config/pricing.yaml`), so the main purpose today is sending hard prompts to the bigger model and building the mechanism a learned router will later use.

## Semantic cache

- **Scope:** an entry is only compared with entries from the same API key, model, system prompt, temperature bucket and prompt version. The model is the one that actually answers (the result of routing).
- **Embeddings:** `all-MiniLM-L6-v2` (local, 384 dimensions, input truncated at 256 tokens, so prompts longer than 800 characters skip the cache). Nearest entries are found with FAISS.
- **Hit rule:** similarity at or above the threshold (`config/cache.yaml`) **and** a passing guard.
- **Guard:** numbers (in order, digits and number words), years, names (capitalized words, compared in any case), negations and polarity words (not, never, illegal, false...), and style words (briefly, detailed, simple...) must match between the new prompt and the cached one.
- **Never cached or never looked up:** time sensitive prompts (today, current, latest, this year...), requests with a high temperature or `max_tokens`, streaming requests, fallback answers, empty answers and short refusals ("I'm sorry, but I can't...").
- **Fails open:** if embedding or lookup breaks, the request simply goes to the provider.

## Measuring the cache

```
python -m scripts.tune_cache                 # threshold sweep on labeled prompt pairs
python -m scripts.cache_report --hours 24    # hit rate, tokens and estimated cost saved, latency
```

`benchmarks/cache_pairs.jsonl` holds 99 hand labeled pairs (36 whose answer can be reused, 63 where it cannot: changed numbers, years, names, negations, style, same topic different question, unrelated). A hit on a pair that cannot be reused is a false hit, a wrong answer.

Result on that sample, with the similarity threshold at 0.88:

| | hits | false hits | reusable pairs served |
|---|---|---|---|
| similarity threshold only | 54 | 21 (38.9%) | 28 of 36 |
| threshold plus guard | 31 | 0 (0.0%, 95% interval 0% to 11%) | 31 of 36 (86.1%) |

The reason for the guard: pairs with a changed year, negation or style word are as similar to the original as real paraphrases (median similarity 0.90 to 0.95 against 0.94 for paraphrases), so no threshold alone can separate them. How to read these numbers:

- The guard rules were developed while looking at these same pairs, so the figures are optimistic. They must be confirmed on new pairs.
- 99 pairs is a small sample, which is why the intervals are wide. The pairs were written and labeled by one person.
- The traffic report cannot measure false hits: traces do not store prompts.

## Design decisions

- **A wrong cached answer is worse than a miss.** The guard is strict and the threshold sits above the lowest safe value on purpose.
- **The cache and tracing never break a request.** Errors are caught, logged and the request continues.
- **The rate limiter fails open.** If Redis is down, requests pass and a warning is logged. The gateway pings Redis at startup.
- **Cost uses `Decimal` and the model that actually answered.** On a cache hit the trace records zero tokens and cost for the request, and `saved_tokens` and `saved_cost_usd` for what the cache avoided.
- **Use `127.0.0.1`, not `localhost`.** On Windows `localhost` can try IPv6 first and add a one second delay. Tests enforce this.
- **Database changes go through Alembic.** A test fails if the models and the migrations disagree.

## Tests

```
cd ai-gateway
pytest -v -m "not slow"        # unit and integration tests (integration needs docker compose up -d)
pytest -v -m slow              # tests that load the real embedding model
```

## Limitations

- **Estimated cost is not real spend.** Actual spend is $0 (free tiers and a local model). Cost is calculated from published list prices in `config/pricing.yaml`. The local model rate is a notional placeholder. Some third party sites list different prices for the same model, so verify before relying on them.
- Free tier limits and models change without notice. No limits are hardcoded.
- The cache is in memory: a restart empties it. Two identical requests at the same moment both miss and both store.
- The guard is English only and heuristic. Known blind spots: names written in lowercase everywhere ("google revenue" and "apple revenue" look the same), aliases ("US" against "United States" is blocked, a lost hit), a short list of polarity and style words, and number words written as several words.
- The refusal filter and the keyword rules are English only.
- The cache threshold was tuned on 99 hand labeled pairs written by one person, and the guard was developed on the same pairs.
- The routing rules have not been evaluated for answer quality.
- The rate limiter uses a fixed window (a client can send up to twice the limit around a minute boundary).
- Traces can be lost if the server stops right after a request or the database is down. Requests rejected by auth (401), validation (422) or the gateway rate limit are not traced.
- API keys are stored in a local JSON file (hashed). No streaming yet.
- The embedding model needs the network on the first start. The development database password is for local use only.

## Roadmap

1. Prompt versioning, A/B traffic splitting and a dashboard
2. Evaluation, judge calibration and regression detection
3. ML router and failure forensics
4. RAG and Text to SQL demos, a benchmark with real measurements, CI