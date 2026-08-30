# Issue #12 — Unblocking the Live Phase 1 Acceptance Run (OpenRouter Free-Tier Research)

_Research date: 2026-08-30. All claims below are cited to a primary source with the URL and, where visible, the source's own "last updated" or as-accessed date. Anything I could not verify against a primary source is explicitly flagged as unverified rather than repeated from a blog/aggregator._

## Summary

Issue #12's live acceptance run for the Planner agent (`analyst_crew/agents/planner.py`, `config/settings.toml` → `llm.model = "nvidia/nemotron-3-super-120b-a12b:free"`) is blocked today by **OpenRouter account-wide free-tier exhaustion**: every `:free` model tried returns HTTP 404 `"This model is unavailable for free. The paid version is available now - use this slug instead: <model>"`. OpenRouter's own docs confirm free (`:free`-suffixed) models are capped at **20 requests/minute** and, per account, **50 requests/day** unless the account has purchased **at least $10 in credits (all-time)**, which raises the daily cap to **1000 requests/day** — this is a documented, one-time lifetime unlock, not a spendable balance requirement (see §2). Separately, and independent of quota, the previously-used reasoning model (nemotron) is unreliable for structured JSON output because it leaks chain-of-thought as prose and can exhaust `max_tokens` before emitting JSON (§3–§4).

**Single best next step, ranked by effort/cost:** top up the OpenRouter account with **$10** (one-time, non-consumed lifetime unlock to 1000 free requests/day) **and** switch `config/settings.toml`'s `llm.model` to a free, non-reasoning-first model that OpenRouter's own model metadata marks as supporting `structured_outputs` — e.g. `google/gemma-4-31b-it:free` or `liquid/lfm-2.5-2.6b:free` — while also passing `reasoning: {enabled: false}` and, if `langchain_openai` version allows, switching `with_structured_output(..., method="json_schema")` instead of `"json_mode"`. This is minutes of effort, $10 one-time cost, and directly removes both failure modes at once. If avoiding OpenRouter entirely is preferred, Google's Gemini API direct (`langchain-google-genai`) or Groq's OpenAI-compatible endpoint are realistic zero-cost swap-ins (§5), at the cost of a code change to the Planner's model client.

---

## 1. OpenRouter free-tier rate limits — current documented numbers

- **Per-minute limit:** 20 requests/minute on `:free`-suffixed models, regardless of credit history.
  Source: OpenRouter API reference — Limits page, `https://openrouter.ai/docs/api-reference/limits` (fetched 2026-08-30).
- **Per-day limit, two-tier:**
  - **50 requests/day** for accounts that have never purchased credits.
  - **1000 requests/day** for accounts that have purchased **at least $10 in credits, all-time** (this is a lifetime unlock — spending the $10 back down to a $0 balance does not revoke the 1000/day tier).
  Source: same page, `https://openrouter.ai/docs/api-reference/limits`, and corroborated by OpenRouter's own FAQ, `https://openrouter.ai/docs/faq` (fetched 2026-08-30, both pages state the $10 threshold and the 50→1000 change explicitly).
- **Reset cadence:** Daily limits reset on UTC calendar-day boundaries; the per-minute limit is a rolling 60-second window. Neither doc page gives a more granular reset mechanism than "UTC day."
  Source: `https://openrouter.ai/docs/api-reference/limits`.
- **Checking remaining quota:** `GET https://openrouter.ai/api/v1/key` returns `limit_remaining` and usage stats (daily/weekly/monthly). OpenRouter's own docs recommend polling this proactively before requests start failing.
  Source: `https://openrouter.ai/docs/api-reference/limits`.
- **The exact 404 message you're hitting** ("This model is unavailable for free. The paid version is available now - use this slug instead: `<model>`") is **not documented on any OpenRouter docs page I could reach** (`/docs/api-reference/limits`, `/docs/faq`, `/docs/models`) — it appears to be an undocumented runtime error string, not a named/explained error code. I could not confirm via a primary source whether this specific message means (a) account-wide quota exhaustion, (b) a per-model daily cap being hit, or (c) the specific model having been withdrawn from the free tier entirely. **Flagging as unverified**: the message text itself was not found in OpenRouter's own docs by direct fetch or targeted web search; only community/forum discussion referenced it.
- **Relevant context that may explain part of it:** OpenRouter has an active pattern of moving models off the free tier or deprecating older free slugs — e.g., community-reported deprecations of several free model slugs around 2026-07-19–21, and reports that all DeepSeek models became paid-only around July 2026. This is **not from an OpenRouter primary changelog fetch** (the changelog page fetch, `https://openrouter.ai/docs/changelog`, was not directly confirmed with these specifics — this point is sourced from aggregated search results, not a verified primary doc, and should be treated as directionally useful context only, not a confirmed fact).
  Given that your test hit the **same error across four unrelated model slugs** (nemotron, llama-3.3-70b, deepseek-v3.1, mistral-small-3.1) on the same account, an **account-wide quota exhaustion** (§1's 50/day cap) is the more parsimonious explanation than four independent per-model deprecations happening simultaneously — but this is my inference from the rate-limit doc, not a confirmed root-cause from OpenRouter.

## 2. Does adding credits raise free-tier limits? Confirmed current policy

Confirmed directly from OpenRouter's own docs (§1 sources): **yes** — purchasing **$10 or more in credits (lifetime, cumulative)** raises the daily free-model quota from **50 → 1000 requests/day**. The per-minute cap (20 req/min) is unaffected by credits. This is a one-time unlock tied to lifetime purchase history, not a live balance check — OpenRouter's docs describe it as unlocked once the $10 threshold is crossed, not something that reverts as the balance is spent down (`https://openrouter.ai/docs/api-reference/limits`, `https://openrouter.ai/docs/faq`, both fetched 2026-08-30). I did not find an explicit "last updated" timestamp on either doc page, so treat this as current-as-of-access rather than dated.

## 3. Free OpenRouter models suited for structured/JSON output (not reasoning-first)

I queried OpenRouter's own models API directly (`GET https://openrouter.ai/api/v1/models`, fetched 2026-08-30) and filtered for `:free`-suffixed models whose `supported_parameters` array includes `structured_outputs` (OpenRouter's native `response_format: {"type":"json_schema", strict: true}` support) — this is a first-party, machine-readable signal, not a scraped page. As of 2026-08-30, 8 of 18 free models declare `structured_outputs` support:

| Model slug | Notes on reasoning behavior |
|---|---|
| `google/gemma-4-31b-it:free` | Instruction-tuned; reasoning/thinking is **configurable** (can be turned off via the `reasoning` param), not on by default the way nemotron's chain-of-thought leakage was. |
| `google/gemma-4-26b-a4b-it:free` | Same family, MoE variant; same configurable-reasoning behavior. |
| `liquid/lfm-2.5-2.6b:free` | Small (2.6B) instruct model, not marketed as a reasoning model; broadest `supported_parameters` list of the free tier (includes `structured_outputs`, `seed`, `top_k`, etc.), likely lowest risk of CoT leakage into JSON output. |
| `z-ai/glm-5.2:free` | GLM-5 is a hybrid reasoning model — has a `reasoning_effort` parameter, meaning reasoning can leak unless explicitly disabled. |
| `minimax/minimax-m2.7:free`, `minimax/minimax-m3:free` | MiniMax M2/M3 are explicitly reasoning models that use `<think>…</think>` interleaved-thinking by design (confirmed via MiniMax's own blog, `https://www.minimax.io/news/why-is-interleaved-thinking-important-for-m2`) — same risk class as nemotron; avoid unless reasoning is disabled. |
| `dots-studio/dots-3-note-preview:free`, `nvidia/nemotron-3-super-120b-a12b:free` | nemotron is the model already causing the CoT-leak failures in issue #12; dots-3 is unverified re: reasoning behavior. |

**Recommendation from this list:** `google/gemma-4-31b-it:free`, `google/gemma-4-26b-a4b-it:free`, or `liquid/lfm-2.5-2.6b:free` — all three declare native `structured_outputs` support and are not reasoning-first models by design (though Gemma 4's reasoning mode should still be explicitly disabled per §4 as a defensive measure).

Source for the model list and `supported_parameters` data: `https://openrouter.ai/api/v1/models` (OpenRouter's own public models API, fetched 2026-08-30).

## 4. Configuration-level fixes: disabling reasoning leakage, and native structured outputs

- **OpenRouter's `reasoning` request parameter** (unified across providers) lets you disable or suppress chain-of-thought:
  - `"reasoning": {"enabled": false}` — fully disables reasoning.
  - `"reasoning": {"effort": "none"}` — equivalent disable via the effort scale (`max`/`xhigh` ~95% of tokens for reasoning → `none` = disabled).
  - `"reasoning": {"effort": "high", "exclude": true}` — lets the model still reason internally but excludes the reasoning trace from the returned response (useful if a model reasons better with it on, but you don't want it polluting JSON parsing).
  - Legacy: `include_reasoning: false` (equivalent to `exclude: true`), `include_reasoning: true`.
  Source: OpenRouter docs, "Reasoning Tokens" page, `https://openrouter.ai/docs/use-cases/reasoning-tokens` (fetched 2026-08-30).
- **OpenRouter's native Structured Outputs feature**: `response_format: {"type": "json_schema", "json_schema": {...}, "strict": true}`. OpenRouter's own docs state strict mode is "provider-native... exact compliance is not guaranteed on every endpoint" — some providers guarantee schema conformance, others treat the schema as a strong hint. Only models that declare `structured_outputs` in `supported_parameters` (§3) should be relied on; OpenRouter recommends setting `require_parameters: true` in provider routing preferences to avoid silently routing to a non-conforming endpoint/provider.
  Source: `https://openrouter.ai/docs/features/structured-outputs` (fetched 2026-08-30).
- **`langchain_openai.ChatOpenAI.with_structured_output` — `method` parameter**, per LangChain's own reference docs (`https://reference.langchain.com/python/langchain-openai/chat_models/base/BaseChatOpenAI/with_structured_output`, fetched 2026-08-30):
  - `method="function_calling"` (default) — uses OpenAI-style tool-calling.
  - `method="json_schema"` — uses OpenAI's Structured Output API (`response_format: {"type":"json_schema", ...}`), supports `strict=True` for stricter validation. This is the LangChain-side equivalent of OpenRouter's native structured-outputs feature described above, and is more robust than `json_mode` specifically because it constrains the schema at decode time rather than just requesting "valid JSON."
  - `method="json_mode"` (what the Planner currently uses) — uses plain JSON mode; LangChain's own docs note explicitly: *"if using JSON mode then you must include instructions for formatting the output into the desired schema into the model call"* — i.e., `json_mode` guarantees syntactically valid JSON but not schema conformance, and depends on the prompt alone to steer the shape, which is a plausible contributor to nemotron's parsing failures beyond the CoT leakage. Since the model swap in §3 targets models that support `structured_outputs`, changing `Plan`'s structured-output call from `method="json_mode"` to `method="json_schema"` (with `strict=True`) is a compatible, low-risk pairing.
  I could not confirm from LangChain's docs whether `method="json_schema"` is validated to work against arbitrary OpenAI-compatible `base_url` endpoints (like OpenRouter) versus only api.openai.com — the reference page doesn't call this out explicitly, so treat this as needing a live smoke test, not a guaranteed compatibility claim.

## 5. Alternatives to OpenRouter's free tier

- **Google Gemini API direct (`ai.google.dev`)**: Google's own rate-limits doc (`https://ai.google.dev/gemini-api/docs/rate-limits`, page states last-updated 2026-08-18, fetched 2026-08-30) does **not** publish fixed numeric free-tier RPM/TPM/RPD values on the docs page itself — it explicitly says limits "depend on a variety of factors (such as your usage tier) and can be viewed in Google AI Studio" (`https://aistudio.google.com/rate-limit`, which requires an authenticated Google account to view and could not be fetched here). **I could not verify current exact numeric limits from a primary source** — community reports (Google's own developer forum threads, not docs) describe Gemini 2.5 Flash's free-tier RPD dropping from 250 to 20 around December 2025, but this is forum chatter, not a docs page, so treat any specific number as unverified until checked live in AI Studio. LangChain integration exists via `langchain-google-genai`, which is a realistic swap-in for `ChatOpenAI`-style calls including structured output (Gemini supports a native JSON schema response mode), but the exact current quota needs to be checked in-console before relying on it, since it appears to have been reduced recently.
- **Groq API (`groq.com` / `console.groq.com/docs`)**: Groq's own rate-limits doc (`https://console.groq.com/docs/rate-limits`, fetched 2026-08-30) does **not** publish fixed numbers on the page either — it directs users to their account's live limits page in the console. Groq's own Structured Outputs doc (`https://console.groq.com/docs/structured-outputs`) confirms first-party support for `response_format: {"type":"json_schema", "json_schema": {...}, "strict": true}` with constrained decoding guaranteeing schema adherence when `strict: true`, which is stronger than OpenRouter's "not guaranteed on every endpoint" caveat. Groq's endpoint is OpenAI-compatible, so it's a low-effort swap for `ChatOpenAI(base_url=...)` (just change `base_url` and `model`). **Numeric free-tier limits I found (30 RPM / ~12k TPM / ~1000 RPD for Llama-3.3-70B) came only from third-party aggregator pages, not Groq's own docs page, which withholds exact numbers in favor of the live console** — flag these numbers as unverified against a primary source.
- **Local model via Ollama (`ollama.com`)**: Ollama exposes an OpenAI-compatible `/v1/chat/completions` endpoint plus its own native `/api/chat` with a `format` parameter that accepts a raw JSON schema for structured output (Ollama's own blog post, `https://ollama.com/blog/structured-outputs`, and docs page `https://docs.ollama.com/capabilities/structured-outputs`, found via search — not directly fetched in this session, so treat as a search-summary rather than a full primary-source read). Important compatibility note found via search: Ollama's OpenAI-compatible layer and its native `/api/chat` format parameter are **not identical** — a request built for OpenAI's `response_format: {"type":"json_schema", ...}` needs translation before hitting Ollama's native endpoint (per a linked Ollama GitHub issue, `https://github.com/ollama/ollama/issues/10001`, discussing improving that compatibility). `langchain-ollama`'s `ChatOllama.with_structured_output` exists as a first-party LangChain integration. Hardware requirements were **not verified in this session** — no primary Ollama page with concrete hardware specs (RAM/VRAM per model size) was successfully fetched; this should be checked against `https://ollama.com/library` model cards before committing to this path, since running a ~70B-class model locally without a capable GPU is not realistic for most dev machines.

---

## Recommended options, ranked

| Option | Effort | Cost | How it unblocks #12 | Risk / caveat |
|---|---|---|---|---|
| **Add $10 OpenRouter credit + switch `llm.model` to `google/gemma-4-31b-it:free` (or `liquid/lfm-2.5-2.6b:free`) + set `reasoning.enabled=false` + use `method="json_schema"`** | Low (config change + $10 purchase) | $10 one-time, lifetime unlock | Fixes both failure modes: raises daily cap 50→1000 (§1–2) and swaps off the leaky reasoning model onto one that declares native `structured_outputs` support (§3–4) | OpenRouter's own docs say strict schema conformance is "not guaranteed on every endpoint" even with `structured_outputs` declared (§4); still needs a live smoke test. The $10 doesn't raise the 20 req/min ceiling. |
| **Switch only the model (no credit purchase)** | Low | $0 | Fixes failure mode 2 (JSON reliability) but does nothing for the 50/day account-wide cap already being hit today | Still capped at 50 free requests/day account-wide; likely re-blocks quickly during iterative testing. |
| **Swap Planner's LLM client to Groq (OpenAI-compatible `base_url`)** | Medium (code change to Planner's client config, plus a Groq API key) | $0 (free tier) | Removes OpenRouter dependency entirely; Groq's own docs confirm first-party `strict` JSON-schema structured outputs, generally more reliable than json_mode | Exact free-tier numeric limits not confirmed from Groq's own docs page in this session (§5) — must check the live console before relying on it for CI/test volume. |
| **Swap to Gemini API direct via `langchain-google-genai`** | Medium (new dependency + client code) | $0 (free tier, but current limits unverified) | Removes OpenRouter dependency; Gemini has native JSON schema output | Free-tier numeric limits are not published on Google's docs page and reportedly dropped substantially in Dec 2025 per forum reports (unverified primary number) — may be more restrictive than OpenRouter's post-$10 tier. |
| **Run a model locally via Ollama** | High (infra setup, hardware-dependent) | $0 (assuming adequate local hardware) | Fully removes external API dependency and rate limits | Hardware requirements not verified in this session; OpenAI-compatible layer vs. native `/api/chat` structured-output format have a documented compatibility gap (§5) that may need extra glue code. |

---

### Sources referenced (primary, fetched 2026-08-30 unless noted)

- OpenRouter API Limits — `https://openrouter.ai/docs/api-reference/limits`
- OpenRouter FAQ — `https://openrouter.ai/docs/faq`
- OpenRouter Structured Outputs — `https://openrouter.ai/docs/features/structured-outputs`
- OpenRouter Reasoning Tokens — `https://openrouter.ai/docs/use-cases/reasoning-tokens`
- OpenRouter Models docs — `https://openrouter.ai/docs/models`
- OpenRouter public Models API — `https://openrouter.ai/api/v1/models` (queried live)
- LangChain `with_structured_output` reference — `https://reference.langchain.com/python/langchain-openai/chat_models/base/BaseChatOpenAI/with_structured_output`
- Google Gemini API rate limits — `https://ai.google.dev/gemini-api/docs/rate-limits` (page states last updated 2026-08-18)
- Groq rate limits docs — `https://console.groq.com/docs/rate-limits`
- Groq Structured Outputs docs — `https://console.groq.com/docs/structured-outputs` (via search summary)
- Ollama Structured Outputs blog — `https://ollama.com/blog/structured-outputs`, `https://docs.ollama.com/capabilities/structured-outputs` (via search summary, not directly fetched)
- MiniMax M2 interleaved thinking — `https://www.minimax.io/news/why-is-interleaved-thinking-important-for-m2`

Items explicitly flagged above as **unverified against a primary source** (do not treat as confirmed): the exact wording/meaning of the OpenRouter 404 message; specific numeric Gemini free-tier RPM/RPD figures; specific numeric Groq free-tier RPM/RPD/TPM figures; Ollama hardware requirements; whether `langchain_openai`'s `method="json_schema"` is validated against non-OpenAI `base_url` endpoints such as OpenRouter.
