# RAG Extractor — Development Session Playbook

Design is **locked** in `CONTEXT.md` + `docs/adr/0001` + `docs/adr/0002`. Do **not** re-grill or re-open the design in implementation sessions — point each session at those three files and build.

## Which Matt Pocock skills apply (and which to skip)

| Skill | Use here? | Why |
|---|---|---|
| `grill-with-docs` | ✅ done | Produced CONTEXT.md + the two ADRs. |
| `prototype` | ✅ optional, recommended first | Extraction quality = prompt × model. Eyeball it on the real local model before committing to tests. |
| `tdd` | ✅ core skill | Extractive selection + integrity-over-budget + defensive parsing are exactly the kind of invariants worth pinning with tests. |
| `verify` | ✅ at the end | Confirm the wired `retrieve_knowledge` behaves in the real running agent, not just unit tests. |
| `handoff` | ✅ as needed | If a session's context fills before the slice is done, compact and continue next session. |
| `to-prd` / `to-issues` | ⏭️ skip | These are for multi-feature efforts or when you need tracked tickets. CONTEXT + ADRs already are the spec for a single feature. Use `to-issues` only if you want tracker tickets. |
| `improve-codebase-architecture` | ⏭️ later | Only after it's shipped and you want a refactor pass. |
| `code-review` | ✅ before merge | Built-in; run on the diff at the end. |

## Session plan

Each session: open `CONTEXT.md`, `docs/adr/0001-*.md`, `docs/adr/0002-*.md` first. End with `/handoff` if not finished.

### Session 1 — Prototype the extraction prompt (optional but recommended)
- **Skill:** `prototype` (throwaway-script branch)
- **Model:** **Sonnet** — fast, cheap iteration on a throwaway script; you're eyeballing output, not reasoning hard.
- **Goal:** A throwaway script that calls your local 70B with a few real retrieved chunks + a sample `task_intent`, and prints the extracted verbatim spans. Iterate the prompt until: (a) it selects only relevant spans, (b) it never paraphrases, (c) payloads/code blocks come back whole. Throw the script away; keep the winning prompt text.
- **Prompt:**
  > Read CONTEXT.md and docs/adr/0002. Use /prototype (throwaway script branch). I want to nail the Extractor's extraction prompt against my real local model before writing production code. Build a throwaway script that: takes a hardcoded list of sample RAG chunks + a sample query + task_intent, calls the local LLM via the existing LLMClient with a candidate extraction prompt, and prints the selected verbatim spans with their doc_id. Iterate the prompt with me until selection is relevant, nothing is paraphrased, and payloads/code blocks survive whole. Do not touch production code — this is throwaway.

### Session 2 — Implement `rag/extractor.py` + config (core, TDD)
- **Skill:** `tdd`
- **Model:** **Opus** — this is the security-sensitive core (integrity-over-budget, defensive parsing of weak-model output). Subtle invariants; worth the stronger model.
- **Goal:** `conf/config.py` gets `EXTRACTOR_RETRIEVE_K` (default 30), `EXTRACTOR_OUTPUT_BUDGET_CHARS` (default 1500), `EXTRACTOR_TIMEOUT` (default 120). `rag/extractor.py` exposes a single function that takes `query`, `task_intent`, raw retrieved chunks → returns extracted spans honoring: extractive-only, span-boundary budget, integrity > budget (payload/code block never split, always ≥1 span), graceful return on malformed LLM output. Tests cover the budget/integrity edge cases (incl. the "payload longer than budget" case).
- **Prompt:**
  > Read CONTEXT.md and docs/adr/0001 and 0002 — design is locked, don't re-open it. Use /tdd to implement rag/extractor.py plus the EXTRACTOR_* constants in conf/config.py. The extractor is a pure function: (query, task_intent, raw_chunks) -> extracted spans, using the winning prompt from the prototype session. Enforce: extractive-only (verbatim + doc_id/score), output budget as a soft span-boundary stop rule, integrity > budget (never split a payload/code block, always return at least the single most-relevant span even if it exceeds budget), and defensive handling of malformed/empty LLM output. Write the tests first, especially the "relevant payload exceeds budget" and "malformed model output" cases. Do not wire it into retrieve_knowledge yet.

### Session 3 — Wire into `retrieve_knowledge` (transparent replacement + degrade)
- **Skill:** `tdd` (or plain implementation)
- **Model:** **Sonnet** — well-scoped wiring once the contract from Session 2 is fixed; mechanical.
- **Goal:** `retrieve_knowledge` gains an optional `task_intent` param, internally retrieves `EXTRACTOR_RETRIEVE_K` candidates, runs the extractor, returns the same JSON shape (`results[].snippet` now = extracted spans). On extractor timeout/failure: fall back to raw `top_k=5`, set `extractor_fallback: true`, and emit a visible console warning + logger entry. `knowledge_service` stays untouched (per ADR-0001).
- **Prompt:**
  > Read docs/adr/0001 and 0002. Use /tdd. Wire rag/extractor into the retrieve_knowledge MCP tool in tools/mcp_service.py as a transparent replacement: add an optional task_intent param, retrieve EXTRACTOR_RETRIEVE_K candidates, run the extractor, return the existing results JSON shape. On extractor timeout/exception/malformed output, degrade to raw top_k=5, add extractor_fallback:true to the JSON, and emit a visible console Panel warning + logger entry (match the existing warning style in agent.py). Do NOT modify rag/knowledge_service.py — extraction stays in the tool layer per ADR-0001.

### Session 4 — Verify in the running agent
- **Skill:** `verify`
- **Model:** **Sonnet** — running and observing, not deep reasoning.
- **Goal:** Run the agent against a task that triggers RAG, confirm: results are cleaner/shorter, payloads intact, and that a forced extractor failure (e.g. wrong model name) degrades to raw top_k with the warning.
- **Prompt:**
  > Use /verify. Run the agent on a scenario that exercises retrieve_knowledge with the new Extractor. Confirm: (1) returned snippets are extractive and shorter than raw, (2) a known exact payload survives verbatim, (3) forcing the extractor to fail degrades to raw top_k=5 with the visible warning and extractor_fallback:true. Report what you observed.

### Session 5 — Review before merge
- **Skill:** `code-review` (built-in)
- **Model:** **Opus** — review benefits from the stronger model, especially the security-sensitive integrity logic.
- **Prompt:**
  > Run /code-review high on the current diff. Pay special attention to ADR-0002's integrity-over-budget invariant: confirm no code path can truncate a payload mid-string, and that malformed local-model output always degrades safely.

## Model selection rule of thumb
- **Opus**: subtle/security-sensitive logic, the extractor core, reviews. (Sessions 2, 5)
- **Sonnet**: throwaway prototyping, mechanical wiring, running/verifying. (Sessions 1, 3, 4)

## Tuning defaults per local model (from ADR-0002)
`EXTRACTOR_RETRIEVE_K`: ~10 (8B) · ~20 (32B) · ~30 (70B). Output budget and timeout are model-agnostic; raise the timeout if a smaller-but-busier server is slow.
