# The Extractor is extractive-only, and content integrity outranks the output budget

## Context

The Extractor compresses retrieved RAG results before they reach the Executor. The knowledge base is a security/CTF corpus full of exact payloads, regexes, and encoding variants (`' OR 1=1--`, `%2e%2e%2f`, `&#x3c;`) where a single wrong character makes the payload useless or dangerous. The Extractor runs on a **local** LLM that must work across model sizes (8B / 32B / 70B) by tuning parameters only — so the contract cannot assume strong-model reasoning.

## Decision

1. **Extractive, never abstractive.** The Extractor selects verbatim spans from retrieved chunks and tags each with its `doc_id`/`score`. It never summarises, paraphrases, or rewrites. Code blocks and payload lines are preserved whole.
2. **Integrity outranks the output budget.** The output budget (`EXTRACTOR_OUTPUT_BUDGET_CHARS`) is a *soft target enforced at span boundaries*, not a truncation rule. A relevant payload is emitted whole even if it exceeds the budget; truncating a payload mid-string is forbidden. At least one most-relevant span is always returned.
3. **Model-agnostic and degradation-safe.** All knobs (`EXTRACTOR_RETRIEVE_K`, `EXTRACTOR_OUTPUT_BUDGET_CHARS`, `EXTRACTOR_TIMEOUT`) live in config with mid-tier defaults. The extraction prompt is simple and strongly constrained so weak models can follow it; malformed output or timeout degrades gracefully to the raw `top_k=5` snippets (strictly no worse than pre-Extractor behaviour) with a visible warning.

## Consequences

- A future reader will wonder "why does this RAG stage never summarise, and why does it sometimes exceed its own budget?" — the answer is payload fidelity in a security corpus, which is non-negotiable.
- This aligns with `RAGClient`'s existing graceful-degradation philosophy and its code-block-verbatim handling in `_normalize_snippet`.
- Suggested `EXTRACTOR_RETRIEVE_K` per model size: ~10 (8B), ~20 (32B), ~30 (70B).
