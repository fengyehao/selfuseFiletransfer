# rag/extractor.py
# RAG Extractor (检索精炼器) — see CONTEXT.md + docs/adr/0001 + docs/adr/0002.
#
# A single-pass, stateless retrieval post-processing stage. Given a `query` and a
# `task_intent`, it uses a local LLM to *extractively* select only the relevant
# verbatim spans from already-retrieved chunks, tagging each with its source
# `doc_id`/`score`. It never summarises, paraphrases, rewrites, loops, or keeps
# state.
#
# Invariants (ADR-0002):
#   - Extractive only: spans are copied verbatim; non-verbatim output is dropped.
#   - Integrity outranks budget: EXTRACTOR_OUTPUT_BUDGET_CHARS is a soft target
#     enforced at span boundaries, never a mid-span truncation. At least one
#     (most-relevant) span is always returned, even if it alone exceeds the budget.
#   - Degradation-safe: malformed / empty / timed-out LLM output degrades to the
#     raw top-k chunks (strictly no worse than pre-Extractor behaviour).

import asyncio
import json
from dataclasses import dataclass

from conf.config import (
    EXTRACTOR_FALLBACK_TOP_K,
    EXTRACTOR_OUTPUT_BUDGET_CHARS,
    EXTRACTOR_TIMEOUT,
)


@dataclass
class ExtractionResult:
    """Carrier for the extractor's output.

    `spans` are plain dicts shaped like the rest of the RAG layer's results
    (`{doc_id, score, span}`). `degraded` is True when the extractor fell back to
    the raw chunks instead of LLM-selected spans; `reason` explains why (for the
    call site's visible warning / log).
    """

    spans: list
    degraded: bool
    reason: str | None = None


def build_extraction_prompt(query: str, task_intent: str, chunks: list) -> str:
    """The validated 7-rule extraction prompt (prototype v4 — see _proto_extract_notes.md)."""
    chunks_text = ""
    for i, c in enumerate(chunks):
        chunks_text += (
            f"\n--- CHUNK {i + 1} | doc_id: {c.get('doc_id')} | score: {c.get('score', 0.0):.2f} ---\n"
            f"{c.get('snippet', '')}\n"
        )

    return f"""You are an extractive selector for a security knowledge base. Your only job is to copy verbatim spans from the chunks below that directly help with the task. You do not explain, summarise, or rewrite anything.

STRICT RULES:
1. Copy spans VERBATIM — character-for-character, including all punctuation and special characters.
2. Each code block (``` ... ```) must be copied whole — never truncate a payload or code line mid-string.
3. Within a chunk, select only the code blocks and surrounding sentences that directly address the task intent. Stop before any code block whose operation is unrelated to the task (e.g. skip a database-dumping block if the task is only about login bypass or injection confirmation).
4. If a chunk contains multiple relevant code blocks, include all of them in one span (preserve any brief prose between them so the span is contiguous).
5. Omit any chunk that has no material directly relevant to the task intent.
6. Output a JSON array. Each element has exactly two keys: "doc_id" and "span".
7. No extra keys, no commentary, no markdown outside the JSON — output the JSON array and nothing else.

QUERY: {query}

TASK INTENT: {task_intent}

CHUNKS:{chunks_text}
Output (JSON array only):"""


async def extract(
    query: str,
    task_intent: str,
    raw_chunks: list,
    llm_client=None,
    budget_chars: int = EXTRACTOR_OUTPUT_BUDGET_CHARS,
    timeout: float = EXTRACTOR_TIMEOUT,
    fallback_top_k: int = EXTRACTOR_FALLBACK_TOP_K,
) -> ExtractionResult:
    """Extractively select relevant verbatim spans from `raw_chunks`.

    Returns an ExtractionResult. On any failure to obtain usable LLM output the
    result is degraded to the raw top-k chunks (see _degrade).
    """
    if llm_client is None:
        from llm.llm_client import LLMClient

        llm_client = LLMClient()

    prompt = build_extraction_prompt(query, task_intent, raw_chunks)
    messages = [{"role": "user", "content": prompt}]
    try:
        raw, _metrics = await asyncio.wait_for(
            llm_client.send_message(messages, role="extractor", expect_json=False),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        return _degrade(raw_chunks, fallback_top_k, f"extractor LLM call timeout after {timeout}s")
    except Exception as exc:  # noqa: BLE001 — any LLM/transport failure must degrade, not propagate
        return _degrade(raw_chunks, fallback_top_k, f"extractor LLM call failed: {exc}")

    selected = _parse_spans(raw)
    if not selected:
        return _degrade(raw_chunks, fallback_top_k, "LLM output was empty or not a JSON array of spans")

    spans = []
    chunk_by_id = {c.get("doc_id"): c for c in raw_chunks}
    for item in selected:
        doc_id = item.get("doc_id")
        span_text = item.get("span")
        chunk = chunk_by_id.get(doc_id)
        if chunk is None or not isinstance(span_text, str) or not span_text:
            continue
        # Extractive-only: the span must be a literal substring of its source chunk.
        # Anything else is a paraphrase/hallucination and must not reach the Executor.
        if span_text not in chunk.get("snippet", ""):
            continue
        spans.append({"doc_id": doc_id, "score": chunk.get("score", 0.0), "span": span_text})

    if not spans:
        return _degrade(raw_chunks, fallback_top_k, "no verbatim spans survived extraction")

    spans = _apply_budget(spans, budget_chars)
    return ExtractionResult(spans=spans, degraded=False)


def _apply_budget(spans: list, budget_chars: int) -> list:
    """Soft span-boundary stop rule (ADR-0002 §2).

    Spans are ordered most-relevant first (by source score) and emitted whole.
    The most-relevant span is ALWAYS emitted, even if it alone exceeds the budget
    (integrity outranks budget). Each subsequent span is emitted only while the
    running total stays within budget; the first span that would exceed it stops
    emission. A span is never split or truncated.
    """
    ordered = sorted(spans, key=lambda s: s.get("score", 0.0), reverse=True)
    kept = []
    total = 0
    for span in ordered:
        length = len(span["span"])
        if kept and total + length > budget_chars:
            break
        kept.append(span)
        total += length
    return kept


def _degrade(raw_chunks: list, fallback_top_k: int, reason: str) -> ExtractionResult:
    """Fall back to the raw top-k chunks (by score), unchanged, as spans.

    This is the degradation-safe path of ADR-0002 §3: strictly no worse than the
    pre-Extractor behaviour. Snippets are passed through verbatim as spans.
    """
    ordered = sorted(raw_chunks, key=lambda c: c.get("score", 0.0), reverse=True)
    spans = [
        {"doc_id": c.get("doc_id"), "score": c.get("score", 0.0), "span": c.get("snippet", "")}
        for c in ordered[: max(0, fallback_top_k)]
    ]
    return ExtractionResult(spans=spans, degraded=True, reason=reason)


def _parse_spans(raw):
    """Defensively parse the LLM's raw text into a list of {doc_id, span} dicts.

    expect_json=False is used at the call site, so all parsing happens here. Any
    deviation from "a JSON array of objects" yields an empty list (→ degradation).
    """
    cleaned = (raw or "").strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(lines[1:] if lines[0].startswith("```") else lines)
    if cleaned.endswith("```"):
        cleaned = "\n".join(cleaned.splitlines()[:-1])
    cleaned = cleaned.strip()
    if not cleaned:
        return []

    try:
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return []

    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]
