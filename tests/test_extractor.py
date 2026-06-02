# tests/test_extractor.py
# Behavior tests for rag/extractor.py — the RAG Extractor (检索精炼器).
# Design is locked in CONTEXT.md + docs/adr/0001 + docs/adr/0002.
#
# The Extractor is a single-pass, stateless function:
#   (query, task_intent, raw_chunks) -> ExtractionResult
# It extractively selects verbatim spans (never paraphrases), tags each with its
# source doc_id/score, treats EXTRACTOR_OUTPUT_BUDGET_CHARS as a soft span-boundary
# stop rule (integrity > budget), and degrades to the raw top-k chunks on any
# malformed/empty/timed-out LLM output.

import asyncio
import json

import pytest


def _json_str(s: str) -> str:
    """Encode a Python string as a JSON string literal (with surrounding quotes)."""
    return json.dumps(s)

from rag.extractor import extract


class FakeLLMClient:
    """Duck-typed stand-in for llm.LLMClient.

    The Extractor only depends on the async ``send_message`` boundary, so we mock
    there (the external dependency), not the Extractor's internals.
    """

    def __init__(self, content, *, raises=None, delay=0.0):
        self._content = content
        self._raises = raises
        self._delay = delay
        self.calls = []

    async def send_message(self, messages, role="default", expect_json=True):
        self.calls.append({"messages": messages, "role": role, "expect_json": expect_json})
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._raises is not None:
            raise self._raises
        return self._content, {"prompt_tokens": 0, "completion_tokens": 0}


def _chunk(doc_id, score, snippet):
    return {"doc_id": doc_id, "score": score, "snippet": snippet}


@pytest.mark.asyncio
async def test_extracts_verbatim_span_and_reattaches_source_score():
    """A valid span is returned verbatim, tagged with the score from its source chunk."""
    chunks = [
        _chunk("sqli-basics.md", 0.92, "Use a tautology:\n```sql\n' OR '1'='1\n```\nto bypass."),
        _chunk("xss.md", 0.61, "Unrelated XSS material."),
    ]
    llm = FakeLLMClient('[{"doc_id": "sqli-basics.md", "span": "```sql\\n\' OR \'1\'=\'1\\n```"}]')

    result = await extract("sql injection login bypass", "bypass /admin/login", chunks, llm_client=llm)

    assert result.degraded is False
    assert len(result.spans) == 1
    span = result.spans[0]
    assert span["doc_id"] == "sqli-basics.md"
    assert span["span"] == "```sql\n' OR '1'='1\n```"
    # score is re-attached from the source chunk, NOT taken from the LLM output
    assert span["score"] == 0.92
    # the Extractor must call the LLM under the dedicated "extractor" role with expect_json=False
    assert llm.calls[0]["role"] == "extractor"
    assert llm.calls[0]["expect_json"] is False


@pytest.mark.asyncio
async def test_parses_json_wrapped_in_markdown_fence():
    """Weak models often wrap the JSON array in a ```json fence; it must still parse."""
    chunks = [_chunk("a.md", 0.8, "payload: admin'--")]
    llm = FakeLLMClient('```json\n[{"doc_id": "a.md", "span": "admin\'--"}]\n```')

    result = await extract("q", "intent", chunks, llm_client=llm)

    assert result.degraded is False
    assert [s["span"] for s in result.spans] == ["admin'--"]


@pytest.mark.asyncio
async def test_malformed_output_degrades_to_raw_top_k():
    """Non-JSON garbage from the model degrades to the raw top-k chunks, ordered by score.

    Degradation must be strictly no worse than the pre-Extractor behaviour: the raw
    snippets come back unchanged, flagged so the call site can surface a warning.
    """
    chunks = [
        _chunk("low.md", 0.40, "low relevance snippet"),
        _chunk("high.md", 0.95, "```sql\n' OR 1=1--\n```"),
        _chunk("mid.md", 0.70, "mid relevance snippet"),
    ]
    llm = FakeLLMClient("Sure! Here are the relevant spans you asked for: ...")

    result = await extract("q", "intent", chunks, llm_client=llm, fallback_top_k=2)

    assert result.degraded is True
    assert result.reason is not None
    # top-2 by score, verbatim snippets preserved as spans
    assert [s["doc_id"] for s in result.spans] == ["high.md", "mid.md"]
    assert result.spans[0]["span"] == "```sql\n' OR 1=1--\n```"
    assert result.spans[0]["score"] == 0.95


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_output", [None, "", "   ", '{"doc_id": "a.md", "span": "x"}', "[1, 2, 3]"])
async def test_empty_or_non_array_output_degrades(bad_output):
    """None/empty output, or valid JSON that isn't an array of objects, degrades."""
    chunks = [_chunk("a.md", 0.9, "some snippet")]
    llm = FakeLLMClient(bad_output)

    result = await extract("q", "intent", chunks, llm_client=llm)

    assert result.degraded is True
    assert [s["doc_id"] for s in result.spans] == ["a.md"]


@pytest.mark.asyncio
async def test_non_verbatim_span_is_dropped():
    """A span that is not a literal substring of its source chunk is paraphrase, not
    extraction — it must be dropped so no rewritten content reaches the Executor."""
    chunks = [
        _chunk("a.md", 0.9, "Bypass with:\n```sql\n' OR '1'='1\n```"),
        _chunk("b.md", 0.8, "Use ```sql\n1; DROP TABLE users--\n``` to drop."),
    ]
    # first span is verbatim; second is a paraphrase ("OR 1 equals 1") not in b.md
    llm = FakeLLMClient(
        '[{"doc_id": "a.md", "span": "```sql\\n\' OR \'1\'=\'1\\n```"},'
        ' {"doc_id": "b.md", "span": "drop the users table with OR 1 equals 1"}]'
    )

    result = await extract("q", "intent", chunks, llm_client=llm)

    assert result.degraded is False
    assert [s["doc_id"] for s in result.spans] == ["a.md"]


@pytest.mark.asyncio
async def test_all_spans_non_verbatim_degrades():
    """If every returned span fails the verbatim check, nothing is extractive → degrade."""
    chunks = [_chunk("a.md", 0.9, "```sql\n' OR '1'='1\n```")]
    llm = FakeLLMClient('[{"doc_id": "a.md", "span": "a paraphrased explanation of the bypass"}]')

    result = await extract("q", "intent", chunks, llm_client=llm)

    assert result.degraded is True
    assert [s["doc_id"] for s in result.spans] == ["a.md"]
    assert result.spans[0]["span"] == "```sql\n' OR '1'='1\n```"


@pytest.mark.asyncio
async def test_span_with_unknown_doc_id_is_dropped():
    """A span citing a doc_id not in the candidate set can't be verified or scored → drop."""
    chunks = [_chunk("a.md", 0.9, "real: admin'--")]
    llm = FakeLLMClient(
        '[{"doc_id": "a.md", "span": "admin\'--"},'
        ' {"doc_id": "hallucinated.md", "span": "anything"}]'
    )

    result = await extract("q", "intent", chunks, llm_client=llm)

    assert result.degraded is False
    assert [s["doc_id"] for s in result.spans] == ["a.md"]


@pytest.mark.asyncio
async def test_budget_stops_at_span_boundary_keeping_whole_spans():
    """Spans accumulate, most-relevant first, until the next span would exceed the
    budget — then we stop. Spans are emitted whole; none is truncated."""
    chunks = [
        _chunk("high.md", 0.9, "A" * 30),
        _chunk("mid.md", 0.8, "B" * 30),
        _chunk("low.md", 0.7, "C" * 30),
    ]
    llm = FakeLLMClient(
        '[{"doc_id": "high.md", "span": "' + "A" * 30 + '"},'
        ' {"doc_id": "mid.md", "span": "' + "B" * 30 + '"},'
        ' {"doc_id": "low.md", "span": "' + "C" * 30 + '"}]'
    )

    # budget 50: high (30) fits; adding mid (→60) would exceed → stop before mid
    result = await extract("q", "intent", chunks, llm_client=llm, budget_chars=50)

    assert result.degraded is False
    assert [s["doc_id"] for s in result.spans] == ["high.md"]
    assert result.spans[0]["span"] == "A" * 30  # whole, never truncated


@pytest.mark.asyncio
async def test_single_relevant_payload_exceeding_budget_returned_whole():
    """Integrity > budget: the single most-relevant span is always returned whole,
    even if it alone exceeds EXTRACTOR_OUTPUT_BUDGET_CHARS. Never split a payload."""
    big_payload = (
        "```sql\n"
        + "' UNION SELECT " + ",".join(f"col{i}" for i in range(400)) + " FROM users--\n"
        + "```"
    )
    assert len(big_payload) > 1500  # genuinely larger than the budget
    chunks = [_chunk("huge.md", 0.99, "Intro.\n" + big_payload + "\nOutro.")]
    llm = FakeLLMClient('[{"doc_id": "huge.md", "span": ' + _json_str(big_payload) + "}]")

    result = await extract("q", "intent", chunks, llm_client=llm, budget_chars=1500)

    assert result.degraded is False
    assert len(result.spans) == 1
    # returned whole and verbatim — not truncated to the budget, fences intact
    assert result.spans[0]["span"] == big_payload
    assert result.spans[0]["span"].endswith("```")
    assert len(result.spans[0]["span"]) > 1500


@pytest.mark.asyncio
async def test_llm_exception_degrades():
    """Any exception from the LLM call degrades to raw top-k instead of propagating."""
    chunks = [_chunk("a.md", 0.9, "snippet a"), _chunk("b.md", 0.5, "snippet b")]
    llm = FakeLLMClient(None, raises=RuntimeError("boom"))

    result = await extract("q", "intent", chunks, llm_client=llm)

    assert result.degraded is True
    assert result.reason is not None
    assert [s["doc_id"] for s in result.spans] == ["a.md", "b.md"]


@pytest.mark.asyncio
async def test_llm_timeout_degrades():
    """If the LLM call exceeds EXTRACTOR_TIMEOUT, degrade to raw top-k."""
    chunks = [_chunk("a.md", 0.9, "snippet a")]
    llm = FakeLLMClient('[{"doc_id": "a.md", "span": "snippet a"}]', delay=1.0)

    result = await extract("q", "intent", chunks, llm_client=llm, timeout=0.05)

    assert result.degraded is True
    assert "timeout" in result.reason.lower()
    assert [s["doc_id"] for s in result.spans] == ["a.md"]
