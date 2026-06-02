# tests/test_retrieve_knowledge_extractor.py
# Behavior tests for the Extractor integration in the `retrieve_knowledge` MCP tool.
#
# Contract (ADR-0001 / ADR-0002):
#   retrieve_knowledge retrieves EXTRACTOR_RETRIEVE_K candidates, runs _extractor_extract,
#   returns the existing results JSON shape.  On any extractor failure (degraded / exception)
#   it falls back to raw top-k=5 and adds extractor_fallback:true to the JSON.
#
# Test strategy: call retrieve_knowledge() directly (FastMCP preserves the coroutine).
#   Patch: _httpx_client (service HTTP), _extractor_extract (LLM extraction step),
#          _console (rich Panel warnings), get_llm_client (LLM client factory).

import json
from unittest.mock import AsyncMock, MagicMock, call, patch

import httpx
import pytest
from rich.panel import Panel

import tools.mcp_service as _m
from conf.config import EXTRACTOR_RETRIEVE_K
from rag.extractor import ExtractionResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _service_response(results: list, query: str = "q") -> MagicMock:
    """Build a fake httpx response from the knowledge service."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "success": True,
        "query": query,
        "total_results": len(results),
        "results": results,
    }
    return resp


def _span(doc_id: str, score: float, span: str) -> dict:
    return {"doc_id": doc_id, "score": score, "span": span}


def _result(id_: str, score: float, snippet: str) -> dict:
    return {"id": id_, "score": score, "snippet": snippet}


# ---------------------------------------------------------------------------
# Slice 1 – tracer bullet: happy path returns extracted spans in existing shape
# ---------------------------------------------------------------------------

async def test_extractor_happy_path():
    """Extractor succeeds: output has the two extracted spans, no extractor_fallback key."""
    candidates = [_result("sqli.md", 0.9, "' OR 1=1--"), _result("xss.md", 0.7, "<script>")]
    extraction = ExtractionResult(
        spans=[_span("sqli.md", 0.9, "' OR 1=1--"), _span("xss.md", 0.7, "<script>")],
        degraded=False,
    )

    with (
        patch.object(_m._httpx_client, "post", new=AsyncMock(return_value=_service_response(candidates, "sql injection"))),
        patch("tools.mcp_service._extractor_extract", new=AsyncMock(return_value=extraction)),
        patch("tools.mcp_service.get_llm_client", return_value=MagicMock()),
    ):
        out = json.loads(await _m.retrieve_knowledge("sql injection"))

    assert out["success"] is True
    assert out["query"] == "sql injection"
    assert out["total_results"] == 2
    assert out["results"] == [
        {"id": "sqli.md", "snippet": "' OR 1=1--", "score": 0.9},
        {"id": "xss.md", "snippet": "<script>", "score": 0.7},
    ]
    assert "extractor_fallback" not in out


# ---------------------------------------------------------------------------
# Slice 2 – service is called with EXTRACTOR_RETRIEVE_K, not the caller's top_k
# ---------------------------------------------------------------------------

async def test_service_called_with_extractor_retrieve_k():
    """The POST to the knowledge service always uses EXTRACTOR_RETRIEVE_K, not top_k."""
    extraction = ExtractionResult(spans=[], degraded=False)

    with (
        patch.object(_m._httpx_client, "post", new=AsyncMock(return_value=_service_response([]))) as mock_post,
        patch("tools.mcp_service._extractor_extract", new=AsyncMock(return_value=extraction)),
        patch("tools.mcp_service.get_llm_client", return_value=MagicMock()),
    ):
        await _m.retrieve_knowledge("q", top_k=3)

    _, kwargs = mock_post.call_args
    posted_json = kwargs.get("json") or mock_post.call_args[0][1] if len(mock_post.call_args[0]) > 1 else kwargs["json"]
    assert posted_json["top_k"] == EXTRACTOR_RETRIEVE_K


# ---------------------------------------------------------------------------
# Slice 3 – task_intent is forwarded to _extractor_extract
# ---------------------------------------------------------------------------

async def test_task_intent_passed_to_extractor():
    """task_intent param is forwarded as the second positional arg to _extractor_extract."""
    extraction = ExtractionResult(spans=[], degraded=False)
    mock_extract = AsyncMock(return_value=extraction)

    with (
        patch.object(_m._httpx_client, "post", new=AsyncMock(return_value=_service_response([]))),
        patch("tools.mcp_service._extractor_extract", new=mock_extract),
        patch("tools.mcp_service.get_llm_client", return_value=MagicMock()),
    ):
        await _m.retrieve_knowledge("q", task_intent="bypass /admin/login")

    _query, task_intent, *_ = mock_extract.call_args[0]
    assert task_intent == "bypass /admin/login"


# ---------------------------------------------------------------------------
# Slice 4 – extractor degraded → extractor_fallback:true + Panel warning
# ---------------------------------------------------------------------------

async def test_extractor_degraded_sets_fallback_flag_and_warns():
    """When extraction.degraded is True, output gains extractor_fallback:true and a Panel is printed."""
    fallback_spans = [_span("a.md", 0.9, "raw snippet a"), _span("b.md", 0.5, "raw snippet b")]
    extraction = ExtractionResult(spans=fallback_spans, degraded=True, reason="LLM output was empty")
    mock_console = MagicMock()

    with (
        patch.object(_m._httpx_client, "post", new=AsyncMock(return_value=_service_response([]))),
        patch("tools.mcp_service._extractor_extract", new=AsyncMock(return_value=extraction)),
        patch("tools.mcp_service.get_llm_client", return_value=MagicMock()),
        patch("tools.mcp_service._console", mock_console),
    ):
        out = json.loads(await _m.retrieve_knowledge("q"))

    assert out["extractor_fallback"] is True
    mock_console.print.assert_called_once()
    printed_arg = mock_console.print.call_args[0][0]
    assert isinstance(printed_arg, Panel)


# ---------------------------------------------------------------------------
# Slice 5 – extractor raises → extractor_fallback:true + raw results[:5] + Panel warning
# ---------------------------------------------------------------------------

async def test_extractor_exception_sets_fallback_flag_and_warns():
    """If _extractor_extract raises, output has extractor_fallback:true, raw results[:5], Panel printed."""
    raw = [_result(f"{i}.md", 1.0 - i * 0.1, f"snippet {i}") for i in range(8)]
    mock_console = MagicMock()

    with (
        patch.object(_m._httpx_client, "post", new=AsyncMock(return_value=_service_response(raw))),
        patch("tools.mcp_service._extractor_extract", new=AsyncMock(side_effect=RuntimeError("boom"))),
        patch("tools.mcp_service.get_llm_client", return_value=MagicMock()),
        patch("tools.mcp_service._console", mock_console),
    ):
        out = json.loads(await _m.retrieve_knowledge("q"))

    assert out["extractor_fallback"] is True
    assert len(out["results"]) == 5        # capped at 5
    assert out["results"][0]["id"] == "0.md"
    mock_console.print.assert_called_once()
    printed_arg = mock_console.print.call_args[0][0]
    assert isinstance(printed_arg, Panel)


# ---------------------------------------------------------------------------
# Slice 6 – service connection error returns existing error shape, no extractor_fallback
# ---------------------------------------------------------------------------

async def test_service_connection_error_returns_error_shape():
    """httpx.RequestError → existing error JSON with success:false, no extractor_fallback."""
    with patch.object(
        _m._httpx_client, "post", new=AsyncMock(side_effect=httpx.RequestError("timeout"))
    ):
        out = json.loads(await _m.retrieve_knowledge("q"))

    assert out["success"] is False
    assert "error" in out
    assert "extractor_fallback" not in out


# ---------------------------------------------------------------------------
# Slice 7 – result items carry id / snippet / score keys
# ---------------------------------------------------------------------------

async def test_result_shape_id_snippet_score():
    """Each result in the happy path has exactly id, snippet, score keys (existing contract)."""
    extraction = ExtractionResult(
        spans=[_span("doc.md", 0.88, "exact payload")],
        degraded=False,
    )

    with (
        patch.object(_m._httpx_client, "post", new=AsyncMock(return_value=_service_response([_result("doc.md", 0.88, "exact payload")]))),
        patch("tools.mcp_service._extractor_extract", new=AsyncMock(return_value=extraction)),
        patch("tools.mcp_service.get_llm_client", return_value=MagicMock()),
    ):
        out = json.loads(await _m.retrieve_knowledge("q"))

    assert len(out["results"]) == 1
    item = out["results"][0]
    assert set(item.keys()) == {"id", "snippet", "score"}
    assert item["id"] == "doc.md"
    assert item["snippet"] == "exact payload"
    assert item["score"] == 0.88
