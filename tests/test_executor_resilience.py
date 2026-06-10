# tests/test_executor_resilience.py
# Behavior tests for PR 2 / item 4 of docs/local-model-optimization-plan.md (graceful
# degradation on parse failure). Local 27B–70B models routinely emit truncated or
# prose-wrapped JSON; before this change a single unparseable Executor reply killed the
# whole subtask (RuntimeError("llm_empty_response")). These tests lock the two halves of
# the fix:
#   1. send_message surfaces the last raw content (via the per-call metrics dict) when
#      JSON parsing is exhausted, so the Executor has something to salvage from.
#   2. _salvage_execution_operations extracts executable operations from that raw text,
#      tolerating truncation / prose / <think> noise — and returns None (→ degrade, don't
#      kill) only when nothing executable can be recovered.

import asyncio
import json

import pytest

from core.executor import _salvage_execution_operations, _scan_balanced
from llm.llm_client import LLMClient

# --- _scan_balanced: payload-safe bracket matching -------------------------------


def test_scan_balanced_respects_brackets_inside_strings():
    """Bracket matching must honour JSON string literals, or a payload value containing
    ']' or '}' (common in SQLi / shell payloads) would truncate the salvaged op."""
    text = '[{"action": {"params": {"cmd": "echo ] } done"}}}] trailing'
    end = _scan_balanced(text, 0)
    assert text[:end] == '[{"action": {"params": {"cmd": "echo ] } done"}}}]'


def test_scan_balanced_returns_none_when_truncated():
    """An unclosed delimiter (the hallmark of a mid-stream truncation) yields None so the
    caller falls back to scavenging whatever complete objects precede the cut."""
    assert _scan_balanced('[{"a": 1}, {"b":', 0) is None


# --- _salvage_execution_operations: the core resilience contract -----------------


def _assert_executable(op):
    """A salvaged op is only useful if it survives the Executor's EXECUTE_NOW + node_id
    filter and carries a tool — otherwise the subtask would just stall."""
    assert str(op["command"]).upper() == "EXECUTE_NOW"
    assert op.get("node_id") and op["node_id"] != "None"
    assert op["action"].get("tool") or op["action"].get("name")


def test_salvage_recovers_complete_op_from_truncated_array():
    """The headline local-model failure: max_tokens cut the reply off mid-array. The
    first, complete op must still be recovered (and the truncated tail dropped)."""
    raw = (
        '{"thought": "facts {with braces} here", '
        '"execution_operations": ['
        '{"command": "EXECUTE_NOW", "node_id": "s1", '
        '"action": {"tool": "http_request", "params": {"url": "http://t/a"}}}, '
        '{"command": "EXECUTE_NOW", "node_id": "s2", "action": {"tool": "shell_exec", "params": {"cmd":'  # cut off
    )
    salvaged = _salvage_execution_operations(raw)
    assert salvaged is not None
    ops = salvaged["execution_operations"]
    assert len(ops) == 1
    assert ops[0]["node_id"] == "s1"
    assert ops[0]["action"]["tool"] == "http_request"
    _assert_executable(ops[0])
    # The salvaged reply must not falsely claim completion (that would end the subtask).
    assert salvaged["is_subtask_complete"] is False


def test_salvage_extracts_array_from_prose_and_fences():
    """Models that ignore 'JSON only' wrap the object in prose / ``` fences. The
    execution_operations array is still balanced and must be extracted."""
    raw = (
        "Sure! Here's the plan:\n```json\n"
        '{"thought": "go", "execution_operations": '
        '[{"command": "EXECUTE_NOW", "node_id": "s1", '
        '"action": {"tool": "http_request", "params": {"url": "http://t"}}}], '
        '"is_subtask_complete": false}\n```\nHope this helps!'
    )
    salvaged = _salvage_execution_operations(raw)
    assert salvaged is not None
    assert len(salvaged["execution_operations"]) == 1
    _assert_executable(salvaged["execution_operations"][0])


def test_salvage_wraps_bare_action_object():
    """When the model drops the wrapper and emits only a bare {tool, params}, salvage
    must promote it to a full EXECUTE_NOW op (synthesising command + node_id)."""
    raw = 'I think we should: {"tool": "http_request", "params": {"url": "http://t"}} ok?'
    salvaged = _salvage_execution_operations(raw)
    assert salvaged is not None
    op = salvaged["execution_operations"][0]
    assert op["action"]["tool"] == "http_request"
    _assert_executable(op)


def test_salvage_preserves_payload_with_literal_brackets():
    """A salvaged op's payload must come through byte-for-byte even when it contains the
    very delimiters the scanner tracks — otherwise we'd corrupt the attack payload."""
    raw = (
        '{"execution_operations": [{"command": "EXECUTE_NOW", "node_id": "s1", '
        '"action": {"tool": "shell_exec", "params": {"cmd": "echo ] } [ {"}}}]}'
    )
    salvaged = _salvage_execution_operations(raw)
    assert salvaged is not None
    assert salvaged["execution_operations"][0]["action"]["params"]["cmd"] == "echo ] } [ {"


def test_salvage_returns_none_for_pure_think_noise():
    """A reply that is only a <think> block (no real JSON) is unsalvageable → None, which
    the Executor turns into a degrade step rather than executing garbage."""
    raw = '<think>let me consider {"tool": "x"} as an option</think>'
    assert _salvage_execution_operations(raw) is None


def test_salvage_returns_none_for_non_string_and_garbage():
    assert _salvage_execution_operations(None) is None
    assert _salvage_execution_operations(12345) is None
    assert _salvage_execution_operations("no json, no actions, just apologies.") is None


# --- send_message: surface raw on JSON-parse exhaustion --------------------------


class _FakeResponse:
    def __init__(self, content):
        self.status_code = 200
        self.text = json.dumps(
            {
                "choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3},
            }
        )


@pytest.mark.asyncio
async def test_send_message_surfaces_raw_when_json_unparseable(monkeypatch):
    """item 4 transport half: after JSON retries are exhausted, the last raw content must
    be handed back through the metrics dict (concurrency-safe, since the LLMClient is
    shared across parallel subtasks) so the Executor can attempt salvage. Previously this
    returned (None, None) and the raw was lost."""
    monkeypatch.setattr("llm.llm_client.LLM_PROVIDER", "openai")
    client = LLMClient()
    unparseable = "I'm sorry, I can't help with that — no JSON here at all."

    async def _fake_post(*args, **kwargs):
        return _FakeResponse(unparseable)

    async def _noop_sleep(*args, **kwargs):
        return None

    monkeypatch.setattr(client.client, "post", _fake_post)
    monkeypatch.setattr(asyncio, "sleep", _noop_sleep)  # skip the inter-retry backoff

    result, metrics = await client.send_message(
        [{"role": "user", "content": "hi"}], role="executor", expect_json=True
    )

    assert result is None
    assert isinstance(metrics, dict)
    assert metrics["raw"] == unparseable


@pytest.mark.asyncio
async def test_send_message_salvage_roundtrip(monkeypatch):
    """End-to-end of the two halves: a truncated reply from the wire is surfaced as raw,
    and that raw is exactly what the salvage path can recover an executable op from."""
    monkeypatch.setattr("llm.llm_client.LLM_PROVIDER", "openai")
    client = LLMClient()
    truncated = (
        '{"thought": "t", "execution_operations": ['
        '{"command": "EXECUTE_NOW", "node_id": "s1", '
        '"action": {"tool": "http_request", "params": {"url": "http://t"}}}, {"command": "EXEC'
    )

    async def _fake_post(*args, **kwargs):
        return _FakeResponse(truncated)

    async def _noop_sleep(*args, **kwargs):
        return None

    monkeypatch.setattr(client.client, "post", _fake_post)
    monkeypatch.setattr(asyncio, "sleep", _noop_sleep)

    result, metrics = await client.send_message(
        [{"role": "user", "content": "hi"}], role="executor", expect_json=True
    )
    assert result is None
    salvaged = _salvage_execution_operations(metrics["raw"])
    assert salvaged is not None
    assert salvaged["execution_operations"][0]["node_id"] == "s1"
