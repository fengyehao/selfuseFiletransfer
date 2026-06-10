# tests/test_llm_payload.py
# Behavior tests for PR 1 of docs/local-model-optimization-plan.md (transport layer
# + config for local 27B–70B models). They lock the request-payload contract that
# local inference backends (vLLM / SGLang / Ollama) depend on, and assert that the
# DEFAULT config preserves the pre-change (cloud-model) behaviour — i.e. every new
# capability is opt-in.
#
# Covered: per-role max_tokens (item 1), JSON mode selection (item 2), <think>
# stripping + disable-thinking injection (item 6), derived compression threshold (item 3).

import pytest

from conf.config import LLM_MAX_TOKENS, _derive_token_compress_threshold
from llm.llm_client import LLMClient, strip_think_blocks

MSGS = [{"role": "user", "content": "hi"}]


@pytest.fixture(autouse=True)
def _reset_llm_globals(monkeypatch):
    """Pin the transport-layer globals to their documented defaults so tests are
    hermetic against whatever the ambient .env sets. Individual tests override the
    one knob they exercise."""
    monkeypatch.setattr("llm.llm_client.LLM_JSON_MODE", "json_object")
    monkeypatch.setattr("llm.llm_client.LLM_EXTRA_BODY_ENABLED", False)
    monkeypatch.setattr("llm.llm_client.LLM_THINKING", {"default": "off"})
    monkeypatch.setattr("llm.llm_client.LLM_DISABLE_THINKING", {"default": False})


def _openai_payload(client, *, role="executor", expect_json=True, max_tokens=4096):
    _headers, payload = client._prepare_openai_payload(MSGS, "local-model", 0.3, role, expect_json, max_tokens)
    return payload


# --- Item 1: per-role max_tokens -------------------------------------------------


def test_openai_payload_injects_max_tokens():
    """The OpenAI path must set max_tokens or nested JSON gets silently truncated
    by some local engines -> parse failure. This is the whole point of item 1."""
    payload = _openai_payload(LLMClient(), max_tokens=4096)
    assert payload["max_tokens"] == 4096


def test_anthropic_payload_uses_configured_max_tokens(monkeypatch):
    """Anthropic's previously-hardcoded 4096 now flows from the same config knob."""
    monkeypatch.setattr("llm.llm_client.LLM_PROVIDER", "anthropic")
    client = LLMClient()
    _headers, payload = client._prepare_anthropic_payload(MSGS, "claude-x", 2048)
    assert payload["max_tokens"] == 2048


def test_max_tokens_config_caps_extractor_lower():
    """extractor is a short extractive task, so it is intentionally capped below the
    large-JSON roles — encoding the cost/latency intent, not just a number."""
    assert LLM_MAX_TOKENS["default"] == 4096
    assert LLM_MAX_TOKENS["planner"] == 4096
    assert LLM_MAX_TOKENS["executor"] == 4096
    assert LLM_MAX_TOKENS["reflector"] == 4096
    assert LLM_MAX_TOKENS["extractor"] == 1024


# --- Item 2: JSON mode -----------------------------------------------------------


def test_json_mode_json_object_is_default_and_backward_compatible(monkeypatch):
    """Default mode reproduces the exact pre-change payload (response_format only)."""
    monkeypatch.setattr("llm.llm_client.LLM_JSON_MODE", "json_object")
    payload = _openai_payload(LLMClient(), expect_json=True)
    assert payload["response_format"] == {"type": "json_object"}
    assert "format" not in payload
    assert "extra_body" not in payload


def test_json_mode_ollama_uses_top_level_format(monkeypatch):
    monkeypatch.setattr("llm.llm_client.LLM_JSON_MODE", "ollama")
    payload = _openai_payload(LLMClient(), expect_json=True)
    assert payload["format"] == "json"
    assert "response_format" not in payload


def test_json_mode_guided_json_uses_extra_body(monkeypatch):
    monkeypatch.setattr("llm.llm_client.LLM_JSON_MODE", "guided_json")
    payload = _openai_payload(LLMClient(), expect_json=True)
    assert payload["extra_body"]["guided_json"] == {"type": "object"}
    assert "response_format" not in payload
    assert "format" not in payload


def test_json_mode_off_injects_no_constraint(monkeypatch):
    """`off` is the escape hatch for backends that reject response_format entirely."""
    monkeypatch.setattr("llm.llm_client.LLM_JSON_MODE", "off")
    payload = _openai_payload(LLMClient(), expect_json=True)
    assert "response_format" not in payload
    assert "format" not in payload
    assert "extra_body" not in payload


def test_json_constraint_skipped_when_not_expecting_json(monkeypatch):
    """No JSON constraint when the caller doesn't expect JSON, whatever the mode."""
    monkeypatch.setattr("llm.llm_client.LLM_JSON_MODE", "guided_json")
    payload = _openai_payload(LLMClient(), expect_json=False)
    assert "response_format" not in payload
    assert "format" not in payload
    assert "extra_body" not in payload


# --- Item 6: <think> stripping + disable-thinking --------------------------------


def test_strip_think_blocks_paired():
    assert strip_think_blocks('<think>reasoning here</think>{"real":true}') == '{"real":true}'


def test_strip_think_blocks_dangling_to_end():
    """An unclosed <think> (truncated reasoning) is removed through end-of-string."""
    assert strip_think_blocks("keep this<think>truncated reasoning...") == "keep this"


def test_strip_think_blocks_noop_without_think():
    assert strip_think_blocks('{"a":1}') == '{"a":1}'


def test_robust_parser_drops_think_wrapper():
    """The plan's exact item-6 verification: <think>{...}</think>{real} -> {real}.
    Without stripping, the parser's first-{-to-last-} heuristic would swallow the
    think block's braces and mis-parse."""
    parsed = LLMClient()._robust_json_parser('<think>{"a":1}</think>{"real":true}')
    assert parsed == {"real": True}


def test_disable_thinking_vllm_style_uses_chat_template_kwargs(monkeypatch):
    monkeypatch.setattr("llm.llm_client.LLM_JSON_MODE", "guided_json")
    monkeypatch.setattr("llm.llm_client.LLM_DISABLE_THINKING", {"default": True})
    payload = _openai_payload(LLMClient(), role="executor", expect_json=True)
    assert payload["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "think" not in payload  # top-level think is the ollama path only


def test_disable_thinking_ollama_style_uses_top_level_think(monkeypatch):
    monkeypatch.setattr("llm.llm_client.LLM_JSON_MODE", "ollama")
    monkeypatch.setattr("llm.llm_client.LLM_DISABLE_THINKING", {"default": True})
    payload = _openai_payload(LLMClient(), role="executor", expect_json=True)
    assert payload["think"] is False
    assert "extra_body" not in payload  # ollama doesn't use chat_template_kwargs


def test_thinking_mode_survives_extra_body_refactor(monkeypatch):
    """The pre-existing extra_body.thinking behaviour must survive being merged into
    the unified _build_extra_body assembler (item 2/6 share extra_body)."""
    monkeypatch.setattr("llm.llm_client.LLM_EXTRA_BODY_ENABLED", True)
    monkeypatch.setattr("llm.llm_client.LLM_THINKING", {"default": "hidden"})
    payload = _openai_payload(LLMClient(), role="executor", expect_json=True)
    assert payload["extra_body"]["thinking"] == "hidden"


# --- Item 3: derived compression threshold ---------------------------------------


def test_threshold_derived_from_context_window_when_unset():
    """Unset -> 70% of the context window, so compression fires before a small local
    window overflows (int() truncates)."""
    assert _derive_token_compress_threshold(None, 32768) == 22937
    assert _derive_token_compress_threshold(None, 8192) == 5734


def test_explicit_threshold_overrides_derivation():
    assert _derive_token_compress_threshold("80000", 8192) == 80000
