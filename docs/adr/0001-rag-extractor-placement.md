# RAG Extractor lives in `rag/` but is invoked from the MCP tool layer

## Context

We are adding an **Extractor** stage between the Executor and the RAG knowledge base: it retrieves a larger candidate pool, then uses a local LLM to extractively return only the query-relevant material, protecting the Executor's context window. The Extractor needs a generative LLM. The `retrieve_knowledge` MCP tool runs in the agent/MCP process where `get_llm_client()` (provider config, model routing, cost metrics) is available; the `knowledge_service` HTTP service (port 8081) is a separate process that only holds the embedding model and `RAGClient`.

## Decision

The Extractor's code belongs to the RAG module (`rag/extractor.py`), but its LLM call executes in the `retrieve_knowledge` MCP tool — not inside the `knowledge_service`. The HTTP service stays a pure retrieval service (FAISS + BM25 + rerank). This follows the existing `expert_analysis` pattern of an MCP tool invoking the LLM client via a dedicated role.

## Considered Options

- **Server-side, inside `knowledge_service`** — most literally "in the RAG service", but would force LLM provider config, keys, model routing, and cost tracking into a currently-thin retrieval service, duplicating the agent's `LLMClient` across two processes. Rejected for a single-process local deployment.

## Consequences

- "Module ownership" (`rag/`) and "execution process" (MCP) are deliberately decoupled — a future reader who expects extraction to run inside the RAG service should know this was an intentional reuse-of-LLM-infrastructure trade-off.
- If the RAG service is ever made independently reusable by other processes, this decision should be revisited (extraction would then belong server-side).
