# LuaN1aoAgent — Domain Language

Autonomous penetration-testing / CTF agent built as a Planner → Executor → Reflector loop over an LLM, with a local RAG knowledge base for attack techniques and writeups.

## Language

### Agents

**Executor**:
The ReAct-style LLM agent that carries out a plan step by step by calling MCP tools. The only consumer of the RAG knowledge base.

**Extractor** (RAG Extractor / 检索精炼器):
A single-pass, stateless retrieval post-processing stage inside the RAG module. Given a `query` and a `task_intent`, it retrieves candidate chunks, then uses a local LLM to **extractively** select only the relevant material — verbatim spans plus their source `doc_id`/`score`, never paraphrased — instead of returning all raw snippets. Code blocks and payload lines are preserved whole. It does **not** reason autonomously, loop, maintain state, summarise, or rewrite.
_Avoid_: "信息提取 agent", "extraction agent" — it is deliberately not an agent, to keep it inside the RAG module and avoid nesting an uncontrolled LLM loop inside the Executor loop.

**Task intent**:
A short, explicit statement of what the Executor is currently trying to achieve (e.g. the current node goal or where it is stuck), passed alongside the `query` into the Extractor so it can judge which retrieved material is relevant. It is **not** the Executor's conversation history.
_Avoid_: "context", "上下文" — too broad; this is one Executor-authored sentence, not the dialogue.

**Reflector**:
The LLM agent that analyses execution failures and revises the plan.

**Planner**:
The LLM agent that produces the initial plan / attack graph.
