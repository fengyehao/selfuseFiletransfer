# 本地模型 (27B–70B) 适配优化计划

> 目标：让 LuaN1aoAgent 在本地部署的 27B–70B 开源模型（Qwen2.5-32B/72B、Gemma-2-27B、
> Llama-3.3-70B、QwQ/Qwen3 等）上稳定运行。当前代码按 GPT-4o / Claude 的能力假设编写，
> 在**严格 JSON 结构化输出、长上下文、复杂嵌套指令跟随**三点上对本地模型缺乏针对性容错。
>
> 范围：本计划覆盖 6 项改动。全部改动**对云端模型向后兼容**（默认行为不变，新能力靠开关启用）。

## TL;DR — 需要几个 Session？

**建议拆成 2 个 Session / 2 个 PR**（即使在同一 session 内也按两个提交单元推进）：

- **PR 1「传输层 + 配置」** — 低风险、向后兼容、不改 P-E-R 逻辑：
  items **1 (max_tokens)**、**2 (guided-JSON, 宽松档)**、**6 (剥离 `<think>` + /no_think)**、**3 (压缩阈值 + 上下文窗口)**。
  全部集中在 `conf/config.py`、`llm/llm_client.py`、`.env.example`，可静态自测，独立可评审。

- **PR 2「韧性 + Schema」** — 改运行时行为与 prompt 契约，建议对着真实本地模型冒烟验证：
  items **4 (解析失败优雅降级)**、**5 (Executor schema 瘦身)**。

**为什么不强行塞进一个 session**：(1) 上下文预算——`core/executor.py` 1023 行、`reflector.py` 782 行、
`graph_manager.py` 较大，一次性读+改易耗尽上下文；(2) 4/5 改变运行时行为，值得独立评审并对真实模型验证；
(3) item 2 与 item 5 存在 schema 耦合（见下方「关键依赖」），分阶段能用「单一数据源」干净解耦。

> 单 session 也能写完代码，但 2 段式更安全、更易回滚。若上下文允许，可在一个 session 内连续完成两个 PR。

## 关键发现（决定风险评估）

1. **顶层 `thought` 块是只写的**：全仓没有任何代码读取 `thought.step1_analysis` 等 5 个子字段。
   唯一被消费的 `thought` 是 `execution_operations[].thought`（逐操作字符串，`core/executor.py:734`）。
   → **item 5 把顶层 `thought` 合并为单字段是低风险改动**，下游无依赖。
2. **`hypothesis_update` 被消费**（`core/executor.py:731`、`core/graph_manager.py:1747-1749`）；
   **`staged_causal_nodes` / `execution_operations` 被消费**（`agent.py`、`graph_manager.py` 多处）。
   → 这些字段瘦身需谨慎，**item 5 只动顶层 `thought`，不动这三者的结构**。
3. **`send_message` 失败只返回 `(None, None)`**（`llm/llm_client.py:382,417`），Executor 收到 None 直接
   `raise RuntimeError("llm_empty_response")`（`core/executor.py:266-274`）杀掉整个子任务。→ item 4 的改造点。
4. **zh 与 en 两套 executor schema 都在**（`core/prompts/templates/{zh,en}/executor/output_schemas/executor_schema.jinja2`）。
   → item 5 必须同步改两份。
5. **`rag/extractor.py` 是「为本地模型设计」的范本**：降级安全 + 抽取式校验 + 按模型规模的旋钮。
   item 4 的降级思路可复用其 `_degrade()` 模式。

## 关键依赖（item 2 ↔ item 5）

guided-JSON 若使用「严格 schema 语法」，该 schema 必须与 Executor 实际输出结构一致；而 item 5 会改 Executor schema。
**解耦方案**：item 2 分两档实现——

- **2a 宽松档（PR 1 落地）**：只强制「输出为合法 JSON 对象」。各后端均支持（Ollama `format:"json"`、
  vLLM `guided_json` 配极简 schema、或 `response_format:{type:json_object}`）。**与具体字段无关**，能跨越 item 5 的 schema 变更存活，已捕获 80% 收益（杜绝截断/夹带散文的坏 JSON）。
- **2b 严格档（可选，PR 2 之后）**：用完整 JSON Schema 约束工具名枚举等。**必须在 item 5 定稿后**，
  并以「单一数据源」生成（prompt 里展示的 schema 与传给后端的 grammar 来自同一处定义）。

---

## PR 1：传输层 + 配置

### Item 1 — OpenAI 路径补 `max_tokens`（per-role 可配）

- **现状**：`_prepare_openai_payload`（`llm/llm_client.py:163-190`）**未设** `max_tokens`；Anthropic 路径硬编码 4096（`:154`）。本地引擎对未设值行为不一，复杂嵌套 JSON 可能被中途截断 → 解析失败；且无法封顶延迟/显存。
- **改动**：
  - `conf/config.py` 新增 `LLM_MAX_TOKENS` 字典（与 `LLM_TEMPERATURES` 同构），env：
    `LLM_DEFAULT_MAX_TOKENS` / `LLM_PLANNER_MAX_TOKENS` / `LLM_EXECUTOR_MAX_TOKENS` / `LLM_REFLECTOR_MAX_TOKENS` / `LLM_EXTRACTOR_MAX_TOKENS` …
  - 默认值：planner/executor/reflector 偏大（建议 `4096`，其 JSON 大）；extractor 可小（如 `1024`）。
  - `_prepare_openai_payload` 注入 `payload["max_tokens"]`；`send_message` 按 role 取值并传入。
  - 顺手把 Anthropic 的硬编码 4096 也改为读同一配置。
- **风险**：极低（纯增量）。**向后兼容**：默认值给足，云模型不受影响。
- **验证**：构造 payload 单测断言含 `max_tokens`；对本地 endpoint 发一次长输出请求确认不被截断。

### Item 2 — 接入 guided-JSON / grammar（宽松档 2a）

- **现状**：仅 `response_format:{type:json_object}`（`:187`），未利用本地引擎的约束解码。
- **改动**：
  - `conf/config.py` 新增 `LLM_JSON_MODE`（env `LLM_JSON_MODE`），取值：
    `off`（保持现状）｜ `json_object`（显式，默认）｜ `guided_json`（vLLM，走 `extra_body`）｜ `ollama`（顶层 `format:"json"`）。
  - `_prepare_openai_payload`：当 `expect_json` 且按 `LLM_JSON_MODE` 注入对应字段：
    - `guided_json` → `payload["extra_body"]["guided_json"] = <极简 schema 或 {"type":"object"}>`（vLLM/SGLang）。
    - `ollama` → `payload["format"] = "json"`。
    - `json_object`/`off` → 现行行为。
  - 注意与 item 6 的 `extra_body` 注入合并，避免互相覆盖（统一一个 `extra_body` 组装函数）。
- **风险**：低。**向后兼容**：默认 `json_object`/`off` = 现行行为。
- **验证**：各 mode 下断言 payload 结构正确；对 vLLM/Ollama 各发一次请求确认返回合法 JSON。
- **备注**：严格档 2b 留到 item 5 之后，见「关键依赖」。

### Item 6 — 剥离 `<think>` 段 + `/no_think` 支持

- **现状**：`LLM_THINKING` 仅走 `extra_body.thinking`（`:181-184`）；本地 reasoning 模型（QwQ/Qwen3）会在 content 直接吐 `<think>...</think>`，而 `_robust_json_parser` 的「首 `{` 到末 `}`」启发式会把含花括号的 think 内容吞进 JSON。
- **改动**：
  - **剥离**：在 `_clean_json_string`（`:511`）前新增一步，移除 `<think>...</think>`（含未闭合的 `<think>` 到串尾）。同时给 extractor 的 `_parse_spans`（`rag/extractor.py:163`）加同样的剥离。
  - **禁用思考**：`conf/config.py` 新增 `LLM_DISABLE_THINKING`（per-role 可选）。注入方式按后端：
    vLLM → `extra_body.chat_template_kwargs = {"enable_thinking": false}`；Ollama → 顶层 `think: false`；
    或在 prompt 末尾追加 `/no_think`（Qwen3）。统一在 `extra_body` 组装函数里处理。
- **风险**：低。剥离是纯净化；禁用思考默认 off。**注意**：剥离正则不要误伤 payload 里字面的 `<think>`（安全知识库少见，但用非贪婪 + 仅在 JSON 外层剥离）。
- **验证**：喂含 `<think>{"a":1}</think>{"real":true}` 的样例，断言解析出 `{"real":true}`。

### Item 3 — 压缩阈值下调 + 暴露上下文窗口

- **现状**：`EXECUTOR_TOKEN_COMPRESS_THRESHOLD=80000`（`conf/config.py:153`），消费点
  `core/executor.py:164`。本地模型多为 8K–32K 窗口，压缩还没触发上下文已溢出。
- **改动**：
  - 新增 `LLM_CONTEXT_WINDOW`（env，默认如 `32768`）表示模型上下文窗口。
  - `EXECUTOR_TOKEN_COMPRESS_THRESHOLD` 默认改为「派生」：未显式设置时取 `int(LLM_CONTEXT_WINDOW * 0.7)`；
    显式设置则尊重显式值（保持可覆盖）。
  - `.env.example` 增加 `LLM_CONTEXT_WINDOW` 与各档建议值注释（见下表）。
- **风险**：低，但会**改变默认压缩时机**（更早压缩）→ 归入 PR 1 但需在说明里标注。
- **验证**：不同 `LLM_CONTEXT_WINDOW` 下断言派生阈值正确；跑一个长子任务确认压缩按预期更早触发。

---

## PR 2：韧性 + Schema

### Item 4 — 解析失败优雅降级（别杀整个子任务）

- **现状**：JSON 三次重试仍失败 → `send_message` 返回 `(None, None)` → Executor `raise RuntimeError("llm_empty_response")`（`core/executor.py:266-274`），整个子任务终止。
- **改动（方案，倾向 A）**：
  - **A. 抢救 `execution_operations`**：`send_message` 失败时，除 `(None,None)` 外，把**原始字符串**透出给调用方（如新增可选返回或 metrics 里带 `raw`）。Executor 在 `if not llm_reply_json` 分支先尝试
    `_salvage_execution_operations(raw)`：用宽松正则/括号匹配抽取首个 `execution_operations` 数组或首个 `{"tool":...}`。
    - 抢救到 ≥1 个可执行 op → 用最小 reply dict 继续本步；
    - 抢救不到 → **降级为单步安全动作**（如插入一次 `think`/重新观察），而非杀子任务；连续 N 次失败才终止（复用 `EXECUTOR_FAILURE_THRESHOLD`）。
  - 参考 `rag/extractor.py:_degrade()` 的「严格不劣于」降级哲学。
- **风险**：中。改动 Executor 控制流；需保证抢救路径不把脏数据写进因果图（抢救出的 op 仍走现有校验）。
- **向后兼容**：云模型几乎不触发此路径，行为不变。
- **验证**：mock `send_message` 返回截断 JSON / 夹带散文 / 纯 `<think>`，断言子任务不被杀且产出合理降级步骤。

### Item 5 — Executor schema 瘦身（顶层 `thought` 合并为单字段）

- **现状**：`core/prompts/templates/{zh,en}/executor/output_schemas/executor_schema.jinja2` 要求一次性产出
  顶层 `thought`（**5 子字段**）+ `execution_operations` + `hypothesis_update`(6 字段) + `previous_steps_status`
  + `is_subtask_complete` + `staged_causal_nodes`。27B 模型常漏字段/格式崩。
- **改动**：
  - 把**顶层 `thought` 对象合并为单个字符串字段**（如 `"thought": "用一段话写出：关键事实→可证伪假设→工具选择理由→（若有）观察归因"`）。
    **已确认下游无人读其子字段**（见「关键发现 1」），低风险。
  - **不动** `execution_operations[].thought`（被消费）、`hypothesis_update`（被消费）、`staged_causal_nodes`（被消费）。
  - 同步更新 `executor_template.jinja2` 的「输出格式/关键要求」说明，明确哪些字段「条件可选」。
  - **zh 与 en 两份 schema 必须同步**。
  - 若做 2b 严格 grammar：以瘦身后的结构为单一数据源生成。
- **风险**：低-中（prompt 契约变化，需回归一次完整 P-E-R）。
- **验证**：跑一条完整任务，断言 Executor 步骤仍能落图、`hypothesis_update`/`staged_causal_nodes` 仍正确写入；
  对 27B 模型对比瘦身前后 JSON 合法率。

---

## 新增 / 变更配置一览

| 配置 (env) | 含义 | 建议默认 | 备注 |
|---|---|---|---|
| `LLM_*_MAX_TOKENS` | per-role 输出上限 | planner/executor/reflector `4096`，extractor `1024` | item 1 |
| `LLM_JSON_MODE` | 约束解码模式 | `json_object` | `off｜json_object｜guided_json｜ollama` (item 2) |
| `LLM_DISABLE_THINKING` | 禁用模型思考段 | `false` | per-role 可选 (item 6) |
| `LLM_CONTEXT_WINDOW` | 模型上下文窗口 | `32768` | 驱动压缩阈值派生 (item 3) |
| `EXECUTOR_TOKEN_COMPRESS_THRESHOLD` | 压缩触发阈值 | 派生 `≈0.7×窗口` | 显式设置可覆盖 (item 3) |

### 本地模型档位建议（写入 `.env.example` 注释）

| 模型档 | `LLM_CONTEXT_WINDOW` | `EXTRACTOR_RETRIEVE_K` | 温度 (planner/executor) | `LLM_JSON_MODE` |
|---|---|---|---|---|
| ~27B (Gemma-2-27B) | 8192–16384 | ~20 | 0.3 / 0.2 | guided_json / ollama |
| ~32B (Qwen2.5-32B) | 32768 | ~20 | 0.3 / 0.2 | guided_json |
| ~70B (Llama-3.3-70B / Qwen2.5-72B) | 32768+ | ~30 | 0.3 / 0.2 | guided_json |

> 混合部署（70B 跑 Planner + 32B 跑 Executor）需要 per-role `base_url`/`api_key`——
> 该项不在本计划 6 项内，作为后续 PR 3 跟进。

## 触及文件清单

- **PR 1**：`conf/config.py`、`llm/llm_client.py`、`rag/extractor.py`（仅 think 剥离）、`.env.example`
- **PR 2**：`core/executor.py`、`llm/llm_client.py`（失败透出 raw）、
  `core/prompts/templates/zh/executor/output_schemas/executor_schema.jinja2`、
  `core/prompts/templates/en/executor/output_schemas/executor_schema.jinja2`、
  `core/prompts/templates/{zh,en}/executor_template.jinja2`

## 验证策略（无保证的本地模型环境）

1. **静态/单测**：payload 组装（max_tokens / json_mode / extra_body 合并）、`<think>` 剥离、阈值派生、salvage 抽取——均可不依赖真实模型。
2. **冒烟**（需本地 endpoint，建议 Ollama 跑一个 ~27B 模型）：PR 1 各开关各发一次请求；PR 2 跑一条完整 P-E-R 任务。
3. **回归**：确保 `LLM_PROVIDER=anthropic` / `LLM_JSON_MODE=off` 时行为与改动前一致。

## 不在本计划范围（后续跟进）

- per-role `base_url`/`api_key`（混合部署路由）— PR 3
- guided-JSON 严格档 2b（完整 schema grammar）— item 5 定稿后可选
- prompt 体积裁剪（按需 include domain_knowledge）— 独立优化
