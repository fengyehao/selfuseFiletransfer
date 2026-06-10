# conf/config.py
# 配置文件：存放API密钥、模型参数等核心配置项
# 注意：请勿将真实API密钥提交到版本控制系统

import os
from dotenv import load_dotenv

# 从.env文件加载环境变量
load_dotenv()

# ============================================================================
# 核心场景配置 (Scenario Configuration)
# ============================================================================

# 运行场景模式
# "general": 通用模式，适用于实战渗透、内网渗透等复杂环境（默认）
# "ctf": CTF模式，针对CTF夺旗赛优化（会禁用部分大规模扫描工具，启用特定Prompt优化）
SCENARIO_MODE = os.getenv("SCENARIO_MODE", "general").lower()

# ============================================================================
# 输出配置 (Output Configuration)
# ============================================================================

# 控制台输出模式: "simple", "default", "debug"
# "simple": 精简输出,只展示核心信息
# "default": 标准输出,提供正常调试所需信息
# "debug": 详细输出,等同于 --verbose,提供所有调试信息
OUTPUT_MODE = os.getenv("OUTPUT_MODE", "default").lower()

# 提示词语言 / Prompt Language: "zh" (中文), "en" (English)
PROMPT_LANGUAGE = os.getenv("PROMPT_LANGUAGE", "zh").lower()

# ============================================================================
# LLM API 配置
# ============================================================================

# 从环境变量读取配置，如果环境变量不存在则使用默认值
LLM_API_BASE_URL = os.getenv("LLM_API_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY")  # 请在.env文件中设置
LLM_FALLBACK_API_KEY = os.getenv("LLM_FALLBACK_API_KEY")  # 备选API密钥，用于处理429错误


# ============================================================================
# LLM模型配置
# ============================================================================

# 为不同角色/模块定义语言模型
# 核心思想：允许为每个关键模块（规划、执行、反思）使用不同的模型，以平衡成本、速度和能力
# 例如，Planner可以使用最强大的模型来确保计划质量，而Executor可以使用速度更快、成本更低的模型
LLM_MODELS = {
    "default": os.getenv("LLM_DEFAULT_MODEL", "gpt-4o"),
    "planner": os.getenv("LLM_PLANNER_MODEL", "gpt-4o"),
    "executor": os.getenv("LLM_EXECUTOR_MODEL", "gpt-4o"),
    "reflector": os.getenv("LLM_REFLECTOR_MODEL", "gpt-4o"),
    "expert_analysis": os.getenv("LLM_EXPERT_MODEL", "gpt-4o"),
    "summarizer": os.getenv("LLM_SUMMARIZER_MODEL", os.getenv("LLM_DEFAULT_MODEL", "gpt-4o")),
    "reflector_validator": os.getenv("LLM_REFLECTOR_VALIDATOR_MODEL", os.getenv("LLM_REFLECTOR_MODEL", "gpt-4o")),
    "planner_crisis_expert": os.getenv("LLM_PLANNER_CRISIS_EXPERT_MODEL", os.getenv("LLM_PLANNER_MODEL", "gpt-4o")),
    # RAG Extractor (检索精炼器)：本地LLM做抽取式选择，确定性任务，使用默认模型即可
    "extractor": os.getenv("LLM_EXTRACTOR_MODEL", os.getenv("LLM_DEFAULT_MODEL", "gpt-4o")),
}

# 为不同角色设置独立的LLM温度参数
# 较高的温度(如0.7)增加输出多样性，适合创造性任务
# 较低的温度(如0.2)增加输出确定性，适合需要精确性的任务
LLM_TEMPERATURES = {
    "default": 0.3,
    "planner": 0.5,  # 规划器需要一定创造性来生成多样化策略
    "executor": 0.3,  # 执行器需要稳定可靠的工具调用
    "reflector": 0.2,  # 反思器需要精确的分析和判断
    "expert_analysis": 0.7,  # 专家分析需要更多创造性思维
    "summarizer": 0.2,  # 摘要需要稳定、简洁输出
    "reflector_validator": 0.1,  # 二值判定需要更高确定性
    "planner_crisis_expert": 0.4,  # 危机重规划需平衡稳定性与探索性
    "extractor": 0.0,  # 抽取式选择是确定性任务，温度取0以最大化稳定性
}

# 为不同角色配置输出 token 上限 (max_tokens)，结构与 LLM_TEMPERATURES / LLM_MODELS 同构。
# 背景：本地推理引擎对“未设 max_tokens”的默认行为不一，复杂嵌套 JSON 可能被中途截断导致解析失败；
# 显式封顶亦可控制单次调用的延迟与显存占用。
# 默认：planner/executor/reflector 等输出 JSON 较大，给足 4096；extractor 为抽取式短输出，给 1024。
LLM_MAX_TOKENS = {
    "default": int(os.getenv("LLM_DEFAULT_MAX_TOKENS", "4096")),
    "planner": int(os.getenv("LLM_PLANNER_MAX_TOKENS", os.getenv("LLM_DEFAULT_MAX_TOKENS", "4096"))),
    "executor": int(os.getenv("LLM_EXECUTOR_MAX_TOKENS", os.getenv("LLM_DEFAULT_MAX_TOKENS", "4096"))),
    "reflector": int(os.getenv("LLM_REFLECTOR_MAX_TOKENS", os.getenv("LLM_DEFAULT_MAX_TOKENS", "4096"))),
    "expert_analysis": int(os.getenv("LLM_EXPERT_MAX_TOKENS", os.getenv("LLM_DEFAULT_MAX_TOKENS", "4096"))),
    "summarizer": int(os.getenv("LLM_SUMMARIZER_MAX_TOKENS", os.getenv("LLM_DEFAULT_MAX_TOKENS", "4096"))),
    "reflector_validator": int(
        os.getenv("LLM_REFLECTOR_VALIDATOR_MAX_TOKENS", os.getenv("LLM_DEFAULT_MAX_TOKENS", "4096"))
    ),
    "planner_crisis_expert": int(
        os.getenv("LLM_PLANNER_CRISIS_EXPERT_MAX_TOKENS", os.getenv("LLM_DEFAULT_MAX_TOKENS", "4096"))
    ),
    "extractor": int(os.getenv("LLM_EXTRACTOR_MAX_TOKENS", "1024")),
}

# ============================================================================
# LLM高级配置
# ============================================================================

# 是否启用OpenAI兼容接口的extra_body字段，用于传递供应商自定义参数（如思考模式thinking）
# 设置为True后，如果下方LLM_THINKING为非off，将在请求payload中注入{"extra_body": {"thinking": "hidden|visible"}}
LLM_EXTRA_BODY_ENABLED = os.getenv("LLM_EXTRA_BODY_ENABLED", "false").lower() == "true"

# 为不同角色配置是否开启“思考模式”（extra_body.thinking）
# 取值说明：
# - off: 不启用思考模式（不注入extra_body）
# - hidden: 启用但不在最终输出中显示思考过程（由具体供应商决定具体行为）
# - visible: 启用并在最终输出中返回思考过程（例如返回reasoning_content或思维链）
# 注意：该参数仅在OpenAI兼容API且供应商支持extra_body.thinking时生效
LLM_THINKING = {
    "default": os.getenv("LLM_DEFAULT_THINKING", "off"),
    "planner": os.getenv("LLM_PLANNER_THINKING", os.getenv("LLM_DEFAULT_THINKING", "off")),
    "executor": os.getenv("LLM_EXECUTOR_THINKING", os.getenv("LLM_DEFAULT_THINKING", "off")),
    "reflector": os.getenv("LLM_REFLECTOR_THINKING", os.getenv("LLM_DEFAULT_THINKING", "off")),
    "expert_analysis": os.getenv("LLM_EXPERT_THINKING", os.getenv("LLM_DEFAULT_THINKING", "off")),
    "summarizer": os.getenv("LLM_SUMMARIZER_THINKING", os.getenv("LLM_DEFAULT_THINKING", "off")),
    "reflector_validator": os.getenv("LLM_REFLECTOR_VALIDATOR_THINKING", os.getenv("LLM_DEFAULT_THINKING", "off")),
    "planner_crisis_expert": os.getenv(
        "LLM_PLANNER_CRISIS_EXPERT_THINKING", os.getenv("LLM_DEFAULT_THINKING", "off")
    ),
    "extractor": os.getenv("LLM_EXTRACTOR_THINKING", os.getenv("LLM_DEFAULT_THINKING", "off")),
}

# 约束解码 / JSON 输出模式（本地引擎适配，宽松档 2a：仅强制“输出为合法 JSON 对象”）。取值：
# - off:         不注入任何 JSON 约束（适配不支持 response_format 的后端，依赖 prompt + 健壮解析器）
# - json_object: 注入 response_format={"type":"json_object"}（默认，等价于改动前行为）
# - guided_json: vLLM/SGLang 约束解码，注入 extra_body.guided_json={"type":"object"}
# - ollama:      注入顶层 format="json"
# 仅在调用方 expect_json=True 时生效。
LLM_JSON_MODE = os.getenv("LLM_JSON_MODE", "json_object").lower()

# 为不同角色配置是否禁用模型“思考段”（reasoning 模型如 QwQ/Qwen3 会在 content 直接吐 <think>...</think>）。
# 注入方式按后端：vLLM/SGLang → extra_body.chat_template_kwargs={"enable_thinking": false}；Ollama → 顶层 think=false。
# 默认 false（不禁用）。与解析侧的 <think> 段剥离相互独立、互补（一个从源头关闭，一个事后净化）。
LLM_DISABLE_THINKING = {
    "default": os.getenv("LLM_DEFAULT_DISABLE_THINKING", "false").lower() == "true",
    "planner": os.getenv(
        "LLM_PLANNER_DISABLE_THINKING", os.getenv("LLM_DEFAULT_DISABLE_THINKING", "false")
    ).lower() == "true",
    "executor": os.getenv(
        "LLM_EXECUTOR_DISABLE_THINKING", os.getenv("LLM_DEFAULT_DISABLE_THINKING", "false")
    ).lower() == "true",
    "reflector": os.getenv(
        "LLM_REFLECTOR_DISABLE_THINKING", os.getenv("LLM_DEFAULT_DISABLE_THINKING", "false")
    ).lower() == "true",
    "expert_analysis": os.getenv(
        "LLM_EXPERT_DISABLE_THINKING", os.getenv("LLM_DEFAULT_DISABLE_THINKING", "false")
    ).lower() == "true",
    "summarizer": os.getenv(
        "LLM_SUMMARIZER_DISABLE_THINKING", os.getenv("LLM_DEFAULT_DISABLE_THINKING", "false")
    ).lower() == "true",
    "reflector_validator": os.getenv(
        "LLM_REFLECTOR_VALIDATOR_DISABLE_THINKING", os.getenv("LLM_DEFAULT_DISABLE_THINKING", "false")
    ).lower() == "true",
    "planner_crisis_expert": os.getenv(
        "LLM_PLANNER_CRISIS_EXPERT_DISABLE_THINKING", os.getenv("LLM_DEFAULT_DISABLE_THINKING", "false")
    ).lower() == "true",
    "extractor": os.getenv(
        "LLM_EXTRACTOR_DISABLE_THINKING", os.getenv("LLM_DEFAULT_DISABLE_THINKING", "false")
    ).lower() == "true",
}

# 模型上下文窗口 (token)，用于派生执行器压缩阈值。本地模型多为 8K–32K 窗口。
#
# ── 按窗口大小调压缩触发的档位建议 ───────────────────────────────────────────────
# 只设 LLM_CONTEXT_WINDOW 不够：策略3（token）阈值会按它自动派生(0.7×)，但策略1/2 是
# “消息条数”触发、与窗口无关，必须随窗口一起手动抬高，否则大窗口也会在十几条消息就被
# 压缩（见 core/executor.py 的 _compress_context_if_needed 三个 OR 触发条件）。
#
#  档位   LLM_CONTEXT_   →派生token   MESSAGE_COMPRESS_   COMPRESS_INTERVAL_   RECENT_MESSAGES_
#         WINDOW         阈值(0.7×)   THRESHOLD           MSG_THRESHOLD        KEEP
#  8k       8192          ≈5734            8                   6                  4
#  16k     16384         ≈11468           12                   8                  6
#  32k     32768         ≈22937           16                  12                  6
#  64k     65536         ≈45875           24                  16                  8
#  128k   131072         ≈91750           32                  20                 10
#
# 当前代码默认已对齐上表 32k 档：window=32768，消息阈值 16/12/6
# （MESSAGE_COMPRESS_THRESHOLD / COMPRESS_INTERVAL_MSG_THRESHOLD / RECENT_MESSAGES_KEEP）。
# 提醒：本地 27B–70B 模型在标称窗口远低处，指令跟随/JSON 合法率就开始下降（见
#       docs/local-model-optimization-plan.md）。消息条数阈值同时是质量护栏，建议温和上调
#       并对真实模型观察 JSON 合法率，别直接把窗口顶满。token 阈值无需手设，设好窗口即自动派生。
# ────────────────────────────────────────────────────────────────────────────────
LLM_CONTEXT_WINDOW = int(os.getenv("LLM_CONTEXT_WINDOW", "32768"))

# ============================================================================
# LLM提供商配置
# ============================================================================

# LLM Provider: "openai"或"anthropic"
# 支持使用不同的LLM提供商，可根据需求切换
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")

# Anthropic API配置（如使用Claude模型）
ANTHROPIC_API_BASE_URL = os.getenv("ANTHROPIC_API_BASE_URL", "https://api.anthropic.com/v1/messages")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", LLM_API_KEY)  # 默认使用主API密钥
ANTHROPIC_FALLBACK_API_KEY = os.getenv("ANTHROPIC_FALLBACK_API_KEY", LLM_FALLBACK_API_KEY)
ANTHROPIC_VERSION = os.getenv("ANTHROPIC_VERSION", "2023-06-01")

# Anthropic模型映射
ANTHROPIC_MODELS = {
    "default": os.getenv("ANTHROPIC_DEFAULT_MODEL", "claude-3-5-sonnet-20240620"),
    "planner": os.getenv("ANTHROPIC_PLANNER_MODEL", "claude-3-5-sonnet-20240620"),
    "executor": os.getenv("ANTHROPIC_EXECUTOR_MODEL", "claude-3-5-sonnet-20240620"),
    "reflector": os.getenv("ANTHROPIC_REFLECTOR_MODEL", "claude-3-5-sonnet-20240620"),
    "expert_analysis": os.getenv("ANTHROPIC_EXPERT_MODEL", "claude-3-5-sonnet-20240620"),
    "summarizer": os.getenv("ANTHROPIC_SUMMARIZER_MODEL", os.getenv("ANTHROPIC_DEFAULT_MODEL", "claude-3-5-sonnet-20240620")),
    "reflector_validator": os.getenv(
        "ANTHROPIC_REFLECTOR_VALIDATOR_MODEL",
        os.getenv("ANTHROPIC_REFLECTOR_MODEL", "claude-3-5-sonnet-20240620"),
    ),
    "planner_crisis_expert": os.getenv(
        "ANTHROPIC_PLANNER_CRISIS_EXPERT_MODEL",
        os.getenv("ANTHROPIC_PLANNER_MODEL", "claude-3-5-sonnet-20240620"),
    ),
    "extractor": os.getenv(
        "ANTHROPIC_EXTRACTOR_MODEL",
        os.getenv("ANTHROPIC_DEFAULT_MODEL", "claude-3-5-sonnet-20240620"),
    ),
}

# ============================================================================
# 执行器行为配置
# ============================================================================

# 执行器最大步数限制
EXECUTOR_MAX_STEPS = int(os.getenv("EXECUTOR_MAX_STEPS", "8"))

# 消息历史压缩阈值
EXECUTOR_MESSAGE_COMPRESS_THRESHOLD = int(os.getenv("EXECUTOR_MESSAGE_COMPRESS_THRESHOLD", "16"))

# Token数量压缩阈值
# 未显式设置 EXECUTOR_TOKEN_COMPRESS_THRESHOLD 时，按 LLM_CONTEXT_WINDOW 的 70% 派生
# （更早触发压缩，避免本地小窗口模型在压缩触发前即上下文溢出）；显式设置则尊重显式值（可覆盖）。
def _derive_token_compress_threshold(explicit: str | None, context_window: int) -> int:
    if explicit is not None:
        return int(explicit)
    return int(context_window * 0.7)


EXECUTOR_TOKEN_COMPRESS_THRESHOLD = _derive_token_compress_threshold(
    os.getenv("EXECUTOR_TOKEN_COMPRESS_THRESHOLD"), LLM_CONTEXT_WINDOW
)

# 无新产出物的耐心值（连续多少步无产出则终止）
# 必须小于 EXECUTOR_MAX_STEPS，否则该机制永远不会触发
EXECUTOR_NO_ARTIFACTS_PATIENCE = int(os.getenv("EXECUTOR_NO_ARTIFACTS_PATIENCE", "4"))

# 失败阈值（连续失败多少次触发策略切换）
EXECUTOR_FAILURE_THRESHOLD = int(os.getenv("EXECUTOR_FAILURE_THRESHOLD", "3"))

# 上下文压缩时保留的最近消息数
EXECUTOR_RECENT_MESSAGES_KEEP = int(os.getenv("EXECUTOR_RECENT_MESSAGES_KEEP", "6"))

# 最小压缩消息阈值
EXECUTOR_MIN_COMPRESS_MESSAGES = int(os.getenv("EXECUTOR_MIN_COMPRESS_MESSAGES", "5"))

# 执行轮次压缩间隔
EXECUTOR_COMPRESS_INTERVAL = int(os.getenv("EXECUTOR_COMPRESS_INTERVAL", "5"))

# 执行轮次压缩时的消息数阈值
EXECUTOR_COMPRESS_INTERVAL_MSG_THRESHOLD = int(os.getenv("EXECUTOR_COMPRESS_INTERVAL_MSG_THRESHOLD", "12"))

# 工具调用超时时间（秒）
EXECUTOR_TOOL_TIMEOUT = int(os.getenv("EXECUTOR_TOOL_TIMEOUT", "120"))

# 各工具独立超时映射（秒）。未列出的工具使用 EXECUTOR_TOOL_TIMEOUT 默认值。
# 重型扫描工具需要更长超时；轻量思考/检索工具应快速返回。
TOOL_TIMEOUTS: dict = {
    # 重型扫描工具
    "sqlmap_tool":       int(os.getenv("TOOL_TIMEOUT_SQLMAP",    "600")),
    "nuclei_scan":       int(os.getenv("TOOL_TIMEOUT_NUCLEI",    "300")),
    "dirsearch_scan":    int(os.getenv("TOOL_TIMEOUT_DIRSEARCH", "300")),
    "concurrency_test":  int(os.getenv("TOOL_TIMEOUT_CONCURRENCY", "180")),
    # 标准工具
    "http_request":      int(os.getenv("TOOL_TIMEOUT_HTTP",       "60")),
    "shell_exec":        int(os.getenv("TOOL_TIMEOUT_SHELL",      "120")),
    "python_exec":       int(os.getenv("TOOL_TIMEOUT_PYTHON",     "300")),
    "web_search":        int(os.getenv("TOOL_TIMEOUT_WEB_SEARCH", "30")),
    "search_exploit":    int(os.getenv("TOOL_TIMEOUT_SEARCH_EXPLOIT", "30")),
    # 轻量工具
    "think":                  int(os.getenv("TOOL_TIMEOUT_THINK",        "30")),
    "formulate_hypotheses":   int(os.getenv("TOOL_TIMEOUT_HYPOTHESES",   "30")),
    "reflect_on_failure":     int(os.getenv("TOOL_TIMEOUT_REFLECT",      "30")),
    "expert_analysis":        int(os.getenv("TOOL_TIMEOUT_EXPERT",       "60")),
    "retrieve_knowledge":     int(os.getenv("TOOL_TIMEOUT_RETRIEVE",     "150")),
    "distill_knowledge":      int(os.getenv("TOOL_TIMEOUT_DISTILL",      "20")),
}

# 执行器观察结果的最大长度（字符），超过此长度将被截断
EXECUTOR_MAX_OUTPUT_LENGTH = int(os.getenv("EXECUTOR_MAX_OUTPUT_LENGTH", "50000"))

# P-E-R 全局循环最大次数（防止无限循环）
GLOBAL_MAX_CYCLES = int(os.getenv("GLOBAL_MAX_CYCLES", "50"))

# 全局最大 Token 消耗限制（安全熔断器）
GLOBAL_MAX_TOKEN_USAGE = int(os.getenv("GLOBAL_MAX_TOKEN_USAGE", "5000000"))

# ============================================================================
# 上下文管理配置
# ============================================================================

# 规划历史保留窗口大小
PLANNER_HISTORY_WINDOW = int(os.getenv("PLANNER_HISTORY_WINDOW", "15"))

# 反思日志保留窗口大小
REFLECTOR_HISTORY_WINDOW = int(os.getenv("REFLECTOR_HISTORY_WINDOW", "15"))

# ============================================================================
# Ablation Configuration
# ============================================================================

# Execution mode
# "default": Standard P-E-R mode
# "linear": Linear mode (No Task Graph), disable dynamic branching
# "react": Pure ReAct mode (Executor Only), disable Planner/Reflector
EXECUTION_MODE = os.getenv("EXECUTION_MODE", "default").lower()

# Whether to disable causal graph
# "true": Disable Reflector's causal updates and Planner's causal reasoning
# "false": Default, use dual-graph
NO_CAUSAL_GRAPH = os.getenv("NO_CAUSAL_GRAPH", "false").lower() == "true"

# ============================================================================
# Web 服务配置
# ============================================================================

# Web UI 服务主机地址
WEB_HOST = os.getenv("WEB_HOST", "127.0.0.1")

# Web UI 服务端口
WEB_PORT = int(os.getenv("WEB_PORT", "8088"))

# ============================================================================
# 知识服务配置
# ============================================================================

# 知识服务端口
KNOWLEDGE_SERVICE_PORT = int(os.getenv("KNOWLEDGE_SERVICE_PORT", "8081"))

# 知识服务 Host
KNOWLEDGE_SERVICE_HOST = os.getenv("KNOWLEDGE_SERVICE_HOST", "127.0.0.1")

# 知识服务 URL
KNOWLEDGE_SERVICE_URL = os.getenv("KNOWLEDGE_SERVICE_URL", f"http://{KNOWLEDGE_SERVICE_HOST}:{KNOWLEDGE_SERVICE_PORT}")

# ============================================================================
# RAG Extractor 配置 (检索精炼器 / ADR-0002)
# ============================================================================
# 所有旋钮均为中档默认值，便于按本地模型规模仅靠调参适配（无需改代码）。

# 候选检索数量：Extractor 先取较大候选池再抽取式精炼
# 建议按本地模型规模调整：~10 (8B) · ~20 (32B) · ~30 (70B)
EXTRACTOR_RETRIEVE_K = int(os.getenv("EXTRACTOR_RETRIEVE_K", "30"))

# 输出预算（字符数）：软目标，仅在 span 边界处生效，绝不在 span 中间截断。
# 完整性优先于预算——单个最相关 span 即使超预算也整段返回（ADR-0002 §2）。
EXTRACTOR_OUTPUT_BUDGET_CHARS = int(os.getenv("EXTRACTOR_OUTPUT_BUDGET_CHARS", "1500"))

# Extractor LLM 调用超时（秒）。超时则优雅降级到原始 top-k 片段。
EXTRACTOR_TIMEOUT = int(os.getenv("EXTRACTOR_TIMEOUT", "120"))

# 降级回退时返回的原始片段数（LLM 输出畸形/为空/超时时，strictly no worse than 精炼前行为）
EXTRACTOR_FALLBACK_TOP_K = int(os.getenv("EXTRACTOR_FALLBACK_TOP_K", "5"))

# ============================================================================
# 人工协同 (HITL) 配置
# ============================================================================

# 是否开启人工介入模式
# 开启后，Agent在生成规划后会暂停，等待Web UI或CLI的人工审批
HUMAN_IN_THE_LOOP = os.getenv("HUMAN_IN_THE_LOOP", "false").lower() == "true"
