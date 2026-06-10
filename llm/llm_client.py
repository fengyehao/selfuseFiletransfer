# llm_client.py
# LLM推理API封装，支持Qwen系列OpenAI兼容API
# 依赖 openai>=1.0.0

import sys
import asyncio

from typing import Any, Dict, List, Optional
import httpx
import json
import re
from rich.panel import Panel
from conf.config import (
    LLM_PROVIDER,
    LLM_API_BASE_URL,
    LLM_API_KEY,
    LLM_FALLBACK_API_KEY,
    LLM_MODELS,
    LLM_TEMPERATURES,
    LLM_MAX_TOKENS,
    LLM_EXTRA_BODY_ENABLED,
    LLM_THINKING,
    LLM_JSON_MODE,
    LLM_DISABLE_THINKING,
    ANTHROPIC_API_BASE_URL,
    ANTHROPIC_API_KEY,
    ANTHROPIC_FALLBACK_API_KEY,
    ANTHROPIC_MODELS,
    ANTHROPIC_VERSION,
)

# 导入事件代理
try:
    from core.events import broker
except ImportError:
    broker = None


# 本地 reasoning 模型（QwQ/Qwen3 等）会在 content 直接吐出 <think>...</think> 段，
# 其中的花括号会被 JSON 解析的“首 { 到末 }”启发式吞入。下面两个正则用于在解析前剥离这些段。
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_DANGLING_RE = re.compile(r"<think>.*$", re.DOTALL | re.IGNORECASE)


def strip_think_blocks(text: str) -> str:
    """移除 reasoning 模型吐出的 <think>...</think> 段，用于 JSON 解析前的净化。

    先以非贪婪方式移除成对的 <think>...</think>（避免吞掉其后的真实内容），再移除未闭合的
    <think>（被截断的思考）到串尾。注意：对 payload 中字面出现且未闭合的 <think> 会误伤，但
    安全知识库中此种情况罕见（见 docs/local-model-optimization-plan.md item 6）。
    """
    if not isinstance(text, str):
        return text
    text = _THINK_BLOCK_RE.sub("", text)
    return _THINK_DANGLING_RE.sub("", text)


class LLMClient:
    def __init__(self, op_id: Optional[str] = None):
        self.provider = LLM_PROVIDER
        self.op_id = op_id  # 用于事件发送

        # Token prices in CNY per 1000 tokens
        # 使用通用预估价格，避免硬编码特定模型。用户应根据实际模型调整。
        self.model_token_prices_cny = {
            "default": {"input": 0.004, "output": 0.016},  # Default placeholder prices
            "gpt-4o": {"input": 0.0175, "output": 0.07}, # ~$2.5/$10 per M
            "deepseek-v3.2": {"input": 0.001, "output": 0.004}, # Approx.
            "deepseek-chat": {"input": 0.001, "output": 0.002}, # Approx.
            "deepseek-reasoner": {"input": 0.004, "output": 0.016}, # Approx.
        }
        # self.cny_to_usd_rate = 7.0 # No longer needed as all costs will be in CNY.
        if self.provider == "anthropic":
            self.api_url = ANTHROPIC_API_BASE_URL
            self.api_key = ANTHROPIC_API_KEY
            self.fallback_api_key = ANTHROPIC_FALLBACK_API_KEY
            self.models = ANTHROPIC_MODELS
            self.anthropic_version = ANTHROPIC_VERSION
            # NOTE: Using placeholder costs for Anthropic models. Replace with actuals.
            self.prompt_token_cost = 0.000003  # $3 per million tokens
            self.completion_token_cost = 0.000015  # $15 per million tokens
        else:  # Default to openai
            self.api_url = LLM_API_BASE_URL
            self.api_key = LLM_API_KEY
            self.fallback_api_key = LLM_FALLBACK_API_KEY
            self.models = LLM_MODELS
            # NOTE: Using placeholder costs for OpenAI-compatible models. Replace with actuals.
            self.prompt_token_cost = 0.000001  # $1 per million tokens
            self.completion_token_cost = 0.000002  # $2 per million tokens

        self.temperatures = LLM_TEMPERATURES
        self.max_tokens = LLM_MAX_TOKENS
        self.client = httpx.AsyncClient()
        self.console = None  # Will initialize console when needed
        self.reset_metrics()

    def _get_console(self):
        """Lazy initialization of console to avoid circular imports."""
        if self.console is None:
            from core.console import console_proxy
            self.console = console_proxy
        return self.console

    def reset_metrics(self):
        """Resets the metrics counters."""
        self.api_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.estimated_cost = 0.0

    def get_and_reset_metrics(self) -> Dict[str, Any]:
        """Retrieves the current metrics and resets the counters."""
        metrics = {
            "api_calls": self.api_calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "estimated_cost_cny": self.estimated_cost,
        }
        self.reset_metrics()
        return metrics

    def _update_metrics(self, usage: Dict[str, Any], model_name: str) -> Dict[str, Any]:
        """Updates metrics from the API response's usage object and returns per-call metrics."""
        if not usage:
            return {}

        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        # Get model-specific costs, default to generic if not found
        model_prices = self.model_token_prices_cny.get(model_name, self.model_token_prices_cny["default"])

        prompt_cost_cny_per_token = model_prices["input"] / 1000
        completion_cost_cny_per_token = model_prices["output"] / 1000

        cost_cny = (prompt_tokens * prompt_cost_cny_per_token) + (completion_tokens * completion_cost_cny_per_token)

        # Update cumulative instance metrics
        self.api_calls += 1
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.estimated_cost += cost_cny

        # Return metrics for this specific call
        return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "cost_cny": cost_cny}

    def _prepare_anthropic_payload(self, current_messages: list, model_name: str, max_tokens: int) -> tuple[dict, dict]:
        """
        为Anthropic API准备请求载荷。

        Returns:
            tuple: (headers, payload)
        """
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.anthropic_version,
            "content-type": "application/json",
        }

        system_message = ""
        other_messages = []
        for msg in current_messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                other_messages.append(msg)

        if not other_messages:
            other_messages.append({"role": "user", "content": system_message})
            system_message = ""
        elif other_messages[0]["role"] != "user":
            other_messages.insert(0, {"role": "user", "content": "Continue."})

        payload = {
            "model": model_name,
            "max_tokens": max_tokens,
            "stream": False,
            "messages": other_messages,
        }
        if system_message:
            payload["system"] = system_message

        return headers, payload

    def _role_disables_thinking(self, role: str) -> bool:
        """该角色是否禁用模型思考段（LLM_DISABLE_THINKING）。"""
        return LLM_DISABLE_THINKING.get(role, LLM_DISABLE_THINKING.get("default", False))

    def _build_extra_body(self, role: str, expect_json: bool, json_mode: str) -> dict:
        """组装 OpenAI 兼容接口的 extra_body（合并思考模式 / guided_json / 禁用思考）。

        统一在此处装配，避免 item 2（guided_json）与 item 6（禁用思考）互相覆盖。
        返回空 dict 表示无需注入 extra_body。
        """
        extra_body: dict = {}

        # 思考模式 (extra_body.thinking)：仅在显式启用 extra_body 且角色配置为 hidden/visible 时注入（沿用原行为）
        if LLM_EXTRA_BODY_ENABLED:
            thinking_mode = LLM_THINKING.get(role, LLM_THINKING.get("default", "off")).lower()
            if thinking_mode in ("hidden", "visible"):
                extra_body["thinking"] = thinking_mode

        # guided-JSON（宽松档 2a）：仅约束“输出为合法 JSON 对象”，与具体字段无关（vLLM/SGLang）
        if expect_json and json_mode == "guided_json":
            extra_body["guided_json"] = {"type": "object"}

        # 禁用模型思考段（vLLM/SGLang 风格）：Ollama 走顶层 think 字段，故此处排除 ollama
        if self._role_disables_thinking(role) and json_mode != "ollama":
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}

        return extra_body

    def _prepare_openai_payload(
        self, current_messages: list, model_name: str, temperature: float, role: str, expect_json: bool, max_tokens: int
    ) -> tuple[dict, dict]:
        """
        为OpenAI API准备请求载荷。

        Returns:
            tuple: (headers, payload)
        """
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        # 部分本地引擎（LM Studio 等）的 jinja 聊天模板在消息列表无 user 角色时直接报
        # "No user query found in messages."；云端 API 容忍 system-only。对齐 Anthropic 路径
        # （_prepare_anthropic_payload 的 :169-171 护栏），确保至少有一条 user 消息再发出。
        if not any(m.get("role") == "user" for m in current_messages):
            kickoff = "请根据以上系统指令开始本步执行" + ("，并仅输出符合要求的合法 JSON。" if expect_json else "。")
            current_messages = current_messages + [{"role": "user", "content": kickoff}]

        payload = {
            "model": model_name,
            "messages": current_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        json_mode = LLM_JSON_MODE.lower()

        # 约束解码 / JSON 输出（按后端选择顶层注入方式；guided_json 走 extra_body）
        if expect_json:
            if json_mode == "ollama":
                payload["format"] = "json"
            elif json_mode in ("guided_json", "off"):
                pass  # guided_json → extra_body（见 _build_extra_body）；off → 不注入约束
            else:  # "json_object"（默认）及未知值，等价于改动前行为
                payload["response_format"] = {"type": "json_object"}

        # 禁用思考：Ollama 走顶层 think 字段（vLLM/SGLang 走 extra_body.chat_template_kwargs）
        if json_mode == "ollama" and self._role_disables_thinking(role):
            payload["think"] = False

        extra_body = self._build_extra_body(role, expect_json, json_mode)
        if extra_body:
            payload["extra_body"] = extra_body

        return headers, payload

    async def _extract_response_content(self, api_response_json: dict, model_name: str) -> tuple[str, dict]:
        """
        从LLM API响应中提取内容和指标。

        Returns:
            tuple: (content_string, call_metrics)
        """
        content_string = ""
        usage_data = api_response_json.get("usage")
        call_metrics = None

        if usage_data:
            call_metrics = self._update_metrics(usage_data, model_name)

        if self.provider == "anthropic":
            content_string = api_response_json["content"][0]["text"]
        else:  # OpenAI
            content_string = api_response_json["choices"][0]["message"]["content"]

        return content_string, call_metrics

    async def _handle_network_error(
        self, e: Exception, network_retries: int, max_retries: int
    ) -> tuple[bool, int]:
        """
        处理网络错误。

        Returns:
            tuple: (should_continue, new_retry_count)
        """
        network_retries += 1
        if network_retries > max_retries:
            self._get_console().print(f"[bold red]网络连接失败，已达最大重试次数: {e}[/bold red]")
            raise e

        wait_time = 5 * network_retries
        self._get_console().print(
            f"[bold yellow]网络连接错误，正在等待 {wait_time} 秒后重试... ({network_retries}/{max_retries})[/bold yellow]"
        )
        await asyncio.sleep(wait_time)
        return True, network_retries

    async def _handle_rate_limit_error(
        self,
        e: Exception,
        current_api_key: str,
        fallback_used: bool,
        api_call_retries: int,
        max_retries: int,
    ) -> tuple[bool, str, bool, int]:
        """
        处理API速率限制错误。

        Returns:
            tuple: (should_continue, new_api_key, new_fallback_used, new_retry_count)
        """
        # 尝试切换到备用key
        if not fallback_used and self.fallback_api_key is not None and self.fallback_api_key != "":
            current_api_key = self.fallback_api_key
            fallback_used = True
            self._get_console().print(
                "[bold yellow]429 error detected, using fallback API key for this request...[/bold yellow]"
            )
            return True, current_api_key, fallback_used, api_call_retries

        # 进行指数退避重试
        api_call_retries += 1
        if api_call_retries > max_retries:
            self._get_console().print("[bold red]API rate limit exceeded. Max retries reached.[/bold red]")
            raise e
        
        # 指数退避: 10s, 20s, 40s, 80s...
        backoff_time = 10 * (2 ** (api_call_retries - 1))
        # 设置上限
        backoff_time = min(backoff_time, 120)
        
        self._get_console().print(
            f"[bold yellow]Rate limit hit. Waiting {backoff_time}s to retry... ({api_call_retries}/{max_retries})[/bold yellow]"
        )
        await asyncio.sleep(backoff_time)
        return True, current_api_key, fallback_used, api_call_retries

    async def send_message(
        self, messages: List[Dict[str, Any]], role: str = "default", expect_json: bool = True
    ) -> tuple[Dict | str | None, Dict | None]:
        """
        通过HTTP POST方式异步向LLM发送多轮消息，并返回解析后的内容和本次调用的指标。
        增加了JSON解析失败时的重试逻辑，并要求LLM修正格式。
        messages: [{"role": "system/user/assistant", "content": "..."}]
        role: 调用者的角色 (e.g., "planner", "executor")，用于选择合适的模型和温度参数
        Returns: A tuple containing:
                 - The parsed dictionary or raw string.
                 - A dictionary with the metrics for this specific call (tokens, cost).
                 Returns (None, None) if all retries fail. 例外：当 expect_json 且 JSON 重试
                 耗尽时，返回 (None, {"raw": <最后一次原始内容>})，以便调用方（Executor）尝试
                 从原文抢救可执行操作（item 4 韧性，见 docs/local-model-optimization-plan.md）。
        """
        model_name = self.models.get(role) or self.models.get("default")
        temperature = self.temperatures.get(role, self.temperatures.get("default", 0.2))
        max_tokens = self.max_tokens.get(role, self.max_tokens.get("default", 4096))

        json_parsing_retries = 0
        MAX_JSON_PARSE_RETRIES = 3  # 允许2次重试 (总共3次尝试)
        last_raw_content: Optional[str] = None  # 最后一次模型原始内容，供解析失败时透出抢救

        api_call_retries = 0
        MAX_API_CALL_RETRIES = 10  # TPM limit retries
        network_retries = 0
        MAX_NETWORK_RETRIES = 3  # Network error retries

        current_messages = list(messages)  # 复制消息列表，以便在重试时修改

        # 修复点：让密钥选择在本次方法的所有重试中保持状态
        current_api_key = self.api_key
        fallback_used_in_this_request = False

        # 发送 LLM 请求事件
        if broker and self.op_id:
            try:
                await broker.emit(
                    "llm.request",
                    {
                        "messages": current_messages,
                        "role": role,
                        "model": model_name,
                        "timestamp": asyncio.get_event_loop().time(),
                    },
                    op_id=self.op_id,
                )
            except Exception:
                pass

        while json_parsing_retries <= MAX_JSON_PARSE_RETRIES:
            try:
                # 准备API请求
                if self.provider == "anthropic":
                    headers, payload = self._prepare_anthropic_payload(current_messages, model_name, max_tokens)
                else:  # OpenAI
                    headers, payload = self._prepare_openai_payload(
                        current_messages, model_name, temperature, role, expect_json, max_tokens
                    )

                # 更新headers中的API key
                if self.provider == "anthropic":
                    headers["x-api-key"] = current_api_key
                else:
                    headers["Authorization"] = f"Bearer {current_api_key}"

                # 发送请求
                response = await self.client.post(self.api_url, headers=headers, json=payload, timeout=1200.0)

                if response.status_code != 200:
                    raise Exception(f"LLM API请求失败: {response.status_code} {response.text}")

                # 提取响应内容
                api_response_json = json.loads(response.text)
                content_string, call_metrics = await self._extract_response_content(api_response_json, model_name)
                last_raw_content = content_string  # 记录原始内容，JSON 解析失败耗尽重试时透出供抢救

                # 发送响应事件
                if broker and self.op_id:
                    try:
                        await broker.emit(
                            "llm.response",
                            {
                                "content": content_string,
                                "role": role,
                                "model": model_name,
                                "metrics": call_metrics,
                                "timestamp": asyncio.get_event_loop().time(),
                            },
                            op_id=self.op_id,
                        )
                    except Exception:
                        pass

                # 处理JSON解析
                if expect_json:
                    final_json = self._robust_json_parser(content_string)
                    if final_json is not None:
                        return final_json, call_metrics
                    else:
                        raise json.JSONDecodeError("LLM content is not valid JSON.", content_string, 0)
                else:
                    return content_string, call_metrics

            except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError) as e:
                should_continue, network_retries = await self._handle_network_error(e, network_retries, MAX_NETWORK_RETRIES)
                if should_continue:
                    continue

            except json.JSONDecodeError:
                json_parsing_retries += 1
                if json_parsing_retries > MAX_JSON_PARSE_RETRIES:
                    self._get_console().print("[bold red]JSON解析最终失败，已达最大重试次数。[/bold red]")
                    # item 4（韧性）：把最后一次原始内容经 metrics 透出（按调用返回，并发安全），
                    # 让 Executor 尝试抢救 execution_operations，而非直接杀掉子任务。
                    # update_*_metrics 只读 token 字段，附带 "raw" 不影响既有计量行为。
                    if last_raw_content:
                        return None, {"raw": last_raw_content}
                    return None, None

                self._get_console().print(
                    f"[bold yellow]警告：LLM返回的JSON格式无效，正在尝试第 {json_parsing_retries}/{MAX_JSON_PARSE_RETRIES} 次重试...[/bold yellow]"
                )
                current_messages.append(
                    {
                        "role": "user",
                        "content": "Your previous response was not valid JSON. Please correct the format and provide the full response again, ensuring it is a single, valid JSON object.",
                    }
                )
                await asyncio.sleep(1)

            except Exception as e:
                if "429" in str(e):
                    should_continue, current_api_key, fallback_used_in_this_request, api_call_retries = (
                        await self._handle_rate_limit_error(
                            e, current_api_key, fallback_used_in_this_request, api_call_retries, MAX_API_CALL_RETRIES
                        )
                    )
                    if should_continue:
                        continue
                elif "tpm rate limit exceeded" in str(e):
                    api_call_retries += 1
                    if api_call_retries > MAX_API_CALL_RETRIES:
                        self.console.print("[bold red]API rate limit exceeded. Max retries reached.[/bold red]")
                        raise e
                    self.console.print(
                        f"[bold yellow]API rate limit exceeded. Waiting 10 seconds to retry... ({api_call_retries}/{MAX_API_CALL_RETRIES})[/bold yellow]"
                    )
                    await asyncio.sleep(10)
                    continue

                self._get_console().print(f"[bold red]发送消息时发生未预期的错误: {e}[/bold red]")
                raise e
        return None, None

    def _generate_preservation_aware_compression_prompt(self, history_to_compress: List[Dict[str, str]]) -> str:
        """
        生成保护探索性思维的压缩提示词，用于LLM总结对话历史。
        Args:
            history_to_compress: 要压缩的对话历史片段。
        Returns:
            用于压缩的系统提示词。
        """
        history_content = ""
        for i, msg in enumerate(history_to_compress):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            history_content += f"\n[消息{i + 1}] {role}:\n{content}\n" + "-" * 50

        return f"""你是LuaN1ao渗透测试智能体的记忆管理专家。请将以下对话历史压缩为简洁的自然语言摘要，用于后续测试决策参考。

## 压缩要求

### 📋 必须保留的关键信息
1. **安全发现** - 所有漏洞、异常响应、错误信息必须详细记录
2. **技术细节** - 目标系统的技术栈、版本、配置信息
3. **攻击面** - 已发现的输入点、参数、端点
4. **测试进展** - 当前测试阶段、已完成的测试类型
5. **有效载荷** - 成功或有价值的攻击payload
6. **探索性思维** - 包含"试试"、"可能"、"简单"、"直接"等探索性表述的思维过程
7. **直觉判断** - 基于经验的快速判断和假设
8. **策略转换节点** - 决定改变测试方向的关键思考

### 🔄 可以适度压缩的内容
1. **重复操作** - 多次相同的工具调用（保留结果即可）
2. **冗长的理论分析** - 保留核心结论
3. **正常的业务功能验证** - 简化描述过程

### 🧠 特别保护原则
- **必须保留原始Payload字符串**：即使失败，也要保留如 `'--`、`OR 1=1`、`SLEEP(5)` 等具体Payload。
- 保持测试思维的"火花"和"直觉"
- 不要过度理性化，保留一些看似"不成熟"但可能有价值的想法
- 特别保护那些提到"简单方法"、"基础绕过"的内容
- 保留探索性的"也许应该试试..."类型的思考

## 📝 输出格式

请用自然语言写成一个连贯的测试报告，包含以下部分：

**测试目标与进展**
- 当前测试的目标系统概况
- 测试进展到的阶段（发现/枚举/漏洞测试/利用）

**关键安全发现**
- 按重要性排序的安全发现
- 每个发现包括：描述、证据、潜在影响

**技术环境分析**
- 目标系统的技术栈和架构
- 已识别的攻击面和入口点

**测试策略与调整**
- 已尝试的测试方法及效果
- 根据发现调整的测试策略

**后续建议**
- 基于当前发现的后续测试方向
- 需要重点关注的潜在风险点

## 📊 待压缩的对话历史：

{history_content}

请将上述历史压缩为一份简洁但完整的测试进展报告，确保所有关键安全信息和探索性思维都得到妥善保留。报告应该让另一个智能体能够基于这份摘要继续进行有效的渗透测试。"""

    async def summarize_conversation(self, messages_to_summarize: List[Dict[str, Any]]) -> tuple[str, Optional[Dict]]:
        """
        使用LLM总结一段对话，提取关键信息，并遵循保护探索性思维的原则。
        Args:
            messages_to_summarize: 要总结的对话历史片段。
        Returns:
            tuple[str, Optional[Dict]]: 对话的简洁摘要和调用指标。
        """
        compression_prompt_content = self._generate_preservation_aware_compression_prompt(messages_to_summarize)

        summarization_messages = [{"role": "user", "content": compression_prompt_content}]
        # Use a specific role for summarization to potentially use a different model/temperature
        summary, metrics = await self.send_message(summarization_messages, role="summarizer", expect_json=False)
        
        # Ensure summary is a string
        if summary is None:
             summary = ""
        elif not isinstance(summary, str):
             summary = str(summary)
             
        return summary, metrics

    def _clean_json_string(self, json_string: str) -> str:
        """
        清理JSON字符串：剥离<think>段、移除BOM、空白、Markdown代码块。

        Returns:
            清理后的字符串
        """
        # 剥离 reasoning 模型吐出的 <think>...</think> 段（须在 BOM/围栏处理之前）
        json_string = strip_think_blocks(json_string)

        # 移除 UTF-8 BOM
        if json_string.startswith("\ufeff"):
            json_string = json_string.lstrip("\ufeff")

        json_string = json_string.strip()

        # 处理 Markdown 围栏代码块
        if json_string.startswith("```json"):
            json_string = json_string[7:]
        if json_string.startswith("```"):
            json_string = json_string[3:]
        if json_string.startswith("~~~"):
            json_string = json_string[3:]

        if json_string.endswith("```"):
            json_string = json_string[:-3]
        if json_string.endswith("~~~"):
            json_string = json_string[:-3]

        return json_string.strip()

    def _try_parse_json(self, json_string: str) -> Dict | None:
        """
        尝试解析JSON字符串，如果是数组则包装为对象。

        Returns:
            解析的字典或None
        """
        try:
            parsed = json.loads(json_string)
            if isinstance(parsed, list):
                return {"list": parsed}
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        return None

    def _extract_json_from_text(self, json_string: str) -> str | None:
        """
        从文本中提取JSON段落。

        Returns:
            提取的JSON字符串或None
        """
        try:
            brace_start = json_string.index("{") if "{" in json_string else None
            bracket_start = json_string.index("[") if "[" in json_string else None

            candidates = [i for i in [brace_start, bracket_start] if i is not None]
            if not candidates:
                self._get_console().print(Panel("JSON解析器在字符串中未找到 '{' 或 '['。", title="警告", style="yellow"))
                return None
            start_index = min(candidates)

            brace_end = json_string.rindex("}") if "}" in json_string else -1
            bracket_end = json_string.rindex("]") if "]" in json_string else -1
            end_index = max(brace_end, bracket_end)

            if end_index < start_index:
                self._get_console().print(Panel("JSON解析器未能定位匹配的结束符 '}' 或 ']'。", title="警告", style="yellow"))
                return None

            return json_string[start_index : end_index + 1].strip()
        except ValueError:
            self._get_console().print(Panel("JSON解析器在字符串中未找到 '{}' 或 '}'。", title="警告", style="yellow"))
            return None

    def _apply_soft_fixes(self, json_str: str) -> str:
        """
        对JSON字符串应用轻度纠错：替换Python字面量、移除尾随逗号。
        注意：已移除激进的单引号替换逻辑，以防止破坏包含引号的Payload（如SQL注入）。

        Returns:
            修复后的字符串
        """
        fixed = json_str
        # 将 Python 布尔/空值替换为 JSON
        fixed = fixed.replace(" None", " null").replace(": None", ": null")
        fixed = fixed.replace(" True", " true").replace(": True", ": true")
        fixed = fixed.replace(" False", " false").replace(": False", ": false")
        # 简单处理尾随逗号
        fixed = re.sub(r",\s*}\s*$", "}", fixed)
        fixed = re.sub(r",\s*]\s*$", "]", fixed)
        
        return fixed

    def _robust_json_parser(self, json_string: str) -> Dict | None:
        """更健壮的 JSON 解析器：
        - 处理 UTF-8 BOM、首尾空白
        - 清理 Markdown 代码块（```json、```、~~~ 等）
        - 从文本中提取第一个完整 JSON 对象或数组
        - 容忍根为数组的合法输出
        - 对常见错误进行轻度纠正（尾随逗号、单引号、None/True/False等 Python 字面量）
        """
        if not isinstance(json_string, str):
            return None

        # 1. 清理字符串
        json_string = self._clean_json_string(json_string)
        if not json_string:
            return None

        # 2. 尝试直接解析
        result = self._try_parse_json(json_string)
        if result is not None:
            return result

        # 3. 从文本中提取JSON
        candidate = self._extract_json_from_text(json_string)
        if candidate is None:
            return None

        # 4. 应用轻度纠错
        candidate_fixed = self._apply_soft_fixes(candidate)

        # 5. 尝试解析修夊后的字符串
        try:
            parsed = json.loads(candidate_fixed)
            if isinstance(parsed, list):
                return {"list": parsed}
            if isinstance(parsed, dict):
                return parsed
            return None
        except json.JSONDecodeError as e:
            self._get_console().print(
                Panel(f"JSON解析失败: {e}\n原始字符串 (清理后): {candidate_fixed[:500]}...", title="警告", style="yellow")
            )
            return None
        except Exception:
            return None


# 示例用法
if __name__ == "__main__":
    import asyncio
    import sys

    async def main():
        provider = "openai"
        if len(sys.argv) > 1 and sys.argv[1] == "anthropic":
            provider = "anthropic"

        # Manually override the provider for testing
        from conf import config

        config.LLM_PROVIDER = provider

        client = LLMClient()
        msgs = [{"role": "user", "content": "hi"}]
        # Test using the specified provider
        print(f"Sending message with provider: {provider}")
        reply = await client.send_message(msgs)
        print("Model reply:", reply)

    asyncio.run(main())
