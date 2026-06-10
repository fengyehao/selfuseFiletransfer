# core/executor.py
import asyncio
import json
import os
import re
import time
import tempfile
from typing import Dict, Any

from conf.i18n import t

import httpx
from rich.errors import MarkupError
from rich.panel import Panel

from core.console import sanitize_for_rich


def _get_console():
    """Lazy initialization of console to avoid circular imports."""
    from core.console import console_proxy
    return console_proxy
from core.events import broker
from core.graph_manager import GraphManager
from core.prompts import PromptManager
from llm.llm_client import LLMClient, strip_think_blocks
from tools.mcp_client import call_mcp_tool_async
from conf.config import (
    EXECUTOR_MAX_STEPS,
    EXECUTOR_MESSAGE_COMPRESS_THRESHOLD,
    EXECUTOR_TOKEN_COMPRESS_THRESHOLD,
    EXECUTOR_NO_ARTIFACTS_PATIENCE,
    EXECUTOR_FAILURE_THRESHOLD,
    EXECUTOR_RECENT_MESSAGES_KEEP,
    EXECUTOR_MIN_COMPRESS_MESSAGES,
    EXECUTOR_COMPRESS_INTERVAL,
    EXECUTOR_COMPRESS_INTERVAL_MSG_THRESHOLD,
    EXECUTOR_TOOL_TIMEOUT,
    EXECUTOR_MAX_OUTPUT_LENGTH,
    TOOL_TIMEOUTS,
)


async def _execute_with_retry(func, *args, max_retries: int = 3, delay: int = 5, **kwargs):
    """
    执行异步函数重试机制。

    在发生特定瞬时网络错误（超时、连接错误、JSON解析错误）时进行重试，
    最多重试max_retries次，每次间隔delay秒。

    Args:
        func: 要执行的异步函数
        *args: 函数的位置参数
        max_retries: 最大重试次数，默认为3
        delay: 重试间隔时间（秒），默认为5
        **kwargs: 函数的关键字参数

    Returns:
        函数执行成功的返回值

    Raises:
        Exception: 达到最大重试次数后的最后一个异常
    """
    for attempt in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except (httpx.ReadTimeout, httpx.ConnectError, httpx.TimeoutException, json.JSONDecodeError) as e:
            if attempt < max_retries - 1:
                _get_console().print(
                    Panel(
                        f"发生瞬时错误: {type(e).__name__} - {e}。将在{delay}秒后重试... (尝试{attempt + 2}/{max_retries})",
                        title="警告",
                        style="yellow",
                    )
                )
                await asyncio.sleep(delay)
            else:
                _get_console().print(
                    Panel(f"达到最大重试次数。放弃执行。错误: {type(e).__name__} - {e}", title="错误", style="bold red")
                )
                raise


async def _check_halt_signal(
    graph_manager: "GraphManager", subtask_id: str, last_step_ids: list, messages: list, cycle_metrics: dict, log_dir: str = None
) -> tuple[bool, tuple]:
    """
    检查终止信号。

    Args:
        graph_manager: 图管理器实例
        subtask_id: 子任务ID
        last_step_ids: 最后的步骤ID列表
        messages: 消息列表
        cycle_metrics: 周期指标
        log_dir: 日志目录

    Returns:
        tuple: (is_halted, return_value)
    """
    halt_file = os.path.join(tempfile.gettempdir(), f"{graph_manager.task_id}.halt")
    if os.path.exists(halt_file):
        _get_console().print(Panel(t("halt_signal_detected"), style="bold yellow"))
        try:
            await broker.emit(
                "execution.halt", {"subtask_id": subtask_id}, op_id=os.path.basename(log_dir) if log_dir else None
            )
        except Exception:
            pass
        for step_id in last_step_ids:
            if graph_manager.graph.has_node(step_id):
                graph_manager.update_node(step_id, {"status": "aborted"})
        graph_manager.update_subtask_conversation_history(subtask_id, messages)
        return True, (subtask_id, "aborted_by_halt_signal", cycle_metrics)
    return False, None


async def _compress_context_if_needed(
    messages: list,
    executed_steps_count: int,
    llm: "LLMClient",
    graph_manager: "GraphManager",
    subtask_id: str,
    log_dir: str = None,
    output_mode: str = "default",
    update_metrics_func: callable = None,
) -> list:
    """
    智能上下文压缩策略。

    Args:
        messages: 消息列表
        executed_steps_count: 已执行步骤数
        llm: LLM客户端
        graph_manager: 图管理器
        subtask_id: 子任务ID
        log_dir: 日志目录

    Returns:
        压缩后的消息列表
    """
    from rich.panel import Panel

    should_compress = False
    compression_reason = ""

    # 策略1: 消息数量阈值
    if len(messages) > EXECUTOR_MESSAGE_COMPRESS_THRESHOLD:
        should_compress = True
        compression_reason = f"消息数量过多 ({len(messages)} > {EXECUTOR_MESSAGE_COMPRESS_THRESHOLD})"

    # 策略2: 执行轮次阈值
    elif (
        executed_steps_count > 0
        and executed_steps_count % EXECUTOR_COMPRESS_INTERVAL == 0
        and len(messages) > EXECUTOR_COMPRESS_INTERVAL_MSG_THRESHOLD
    ):
        should_compress = True
        compression_reason = f"定期压缩 (第{executed_steps_count}轮)"

    # 策略3: 估算token超限
    else:
        total_chars = sum(len(msg.get("content", "")) for msg in messages)
        estimated_tokens = total_chars // 4
        if estimated_tokens > EXECUTOR_TOKEN_COMPRESS_THRESHOLD:
            should_compress = True
            compression_reason = f"估算token超限 ({estimated_tokens} > {EXECUTOR_TOKEN_COMPRESS_THRESHOLD})"

    if should_compress:
        try:
            if output_mode in ["default", "debug"]:
                _get_console().print(Panel(t("context_compression_trigger", reason=compression_reason), style="blue"))

            system_prompt_msg = messages[0] if messages and messages[0]["role"] == "system" else {"role": "system", "content": ""}

            recent_messages = (
                messages[-EXECUTOR_RECENT_MESSAGES_KEEP:] if len(messages) > EXECUTOR_RECENT_MESSAGES_KEEP else messages.copy()
            )

            messages_to_compress = []
            if len(messages) > EXECUTOR_RECENT_MESSAGES_KEEP:
                messages_to_compress = messages[1:-EXECUTOR_RECENT_MESSAGES_KEEP]
            else:
                messages_to_compress = messages[1:] if len(messages) > 1 else []

            if messages_to_compress and len(messages_to_compress) >= EXECUTOR_MIN_COMPRESS_MESSAGES:
                compressed_summary, compress_metrics = await llm.summarize_conversation(messages_to_compress)

                if update_metrics_func and compress_metrics:
                    update_metrics_func(compress_metrics)

                if compressed_summary:
                    compressed_message = {
                        "role": "system",
                        "content": f"📊 智能上下文摘要（压缩自{len(messages_to_compress)}条历史消息）:\n\n{compressed_summary}",
                    }

                    messages = [system_prompt_msg, compressed_message]
                    messages.extend(recent_messages)

                    graph_manager.update_subtask_conversation_history(subtask_id, messages)

                    if output_mode in ["default", "debug"]:
                        _get_console().print(
                            Panel(
                                f"✅ 智能压缩完成: {len(messages_to_compress)}条历史 → 1条摘要 + {len(recent_messages)}条近期消息",
                                style="green",
                            )
                        )
                else:
                    if output_mode in ["default", "debug"]:
                        _get_console().print(Panel(t("compression_empty"), style="yellow"))
            else:
                if output_mode in ["default", "debug"]:
                    _get_console().print(Panel(t("compression_unnecessary"), style="yellow"))

        except Exception as e:
            _get_console().print(Panel(t("compression_failed", error=str(e)), style="red"))
            if log_dir:
                error_log_path = os.path.join(log_dir, "compression_errors.log")
                try:
                    with open(error_log_path, "a", encoding="utf-8") as f:
                        f.write(f"[{time.time()}] Compression error: {e}\n")
                except Exception:
                    pass

    return messages


# --- item 4（韧性）：解析失败抢救 -------------------------------------------------
# 本地 27B–70B 模型常产出截断 / 夹带散文的 JSON，导致 send_message 三次重试后仍解析失败。
# 与其直接杀掉子任务，先尝试从原始文本中"宽松抽取"可执行的 execution_operations；抢救出的
# op 仍会走下游既有的规范化与校验（不直接写入因果图）。参考 rag/extractor.py 的"严格不劣于"
# 降级哲学（见 docs/local-model-optimization-plan.md item 4）。


def _scan_balanced(text: str, start: int) -> int | None:
    """从 text[start]（须为 '{' 或 '['）起做括号配对扫描，返回匹配闭合符之后的索引。

    扫描时尊重 JSON 字符串字面量与转义，避免字符串内部的括号 / 引号干扰（payload 常含
    []、{}、引号）。若到串尾仍未闭合（典型的中途截断）返回 None。
    """
    open_ch = text[start]
    close_ch = {"{": "}", "[": "]"}.get(open_ch)
    if close_ch is None:
        return None
    depth = 0
    in_str = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return i + 1
    return None


def _scan_json_objects(text: str, start: int) -> list:
    """从 text[start] 起逐个抽取顶层 `{...}` 对象并 json.loads，返回成功解析的 dict 列表。

    遇到第一个无法闭合的 `{`（截断）即停止——这正是要抢救"截断前已完整的对象"。
    """
    objects = []
    i = start
    n = len(text)
    while i < n:
        if text[i] == "{":
            end = _scan_balanced(text, i)
            if end is None:
                break  # 截断：之后不再有完整对象
            try:
                obj = json.loads(text[i:end])
                if isinstance(obj, dict):
                    objects.append(obj)
            except Exception:
                pass
            i = end
        else:
            i += 1
    return objects


def _coerce_to_op(obj, index: int) -> dict | None:
    """把候选 dict 规整为可执行的 EXECUTE_NOW 操作；无法构成有效操作时返回 None。

    既接受完整 op（含 action / command），也接受裸 action（顶层带 tool / name）。补齐缺失的
    command 与 node_id，确保能通过 Executor 既有的 "EXECUTE_NOW + node_id" 过滤。
    """
    if not isinstance(obj, dict):
        return None
    if isinstance(obj.get("action"), dict) or str(obj.get("command", "")).upper() == "EXECUTE_NOW":
        op = dict(obj)
    elif obj.get("tool") or obj.get("name"):  # 裸 action 对象
        op = {"action": {k: obj[k] for k in ("tool", "name", "params", "arguments") if k in obj}}
    else:
        return None
    op.setdefault("command", "EXECUTE_NOW")
    if not op.get("node_id") or op.get("node_id") == "None":
        op["node_id"] = f"salvaged_{index + 1}"
    action = op.get("action")
    if not isinstance(action, dict) or not (action.get("tool") or action.get("name")):
        return None
    return op


def _salvage_execution_operations(raw) -> dict | None:
    """从无法整体解析的 LLM 原文中宽松抽取可执行的 execution_operations。

    策略：先剥离 <think> 段；优先定位 "execution_operations" 数组（整体解析，失败则在数组区
    域逐元素抽取以应对截断）；若无该键则全文扫描，抢救裸 op / action 对象。抢救到 ≥1 个可执行
    op 时返回最小 reply dict，否则返回 None 交由调用方降级。返回的 op 仍走 Executor 既有的
    规范化与校验，不直接入图。
    """
    if not isinstance(raw, str):
        return None
    text = strip_think_blocks(raw)

    candidates: list = []
    match = re.search(r'"execution_operations"\s*:\s*\[', text)
    if match:
        arr_start = match.end() - 1  # 指向匹配到的 '['
        end = _scan_balanced(text, arr_start)
        if end is not None:
            try:
                arr = json.loads(text[arr_start:end])
                if isinstance(arr, list):
                    candidates = [x for x in arr if isinstance(x, dict)]
            except Exception:
                candidates = []
        if not candidates:
            # 数组被截断 / 夹带噪声：在数组区域逐个抽取已完整的对象
            candidates = _scan_json_objects(text, arr_start)
    if not candidates:
        # 没有 execution_operations 键：全文扫描，抢救裸 op / action 对象
        candidates = _scan_json_objects(text, 0)

    ops = []
    for idx, candidate in enumerate(candidates):
        op = _coerce_to_op(candidate, idx)
        if op is not None:
            ops.append(op)
    if not ops:
        return None
    return {
        "thought": "（系统抢救）原始输出无法整体解析，已从中抽取可执行操作以继续本步。",
        "execution_operations": ops,
        "is_subtask_complete": False,
    }


async def _call_llm_and_parse_response(
    llm: "LLMClient",
    messages: list,
    update_cycle_metrics_func: callable,
    subtask_id: str,
    console_output_path: str = None,
    output_mode: str = "default", # Add this parameter
) -> tuple[dict, list]:
    """
    调用LLM并解析响应。

    Returns:
        tuple: (llm_reply_json, updated_messages) 或引发退出
    """
    llm_reply_json, call_metrics = None, None
    try:
        llm_reply_json, call_metrics = await _execute_with_retry(llm.send_message, messages, role="executor")
        update_cycle_metrics_func(call_metrics)
    except httpx.ReadTimeout:
        _get_console().print("LLM调用超时，无法继续执行。", style="red")
        if console_output_path:
            try:
                with open(console_output_path, "a", encoding="utf-8") as f:
                    f.write(f"[ERROR] LLM调用超时，子任务 {subtask_id} 终止。\n")
            except Exception:
                pass
        raise RuntimeError("llm_timeout")
    except Exception as e:
        _get_console().print(f"LLM输出或解析失败: {e}", style="red")
        if console_output_path:
            try:
                with open(console_output_path, "a", encoding="utf-8") as f:
                    f.write(f"[ERROR] LLM输出或解析失败: {type(e).__name__}: {e}\n")
            except Exception:
                pass
        raise RuntimeError("llm_parse_error")

    if not llm_reply_json:
        # item 4（韧性）：整体解析失败时，先尝试从原始文本抢救可执行的 execution_operations，
        # 而非直接终止子任务。raw 由 send_message 在 JSON 重试耗尽时经 metrics 透出（并发安全）。
        raw = call_metrics.get("raw") if isinstance(call_metrics, dict) else None
        salvaged = _salvage_execution_operations(raw)
        if salvaged is not None:
            _get_console().print("⚠️ LLM输出无法整体解析，已抢救出可执行操作并继续本步。", style="yellow")
            if console_output_path:
                try:
                    with open(console_output_path, "a", encoding="utf-8") as f:
                        f.write(f"[WARN] LLM输出解析失败，已抢救 execution_operations 继续子任务 {subtask_id}。\n")
                except Exception:
                    pass
            llm_reply_json = salvaged
        else:
            _get_console().print("LLM输出解析失败且无法抢救，触发降级处理。", style="red")
            if console_output_path:
                try:
                    with open(console_output_path, "a", encoding="utf-8") as f:
                        f.write(f"[ERROR] LLM输出解析失败且无法抢救，子任务 {subtask_id} 进入降级处理。\n")
                except Exception:
                    pass
            raise RuntimeError("llm_empty_response")

    messages.append({"role": "assistant", "content": json.dumps(llm_reply_json, ensure_ascii=False)})

    # 打印LLM响应
    json_str = json.dumps(llm_reply_json, indent=2, ensure_ascii=False)
    safe_json_str = sanitize_for_rich(json_str)

    # 只有在 default 或 debug 模式下才打印 LLM 思考过程
    if output_mode in ["default", "debug"]:
        try:
            _get_console().print(Panel(safe_json_str, title=t("llm_thought_title"), style="cyan"))
        except MarkupError:
            _get_console().print(f"LLM思考 (结构化 - 原始):\n{safe_json_str}")

    return llm_reply_json, messages


def _update_previous_steps_status(
    llm_reply_json: dict,
    last_step_ids: list,
    graph_manager: "GraphManager",
) -> None:
    """
    使用LLM的判断更新上一步的状态。
    """
    raw_prev_status = llm_reply_json.get("previous_steps_status", {})
    if isinstance(raw_prev_status, str):
        try:
            parsed_prev = json.loads(raw_prev_status)
            previous_steps_status = parsed_prev if isinstance(parsed_prev, dict) else {}
        except Exception:
            previous_steps_status = {}
    elif isinstance(raw_prev_status, dict):
        previous_steps_status = raw_prev_status
    else:
        previous_steps_status = {}

    if last_step_ids:
        for step_id in last_step_ids:
            status = previous_steps_status.get(step_id)
            # Normalize 'executed' to 'completed' for frontend compatibility
            if status == "executed":
                status = "completed"
            
            if status in ["completed", "failed"]:
                graph_manager.update_node(step_id, {"status": status})
            else:
                # If LLM returns something else or nothing, keep the status set by tool execution (usually 'completed' or 'failed')
                pass


def _check_failure_patterns_and_trigger_reflection(
    llm_reply_json: dict,
    last_step_ids: list,
    graph_manager: "GraphManager",
    failure_counts_per_parent: dict,
    messages: list,
) -> list:
    """
    检查失败模式并触发强制反思。

    Returns:
        更新后的messages列表
    """
    # 2.1. 更新失败计数器
    if last_step_ids:
        parent_to_current_steps = {}
        for step_id in last_step_ids:
            parent_id = graph_manager.graph.nodes[step_id].get("parent")
            if parent_id:
                if parent_id not in parent_to_current_steps:
                    parent_to_current_steps[parent_id] = []
                parent_to_current_steps[parent_id].append(step_id)

        for parent_id, current_steps in parent_to_current_steps.items():
            all_failed = all(graph_manager.graph.nodes[step_id].get("status") == "failed" for step_id in current_steps)
            if all_failed:
                failure_counts_per_parent[parent_id] = failure_counts_per_parent.get(parent_id, 0) + 1
                _get_console().print(
                    f"🔍 Executor: parent_id '{parent_id}' 的连续失败次数计为 {failure_counts_per_parent[parent_id]}",
                    style="dim",
                )
            else:
                if parent_id in failure_counts_per_parent:
                    failure_counts_per_parent[parent_id] = 0

        # 2.2. 检查是否需要强制反思 (基于失败计数器)
        for parent_id, count in failure_counts_per_parent.items():
            if count >= EXECUTOR_FAILURE_THRESHOLD:
                forced_reflection_message = (
                    f"⚠️ 警告：你在 parent_id '{parent_id}' 下连续 {count} 次执行操作均失败。"
                    f"你必须立即调用 'formulate_hypotheses' 工具来重新审视你的假设并制定新策略，"
                    f"或者切换到不同的测试方向，不要再重复当前策略。"
                )
                messages.append({"role": "user", "content": forced_reflection_message})
                _get_console().print(f"🤖 Executor: 向 LLM 发送强制反思指令，针对 parent_id '{parent_id}'。", style="bold yellow")
                failure_counts_per_parent[parent_id] = 0

    # 2.3. 检查是否需要强制反思 (基于矛盾检测)
    raw_hypothesis_update = llm_reply_json.get("hypothesis_update", {})
    if isinstance(raw_hypothesis_update, str):
        try:
            parsed_hyp = json.loads(raw_hypothesis_update)
            hypothesis_update_data = parsed_hyp if isinstance(parsed_hyp, dict) else {}
        except Exception:
            hypothesis_update_data = {}
    elif isinstance(raw_hypothesis_update, dict):
        hypothesis_update_data = raw_hypothesis_update
    else:
        hypothesis_update_data = {}

    if hypothesis_update_data.get("contradiction_detected"):
        contradiction_message = hypothesis_update_data.get("contradiction_detected")
        forced_reflection_message = (
            f"⚠️ 警告：Executor 检测到矛盾: {contradiction_message}。"
            f"你必须立即调用 'formulate_hypotheses' 工具来重新审视你的假设并制定新策略，"
            f"或者切换到不同的测试方向，不要再重复当前策略。"
        )
        messages.append({"role": "user", "content": forced_reflection_message})
        _get_console().print("🤖 Executor: 向 LLM 发送强制反思指令，针对检测到的矛盾。", style="bold yellow")

    return messages


async def _build_executor_prompt(
    graph_manager: "GraphManager",
    subtask_id: str,
    main_goal: str,
    global_mission_briefing: str,
    messages: list,
) -> tuple[str, list]:
    """
    构建执行器提示词。

    Returns:
        tuple: (system_prompt, updated_messages)
    """
    subtask_data = graph_manager.graph.nodes[subtask_id]
    prompt_context = graph_manager.build_prompt_context(
        subtask_id, include_relevant_causal_context=False
    )

    manager = PromptManager()
    subtask = {
        "id": subtask_id,
        "description": subtask_data["description"],
        "completion_criteria": prompt_context.get("subtask", {}).get("completion_criteria", "N/A") if prompt_context else "N/A",
    }
    context = {
        "causal_context": prompt_context.get("causal_context", {}) if prompt_context else {},
        "dependencies": prompt_context.get("dependencies", []) if prompt_context else [],
        "causal_graph_summary": prompt_context.get("causal_graph_summary", "因果链图谱为空。") if prompt_context else "因果链图谱为空。",
        "key_facts": prompt_context.get("key_facts", []) if prompt_context else [],
        "active_hypotheses": subtask_data.get("active_hypotheses") or [],
    }
    system_prompt = manager.build_executor_prompt(
        main_goal=main_goal, subtask=subtask, context=context, global_mission_briefing=global_mission_briefing
    )

    if not messages or messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": system_prompt})
    else:
        messages[0] = {"role": "system", "content": system_prompt}

    return system_prompt, messages


# ── P3-1: 本地工具处理 ─────────────────────────────────────────────────────────

_LOCAL_TOOLS = {"query_causal_graph"}


async def _handle_local_tool(tool_name: str, tool_params: dict, graph_manager: GraphManager) -> str:
    """处理不经过 MCP 的本地工具调用。当前支持：query_causal_graph。"""
    if tool_name == "query_causal_graph":
        node_type_filter: str = tool_params.get("node_type", "")
        query: str = tool_params.get("query", "")
        limit: int = int(tool_params.get("limit", 10))

        results = []
        for node_id, data in graph_manager.causal_graph.nodes(data=True):
            nt = data.get("node_type", "")
            if node_type_filter and nt != node_type_filter:
                continue
            text = " ".join(filter(None, [
                data.get("title", ""), data.get("description", ""),
                data.get("vulnerability", ""), data.get("hypothesis", ""),
            ])).lower()
            if query and query.lower() not in text:
                continue
            results.append({
                "id": node_id,
                "node_type": nt,
                "title": data.get("title", ""),
                "description": data.get("description", ""),
                "confidence": data.get("confidence"),
                "status": data.get("status"),
                "evidence": data.get("evidence"),
                "vulnerability": data.get("vulnerability"),
            })
            if len(results) >= limit:
                break

        return json.dumps({
            "success": True,
            "count": len(results),
            "results": results,
            "note": f"查询参数: node_type={node_type_filter!r}, query={query!r}, limit={limit}",
        }, ensure_ascii=False, indent=2)

    return json.dumps({"success": False, "error": f"未知本地工具: {tool_name}"}, ensure_ascii=False)


async def run_executor_cycle(
    main_goal: str,
    subtask_id: str,
    llm: LLMClient,
    graph_manager: GraphManager,
    global_mission_briefing: str = "",

    log_dir: str = None,
    save_callback: callable = None,
    output_mode: str = "default",
    max_steps: int = None,
    disable_artifact_check: bool = False,
) -> tuple[str, str, dict]:
    """
    执行器循环：为子任务执行思想树探索循环。

    该函数实现了执行器的核心功能，包括：
    - 智能上下文压缩：基于消息数量、轮次和内容复杂度的多维度压缩
    - 动态终止逻辑：基于步数限制、无产出物检测和失败阈值
    - 工具调用执行：支持MCP工具调用和结果处理
    - 错误处理和重试机制
    - 详细的指标追踪和日志记录

    Args:
        main_goal: 主要目标描述
        subtask_id: 子任务ID
        llm: LLM客户端实例
        graph_manager: 图管理器实例
        global_mission_briefing: 全局任务简报（可选）
        verbose: 是否启用详细输出，默认为False
        log_dir: 日志目录路径（可选）
        save_callback: 保存回调函数（可选）

    Returns:
        tuple: 包含以下元素
            - subtask_id (str): 子任务ID
            - status (str): 执行结果状态（success/aborted_by_halt_signal等）
            - cycle_metrics (dict): 执行周期指标字典
    """
    from rich.panel import Panel
    from collections import defaultdict

    messages = graph_manager.get_subtask_conversation_history(subtask_id)

    # 初始化本周期的指标
    cycle_metrics = {"prompt_tokens": 0, "completion_tokens": 0, "cost_cny": 0, "tool_calls": defaultdict(int)}

    def update_cycle_metrics(call_metrics: Dict[str, Any]) -> None:
        """
        更新执行周期指标。

        Args:
            call_metrics: LLM调用指标字典，包含token使用和成本信息

        Returns:
            None
        """
        if call_metrics:
            cycle_metrics["prompt_tokens"] += call_metrics.get("prompt_tokens", 0)
            cycle_metrics["completion_tokens"] += call_metrics.get("completion_tokens", 0)
            cycle_metrics["cost_cny"] += call_metrics.get("cost_cny", 0)

    executed_steps_count = 0
    # 初始化每步详细日志
    # run_log_path (已移除，不再使用)
    console_output_path = None
    if log_dir:
        console_output_path = os.path.join(log_dir, "console_output.log")
    consecutive_no_new_artifacts = 0
    # 从子任务节点读取持久化的执行链，确保子任务恢复执行时能续接上一次的执行链
    last_step_ids = graph_manager.get_subtask_last_step_ids(subtask_id)
    failure_counts_per_parent = {}
    consecutive_parse_failures = 0  # item 4：连续"无法解析且无法抢救"次数，达阈值才终止子任务

    while True:
        # 检查终止信号
        is_halted, halt_result = await _check_halt_signal(graph_manager, subtask_id, last_step_ids, messages, cycle_metrics, log_dir)
        if is_halted:
            return halt_result

        # 智能上下文压缩
        messages = await _compress_context_if_needed(messages, executed_steps_count, llm, graph_manager, subtask_id, log_dir, output_mode=output_mode, update_metrics_func=update_cycle_metrics)

        _get_console().print(
            Panel(t("subtask_step", subtask_id=subtask_id, step=executed_steps_count + 1), title_align="left", style="green")
        )

        # P1-2: 并行任务共享公告板 — 读取其他并行子任务新增的高价值发现并注入 prompt
        if executed_steps_count > 0:
            new_findings = graph_manager.get_new_shared_findings(subtask_id)
            if new_findings:
                bulletin_lines = []
                for f in new_findings:
                    bulletin_lines.append(
                        f"  - [{f['node_type']}] 来自子任务 {f['from_subtask']}: **{f['title']}** — {f['description'][:120]}"
                    )
                bulletin_msg = (
                    f"📢 [共享公告板] 其他并行子任务新增了 {len(new_findings)} 条线索（暂存状态，尚未经 Reflector 审核，请作为参考而非已确认事实）：\n"
                    + "\n".join(bulletin_lines)
                )
                messages.append({"role": "user", "content": bulletin_msg})
                _get_console().print(f"📢 [共享公告板] 向子任务 {subtask_id} 注入 {len(new_findings)} 条跨任务发现", style="dim cyan")

        # P1-3: 首步假设制定软性提示 — 仅在第 0 步且因果图无明确 ConfirmedVulnerability 时提示
        if executed_steps_count == 0:
            has_confirmed_vuln = any(
                d.get("node_type") == "ConfirmedVulnerability"
                for _, d in graph_manager.causal_graph.nodes(data=True)
            )
            if not has_confirmed_vuln:
                messages.append({
                    "role": "user",
                    "content": (
                        "💡 [建议] 这是你的第一步。当前因果图中尚无 ConfirmedVulnerability 节点。"
                        "建议优先调用 `formulate_hypotheses` 来明确你的攻击假设和测试路径，"
                        "这将使后续步骤更有目的性。如果你已有明确的行动计划，可以直接执行。"
                    ),
                })
                _get_console().print(f"💡 [P1-3] 首步假设提示已注入子任务 {subtask_id}", style="dim cyan")

        # 构建提示词
        system_prompt, messages = await _build_executor_prompt(graph_manager, subtask_id, main_goal, global_mission_briefing, messages)

        # 调用LLM并解析响应
        try:
            llm_reply_json, messages = await _call_llm_and_parse_response(
                llm, messages, update_cycle_metrics, subtask_id, console_output_path, output_mode=output_mode
            )
            consecutive_parse_failures = 0  # 成功（含抢救成功）→ 重置降级计数
        except RuntimeError as e:
            # item 4（韧性）：JSON 解析彻底失败且无法抢救（llm_empty_response）时优雅降级——
            # 注入一次"重新观察"提示并重试本步，而非杀掉子任务；连续达阈值才终止。
            # 其他 RuntimeError（llm_timeout / llm_parse_error，即拿不到响应）维持原行为，直接终止。
            if str(e) == "llm_empty_response":
                consecutive_parse_failures += 1
                if consecutive_parse_failures >= EXECUTOR_FAILURE_THRESHOLD:
                    _get_console().print(
                        Panel(
                            f"连续 {consecutive_parse_failures} 次无法解析或抢救 LLM 输出，终止子任务。",
                            title="智能终止",
                            style="bold red",
                        )
                    )
                    graph_manager.update_subtask_conversation_history(subtask_id, messages)
                    return (subtask_id, "error", cycle_metrics)
                _get_console().print(
                    Panel(
                        f"LLM 输出无法解析（第 {consecutive_parse_failures}/{EXECUTOR_FAILURE_THRESHOLD} 次），"
                        "降级为重新观察并重试本步，不终止子任务。",
                        title="降级处理",
                        style="yellow",
                    )
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "⚠️ 系统提示：你上一次的回复经多次重试仍无法解析为合法 JSON，且无法从中抢救出可执行操作。"
                            "请重新输出一个**最小化的合法 JSON 对象**，仅包含：一个 `thought`（一句话说明当前最稳妥的下一步），"
                            "以及一个 `execution_operations` 数组（其中只含一个最简单、最安全的 `EXECUTE_NOW` 操作，"
                            "例如对当前目标重新发起一次最小探针或观察）。"
                            "禁止输出任何 JSON 以外的文本、解释或 `<think>` 段。"
                        ),
                    }
                )
                continue
            return (subtask_id, "error", cycle_metrics)

        # 更新上一步状态
        _update_previous_steps_status(llm_reply_json, last_step_ids, graph_manager)

        # 检查失败模式并触发反思
        messages = _check_failure_patterns_and_trigger_reflection(
            llm_reply_json, last_step_ids, graph_manager, failure_counts_per_parent, messages
        )

        # Store the completion flag from this turn's response, with robust coercion
        raw_complete = llm_reply_json.get("is_subtask_complete", False)
        if isinstance(raw_complete, str):
            is_final_step = raw_complete.strip().lower() in ("true", "yes", "1")
        else:
            is_final_step = bool(raw_complete) is True

        # 3. 处理产出物提议（健壮化列表与元素类型）
        artifact_proposals_raw = llm_reply_json.get("staged_causal_nodes", [])
        artifact_proposals: list = []
        if isinstance(artifact_proposals_raw, str):
            try:
                loaded = json.loads(artifact_proposals_raw)
                if isinstance(loaded, list):
                    artifact_proposals = [x for x in loaded if isinstance(x, dict)]
                elif isinstance(loaded, dict):
                    artifact_proposals = [loaded]
            except Exception:
                artifact_proposals = []
        elif isinstance(artifact_proposals_raw, list):
            artifact_proposals = [x for x in artifact_proposals_raw if isinstance(x, dict)]
        elif isinstance(artifact_proposals_raw, dict):
            artifact_proposals = [artifact_proposals_raw]
        else:
            artifact_proposals = []

        if artifact_proposals:
            graph_manager.stage_proposed_causal_nodes(subtask_id, artifact_proposals)

        # ... (rest of the logic for processing LLM response, updating graph, etc.)

        # 规范化执行指令，防止将字符串或不合规结构当作字典访问
        exec_ops_raw = llm_reply_json.get("execution_operations", [])
        normalized_exec_ops = []

        def _normalize_op_item(item):
            """
            将不同类型的执行指令规整为字典列表。

            Args:
                item: 执行指令项，可能是dict、list或str

            Returns:
                规范化后的字典列表
            """
            if isinstance(item, dict):
                return [item]
            if isinstance(item, list):
                return [x for x in item if isinstance(x, dict)]
            if isinstance(item, str):
                s = item.strip()
                try:
                    loaded = json.loads(s)
                    if isinstance(loaded, dict):
                        return [loaded]
                    if isinstance(loaded, list):
                        return [x for x in loaded if isinstance(x, dict)]
                except Exception:
                    pass
                _get_console().print(f"⚠️ 非结构化执行指令，已忽略: {s[:200]}", style="yellow")
                return []
            return []

        if isinstance(exec_ops_raw, list):
            for v in exec_ops_raw:
                normalized_exec_ops.extend(_normalize_op_item(v))
        else:
            normalized_exec_ops.extend(_normalize_op_item(exec_ops_raw))

        current_step_ops = [
            op
            for op in normalized_exec_ops
            if isinstance(op, dict) and str(op.get("command", "")).upper() == "EXECUTE_NOW"
        ]

        halt_file = os.path.join(tempfile.gettempdir(), f"{graph_manager.task_id}.halt")
        if os.path.exists(halt_file):
            return (subtask_id, "aborted_by_halt_signal", cycle_metrics)

        if not current_step_ops and not is_final_step:
            _get_console().print("LLM在本轮思考中未提供可执行的动作（EXECUTE_NOW），子任务结束。", style="yellow")
            return (subtask_id, "stalled_no_plan", cycle_metrics)

        # Execute tools in parallel
        execution_tasks = []
        potential_parent = last_step_ids[0] if last_step_ids else subtask_id
        current_cycle_step_ids = []

        for i, op in enumerate(current_step_ops):
            step_id = op.get("node_id")
            if not step_id or step_id == "None":
                _get_console().print(f"⚠️ 跳过无效EXECUTE_NOW操作（缺少node_id）: {op}", style="yellow")
                continue
            
            # Ensure step_id is globally unique by prepending subtask_id
            # This is crucial to prevent node_id collisions across different subtasks
            original_step_id = step_id
            step_id = f"{subtask_id}_{original_step_id}"

            current_cycle_step_ids.append(step_id)
            
            parent_id = op.get("parent_id") or potential_parent
            if not graph_manager._is_valid_parent_for_subtask(parent_id, subtask_id):
                parent_id = potential_parent

            hypothesis_update = llm_reply_json.get("hypothesis_update", {})
            if not isinstance(hypothesis_update, dict):
                hypothesis_update = {}
            thought = op.get("thought")

            action = op.get("action") or {}
            if isinstance(action, str):
                try:
                    action = json.loads(action)
                except:
                    action = {"tool": str(action)}
            
            # Add execution step to graph
            graph_manager.add_execution_step(
                step_id, parent_id, thought, action, "in_progress", hypothesis_update=hypothesis_update
            )
            try:
                await broker.emit(
                    "graph.changed",
                    {"reason": "execution_step_added", "step_id": step_id},
                    op_id=os.path.basename(log_dir) if log_dir else None,
                )
            except Exception:
                pass
            
            tool_name = action.get("tool") or action.get("name") or "unknown_tool"
            tool_params = action.get("params") or action.get("arguments") or {}
            
            cycle_metrics["tool_calls"][tool_name] += 1
            
            _get_console().print(
                Panel(
                    f"准备并行执行动作: {tool_name}\n参数: {json.dumps(tool_params, ensure_ascii=False)}",
                    title=f"准备动作{step_id}",
                    style="magenta",
                )
            )
            
            execution_tasks.append(
                asyncio.wait_for(
                    _handle_local_tool(tool_name, tool_params, graph_manager)
                    if tool_name in _LOCAL_TOOLS
                    else _execute_with_retry(call_mcp_tool_async, tool_name, tool_params),
                    timeout=TOOL_TIMEOUTS.get(tool_name, EXECUTOR_TOOL_TIMEOUT)
                )
            )

        # Real-time metrics update
        if log_dir:
            metrics_path = os.path.join(log_dir, "metrics.json")
            try:
                if os.path.exists(metrics_path):
                    with open(metrics_path, "r", encoding="utf-8") as f:
                        current_metrics = json.load(f)
                else:
                    current_metrics = {}
                
                current_metrics.setdefault("tool_calls", {})
                for tool, count in cycle_metrics["tool_calls"].items():
                    current_metrics["tool_calls"][tool] = current_metrics["tool_calls"].get(tool, 0) + count
                    
                with open(metrics_path, "w", encoding="utf-8") as f:
                    json.dump(current_metrics, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

        last_step_ids = current_cycle_step_ids
        # 持久化执行链到子任务节点，确保子任务被中断后恢复时能续接执行链
        graph_manager.update_subtask_last_step_ids(subtask_id, current_cycle_step_ids)

        if execution_tasks:
            tool_results = await asyncio.gather(*execution_tasks, return_exceptions=True)

            # Process results and check for immediate failures
            has_correctable_error = False
            correction_feedback = []
            observations = []
            MAX_OBSERVATION_LENGTH = EXECUTOR_MAX_OUTPUT_LENGTH
            truncated_steps = []

            for i, result in enumerate(tool_results):
                step_id = last_step_ids[i]
                tool_name = current_step_ops[i].get("action", {}).get("tool", "unknown_tool")
                step_status = "completed"
                
                # Handle errors
                if isinstance(result, Exception):
                    result_str = f"Error executing tool: {result}"
                    step_status = "failed"
                    if console_output_path:
                        try:
                            with open(console_output_path, "a", encoding="utf-8") as f:
                                f.write(f"[ERROR] 工具 {tool_name} 执行异常: {result}\n")
                        except Exception:
                            pass
                else:
                    result_str = str(result)
                    # Check for soft errors in JSON response
                    try:
                        data = json.loads(result_str)
                        if isinstance(data, dict) and data.get("success") is False:
                            error_type = data.get("error_type")
                            if error_type in ["SYNTAX", "MISSING_TOOL"]:
                                has_correctable_error = True
                                feedback = f"- Step {step_id} (Tool: {tool_name}) failed: {data.get('message')} -> {data.get('fix_suggestion')}"
                                correction_feedback.append(feedback)
                                step_status = "failed"
                    except:
                        pass

                # Truncation logic
                original_length = len(result_str)
                was_truncated = False
                if original_length > MAX_OBSERVATION_LENGTH:
                    result_str = result_str[:MAX_OBSERVATION_LENGTH] + f"\n... (Truncated from {original_length})"
                    was_truncated = True
                    _get_console().print(Panel(f"⚠️ 动作 {step_id} 结果过长已截断", title="警告", style="yellow"))
                    truncated_steps.append({
                        "step_id": step_id, 
                        "tool_name": tool_name, 
                        "original_length": original_length,
                        "sent_length": MAX_OBSERVATION_LENGTH
                    })

                observations.append(f"动作 {step_id} (工具={tool_name}) 的结果: {result_str}")
                
                # Update graph and logs
                graph_manager.update_node(
                    step_id,
                    {
                        "observation": observations[-1],
                        "observation_truncated": was_truncated,
                        "observation_original_length": original_length,
                        "status": step_status,
                    },
                )

                # P1-1: 假设跨步持久化 — 将 formulate_hypotheses 输出写入子任务节点
                if tool_name == "formulate_hypotheses" and step_status == "completed":
                    try:
                        hyp_result = json.loads(result_str)
                        new_hypotheses = hyp_result.get("hypotheses_record", {}).get("hypotheses", [])
                        if new_hypotheses:
                            graph_manager.update_node(subtask_id, {"active_hypotheses": new_hypotheses})
                            _get_console().print(
                                f"💡 [假设持久化] 已将 {len(new_hypotheses)} 条假设写入节点 {subtask_id}",
                                style="dim cyan",
                            )
                    except Exception:
                        pass
                
                try:
                    run_log_entry = {
                        "event": "executor_step_completed",
                        "step_id": step_id,
                        "tool_name": tool_name,
                        "result": result_str,
                        "timestamp": time.time(),
                    }
                    await broker.emit("execution.step.completed", run_log_entry, op_id=os.path.basename(log_dir) if log_dir else None)
                except:
                    pass

            # Handle immediate corrections
            if has_correctable_error:
                correction_prompt = f"检测到工具调用错误，请立即修正:\n" + "\n".join(correction_feedback)
                _get_console().print(Panel(correction_prompt, title="🤖 Executor: 请求修正", style="bold yellow"))
                messages.append({"role": "user", "content": correction_prompt})
                continue

            full_observation = "\n".join(observations)
            messages.append({"role": "user", "content": f"你并行执行了 {len(last_step_ids)} 个动作，观察到：\n{full_observation}"})

            if output_mode == "debug": # Changed from if verbose:
                _get_console().print(
                    Panel(
                        f"工具执行结果:\n{full_observation}",
                        title="[bold green]Debug Tool Results[/bold green]", # Changed title
                        style="green"
                    )
                )

            if truncated_steps:
                messages.append({"role": "user", "content": f"⚠️ 注意：{len(truncated_steps)} 个观察结果已被截断。"})

        if is_final_step:
            _get_console().print(Panel(t("subtask_complete", subtask_id=subtask_id), style="green"))
            graph_manager.update_node(subtask_id, {"status": "completed"})
            return (subtask_id, "completed", cycle_metrics)

        # --- 动态终止逻辑 ---
        # 1. 最大步数限制（安全网）
        effective_max_steps = max_steps if max_steps is not None else EXECUTOR_MAX_STEPS
        if executed_steps_count >= effective_max_steps:
            _get_console().print(
                Panel(
                    f"达到最大步数限制 ({effective_max_steps})，为安全起见终止子任务。",
                    title="智能终止",
                    style="bold red",
                )
            )
            # 将终止原因写入子任务节点，供 Reflector/Planner 使用
            graph_manager.update_node(
                subtask_id, {"termination_reason": "max_steps_reached", "executed_steps": executed_steps_count}
            )
            break

        # 2. 检查新产出物
        if not llm_reply_json.get("staged_causal_nodes", []):
            consecutive_no_new_artifacts += 1
        else:
            consecutive_no_new_artifacts = 0  # 有新产出物时重置

        if not disable_artifact_check and consecutive_no_new_artifacts >= EXECUTOR_NO_ARTIFACTS_PATIENCE:
            _get_console().print(
                Panel(
                    f"连续 {EXECUTOR_NO_ARTIFACTS_PATIENCE} 步没有新的产出物提议，探索已停滞。终止子任务。",
                    title="智能终止",
                    style="bold yellow",
                )
            )
            # 记录缺乏新产物导致的终止原因
            graph_manager.update_node(
                subtask_id, {"termination_reason": "no_new_artifacts", "executed_steps": executed_steps_count}
            )
            break

        # 8. 检查外部终止信号
        halt_file = os.path.join(tempfile.gettempdir(), f"{graph_manager.task_id}.halt")
        if os.path.exists(halt_file):
            try:
                with open(halt_file, "r", encoding="utf-8") as f:
                    halt_payload = json.load(f) # Read payload to include in metrics if needed
                _get_console().print(
                    Panel(
                        f"🚩 在 {subtask_id} 执行期间检测到外部终止信号！正在中断...",
                        style="bold yellow",
                    )
                )
                for step_id in last_step_ids:
                    if graph_manager.graph.has_node(step_id):
                        graph_manager.update_node(step_id, {"status": "aborted"})
                # Save current messages state and turn counter before returning
                graph_manager.update_subtask_conversation_history(subtask_id, messages)
                return (subtask_id, "aborted_by_external_halt_signal", cycle_metrics)
            except Exception:
                _get_console().print(Panel("读取终止信号文件失败或格式无效，继续执行。", title="警告", style="red"))

        # Save logs after each step
        if save_callback:
            save_callback(cycle_metrics=cycle_metrics)
        # 新增：实时维护 metrics.json 的 execution_steps 字段
        if log_dir:
            metrics_path = os.path.join(log_dir, "metrics.json")
            try:
                if os.path.exists(metrics_path):
                    with open(metrics_path, "r", encoding="utf-8") as f:
                        metrics = json.load(f)
                else:
                    metrics = {}
                
                # 直接设置执行步数（不累加）
                metrics["execution_steps"] = executed_steps_count
                
                # 实时更新tool_calls（使用cycle_metrics中的累计值）
                if "tool_calls" not in metrics:
                    metrics["tool_calls"] = {}
                # 实时更新 cost 和 token 信息
                if "cost_cny" not in metrics:
                    metrics["cost_cny"] = 0
                metrics["cost_cny"] = max(metrics.get("cost_cny", 0), cycle_metrics.get("cost_cny", 0))
                metrics["total_tokens"] = cycle_metrics.get("prompt_tokens", 0) + cycle_metrics.get("completion_tokens", 0)
                
                # Atomic write
                tmp_path = metrics_path + ".tmp"
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(metrics, f, ensure_ascii=False, indent=2)
                os.replace(tmp_path, metrics_path)
            except Exception:
                pass

        # Increment executed steps count for the next iteration
        executed_steps_count += 1


    # End of while loop
    _get_console().print(Panel(t("max_steps_reached", steps=executed_steps_count), style="yellow"))
    for step_id in last_step_ids:
        if graph_manager.graph.has_node(step_id):
            graph_manager.update_node(step_id, {"status": "completed"})
    graph_manager.update_subtask_conversation_history(subtask_id, messages)
    cycle_metrics["execution_steps"] = executed_steps_count
    return (subtask_id, "completed", cycle_metrics)
