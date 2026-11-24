"""LangGraph 版本的對話迴圈（與 loop.run_agent 行為等價，流程改成看得見的狀態圖）。

    python -m agent_v2.graph "台積電 2025Q4 的營收是多少？"     # 問一題
    python -m agent_v2.graph --mermaid                          # 印出流程圖（Mermaid）

設計原則：
1. loop.py 不動，當作等價性比對的基準；守衛函式、提示詞、AgentResult 都直接 import，不複製。
2. Gemini 函式呼叫與 tools.py 維持手寫，LangGraph 只負責流程。
3. 節點只回傳「要更新的欄位」，不原地修改 list（日後接 checkpointer 才安全）。

流程：
  call_model ─┬─ 有要呼叫工具 → run_tools ─┬→ call_model（步數未滿）
              │                            └→ max_steps（步數用完）
              └─ 最終答案 → guard_empty → guard_ungrounded → guard_leak → guard_docs → finish
                  任一守衛要求重試 → nudge → call_model（或步數用完 → max_steps）
"""
import json
import logging
import os
from datetime import date
from typing import Any, TypedDict

from google.genai import types
from langgraph.graph import END, START, StateGraph

from agent_v2 import memory, tools
from agent_v2.loop import (EMPTY_RETRY_TEMPERATURES, LEAK_REFUSAL, MAX_STEPS, NUDGE, NUDGE_DOCS, SYSTEM_PROMPT,
                           UNGROUNDED_ANSWER, AgentResult, _latest_period, claims_no_docs_without_search,
                           get_client, has_unsourced_numbers, leaks_system_prompt)

logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    system: str
    contents: list           # 送給 Gemini 的對話紀錄
    tool_log: list           # 每次工具呼叫的紀錄（評估與追蹤用）
    seen: list               # 已呼叫過的「工具名+參數」簽章（重複呼叫偵測）
    step: int                # 已經呼叫過幾次 call_model（含 nudge 後的重試）
    max_steps: int
    nudged: bool             # 「沒查資料就報數字」的提醒是否用過
    docs_nudged: bool        # 「沒搜尋就說文件庫沒有」的提醒是否用過
    empty_retries: int
    usage: dict
    response: Any            # 最近一次模型回應
    pending_nudge: str       # 守衛要求加進對話的提醒訊息；空字串 = 沒有
    result: AgentResult      # 一旦有值，圖就結束


def _make_config(system: str, temperature: float) -> types.GenerateContentConfig:
    thinking = os.getenv("GEMINI_THINKING_BUDGET")
    extra = {"thinking_config": types.ThinkingConfig(thinking_budget=int(thinking))} if thinking not in (None, "") else {}
    return types.GenerateContentConfig(
        system_instruction=system, temperature=temperature, tools=tools.GEMINI_TOOLS,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True), **extra)


def _result(state: dict, answer: str, reason: str, **kw) -> AgentResult:
    return AgentResult(answer, reason, state["tool_log"], guard_triggered=state["nudged"],
                       empty_retries=state["empty_retries"], usage=state["usage"], **kw)


def _text(state: dict) -> str:
    return (state["response"].text or "").strip()


def build_graph(client, model: str):
    """client 與 model 放在閉包裡，不進狀態（客戶端不能序列化）"""

    # ---------------------------------------------------------------- 節點
    def call_model(state: AgentState) -> dict:
        step, usage, empty_retries = state["step"] + 1, dict(state["usage"]), state["empty_retries"]

        def generate(temperature: float):
            resp = client.models.generate_content(model=model, contents=state["contents"], config=_make_config(state["system"], temperature))
            um = getattr(resp, "usage_metadata", None)
            usage["calls"] += 1
            if um:
                usage["prompt"] += getattr(um, "prompt_token_count", 0) or 0
                usage["cached"] += getattr(um, "cached_content_token_count", 0) or 0
                usage["output"] += getattr(um, "candidates_token_count", 0) or 0
                usage["thinking"] += getattr(um, "thoughts_token_count", 0) or 0
            return resp

        try:
            response = generate(0)
            for temp in EMPTY_RETRY_TEMPERATURES:
                if response.function_calls or (response.text or "").strip():
                    break
                empty_retries += 1
                cand = response.candidates[0] if getattr(response, "candidates", None) else None
                logger.warning("模型回傳空白答案（finish_reason=%s，content=%s），以 temperature=%s 重試",
                               getattr(cand, "finish_reason", None), str(getattr(cand, "content", None))[:120], temp)
                response = generate(temp)
        except Exception as e:
            logger.error(f"第 {step} 輪呼叫 Gemini 失敗：{type(e).__name__}: {e}")
            merged = {**state, "usage": usage, "empty_retries": empty_retries}
            return {"step": step, "usage": usage, "empty_retries": empty_retries,
                    "result": _result(merged, "抱歉，系統暫時無法處理這個問題，請稍後再試。", "error")}
        return {"step": step, "usage": usage, "empty_retries": empty_retries, "response": response}

    def run_tools(state: AgentState) -> dict:
        response = state["response"]
        # 必須把模型這一輪「原封不動」放回對話紀錄（內含函式呼叫與思考簽章）
        contents = state["contents"] + [response.candidates[0].content]
        seen, log, parts = list(state["seen"]), list(state["tool_log"]), []
        for call in response.function_calls:                 # 模型可能一輪要求多個工具，全部回覆、放同一個 user 訊息
            args = dict(call.args or {})
            signature = call.name + json.dumps(args, sort_keys=True, ensure_ascii=False)
            if signature in seen:
                result = {"error": "duplicate_call", "message": "已用相同參數查詢過，請直接使用先前的結果回答"}
            else:
                seen.append(signature)
                result = tools.execute_tool(call.name, args)
            log.append({"name": call.name, "args": args, "error": result.get("error") if isinstance(result, dict) else None, "result": result})
            parts.append(types.Part.from_function_response(name=call.name, response={"result": result}))
        return {"contents": contents + [types.Content(role="user", parts=parts)], "seen": seen, "tool_log": log}

    def guard_empty(state: AgentState) -> dict:
        if _text(state):
            return {}
        response = state["response"]
        cand = response.candidates[0] if getattr(response, "candidates", None) else None
        logger.warning("模型回傳空白答案：finish_reason=%s｜finish_message=%s｜prompt_feedback=%s｜候選數=%s",
                       getattr(cand, "finish_reason", None), getattr(cand, "finish_message", None),
                       getattr(response, "prompt_feedback", None), len(getattr(response, "candidates", None) or []))
        logger.warning("空白回覆的候選內容：%s", str(getattr(cand, "content", None))[:400])
        return {"result": _result(state, "抱歉，這次沒有產生有效的回答，請換個方式再問一次。", "empty")}

    def guard_ungrounded(state: AgentState) -> dict:
        if not has_unsourced_numbers(_text(state), state["tool_log"]):
            return {}
        logger.warning("答案含數字但沒有呼叫任何工具，疑似憑記憶捏造")
        if state["nudged"]:                                  # 已經提醒過一次還是這樣 → 放棄這個答案
            return {"result": _result(state, UNGROUNDED_ANSWER, "ungrounded")}
        return {"nudged": True, "pending_nudge": NUDGE}

    def guard_leak(state: AgentState) -> dict:
        if not leaks_system_prompt(_text(state), state["system"]):
            return {}
        logger.warning("答案與系統提示高度重複，已攔截")
        return {"result": _result(state, LEAK_REFUSAL, "done", leak_blocked=True)}

    def guard_docs(state: AgentState) -> dict:
        if state["docs_nudged"] or not claims_no_docs_without_search(_text(state), state["tool_log"]):
            return {}
        logger.warning("答案說文件庫沒有資料，但沒有呼叫過 search_documents，提醒後重試")
        return {"docs_nudged": True, "pending_nudge": NUDGE_DOCS}

    def nudge(state: AgentState) -> dict:
        contents = state["contents"] + [state["response"].candidates[0].content,
                                        types.Content(role="user", parts=[types.Part.from_text(text=state["pending_nudge"])])]
        return {"contents": contents, "pending_nudge": ""}

    def finish(state: AgentState) -> dict:
        logger.info(f"第 {state['step']} 輪完成，共呼叫工具 {len(state['tool_log'])} 次")
        return {"result": _result(state, _text(state), "done")}

    def max_steps(state: AgentState) -> dict:
        logger.warning(f"超過 {state['max_steps']} 輪仍未得到答案，中止")
        return {"result": _result(state, "這個問題需要的查詢步驟過多，我沒能在限制內完成。請把問題拆小一點再問。", "max_steps")}

    # ---------------------------------------------------------------- 邊
    def after_model(state: AgentState) -> str:
        if state.get("result") is not None:
            return END
        return "run_tools" if state["response"].function_calls else "guard_empty"

    def next_guard(following: str):
        def route(state: AgentState) -> str:
            if state.get("result") is not None:
                return END
            return "nudge" if state.get("pending_nudge") else following
        return route

    def retry_or_stop(state: AgentState) -> str:           # 回到 call_model 前先檢查步數（原本 for 迴圈的上限）
        return "call_model" if state["step"] < state["max_steps"] else "max_steps"

    g = StateGraph(AgentState)
    for name, fn in [("call_model", call_model), ("run_tools", run_tools), ("guard_empty", guard_empty),
                     ("guard_ungrounded", guard_ungrounded), ("guard_leak", guard_leak), ("guard_docs", guard_docs),
                     ("nudge", nudge), ("finish", finish), ("max_steps", max_steps)]:
        g.add_node(name, fn)
    g.add_edge(START, "call_model")
    g.add_conditional_edges("call_model", after_model, ["run_tools", "guard_empty", END])
    g.add_conditional_edges("run_tools", retry_or_stop, ["call_model", "max_steps"])
    g.add_conditional_edges("guard_empty", next_guard("guard_ungrounded"), ["guard_ungrounded", "nudge", END])
    g.add_conditional_edges("guard_ungrounded", next_guard("guard_leak"), ["guard_leak", "nudge", END])
    g.add_conditional_edges("guard_leak", next_guard("guard_docs"), ["guard_docs", "nudge", END])
    g.add_conditional_edges("guard_docs", next_guard("finish"), ["finish", "nudge", END])
    g.add_conditional_edges("nudge", retry_or_stop, ["call_model", "max_steps"])
    g.add_edge("finish", END)
    g.add_edge("max_steps", END)
    # checkpointer=False：明確不存檔。否則在外層聊天圖（chat.py）的節點裡呼叫時，會繼承外層的 checkpointer，
    # 把 Gemini 回應物件、思考簽章這些不能序列化的東西拿去存檔而出錯；內層本來就不需要被記住。
    return g.compile(checkpointer=False)


def initial_state(question: str, max_steps: int = MAX_STEPS, history: list | None = None) -> AgentState:
    """一次問答的起始狀態。run_agent_graph 與 trace.py 共用，確保「追蹤」跑的是同一個起點。"""
    system = SYSTEM_PROMPT.format(today=date.today().isoformat(), latest=_latest_period())
    return {
        "system": system, "contents": memory.to_contents(history) + [types.Content(role="user", parts=[types.Part.from_text(text=question)])],
        "tool_log": [], "seen": [], "step": 0, "max_steps": max_steps, "nudged": False, "docs_nudged": False,
        "empty_retries": 0, "usage": {"calls": 0, "prompt": 0, "cached": 0, "output": 0, "thinking": 0}, "pending_nudge": "",
    }


def run_config(max_steps: int) -> dict:
    # 預設 recursion_limit 是 25；每一步最多經過 call_model + 4 個守衛 + nudge，步數大時會誤觸，所以依步數放大
    return {"recursion_limit": max_steps * 8 + 10}


def run_agent_graph(question: str, client, model: str, max_steps: int = MAX_STEPS, history: list | None = None) -> AgentResult:
    """與 loop.run_agent 相同的簽名與回傳型別，可以直接互換；history 為 None 時行為與舊版完全一致。
    history：[{"q","a"},...]，先前幾輪的純文字問答（見 memory.py），放在這一輪的問題之前"""
    final = build_graph(client, model).invoke(initial_state(question, max_steps, history), config=run_config(max_steps))
    return final["result"]


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if "--mermaid" in sys.argv:
        print(build_graph(None, "").get_graph().draw_mermaid())
        sys.exit(0)
    tools.check_tools_in_sync()
    client, model = get_client()
    q = " ".join(sys.argv[1:]) or "台積電 2025Q4 的營收是多少？"
    result = run_agent_graph(q, client, model)
    print(f"\n問題：{q}\n")
    for i, c in enumerate(result.tool_calls, 1):
        print(f"  工具 {i}: {c['name']}({c['args']})" + (f" ❌{c['error']}" if c["error"] else ""))
    print(f"\n[{result.stop_reason}]\n{result.answer}")
