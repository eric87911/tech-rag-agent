"""追蹤一個問題實際走過哪些節點，輸出文字清單與 Mermaid 流程圖（可直接貼進 README）。

    python -m agent_v2.trace "台積電 2025Q4 的營收是多少？"
    python -m agent_v2.trace "台積電毛利率為什麼一直上升？"
    python -m agent_v2.trace "台積電 2025Q4 的營收是多少？" --then "那毛利率呢？"   # 第二題帶著第一題的記憶

跟 `python -m agent_v2.graph` 一樣會呼叫 Gemini 並計費。追蹤的是「真的走過的路徑」，不是從程式推導出來的預期路徑：
用 LangGraph 的 stream_mode="updates"，每個節點執行完就吐出它更新了什麼。
這裡的邏輯（把更新整理成一行說明、組 Mermaid）是純函式，可離線測試；只有 `trace_graph` 會真的跑圖。
"""
import argparse
import logging

from agent_v2 import graph, memory, tools, ui_format
from agent_v2.loop import MAX_STEPS, get_client

NODE_CLASS = {"call_model": "model", "run_tools": "model", "nudge": "guard", "max_steps": "done", "finish": "done"}


def describe_update(node: str, update, tool_log_before: int, state_tool_log: list) -> str:
    """一個節點執行完的一行說明。update 為 None 或 {} 代表節點什麼都沒改（守衛通過）。"""
    update = update or {}
    result = update.get("result")
    if node == "call_model":
        if result is not None:
            return f"呼叫 Gemini 失敗（{result.stop_reason}）"
        calls = update["response"].function_calls
        step = update.get("step", "?")
        if calls:
            names = "、".join(f"{c.name}({ui_format.format_args(dict(c.args or {}))})" for c in calls)
            return f"第 {step} 次呼叫：要求工具 {names}"
        return f"第 {step} 次呼叫：給出文字答案"
    if node == "run_tools":
        new = state_tool_log[tool_log_before:]
        return "；".join(ui_format.tool_line(c) for c in new) or "沒有新的工具呼叫"
    if node.startswith("guard_"):
        if update.get("pending_nudge"):
            return "未通過：要求補查後重試"
        if result is not None:
            return f"未通過：放棄這個答案（{result.stop_reason}）"
        return "通過"
    if node == "nudge":
        return "把提醒加進對話，回到 call_model"
    if node == "finish":
        return "全部守衛通過，回傳答案"
    if node == "max_steps":
        return "步數用完，中止"
    return ""


def trace_graph(question: str, client, model: str, max_steps: int = MAX_STEPS, history: list | None = None):
    """跑一次，回傳 (步驟列表, AgentResult)。步驟：{"i","node","note"}"""
    g = graph.build_graph(client, model)
    state_tool_log: list = []
    steps, final = [], None
    for chunk in g.stream(graph.initial_state(question, max_steps, history), config=graph.run_config(max_steps), stream_mode="updates"):
        for node, update in chunk.items():
            before = len(state_tool_log)
            if update and "tool_log" in update:
                state_tool_log = list(update["tool_log"])
            steps.append({"i": len(steps) + 1, "node": node, "note": describe_update(node, update, before, state_tool_log)})
            if update and update.get("result") is not None:
                final = update["result"]
    return steps, final


def _label(text: str) -> str:
    """Mermaid 節點文字：雙引號、角括號會破壞語法，換成安全的字元"""
    return str(text).replace('"', "'").replace("<", "＜").replace(">", "＞")


def to_mermaid(question: str, steps: list, answer: str, max_answer_chars: int = 60) -> str:
    """實際走過的路徑畫成由上而下的流程圖。節點依序編號，同一個節點走過兩次會出現兩次。"""
    short = answer.strip().replace("\n", " ")
    if len(short) > max_answer_chars:
        short = short[:max_answer_chars] + "…"
    lines = ["flowchart TD", f'    q(["問題<br/>{_label(question)}"])']
    prev = "q"
    for s in steps:
        nid = f"n{s['i']}"
        cls = NODE_CLASS.get(s["node"], "guard" if s["node"].startswith("guard_") else "model")
        lines.append(f'    {nid}["{s["i"]}. {s["node"]}<br/>{_label(s["note"])}"]:::{cls}')
        lines.append(f"    {prev} --> {nid}")
        prev = nid
    lines.append(f'    a(["答案<br/>{_label(short)}"])')
    lines.append(f"    {prev} --> a")
    lines += ["    classDef model fill:#e0f2fe,stroke:#0369a1,color:#0c4a6e",
              "    classDef guard fill:#ffedd5,stroke:#c2410c,color:#7c2d12",
              "    classDef done fill:#f3f4f6,stroke:#6b7280,color:#111827"]
    return "\n".join(lines)


def to_text(steps: list) -> str:
    return "\n".join(f"{s['i']:>2}. {s['node']:<17} {s['note']}" for s in steps)


def render(question: str, steps: list, result) -> str:
    return (f"問題：{question}\n\n{to_text(steps)}\n\n結果：{result.stop_reason}\n答案：{result.answer}\n\n"
            f"```mermaid\n{to_mermaid(question, steps, result.answer)}\n```\n")


def main():
    ap = argparse.ArgumentParser(description="追蹤一個問題走過哪些節點（會呼叫 Gemini 並計費）")
    ap.add_argument("question")
    ap.add_argument("--then", help="接著問的第二題：會帶著第一題的問答當記憶，一併追蹤")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    tools.check_tools_in_sync()
    client, model = get_client()

    steps, res = trace_graph(args.question, client, model)
    print(render(args.question, steps, res))
    if args.then:
        history = memory.remember([], args.question, res)
        if not history:
            print("（第一題沒有被寫入記憶，所以第二題不帶歷史；原因見上面的結果）\n")
        steps2, res2 = trace_graph(args.then, client, model, history=history)
        print(f"（第二題帶著 {len(history)} 輪記憶）\n")
        print(render(args.then, steps2, res2))


if __name__ == "__main__":
    main()
