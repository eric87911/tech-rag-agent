"""多輪對話：外層聊天圖（掛 checkpointer、依 thread_id 區分對話），內層沿用 graph.run_agent_graph。

    python -m agent_v2.chat                 # 互動；/new 開新對話、/history 看記憶、/quit 離開

為什麼分兩層：checkpointer 每一步都會把「完整狀態」序列化。內層狀態含 Gemini 回應物件與思考簽章，
不是純資料、也不需要被記住；外層狀態只有字串與字典（問題、答案、問答歷史），才適合存。

    START → answer_and_remember → END

一個節點做兩件事（回答、決定要不要記住），因為中間產物是 AgentResult，不是純資料，不能放進會被存檔的狀態。
"""
import logging
import uuid
from dataclasses import dataclass, field
from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agent_v2 import graph, memory, tools
from agent_v2.loop import MAX_STEPS, get_client

logger = logging.getLogger(__name__)


class ChatState(TypedDict, total=False):
    """全部是純資料（可序列化）。history 之外的欄位每一輪都會被覆寫。"""
    history: list            # [{"q","a"},...]，唯一會跨輪累積的欄位
    question: str
    answer: str
    stop_reason: str
    tools: list              # 這一輪呼叫過的工具名稱
    guard_triggered: bool
    leak_blocked: bool
    remembered: bool
    usage: dict


@dataclass
class TurnResult:
    answer: str
    stop_reason: str
    tools: list = field(default_factory=list)
    guard_triggered: bool = False
    leak_blocked: bool = False
    remembered: bool = False
    history_len: int = 0
    usage: dict = field(default_factory=dict)


def build_chat_graph(client, model: str, checkpointer, max_steps: int = MAX_STEPS, on_result=None):
    def answer_and_remember(state: ChatState) -> dict:
        history = state.get("history", [])
        res = graph.run_agent_graph(state["question"], client, model, max_steps=max_steps, history=history)
        if on_result:
            on_result(res)          # 評估用：拿到完整的 AgentResult（含工具回傳內容）；狀態本身仍只存純資料
        keep = memory.should_remember(res)
        if not keep:
            logger.info("這一輪不寫入記憶（stop_reason=%s，leak_blocked=%s）", res.stop_reason, res.leak_blocked)
        return {"history": memory.remember(history, state["question"], res), "answer": res.answer, "stop_reason": res.stop_reason,
                "tools": [c["name"] for c in res.tool_calls], "guard_triggered": res.guard_triggered,
                "leak_blocked": res.leak_blocked, "remembered": keep, "usage": res.usage}

    g = StateGraph(ChatState)
    g.add_node("answer_and_remember", answer_and_remember)
    g.add_edge(START, "answer_and_remember")
    g.add_edge("answer_and_remember", END)
    return g.compile(checkpointer=checkpointer)


class Chat:
    """對外介面。checkpointer 可替換（預設存在記憶體，程式關掉就沒了）。"""

    def __init__(self, client, model: str, checkpointer=None, max_steps: int = MAX_STEPS, on_result=None):
        self._graph = build_chat_graph(client, model, checkpointer or MemorySaver(), max_steps, on_result)

    @staticmethod
    def new_thread() -> str:
        return uuid.uuid4().hex[:8]

    def ask(self, question: str, thread_id: str) -> TurnResult:
        cfg = {"configurable": {"thread_id": thread_id}}
        # 只傳 question：history 由 checkpointer 還原；其餘欄位都會被這一輪覆寫，不會殘留上一輪的內容
        out = self._graph.invoke({"question": question}, config=cfg)
        return TurnResult(answer=out["answer"], stop_reason=out["stop_reason"], tools=out["tools"], guard_triggered=out["guard_triggered"],
                          leak_blocked=out["leak_blocked"], remembered=out["remembered"], history_len=len(out["history"]), usage=out["usage"])

    def history(self, thread_id: str) -> list:
        snap = self._graph.get_state({"configurable": {"thread_id": thread_id}})
        return list((snap.values or {}).get("history", []))


def main():
    logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    tools.check_tools_in_sync()
    client, model = get_client()
    chat = Chat(client, model)
    tid = chat.new_thread()
    print("多輪對話（/new 開新對話｜/history 看目前記住的內容｜/quit 離開）")
    while True:
        try:
            q = input(f"\n[{tid}] 你：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q in ("/quit", "/exit"):
            break
        if q == "/new":
            tid = chat.new_thread()
            print(f"已開新對話 {tid}")
            continue
        if q == "/history":
            for i, t in enumerate(chat.history(tid), 1):
                print(f"  {i}. 問：{t['q']}\n     答：{t['a'][:80]}{'…' if len(t['a']) > 80 else ''}")
            continue
        try:
            r = chat.ask(q, tid)
        except Exception as e:
            logger.error("這一輪失敗：%s: %s", type(e).__name__, e)
            print("抱歉，這一輪發生錯誤，沒有寫入記憶。")
            continue
        print(f"\n{r.answer}\n\n  [工具：{', '.join(r.tools) or '無'}｜{'已記住' if r.remembered else '未寫入記憶'}｜記憶 {r.history_len} 輪｜{r.stop_reason}]")


if __name__ == "__main__":
    main()
