"""多輪對話：外層聊天圖（掛 checkpointer、依 thread_id 區分對話），內層沿用 graph.run_agent_graph。

    python -m agent_v2.chat                 # 互動；/new 開新對話、/history 看記憶、/quit 離開（記憶只在這次執行中）
    python -m agent_v2.chat --db data/chat_memory.db            # 對話存進 SQLite，下次還在
    python -m agent_v2.chat --db data/chat_memory.db --thread abc123   # 接續某個對話；/forget 刪除目前對話

為什麼分兩層：checkpointer 每一步都會把「完整狀態」序列化。內層狀態含 Gemini 回應物件與思考簽章，
不是純資料、也不需要被記住；外層狀態只有字串與字典（問題、答案、問答歷史），才適合存。

    START → answer_and_remember → END

一個節點做兩件事（回答、決定要不要記住），因為中間產物是 AgentResult，不是純資料，不能放進會被存檔的狀態。
"""
import argparse
import logging
import sqlite3
import uuid
from pathlib import Path
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


def open_sqlite_saver(path):
    """開（必要時建立）一個 SQLite 檔當 checkpointer，對話重啟後還在。呼叫端負責在結束時 close_sqlite_saver。

    check_same_thread=False：LangGraph 可能在別的執行緒讀寫（例如 API 的 worker 執行緒）。
    WAL + busy_timeout：多個行程同時寫同一個檔案時，先等鎖而不是立刻報 database is locked。
    套件延遲載入：只用記憶體版的人不需要裝 langgraph-checkpoint-sqlite。"""
    from langgraph.checkpoint.sqlite import SqliteSaver
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    return SqliteSaver(conn)


def close_sqlite_saver(saver) -> None:
    try:
        saver.conn.close()
    except Exception as e:      # 關檔失敗不該讓程式在結束時再炸一次
        logger.warning("關閉對話資料庫失敗：%s: %s", type(e).__name__, e)


class Chat:
    """對外介面。checkpointer 可替換（預設存在記憶體，程式關掉就沒了；要持久化就傳 open_sqlite_saver(...)）。"""

    def __init__(self, client, model: str, checkpointer=None, max_steps: int = MAX_STEPS, on_result=None):
        self._checkpointer = checkpointer or MemorySaver()
        self._graph = build_chat_graph(client, model, self._checkpointer, max_steps, on_result)

    @staticmethod
    def new_thread() -> str:
        return uuid.uuid4().hex[:8]

    def ask(self, question: str, thread_id: str) -> TurnResult:
        cfg = {"configurable": {"thread_id": thread_id}}
        # 只傳 question：history 由 checkpointer 還原；其餘欄位都會被這一輪覆寫，不會殘留上一輪的內容
        out = self._graph.invoke({"question": question}, config=cfg)
        return TurnResult(answer=out["answer"], stop_reason=out["stop_reason"], tools=out["tools"], guard_triggered=out["guard_triggered"],
                          leak_blocked=out["leak_blocked"], remembered=out["remembered"], history_len=len(out["history"]), usage=out["usage"])

    def forget(self, thread_id: str) -> None:
        """刪除某個對話的全部紀錄（使用者要求清除時用）。"""
        self._checkpointer.delete_thread(thread_id)

    def history(self, thread_id: str) -> list:
        snap = self._graph.get_state({"configurable": {"thread_id": thread_id}})
        return list((snap.values or {}).get("history", []))


def main():
    ap = argparse.ArgumentParser(description="多輪對話")
    ap.add_argument("--db", help="把對話存進這個 SQLite 檔（沒給就只存在記憶體，結束即消失）")
    ap.add_argument("--thread", help="接續這個對話編號（要搭配 --db）")
    args = ap.parse_args()
    if args.thread and not args.db:
        ap.error("--thread 要搭配 --db 才有意義（記憶體版重開就沒有舊對話）")
    logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    tools.check_tools_in_sync()
    client, model = get_client()
    saver = open_sqlite_saver(args.db) if args.db else None
    chat = Chat(client, model, checkpointer=saver)
    tid = args.thread or chat.new_thread()
    if args.db:
        print(f"對話會存在 {args.db}（內含你的提問與答案；/forget 可刪除目前對話）")
        if args.thread:
            print(f"接續對話 {tid}，目前記得 {len(chat.history(tid))} 輪")
    print("多輪對話（/new 開新對話｜/history 看目前記住的內容｜/forget 刪除目前對話｜/quit 離開）")
    try:
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
            if q == "/forget":
                chat.forget(tid)
                print(f"已刪除對話 {tid} 的紀錄")
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
    finally:
        if saver is not None:
            close_sqlite_saver(saver)
            print(f"（下次接續：--db {args.db} --thread {tid}）")


if __name__ == "__main__":
    main()
