"""SqliteSaver 持久化的離線測試：換掉 checkpointer，其他邏輯不動；「重開」用關檔再開一個新的 saver 模擬行程重啟。"""
import sqlite3
import threading

import pytest

from agent_v2 import chat as chat_mod
from agent_v2 import tools
from test_chat import FakeClient, Resp, call  # noqa: E402


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv("GEMINI_THINKING_BUDGET", raising=False)


def session(db, script):
    """模擬一次『行程』：開檔、建 Chat。回傳 (chat, client, saver)。"""
    saver = chat_mod.open_sqlite_saver(db)
    c = FakeClient(script)
    return chat_mod.Chat(c, "m", checkpointer=saver), c, saver


def test_history_survives_restart_and_next_turn_sees_it(tmp_path):
    db = tmp_path / "sub" / "chat.db"            # 資料夾不存在也要能建立
    chat, _, saver = session(db, [Resp("答案一")])
    chat.ask("問題一", "t1")
    chat_mod.close_sqlite_saver(saver)

    chat2, c2, saver2 = session(db, [Resp("答案二")])
    assert chat2.history("t1") == [{"q": "問題一", "a": "答案一"}]
    r = chat2.ask("那第二題呢", "t1")
    assert c2.seen == [3] and r.history_len == 2          # 模型看到：舊問、舊答、新問
    chat_mod.close_sqlite_saver(saver2)


def test_threads_stay_isolated_across_restart(tmp_path):
    db = tmp_path / "c.db"
    chat, _, saver = session(db, [Resp("a"), Resp("b")])
    chat.ask("問A", "tA")
    chat.ask("問B", "tB")
    chat_mod.close_sqlite_saver(saver)
    chat2, c2, saver2 = session(db, [Resp("c")])
    assert [t["q"] for t in chat2.history("tA")] == ["問A"] and [t["q"] for t in chat2.history("tB")] == ["問B"]
    chat2.ask("再問A", "tA")
    assert c2.seen == [3]                                  # 只看到 tA 自己的歷史，沒有混入 tB
    chat_mod.close_sqlite_saver(saver2)


def test_unremembered_turns_stay_unremembered_after_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "execute_tool", lambda n, a: {"error": "service_error", "message": "x"})
    db = tmp_path / "c.db"
    chat, _, saver = session(db, [Resp(calls=[call("search_documents", query="x")]), Resp("文件檢索暫時無法使用")])
    r = chat.ask("問", "t")
    assert not r.remembered
    chat_mod.close_sqlite_saver(saver)
    chat2, _, saver2 = session(db, [])
    assert chat2.history("t") == []
    chat_mod.close_sqlite_saver(saver2)


def test_forget_deletes_one_thread_only_and_it_persists(tmp_path):
    db = tmp_path / "c.db"
    chat, _, saver = session(db, [Resp("a"), Resp("b")])
    chat.ask("問A", "tA")
    chat.ask("問B", "tB")
    chat.forget("tA")
    chat_mod.close_sqlite_saver(saver)
    chat2, _, saver2 = session(db, [])
    assert chat2.history("tA") == [] and len(chat2.history("tB")) == 1
    chat_mod.close_sqlite_saver(saver2)


def test_stored_data_is_plain_text_only(tmp_path):
    """資料庫裡不該有工具結果或 Gemini 物件；我們只存問答文字。"""
    db = tmp_path / "c.db"
    chat, _, saver = session(db, [Resp(calls=[call("get_ratio", code="2303")]), Resp("答案")])
    tools_called = []
    import agent_v2.tools as T
    orig = T.execute_tool
    T.execute_tool = lambda n, a: (tools_called.append(n), {"data": [{"SECRET_TOOL_PAYLOAD": 1}]})[1]
    try:
        chat.ask("問", "t")
    finally:
        T.execute_tool = orig
    chat_mod.close_sqlite_saver(saver)
    raw = b"".join(bytes(r[0]) for r in sqlite3.connect(db).execute("select checkpoint from checkpoints") if r[0])
    assert b"SECRET_TOOL_PAYLOAD" not in raw and tools_called == ["get_ratio"]


def test_works_from_another_thread(tmp_path):
    """check_same_thread=False 要生效：在別的執行緒呼叫不會丟 ProgrammingError。"""
    chat, _, saver = session(tmp_path / "c.db", [Resp("a")])
    out = []
    t = threading.Thread(target=lambda: out.append(chat.ask("問", "t")))
    t.start(); t.join()
    assert out and out[0].remembered
    chat_mod.close_sqlite_saver(saver)


def test_memory_default_still_works_without_sqlite():
    c = FakeClient([Resp("a"), Resp("b")])
    chat = chat_mod.Chat(c, "m")
    chat.ask("1", "t"); chat.ask("2", "t")
    assert c.seen == [1, 3]
