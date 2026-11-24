"""app_v2.py 冒煙測試：用 Streamlit 的 AppTest 在記憶體裡跑，Chat、API、Gemini 全部換成假的。
驗證的是「介面接線」（輸入 → 顯示答案與細節 → 記憶面板 → 錯誤處理），不是答案品質。"""
import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = st_testing.AppTest

import streamlit as st  # noqa: E402

from agent_v2 import chat as chat_mod, loop, tools  # noqa: E402
from agent_v2.loop import AgentResult  # noqa: E402

APP = str(__import__("pathlib").Path(__file__).resolve().parent.parent / "app_v2.py")


class FakeChat:
    fail = False
    stores: dict = {}          # thread_id -> history；跨 FakeChat 實例共用，模擬 SQLite
    counter = 0                # 編號只增不減（真的 new_thread 是 uuid，刪除後不會撞號）

    def __init__(self, client, model, checkpointer=None, max_steps=6, on_result=None):
        self.on_result = on_result

    @staticmethod
    def new_thread():
        FakeChat.counter += 1
        return "t%d" % FakeChat.counter

    def ask(self, question, thread_id):
        if FakeChat.fail:
            raise RuntimeError("boom")
        hist = FakeChat.stores.setdefault(thread_id, [])
        res = AgentResult(f"答：{question}", "done",
                          tool_calls=[{"name": "get_financials", "args": {"code": "2330"}, "error": None, "result": {"v": 1}}],
                          usage={"calls": 1, "prompt": 5, "output": 2, "thinking": 0})
        if self.on_result:
            self.on_result(res)
        hist.append({"q": question, "a": res.answer})
        return chat_mod.TurnResult(answer=res.answer, stop_reason="done", tools=["get_financials"], remembered=True,
                                   history_len=len(hist), usage=res.usage)

    def history(self, thread_id):
        return list(FakeChat.stores.get(thread_id, []))

    def forget(self, thread_id):
        FakeChat.stores.pop(thread_id, None)


class FakeResp:
    def raise_for_status(self):
        pass

    def json(self):
        return {"latest_period": "2025Q4"}


@pytest.fixture
def app(monkeypatch):
    FakeChat.fail = False
    FakeChat.stores = {}
    st.cache_resource.clear()
    monkeypatch.setattr(chat_mod, "Chat", FakeChat)
    monkeypatch.setattr(chat_mod, "open_sqlite_saver", lambda path: object())
    monkeypatch.setattr(loop, "get_client", lambda: (object(), "fake-model"))
    monkeypatch.setattr(tools, "check_tools_in_sync", lambda: None)
    import httpx
    monkeypatch.setattr(httpx, "get", lambda *a, **k: FakeResp())
    return AppTest.from_file(APP, default_timeout=30)


def _texts(at):
    return " ".join(m.value for m in at.markdown)


def test_first_load_shows_api_ok_and_empty_state(app):
    at = app.run()
    assert not at.exception
    assert any("資料 API 正常" in s.value for s in at.success)
    assert at.chat_input[0].disabled is False


def test_question_shows_answer_details_and_memory(app):
    at = app.run()
    at.chat_input[0].set_value("台積電營收").run()
    assert not at.exception
    text = _texts(at)
    assert "答：台積電營收" in text
    assert "get_financials(code=2330)" in text          # 運作細節裡的工具呼叫
    assert "記憶：已記住（共 1 輪）" in text
    assert "台積電營收" in " ".join(m.value for m in at.sidebar.markdown)   # 「目前記住的內容」面板


def test_failed_turn_is_not_kept_on_screen(app):
    at = app.run()
    FakeChat.fail = True
    at.chat_input[0].set_value("會失敗的問題").run()
    assert any("沒有寫入記憶" in e.value for e in at.error)
    assert "會失敗的問題" not in " ".join(m.value for m in at.sidebar.markdown)
    assert all(m["content"] != "會失敗的問題" for m in at.session_state["messages"])


def test_resume_restores_history(app):
    FakeChat.stores["old1"] = [{"q": "舊問題", "a": "舊答案"}]
    at = app.run()
    at.sidebar.text_input(key="resume_id").set_value("old1")
    at.sidebar.button[0].click().run()
    assert at.session_state["thread_id"] == "old1"
    assert "舊問題" in _texts(at) and "舊答案" in _texts(at)


def test_forget_deletes_thread_and_starts_new(app):
    at = app.run()
    at.chat_input[0].set_value("要被刪的").run()
    tid = at.session_state["thread_id"]
    assert FakeChat.stores.get(tid)
    forget = [b for b in at.sidebar.button if b.label == "刪除此對話"][0]
    forget.click().run()
    assert tid not in FakeChat.stores
    assert at.session_state["thread_id"] != tid and at.session_state["messages"] == []


def test_api_down_disables_input(app, monkeypatch):
    import httpx

    def down(*a, **k):
        raise httpx.ConnectError("refused")
    monkeypatch.setattr(httpx, "get", down)
    at = app.run()
    assert any("資料 API 無法連線" in e.value for e in at.sidebar.error)
    assert at.chat_input[0].disabled is True
