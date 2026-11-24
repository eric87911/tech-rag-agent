"""差分測試：同一份假模型腳本，分別餵給 loop.run_agent 與 graph.run_agent_graph，所有可觀察的輸出都必須相同。
不呼叫 Gemini、不連 API。"""
from types import SimpleNamespace as NS

import pytest

from agent_v2 import graph, loop, tools


class Resp:
    def __init__(self, text=None, calls=None, usage=(10, 2, 5, 3)):
        self.function_calls = calls or None
        self.text = text
        self.candidates = [NS(content=("model-content", id(self)), finish_reason="STOP", finish_message=None)]
        self.prompt_feedback = None
        self.usage_metadata = NS(prompt_token_count=usage[0], cached_content_token_count=usage[1],
                                 candidates_token_count=usage[2], thoughts_token_count=usage[3])


def call(name, **args):
    return NS(name=name, args=args)


class FakeClient:
    def __init__(self, script):
        self.script, self.seen = list(script), []
        self.models = NS(generate_content=self._gen)

    def _gen(self, model, contents, config):
        self.seen.append((len(contents), config.temperature))         # 模型每次「看到的對話長度」與 temperature
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(tools, "execute_tool", lambda name, args: {"echo": name, "args": args})
    monkeypatch.setattr(loop, "_latest_period", lambda: "2025Q4")
    monkeypatch.setattr(graph, "_latest_period", lambda: "2025Q4")
    monkeypatch.delenv("GEMINI_THINKING_BUDGET", raising=False)


def observe(res, client):
    return {"answer": res.answer, "stop": res.stop_reason, "guard": res.guard_triggered, "leak": res.leak_blocked,
            "empty": res.empty_retries, "usage": res.usage,
            "tools": [(c["name"], c["args"], c["error"], c["result"]) for c in res.tool_calls], "model_saw": client.seen}


def both(make_script, **kw):
    c1, c2 = FakeClient(make_script()), FakeClient(make_script())
    a = observe(loop.run_agent("問題", c1, "m", **kw), c1)
    b = observe(graph.run_agent_graph("問題", c2, "m", **kw), c2)
    assert a == b, f"\nloop : {a}\ngraph: {b}"
    assert not c1.script and not c2.script, "腳本沒用完：兩邊呼叫模型的次數與預期不同"
    return a


def test_direct_answer():
    r = both(lambda: [Resp("你好，我是財報助理。")])
    assert r["stop"] == "done" and r["tools"] == []


def test_tool_then_answer():
    r = both(lambda: [Resp(calls=[call("get_financials", code="2330")]), Resp("營收請見工具回傳。")])
    assert r["stop"] == "done" and len(r["tools"]) == 1 and r["model_saw"] == [(1, 0), (3, 0)]


def test_parallel_tool_calls_in_one_turn():
    r = both(lambda: [Resp(calls=[call("a", x=1), call("b", y=2)]), Resp("完成")])
    assert [t[0] for t in r["tools"]] == ["a", "b"]


def test_numbers_without_tools_nudge_then_tool():
    r = both(lambda: [Resp("毛利率是 59.2%"), Resp(calls=[call("get_ratio", code="2330")]), Resp("查詢後的答案")])
    assert r["guard"] is True and r["stop"] == "done"


def test_numbers_without_tools_twice_is_ungrounded():
    r = both(lambda: [Resp("毛利率是 59.2%"), Resp("營收 1.05 兆元")])
    assert r["stop"] == "ungrounded"


def test_empty_response_retries_then_text():
    r = both(lambda: [Resp(""), Resp(None), Resp("終於有答案")])
    assert r["empty"] == 2 and r["stop"] == "done" and r["model_saw"] == [(1, 0), (1, 0.3), (1, 0.7)]


def test_all_empty_gives_empty_result():
    r = both(lambda: [Resp(""), Resp(""), Resp("")])
    assert r["stop"] == "empty" and r["empty"] == 2


def test_gemini_exception_becomes_error():
    r = both(lambda: [RuntimeError("503 UNAVAILABLE")])
    assert r["stop"] == "error"


def test_exception_after_empty_retry_keeps_counts():
    r = both(lambda: [Resp(""), RuntimeError("boom")])
    assert r["stop"] == "error" and r["empty"] == 1 and r["usage"]["calls"] == 1   # 例外發生在呼叫本身，失敗的那次不計入


def test_leak_is_blocked():
    r = both(lambda: [Resp(calls=[call("get_ratio", code="2330")]), Resp("我有 search_company 與 get_financials 這些工具。")])
    assert r["leak"] is True


def test_docs_claim_without_search_nudges_once():
    r = both(lambda: [Resp(calls=[call("get_financials", code="2303")]), Resp("文件庫中沒有相關資料。"), Resp("文件庫中沒有相關資料。")])
    assert r["stop"] == "done" and r["model_saw"] == [(1, 0), (3, 0), (5, 0)]


def test_duplicate_call_is_flagged():
    r = both(lambda: [Resp(calls=[call("get_ratio", code="2330")]), Resp(calls=[call("get_ratio", code="2330")]), Resp("好")])
    assert r["tools"][1][2] == "duplicate_call"


def test_max_steps_after_tool_round():
    r = both(lambda: [Resp(calls=[call("t", i=i)]) for i in range(6)])
    assert r["stop"] == "max_steps" and len(r["tools"]) == 6


def test_max_steps_small_limit():
    r = both(lambda: [Resp(calls=[call("t", i=i)]) for i in range(2)], max_steps=2)
    assert r["stop"] == "max_steps"


def test_nudge_on_last_step_counts_as_max_steps():
    r = both(lambda: [Resp("毛利率是 59.2%")], max_steps=1)
    assert r["stop"] == "max_steps" and r["guard"] is True


def test_large_step_limit_does_not_hit_recursion_limit():
    r = both(lambda: [Resp(calls=[call("t", i=i)]) for i in range(40)], max_steps=40)
    assert r["stop"] == "max_steps"


def test_thinking_budget_env_is_applied(monkeypatch):
    monkeypatch.setenv("GEMINI_THINKING_BUDGET", "0")
    cfg = graph._make_config("sys", 0.3)
    assert cfg.thinking_config.thinking_budget == 0 and cfg.temperature == 0.3
    assert cfg.automatic_function_calling.disable is True


def test_graph_structure_has_expected_nodes():
    mermaid = graph.build_graph(None, "m").get_graph().draw_mermaid()
    for node in ("call_model", "run_tools", "guard_empty", "guard_ungrounded", "guard_leak", "guard_docs", "nudge", "finish", "max_steps"):
        assert node in mermaid


def test_history_is_placed_before_the_new_question():
    history = [{"q": "台積電 2025Q4 營收？", "a": "1.05 兆元"}, {"q": "那毛利率？", "a": "59%"}]
    c = FakeClient([Resp("好的")])
    res = graph.run_agent_graph("那聯電呢？", c, "m", history=history)
    assert res.stop_reason == "done" and c.seen == [(5, 0)]          # 2 輪 × 2 則 + 新問題


def test_history_none_or_empty_is_identical_to_no_history():
    for h in (None, []):
        c1, c2 = FakeClient([Resp("你好")]), FakeClient([Resp("你好")])
        a = observe(graph.run_agent_graph("問題", c1, "m"), c1)
        b = observe(graph.run_agent_graph("問題", c2, "m", history=h), c2)
        assert a == b


def test_stale_numbers_from_history_still_trip_the_unsourced_guard():
    # 使用者追問，模型直接沿用上一輪的數字、沒有重新查 → 守衛要攔下並要求重查（記憶不能繞過「數字必須是這一輪查來的」）
    history = [{"q": "台積電毛利率？", "a": "台積電 2025Q4 毛利率為 59.2%"}]
    c = FakeClient([Resp("毛利率是 59.2%"), Resp(calls=[call("get_ratio", code="2330")]), Resp("已重新查詢")])
    res = graph.run_agent_graph("剛剛那個數字再說一次", c, "m", history=history)
    assert res.guard_triggered is True and len(res.tool_calls) == 1 and res.stop_reason == "done"
