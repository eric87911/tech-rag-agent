"""介面用的純函式測試：離線、不碰 Gemini 與 API。"""
from agent_v2 import ui_format as F
from agent_v2.chat import TurnResult
from agent_v2.loop import AgentResult


def _turn(**kw):
    base = dict(answer="a", stop_reason="done", tools=[], guard_triggered=False, leak_blocked=False,
                remembered=True, history_len=2, usage={})
    base.update(kw)
    return TurnResult(**base)


def test_format_args_sorted_and_truncated():
    assert F.format_args({"item": "revenue", "code": "2330"}) == "code=2330, item=revenue"
    assert F.format_args({"q": "x" * 100}) == "q=" + "x" * F.MAX_VALUE_CHARS + "…"
    assert F.format_args(None) == ""


def test_tool_line_marks_error():
    ok = {"name": "get_ratio", "args": {"code": "2330"}, "error": None}
    bad = {"name": "get_ratio", "args": {"code": "9999"}, "error": "not_found"}
    assert F.tool_line(ok) == "✅ get_ratio(code=2330)"
    assert F.tool_line(bad) == "❌ get_ratio(code=9999) → not_found"


def test_result_preview_keeps_chinese_and_truncates():
    assert "台積電" in F.result_preview({"result": {"name": "台積電"}})
    long = F.result_preview({"result": "字" * 1000})
    assert len(long) == F.MAX_PREVIEW_CHARS + 1 and long.endswith("…")


def test_result_preview_survives_unserializable():
    assert "object" in F.result_preview({"result": object()})


def test_usage_line():
    assert F.usage_line({}) == ""
    assert F.usage_line({"calls": 0}) == ""
    line = F.usage_line({"calls": 2, "prompt": 100, "output": 20, "thinking": 5, "cached": 0})
    assert "呼叫 2 次" in line and "輸入 100" in line and "快取" not in line
    assert "快取 7" in F.usage_line({"calls": 1, "cached": 7})


def test_badges_normal_turn():
    b = F.badges(_turn(), elapsed=3.14)
    assert b[0] == "結果：完成"
    assert "記憶：已記住（共 2 輪）" in b
    assert "耗時 3.1 秒" in b
    assert not any("守衛" in x for x in b)


def test_badges_guard_and_blocked_turn():
    b = F.badges(_turn(stop_reason="ungrounded", guard_triggered=True, leak_blocked=True, remembered=False, history_len=1))
    assert "結果：數字來源無法確認" in b
    assert "守衛：已要求補查資料" in b and "守衛：已攔截洩漏" in b
    assert "記憶：未寫入（共 1 輪）" in b


def test_badges_unknown_stop_reason_shown_as_is():
    assert F.badges(_turn(stop_reason="weird"))[0] == "結果：weird"


def test_build_details_with_full_result():
    res = AgentResult("a", "done", tool_calls=[{"name": "search_company", "args": {"q": "台積電"}, "error": None, "result": [{"code": "2330"}]}],
                      usage={"calls": 1, "prompt": 10, "output": 2, "thinking": 0})
    d = F.build_details(_turn(tools=["search_company"]), res, 1.0)
    assert d["tools"][0]["line"] == "✅ search_company(q=台積電)"
    assert "2330" in d["tools"][0]["preview"]
    assert "呼叫 1 次" in d["usage"]


def test_build_details_without_full_result_falls_back_to_names():
    d = F.build_details(_turn(tools=["get_ranking"], usage={"calls": 3, "prompt": 1, "output": 1, "thinking": 0}))
    assert d["tools"] == [{"line": "✅ get_ranking", "preview": ""}]
    assert "呼叫 3 次" in d["usage"]


def test_history_to_messages():
    msgs = F.history_to_messages([{"q": "q1", "a": "a1"}, {"q": "q2", "a": "a2"}])
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert msgs[2]["content"] == "q2"
    assert F.history_to_messages(None) == []
