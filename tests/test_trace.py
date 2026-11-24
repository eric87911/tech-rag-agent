"""追蹤腳本測試：同一套假 Gemini 腳本，確認「走過的節點」與說明、Mermaid 語法。離線，不呼叫 Gemini 與 API。"""
import re
from types import SimpleNamespace as NS

import pytest

from agent_v2 import graph, loop, memory, tools, trace
from test_graph import FakeClient, Resp, call  # noqa: E402  （同一個 tests 目錄，假模型腳本沿用差分測試的）


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(tools, "execute_tool", lambda name, args: {"echo": name, "args": args})
    monkeypatch.setattr(graph, "_latest_period", lambda: "2025Q4")
    monkeypatch.delenv("GEMINI_THINKING_BUDGET", raising=False)


def nodes(steps):
    return [s["node"] for s in steps]


GUARDS = ["guard_empty", "guard_ungrounded", "guard_leak", "guard_docs"]


def test_tool_then_answer_path():
    c = FakeClient([Resp(calls=[call("get_financials", code="2330", item="revenue")]), Resp("台積電營收如工具所示")])
    steps, res = trace.trace_graph("問題", c, "m")
    assert nodes(steps) == ["call_model", "run_tools", "call_model"] + GUARDS + ["finish"]
    assert res.stop_reason == "done"
    assert "get_financials(code=2330, item=revenue)" in steps[0]["note"]
    assert steps[1]["note"].startswith("✅ get_financials(code=2330, item=revenue)")
    assert steps[2]["note"].endswith("給出文字答案")
    assert all(s["note"] == "通過" for s in steps if s["node"].startswith("guard_"))
    assert [s["i"] for s in steps] == list(range(1, len(steps) + 1))


def test_parallel_tools_listed_in_one_step():
    c = FakeClient([Resp(calls=[call("search_company", q="台積電"), call("get_ratio", code="2330", metric="gross_margin")]), Resp("答案")])
    steps, _ = trace.trace_graph("問題", c, "m")
    assert steps[1]["note"].count("✅") == 2


def test_nudge_path_shows_failed_guard_and_loop_back():
    c = FakeClient([Resp("毛利率是 59.2%"), Resp(calls=[call("get_ratio", code="2330")]), Resp("查詢後的答案")])
    steps, res = trace.trace_graph("問題", c, "m")
    n = nodes(steps)
    assert n[:4] == ["call_model", "guard_empty", "guard_ungrounded", "nudge"]
    assert steps[2]["note"] == "未通過：要求補查後重試"
    assert n[4] == "call_model" and n.count("call_model") == 3 and n[-1] == "finish"
    assert res.guard_triggered is True


def test_guard_giving_up_ends_without_finish():
    c = FakeClient([Resp("59.2%"), Resp("還是 59.2%")])
    steps, res = trace.trace_graph("問題", c, "m")
    assert nodes(steps)[-1] == "guard_ungrounded" and "放棄" in steps[-1]["note"] and res.stop_reason == "ungrounded"


def test_gemini_error_is_one_step():
    c = FakeClient([RuntimeError("boom")])
    steps, res = trace.trace_graph("問題", c, "m")
    assert nodes(steps) == ["call_model"] and "失敗" in steps[0]["note"] and res.stop_reason == "error"


def test_max_steps_path():
    c = FakeClient([Resp(calls=[call("get_ratio", code="2330")])] * 2)
    steps, res = trace.trace_graph("問題", c, "m", max_steps=1)
    assert nodes(steps) == ["call_model", "run_tools", "max_steps"] and res.stop_reason == "max_steps"


def test_trace_agrees_with_run_agent_graph():
    script = lambda: [Resp(calls=[call("get_ratio", code="2330")]), Resp("答案")]
    steps, res = trace.trace_graph("問題", FakeClient(script()), "m")
    ref = graph.run_agent_graph("問題", FakeClient(script()), "m")
    assert (res.answer, res.stop_reason, res.tool_calls, res.usage) == (ref.answer, ref.stop_reason, ref.tool_calls, ref.usage)


def test_history_is_passed_through():
    c = FakeClient([Resp("答案")])
    hist = [{"q": "上一題", "a": "上一答"}]
    trace.trace_graph("追問", c, "m", history=hist)
    assert c.seen[0][0] == 3          # 模型看到：上一題、上一答、追問


def test_mermaid_structure_and_escaping():
    steps = [{"i": 1, "node": "call_model", "note": '呼叫 "x" <b>'}, {"i": 2, "node": "guard_empty", "note": "通過"},
             {"i": 3, "node": "finish", "note": "完成"}]
    m = trace.to_mermaid('問題 "含引號"', steps, "答案" * 100)
    assert m.startswith("flowchart TD")
    body = m.split("classDef")[0]
    assert '"x"' not in body and "<b>" not in body                      # 引號、角括號已換掉（<br/> 是我們自己加的）
    assert "n1" in m and "n2" in m and "n3" in m and "q --> n1" in m and "n3 --> a" in m
    assert ":::model" in m and ":::guard" in m and ":::done" in m
    assert "classDef end" not in m                                        # end 是 Mermaid 保留字
    assert "…" in m                                                       # 長答案被截斷


def test_mermaid_ids_unique_when_node_repeats():
    c = FakeClient([Resp(calls=[call("get_ratio", code="2330")]), Resp("答案")])
    steps, res = trace.trace_graph("問題", c, "m")
    m = trace.to_mermaid("問題", steps, res.answer)
    ids = re.findall(r"^    (n\d+)\[", m, flags=re.M)
    assert len(ids) == len(set(ids)) == len(steps)


def test_render_contains_text_answer_and_fenced_mermaid():
    c = FakeClient([Resp("直接回答")])
    steps, res = trace.trace_graph("問題", c, "m")
    out = trace.render("問題", steps, res)
    assert "1. call_model" in out and "答案：直接回答" in out and "```mermaid" in out
