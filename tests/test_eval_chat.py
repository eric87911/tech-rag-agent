from types import SimpleNamespace as NS

import pytest

from agent_v2 import chat as chat_mod
from agent_v2 import eval_chat, graph, tools
from test_chat import FakeClient, Resp, call  # noqa: E402  （同一個 tests 目錄）


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(tools, "execute_tool", lambda name, args: {"echo": name, "ratio_pct": 59.2})
    monkeypatch.setattr(graph, "_latest_period", lambda: "2025Q4")
    monkeypatch.delenv("GEMINI_THINKING_BUDGET", raising=False)


def run(script, d):
    sink, c = [], FakeClient(script)
    chat = chat_mod.Chat(c, "m", on_result=sink.append)
    return eval_chat.run_dialogue(chat, sink, d), c


D = dict(id="d", turns=["問一", "問二"], tools_any=["get_ratio"], facts=[["59.2"]])


def test_pass_when_last_answer_meets_checks():
    rec, c = run([Resp("第一個答案"), Resp(calls=[call("get_ratio", code="2303")]), Resp("毛利率 59.2%")], D)
    assert rec["passed"] and not rec["infra_error"] and len(rec["turns"]) == 2 and c.seen == [1, 3, 5]


def test_fail_when_last_turn_misses_required_tool_or_fact():
    rec, _ = run([Resp("第一個答案"), Resp("不知道")], D)
    assert not rec["passed"] and any("get_ratio" in f for f in rec["failures"]) and any("59.2" in f for f in rec["failures"])


def test_gemini_error_in_any_turn_is_infra_not_failure():
    rec, _ = run([RuntimeError("503")], D)
    assert rec["infra_error"] and not rec["passed"] and len(rec["turns"]) == 1
    rec, _ = run([Resp("第一個答案"), RuntimeError("503")], D)
    assert rec["infra_error"]


def test_intermediate_turn_not_done_is_a_failure_not_infra():
    script = [Resp(calls=[call("t", i=i)]) for i in range(6)]            # 第一句就步數用完
    rec, _ = run(script, D)
    assert not rec["passed"] and not rec["infra_error"] and "第 1 輪未正常結束" in rec["failures"][0]


def test_each_run_uses_a_fresh_thread():
    sink, c = [], FakeClient([Resp("a1"), Resp("a2"), Resp("b1"), Resp("b2")])
    chat = chat_mod.Chat(c, "m", on_result=sink.append)
    d = dict(id="x", turns=["q1", "q2"], facts=[["a"]])
    eval_chat.run_dialogue(chat, sink, d)
    eval_chat.run_dialogue(chat, sink, d)
    assert c.seen == [1, 3, 1, 3]                                         # 第二次對話沒有看到第一次的記憶


def test_summarize_excludes_infra_errors():
    recs = [{"set": "dev", "passed": True, "infra_error": False}, {"set": "dev", "passed": False, "infra_error": False}, {"set": "dev", "passed": False, "infra_error": True}]
    assert eval_chat.summarize(recs)["dev"] == {"runs": 3, "infra_errors": 1, "passed": 1, "valid": 2}


def test_report_prints():
    lines = []
    eval_chat.print_report([{"set": "dev", "id": "x", "run": 1, "passed": False, "infra_error": False, "failures": ["答案缺少"]}], out=lines.append)
    assert any("x" in l for l in lines)


def test_check_order_requires_increasing_first_positions():
    g = [["台積電", "2330"], ["聯發科", "2454"], ["聯電", "2303"]]
    assert eval_chat.check_order("依序是台積電、聯發科、聯電。", g) == []
    assert eval_chat.check_order("聯電最高，其次聯發科，最後台積電", g)
    assert eval_chat.check_order("台積電最高，聯電其次", g)                  # 少提一家
    assert eval_chat.check_order("2330、2454、2303", g) == []               # 接受代號寫法


def test_order_failure_makes_dialogue_fail():
    d = dict(id="o", turns=["問一", "問二"], order=[["甲"], ["乙"]])
    rec, _ = run([Resp("a"), Resp("乙在前、甲在後")], d)
    assert not rec["passed"] and any("排序" in f for f in rec["failures"])
    rec, _ = run([Resp("a"), Resp("甲在前、乙在後")], d)
    assert rec["passed"]


def test_pct_forms_accepts_dropped_trailing_zeros():
    from agent_v2.eval_chat import _pct_forms
    assert _pct_forms("54.00") == ["54.00", "54.0", "54"]
    assert _pct_forms("62.33") == ["62.33"]


def _flaky(monkeypatch, code="service_error"):
    monkeypatch.setattr(tools, "execute_tool", lambda name, args: {"error": code, "message": "暫時無法使用"})


def test_transient_tool_error_on_last_turn_is_infra(monkeypatch):
    _flaky(monkeypatch)
    rec, _ = run([Resp("第一個答案"), Resp(calls=[call("get_ratio", code="2303")]), Resp("59.2，但資料服務暫時無法使用")], D)
    assert rec["infra_error"] and not rec["passed"] and "service_error" in rec["failures"][0]


def test_transient_tool_error_in_middle_turn_voids_the_dialogue(monkeypatch):
    _flaky(monkeypatch, "api_unavailable")
    rec, _ = run([Resp(calls=[call("search_documents", query="x")]), Resp("文件檢索暫時無法使用")], D)
    assert rec["infra_error"] and len(rec["turns"]) == 1 and "第 1 輪" in rec["failures"][0]


def test_non_transient_tool_error_still_counts_as_model_failure(monkeypatch):
    _flaky(monkeypatch, "not_found")       # 查不到資料是正常結果，答案對錯照算
    rec, _ = run([Resp("第一個答案"), Resp(calls=[call("get_ratio", code="9999")]), Resp("查無資料")], D)
    assert not rec["infra_error"] and not rec["passed"]


def test_check_run_flags_transient_error_for_single_question(monkeypatch):
    from agent_v2 import eval as E
    from agent_v2.loop import AgentResult
    res = AgentResult(answer="文件檢索暫時無法使用。", stop_reason="done", tool_calls=[{"name": "search_documents", "args": {}, "error": "api_unavailable", "result": {}}])
    chk = E.check_run({"id": "q", "question": "x", "tools_any": ["search_documents"], "cite_required": True}, res)
    assert chk["infra_error"] and not chk["passed"]
    ok = AgentResult(answer="a", stop_reason="done", tool_calls=[{"name": "t", "args": {}, "error": None, "result": {}}])
    assert not E.check_run({"id": "q", "question": "x"}, ok)["infra_error"]
