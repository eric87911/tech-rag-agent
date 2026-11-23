import threading
from unittest.mock import MagicMock

from agent_v2 import compare as C
from agent_v2 import eval as E


def _qs():
    return {q["id"]: q for fn, _ in C.SETS.values() for q in fn()}


def test_all_questions_neutralize_and_have_a_check():
    for fn, _ in C.SETS.values():
        for q in fn():
            n = C.neutralize(q)
            assert n["facts"] or n["cite_any"] or n["forbid_numbers"] or n["forbid_cite"] or n["forbid_text"], q["id"]
            assert not any(k in n for k in ("tools_any", "tools_all", "no_tools"))


def test_refusal_wording_is_neutral():
    q = {"id": "x", "question": "q", "facts": [["沒有", "無法"]]}
    n = C.neutralize(q)
    assert C.check_neutral(n, "這個問題超出了我能回答的範圍。", "v1") == []
    assert C.check_neutral(n, "無法提供這項資料。", "v2") == []
    assert C.check_neutral(n, "台積電毛利率是 59%。", "v1")          # 沒拒答 → 失敗


def test_out_of_scope_requires_refusal():
    n = C.neutralize({"id": "o", "question": "q", "no_tools": True})
    assert C.check_neutral(n, "超出範圍，我只能回答財報相關問題。", "v1") == []
    assert C.check_neutral(n, "今天天氣很好，適合出門。", "v1")


def test_dropped_groups_are_recorded():
    n = C.neutralize({"id": "w", "question": "q", "facts": [["來源"], ["42", "缺", "沒有資料", "未參與", "資料缺"], ["台積電"]]})
    assert len(n["dropped"]) == 2 and n["facts"] == [["台積電"]]


def test_rounding_variants_cover_both_modes():
    assert C._rounded_variants("12.25") == ["12.2", "12.3"]
    assert C._rounded_variants("-3.45") == ["-3.4", "-3.5"]
    assert C._rounded_variants("12") == []


def test_wrong_number_fails_right_number_passes():
    n = C.neutralize({"id": "n", "question": "q", "facts": [["59.20"]]})
    assert C.check_neutral(n, "毛利率為 59.2%。", "v1") == []
    assert C.check_neutral(n, "毛利率為 61.0%。", "v1")


def test_cite_format_is_system_specific():
    v1_ans = "台積電資本支出將提高 [1]。" + "內容" * 40
    v2_ans = "台積電資本支出將提高（來源：法說會 p.3）。" + "內容" * 40
    n = C.neutralize({"id": "d", "question": "q", "cite_required": True})
    assert C.check_neutral(n, v1_ans, "v1") == []
    assert C.check_neutral(n, v2_ans, "v2") == []
    assert C.check_neutral(n, v1_ans, "v2")          # v1 格式不算 v2 的來源
    assert C.check_neutral(n, "沒有來源的長答案" * 20, "v1")
    assert C.check_neutral(n, "短 [1]", "v1")        # 太短


def test_fake_cite_detected_when_no_docs():
    n = C.neutralize({"id": "nd", "question": "q", "forbid_text": E.NO_CITE, "facts": [E.NO_DOCS]})
    assert C.check_neutral(n, "文件中沒有提到這件事。", "v1") == []
    assert C.check_neutral(n, "文件中沒有提到，但據說如此 [1]。", "v1")


def test_empty_answer_fails():
    assert C.check_neutral(C.neutralize({"id": "e", "question": "q", "facts": [["a"]]}), "  ", "v1") == ["答案是空的"]


class _Models:
    def generate_content(self, *a, **k):
        r = MagicMock()
        r.usage_metadata.prompt_token_count = 100
        r.usage_metadata.cached_content_token_count = 10
        r.usage_metadata.candidates_token_count = 20
        r.usage_metadata.thoughts_token_count = 5
        return r


def test_meter_counts_across_threads_and_restores():
    orig = _Models.generate_content
    with C.UsageMeter(_Models) as m:
        ts = [threading.Thread(target=lambda: [_Models().generate_content() for _ in range(25)]) for _ in range(8)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        snap = m.snapshot()
    assert snap["calls"] == 200 and snap["prompt"] == 20000 and snap["thinking"] == 1000
    assert _Models.generate_content is orig


def test_run_v1_classifies_errors():
    ok = MagicMock(); ok.ask.return_value = MagicMock(answer="好", route="sql", sql="select 1")
    assert not C.run_v1(ok, "q")["infra_error"]
    boom = MagicMock(); boom.ask.side_effect = RuntimeError("503 UNAVAILABLE")
    r = C.run_v1(boom, "q"); assert r["infra_error"] and not r["internal_error"]
    bug = MagicMock(); bug.ask.side_effect = KeyError("x")
    r = C.run_v1(bug, "q"); assert r["internal_error"] and not r["infra_error"]


def test_summarize_excludes_infra_errors():
    base = {"set": "core", "system": "v1", "internal_error": False, "seconds": 1, "usage": {"calls": 2}, "cost": 0.01}
    recs = [{**base, "passed": True, "infra_error": False}, {**base, "passed": False, "infra_error": False},
            {**base, "passed": False, "infra_error": True}]
    s = C.summarize(recs)["core|v1"]
    assert s["valid"] == 2 and s["passed"] == 1 and s["pass_rate"] == 0.5 and s["infra_errors"] == 1


def test_regression_refusal_group_with_extra_words_is_widened():
    # 實測抓到的 bug：原群組含「預測」「不會」等詞時，放寬沒套用，v1 的「超出範圍」被誤判失敗
    q = {"id": "h", "question": "q", "forbid_numbers": True, "facts": [["無法", "不能", "沒有", "不提供", "預測", "不會"]]}
    n = C.neutralize(q)
    assert C.check_neutral(n, "這個問題超出了我能回答的範圍。", "v1") == []
    q2 = {"id": "h2", "question": "q", "facts": [["無法", "不能", "不提供", "抱歉", "只能"]]}
    assert C.check_neutral(C.neutralize(q2), "這個問題超出了我能回答的範圍。", "v1") == []


def test_regression_refusal_still_required():
    q = {"id": "h", "question": "q", "facts": [["無法", "不能", "沒有", "預測"]]}
    assert C.check_neutral(C.neutralize(q), "台積電明年營收約 2 兆元。", "v1")


def test_regression_digit_groups_are_not_treated_as_refusal():
    n = C.neutralize({"id": "x", "question": "q", "facts": [["1.05", "沒有"]]})
    assert not any("超出" in alt for g in n["facts"] for alt in g)


def test_v1_sql_source_line_is_not_a_fake_citation():
    n = C.neutralize({"id": "nd", "question": "q", "forbid_text": E.NO_CITE, "facts": [E.NO_DOCS]})
    ans = "文件中沒有提到原因。\n📎 資料來源：公開資訊觀測站 財務報表"
    assert C.check_neutral(n, ans, "v1") == []
    assert C.check_neutral(n, "沒有提到 [1]", "v1")
