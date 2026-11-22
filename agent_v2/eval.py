"""評估腳本：用同一份題目、重複多次，量化 agent 的表現。

標準答案由程式向 API 即時查出（不是手寫），資料更新後題目不會過期。
執行：  python -m agent_v2.eval --runs 3        （需要 API 開著，並讀得到 GEMINI_API_KEY）
"""
import argparse
import json
import logging
import re
import statistics
import time
from datetime import datetime
from pathlib import Path

import httpx

from agent_v2 import loop, tools

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
NICKNAMES = {"2330": "台積電"}


# ---------- 1. 向 API 查標準答案 ----------
def _get(path: str, **params) -> dict:
    r = httpx.get(f"{tools.API_BASE_URL}{path}", params=params, timeout=tools.HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()


def _fin(code, item, period) -> dict:
    return _get("/financials", code=code, item=item, period_from=period, period_to=period)["data"][0]


def _money(rec) -> list:           # 金額的可接受寫法：易讀版、原值
    return [rec["value_display"], f"{int(rec['value']):,}"]


def check_api_version() -> None:
    """標準答案要靠 API 的新欄位；如果 API 是用舊程式啟動的（沒加 --reload），在這裡就說清楚"""
    try:
        r = _get("/ratios", code="2330", metric="gross_margin", period_from="2025Q4", period_to="2025Q4")
    except httpx.HTTPError as e:
        raise RuntimeError(f"API 沒有正常回應（{type(e).__name__}）。請確認 uvicorn api.main:app --reload 正在執行。") from e
    if "summary" not in r:
        raise RuntimeError("API 還在執行舊版程式（回傳裡沒有 summary 欄位）。"
                           "請按 Ctrl+C 停掉 uvicorn，再用 uvicorn api.main:app --reload 重新啟動。")


def build_questions() -> list:
    """每題：tools_any=至少要呼叫其中一個；forbid_tools / no_tools；facts=多組「任一寫法出現即可」，全部組都要命中；
    forbid_numbers=答案不可出現帶單位的數字（查不到資料時不可編造）"""
    check_api_version()
    tsmc_rev = _fin("2330", "revenue", "2025Q4")
    tw3_rev = _fin("3045", "revenue", "2025Q3")
    eps = _fin("2454", "eps", "2025Q4")
    hon_oi = _fin("2317", "operating_income", "2025Q4")
    gm = _get("/ratios", code="2330", metric="gross_margin", period_from="2025Q1", period_to="2025Q4")
    gm_all = _get("/ratios", code="2330", metric="gross_margin")["summary"]
    pm = _get("/ratios", code="1301", metric="operating_margin", period_from="2025Q4", period_to="2025Q4")["data"][0]
    top3 = _get("/rankings", item="revenue", period="2025Q4", top_n=3)["data"]
    umc_vs_tsmc = tsmc_rev

    rank_groups = []
    for r in top3:
        alts = [r["code"], r["company_name"].replace("股份有限公司", "")[:2]]
        if r["code"] in NICKNAMES:
            alts.append(NICKNAMES[r["code"]])
        rank_groups.append(alts)

    NO_DATA = ["沒有", "無法", "查無", "缺", "無資料", "尚無"]
    return [
        dict(id="single_money", question="台積電 2025Q4 的營收是多少？",
             tools_any=["get_financials"], facts=[_money(tsmc_rev)]),
        dict(id="alias", question="台哥大 2025Q3 的營收是多少？",
             tools_any=["get_financials"], facts=[_money(tw3_rev)]),
        dict(id="eps", question="聯發科 2025Q4 的 EPS 是多少？",
             tools_any=["get_financials"], facts=[[f"{eps['value']:.2f}"]]),
        dict(id="operating_income", question="鴻海 2025Q4 的營業利益是多少？",
             tools_any=["get_financials"], facts=[_money(hon_oi)]),
        dict(id="trend", question="台積電 2025 年每一季的毛利率",
             tools_any=["get_ratio"], facts=[[f"{x['ratio_pct']:.2f}"] for x in gm["data"]]),
        dict(id="highest", question="台積電毛利率最高是哪一季？",
             tools_any=["get_ratio"],
             facts=[[gm_all["highest"]["period"]], [f"{gm_all['highest']['ratio_pct']:.2f}"]]),
        dict(id="negative_margin", question="台灣塑膠 2025Q4 的營業利益率是多少？",
             tools_any=["get_ratio"], facts=[[f"{abs(pm['ratio_pct']):.2f}"]]),
        dict(id="ranking", question="2025Q4 營收前三名是哪些公司？",
             tools_any=["get_ranking"],
             facts=rank_groups + [["42", "缺", "沒有資料", "未參與", "資料缺"]]),   # 最後一組：必須揭露有公司缺資料
        dict(id="compare", question="台積電和聯電 2025Q4 的營收，哪一家比較高？",
             tools_any=["get_financials"], facts=[["台積電", "台灣積體電路", "2330"], _money(umc_vs_tsmc)]),
        dict(id="ambiguous", question="長榮的營收是多少？",
             tools_any=["search_company"],                       # 規則 2：公司身分必須查證，不能憑記憶
             forbid_tools=["get_financials"], facts=[["海運", "2603"], ["航空", "2618"]]),
        dict(id="missing_data", question="長榮海運 2025Q4 的毛利率是多少？",
             tools_any=["get_ratio", "get_financials"], forbid_numbers=True, facts=[NO_DATA]),
        dict(id="out_of_range", question="台積電 2026Q2 的營收是多少？",
             forbid_numbers=True, facts=[NO_DATA + ["2025Q4"]]),
        dict(id="why_question", question="台積電毛利率為什麼一直上升？",
             tools_any=["get_ratio"], facts=[[f"{gm_all['highest']['ratio_pct']:.2f}"], ["無法", "沒有", "不能", "尚無"]]),
        dict(id="advice", question="台積電現在值不值得買？",
             facts=[["投資建議", "無法", "不提供", "不能"]]),
        dict(id="out_of_scope", question="今天台北天氣如何？", no_tools=True),
    ]


# ---------- 2. 檢查單次結果 ----------
def _norm(text: str) -> str:
    return re.sub(r"[,，\s]", "", text)


def _number_tokens(text: str) -> set:
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)               # 先去掉千分位逗號：1,046,090,421 → 1046090421
    return {round(float(m), 6) for m in re.findall(r"-?\d+(?:\.\d+)?", text)}


def unsourced_numbers(answer: str, tool_calls: list) -> list:
    """答案中「帶單位的數字」若沒出現在任何工具回傳內容裡 → 疑似未溯源（需人工複查，不等於一定錯）。
    用「數值」比對而不是字串：答案寫 59.00、工具回傳 59.0 是同一個數字。"""
    sourced = _number_tokens(json.dumps([c.get("result") for c in tool_calls], ensure_ascii=False))
    out = []
    for m in loop.NUMBER_PATTERN.finditer(answer):
        n = re.match(r"[\d,]*\.?\d+", m.group(0)).group(0)
        if round(float(n.replace(",", "")), 6) not in sourced:
            out.append(n)
    return out


def check_run(q: dict, res) -> dict:
    names = [c["name"] for c in res.tool_calls]
    answer, failures = _norm(res.answer), []
    if res.stop_reason == "error":
        return {"passed": False, "infra_error": True, "failures": ["Gemini 呼叫失敗（不計入準確率）"], "unsourced": []}
    if res.stop_reason != "done":
        failures.append(f"未正常結束：{res.stop_reason}")
    if q.get("tools_any") and not set(names) & set(q["tools_any"]):
        failures.append(f"應呼叫 {q['tools_any']} 之一，實際：{names or '無'}")
    if q.get("no_tools") and names:
        failures.append(f"不該呼叫工具，實際：{names}")
    bad = set(names) & set(q.get("forbid_tools", []))
    if bad:
        failures.append(f"不該呼叫：{sorted(bad)}")
    for group in q.get("facts", []):
        if not any(_norm(alt) in answer for alt in group):
            failures.append(f"答案缺少：{group[:3]}")
    if q.get("forbid_numbers") and loop.NUMBER_PATTERN.search(res.answer):
        failures.append("查不到資料卻出現數字（疑似捏造）")
    return {"passed": not failures, "infra_error": False, "failures": failures,
            "unsourced": unsourced_numbers(res.answer, res.tool_calls)}


# ---------- 3. 執行與統計 ----------
def run_eval(client, model, runs=3, only=None, sleep=0.0) -> dict:
    questions = [q for q in build_questions() if not only or q["id"] in only]
    records = []
    for q in questions:
        for i in range(1, runs + 1):
            start = time.perf_counter()
            res = loop.run_agent(q["question"], client, model)
            seconds = time.perf_counter() - start
            chk = check_run(q, res)
            records.append({"id": q["id"], "run": i, "question": q["question"], "answer": res.answer,
                            "stop_reason": res.stop_reason, "tools": [c["name"] for c in res.tool_calls],
                            "guard_triggered": res.guard_triggered, "seconds": round(seconds, 1), **chk})
            print(f"  {'✅' if chk['passed'] else ('⚠️' if chk['infra_error'] else '❌')} {q['id']} #{i}"
                  + ("" if chk["passed"] else f"  {chk['failures']}"))
            time.sleep(sleep)
    return {"model": model, "runs": runs, "records": records, "summary": summarize(records, questions)}


def summarize(records, questions) -> dict:
    valid = [r for r in records if not r["infra_error"]]
    need_tools = {q["id"] for q in questions if q.get("tools_any")}
    nt = [r for r in valid if r["id"] in need_tools]
    per_q = {}
    for q in questions:
        rs = [r for r in valid if r["id"] == q["id"]]
        per_q[q["id"]] = {"passed": sum(r["passed"] for r in rs), "total": len(rs),
                          "avg_tools": round(statistics.mean(len(r["tools"]) for r in rs), 1) if rs else None,
                          "avg_seconds": round(statistics.mean(r["seconds"] for r in rs), 1) if rs else None}
    return {
        "total_runs": len(records), "infra_errors": len(records) - len(valid),
        "pass_rate": round(sum(r["passed"] for r in valid) / len(valid), 3) if valid else None,
        "tool_call_rate_on_data_questions": round(sum(bool(r["tools"]) for r in nt) / len(nt), 3) if nt else None,
        "guard_triggered_runs": sum(r["guard_triggered"] for r in valid),
        "ungrounded_stops": sum(r["stop_reason"] == "ungrounded" for r in valid),
        "runs_with_unsourced_numbers": sum(bool(r["unsourced"]) for r in valid),
        "per_question": per_q,
    }


def print_summary(result: dict) -> None:
    s = result["summary"]
    print(f"\n模型：{result['model']}｜每題 {result['runs']} 次｜共 {s['total_runs']} 次（Gemini 失敗 {s['infra_errors']} 次，不計入）\n")
    print(f"{'題目':<18}{'通過':>8}{'平均工具數':>12}{'平均秒數':>10}")
    for qid, v in s["per_question"].items():
        print(f"{qid:<18}{str(v['passed']) + '/' + str(v['total']):>8}{str(v['avg_tools']):>12}{str(v['avg_seconds']):>10}")
    pct = lambda x: "n/a" if x is None else f"{x:.0%}"
    print(f"\n整體通過率：{pct(s['pass_rate'])}")
    print(f"需要查資料的題目中，有呼叫工具的比率：{pct(s['tool_call_rate_on_data_questions'])}")
    print(f"「沒查資料就報數字」保險出手次數：{s['guard_triggered_runs']}（其中最後仍被拒絕：{s['ungrounded_stops']}）")
    print(f"答案含「疑似未溯源數字」的次數：{s['runs_with_unsourced_numbers']}（需人工複查 JSON）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--only", nargs="*", help="只跑指定題目 id")
    ap.add_argument("--sleep", type=float, default=0.5, help="每次呼叫之間暫停秒數，避免被限流")
    ap.add_argument("--out", help="結果 JSON 路徑")
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    tools.check_tools_in_sync()
    client, model = loop.get_client()
    result = run_eval(client, model, a.runs, a.only, a.sleep)
    print_summary(result)
    out = Path(a.out) if a.out else ROOT / "eval_results" / f"v2_{datetime.now():%Y%m%d_%H%M}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n完整結果（含每次的答案）已存到：{out}")
