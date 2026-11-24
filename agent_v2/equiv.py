"""等價性驗證：同一批題目，分別用手寫迴圈（loop）與 LangGraph（graph）實作各跑幾次。

    python -m agent_v2.equiv --dry-run
    python -m agent_v2.equiv --sets holdout docs_holdout --runs 3
    python -m agent_v2.equiv --runs 3                      # 全部題組

怎麼判斷「等價」：Gemini 即使 temperature=0 也不完全確定，所以不要求逐字相同，而是比較三件事——
1. 通過率（用 eval.py 原本的完整檢查，含工具呼叫條件）
2. 工具呼叫序列是否相同
3. 「graph 對 loop」的不一致程度，是否落在「loop 自己跑兩次之間」的不一致程度之內（雜訊基準）
這些數字只能支持「看不出差異」，不能證明兩者完全相同。
"""
import argparse
import json
import logging
import time
from datetime import datetime
from itertools import combinations
from pathlib import Path

from agent_v2 import eval as E
from agent_v2 import graph, loop, tools
from agent_v2.compare import UsageMeter, usage_delta

ROOT = Path(__file__).resolve().parent.parent
IMPLS = ("loop", "graph")
SETS = {"core": E.build_questions, "holdout": E.build_holdout_questions, "docs": E.build_docs_questions,
        "docs_holdout": E.build_docs_holdout_questions, "injection": E.build_injection_questions}


def run_equiv(plan: dict, runners: dict, runs: int, check=E.check_run, meter=None, max_cost: float = 1.5, sleep: float = 0.0, out=print):
    """plan: {組名: [題目]}；runners: {"loop": fn(question)->AgentResult, "graph": fn}；回傳 (records, stopped)"""
    records, stopped = [], False
    for r in range(1, runs + 1):
        order = IMPLS if r % 2 else tuple(reversed(IMPLS))                 # 輪流先後，避免順序效應
        for set_name, qs in plan.items():
            for q in qs:
                for impl in order:
                    before, t0 = (meter.snapshot() if meter else None), time.perf_counter()
                    try:
                        res, err = runners[impl](q["question"]), None
                    except Exception as e:
                        res, err = None, f"{type(e).__name__}: {e}"
                    rec = {"set": set_name, "impl": impl, "id": q["id"], "run": r, "seconds": round(time.perf_counter() - t0, 1),
                           "usage": usage_delta(before, meter.snapshot()) if meter else {}}
                    if res is None:
                        rec.update(passed=False, infra_error=True, failures=[err], tools=[], guard=False, leak=False, empty=0, stop="exception", answer="")
                    else:
                        chk = check(q, res)
                        rec.update(passed=chk["passed"], infra_error=chk["infra_error"], failures=chk["failures"], tools=[c["name"] for c in res.tool_calls],
                                   guard=res.guard_triggered, leak=res.leak_blocked, empty=res.empty_retries, stop=res.stop_reason, answer=res.answer)
                    records.append(rec)
                    out(f"  {'⚠️' if rec['infra_error'] else ('✅' if rec['passed'] else '❌')} {set_name}/{q['id']} {impl} #{r}"
                        + ("" if rec["passed"] else f"  {rec['failures']}"))
                    time.sleep(sleep)
                    if meter and E.estimate_cost(meter.snapshot()) > max_cost:
                        out(f"\n⛔ 累計估計費用超過上限 ${max_cost:.2f}，停止")
                        return records, True
    return records, stopped


def _by(records, *keys):
    d = {}
    for r in records:
        d.setdefault(tuple(r[k] for k in keys), []).append(r)
    return d


def seq_mismatch_rates(records) -> dict:
    """工具呼叫序列不一致的比例。cross = 同一題同一次執行，loop 與 graph 的序列不同；
    within_X = 同一個實作、同一題，不同次執行之間序列不同（雜訊基準）。基礎設施失敗的執行不納入。"""
    ok = [r for r in records if not r["infra_error"]]
    cross_n = cross_diff = 0
    for (_, _, _), rs in _by(ok, "set", "id", "run").items():
        m = {r["impl"]: r for r in rs}
        if len(m) == 2:
            cross_n += 1
            cross_diff += tuple(m["loop"]["tools"]) != tuple(m["graph"]["tools"])
    res = {"cross": (cross_diff, cross_n)}
    for impl in IMPLS:
        n = diff = 0
        for rs in _by([r for r in ok if r["impl"] == impl], "set", "id").values():
            for a, b in combinations(rs, 2):
                n += 1
                diff += tuple(a["tools"]) != tuple(b["tools"])
        res[f"within_{impl}"] = (diff, n)
    return res


def summarize(records) -> dict:
    out = {}
    for (set_name, impl), rs in _by(records, "set", "impl").items():
        valid = [r for r in rs if not r["infra_error"]]
        out[f"{set_name}|{impl}"] = {"runs": len(rs), "infra_errors": len(rs) - len(valid), "passed": sum(r["passed"] for r in valid), "valid": len(valid),
                                     "guard": sum(r["guard"] for r in valid), "leak": sum(r["leak"] for r in valid), "empty_retries": sum(r["empty"] for r in valid),
                                     "avg_calls": round(sum(r["usage"].get("calls", 0) for r in rs) / len(rs), 2) if rs and rs[0]["usage"] else None}
    return out


def print_report(records, out=print):
    summ = summarize(records)
    out("\n" + "=" * 70 + "\n等價性驗證：loop（手寫迴圈）vs graph（LangGraph）\n" + "=" * 70)
    for set_name in dict.fromkeys(r["set"] for r in records):
        out(f"\n【{set_name}】")
        out(f"  {'實作':<6}{'通過':>10}{'基礎設施失敗':>12}{'守衛提醒':>9}{'洩漏攔截':>9}{'空回應重試':>10}{'平均呼叫':>9}")
        for impl in IMPLS:
            s = summ.get(f"{set_name}|{impl}")
            if s:
                out(f"  {impl:<6}{s['passed']:>5}/{s['valid']:<4}{s['infra_errors']:>12}{s['guard']:>9}{s['leak']:>9}{s['empty_retries']:>10}{str(s['avg_calls']):>9}")
        diffs = []
        for (sid,), rs in _by([r for r in records if r["set"] == set_name and not r["infra_error"]], "id").items():
            p = {impl: sum(r["passed"] for r in rs if r["impl"] == impl) for impl in IMPLS}
            n = {impl: sum(1 for r in rs if r["impl"] == impl) for impl in IMPLS}
            if n["loop"] and n["graph"] and p["loop"] / n["loop"] != p["graph"] / n["graph"]:
                diffs.append(f"    {sid:<20} loop {p['loop']}/{n['loop']}   graph {p['graph']}/{n['graph']}")
        out("  通過次數不同的題目：" + ("無" if not diffs else "\n" + "\n".join(diffs)))
    m = seq_mismatch_rates(records)
    out("\n工具呼叫序列不一致比例（越低越好；graph 對 loop 若不高於 loop 自己兩次之間，就在雜訊範圍內）：")
    for k, label in (("cross", "graph vs loop（同題同次）"), ("within_loop", "loop 自己兩次之間（雜訊基準）"), ("within_graph", "graph 自己兩次之間")):
        d, n = m[k]
        out(f"  {label:<26} {d}/{n}" + (f" = {d / n:.1%}" if n else "（沒有可比較的資料）"))
    out("\n注意：這只能說「在這批題目與這些次數下，看不出差異」，不能證明兩者完全相同；題數與次數都有限。"
        "逐步行為的等價性由 tests/test_graph.py 的差分測試保證。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="*", choices=list(SETS), default=list(SETS))
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--sleep", type=float, default=0.3)
    ap.add_argument("--max-cost", type=float, default=1.5, help="累計估計費用（美元）超過就停止")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(name)s: %(message)s")
    E.check_api_version()
    plan = {s: [q for q in SETS[s]() if not args.only or q["id"] in args.only] for s in args.sets}
    n_q = sum(len(v) for v in plan.values())
    total = n_q * 2 * args.runs
    print(f"題目 {n_q} 題 × 2 種實作 × 每題 {args.runs} 次 = {total} 次執行；預估約 ${total * 0.0035:.2f}（以最近量到的每次約 $0.0035 估算），上限 ${args.max_cost:.2f}")
    for s, qs in plan.items():
        print(f"  {s}：{len(qs)} 題")
    if args.dry_run:
        return
    tools.check_tools_in_sync()
    client, model = loop.get_client()
    runners = {"loop": lambda q: loop.run_agent(q, client, model), "graph": lambda q: graph.run_agent_graph(q, client, model)}
    with UsageMeter() as meter:
        try:
            records, stopped = run_equiv(plan, runners, args.runs, meter=meter, max_cost=args.max_cost, sleep=args.sleep)
        except KeyboardInterrupt:
            print("\n中斷")
            return
        total_usage = meter.snapshot()
    print_report(records)
    print(f"\nGemini 呼叫 {total_usage['calls']} 次｜估計總費用 ${E.estimate_cost(total_usage):.3f}" + ("（因超過上限而提前停止）" if stopped else ""))
    path = Path(args.out) if args.out else ROOT / "eval_results" / f"equiv_{datetime.now():%Y%m%d_%H%M}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({"runs": args.runs, "records": records, "summary": summarize(records), "seq_mismatch": seq_mismatch_rates(records),
                                "total_usage": total_usage}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"完整結果已存到：{path}")


if __name__ == "__main__":
    main()
