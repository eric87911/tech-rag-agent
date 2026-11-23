"""v1 vs v2 同題對照。

    python -m agent_v2.compare --dry-run        # 只列出計畫與預估費用，不呼叫 Gemini
    python -m agent_v2.compare --runs 1         # 實際執行（需要 API 開著；讀得到 GEMINI_API_KEY）

公平性規則（同樣會印在報告裡）：
1. 只比「看答案」的檢查。v2 的評分有一半在看工具呼叫與引用對照，v1 沒有這些概念，用同一把尺只會讓 v1 無條件失分。
2. 拆掉措辭偏見：拒答接受任何明確的拒答措辭（v1 說「超出範圍」、v2 說「無法」都算）；
   「有標註來源」各用自己的格式判斷（v1 是 [1]、v2 是（來源：…））。
3. 每一條被刪除或放寬的評分條件都會列出（neutralize 的 dropped / widened）。
4. 兩邊用同一個計量器統計 Gemini 用量，並有費用上限（--max-cost），超過就停。
5. 文件類開放式問題沒有標準答案：自動評分只看「有標註來源、沒有編造引用」，內容正確性要人工抽查。
"""
import argparse
import json
import logging
import re
import sys
import threading
import time
from datetime import datetime
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal
from pathlib import Path

from agent_v2 import eval as E
from agent_v2 import loop, tools

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent

# 各組的性質：誰有利、有多乾淨（誠實標示，報告會一併印出）
SETS = {
    "core": (E.build_questions, "v2 開發集（v2 反覆調整過，偏向 v2）"),
    "holdout": (E.build_holdout_questions, "財報留存集（其中 h_prompt_leak 的失敗曾促成修正，並非完全盲測）"),
    "docs": (E.build_docs_questions, "文件開發集（v2 反覆調整過，偏向 v2）"),
    "docs_holdout": (E.build_docs_holdout_questions, "文件留存集（調整前只跑過一次）"),
}

# ---------------------------------------------------------------- 中立評分
# 只涵蓋原題目裡「拒答／查無資料」群組實際用到的詞；放寬時只「額外」加入 v1 會用的措辭，範圍刻意收緊，
# 避免「沒拒答、卻剛好出現『不會』『只能』」的答案被誤判通過
REFUSAL_WORDS = {"沒有", "無法", "查無", "缺", "無資料", "尚無", "不能", "不提供", "投資建議",
                 "未提及", "沒找到", "未涵蓋", "沒有相關"}
CORE_REFUSAL = {"無法", "不能", "不提供", "沒有", "查無", "未提及", "沒找到", "未涵蓋", "無資料", "尚無"}
NEW_REFUSAL_PHRASES = {"超出", "不在", "不屬於", "無相關"}      # v1 的 router 說「超出了我能回答的範圍」
OUT_OF_SCOPE_REFUSAL = sorted(NEW_REFUSAL_PHRASES | {"無法", "不能", "不提供", "沒有"})
CORRECTION_WORDS = {"不對", "不是"}
CORRECTION_GENERIC = sorted(CORRECTION_WORDS | {"不正確", "並非", "有誤", "錯誤", "不符", "不準確"})
# 只有 v2 的資料層／格式才做得到的條件：不拿來比
DROP_GROUPS = {frozenset({"來源"}): "答案必須出現「來源」二字（v2 的引用格式）",
               frozenset({"42", "缺", "沒有資料", "未參與", "資料缺"}): "必須說明有公司缺資料（取決於 v2 API 回傳的 missing_companies）"}
V1_CITE = re.compile(r"\[\d+\]")
MIN_CITED_CHARS = 80
_DEC2 = re.compile(r"^([-−]?)(\d+)\.(\d{2})$")


def _rounded_variants(alt: str) -> list:
    """兩位小數的數字，也接受四捨五入到一位小數的寫法（兩種進位規則都收，因為底層原值未知）"""
    m = _DEC2.match(alt)
    if not m:
        return []
    sign, whole, frac = m.groups()
    d = Decimal(f"{whole}.{frac}")
    return sorted({f"{sign}{d.quantize(Decimal('0.1'), rounding=mode)}" for mode in (ROUND_HALF_UP, ROUND_HALF_EVEN)})


def neutralize(q: dict) -> dict:
    """把 v2 的題目轉成「只看答案」的中立版本，並記下刪了／放寬了什麼"""
    n = {"id": q["id"], "question": q["question"], "facts": [], "forbid_text": [], "forbid_cite": False,
         "forbid_numbers": bool(q.get("forbid_numbers")), "cite_any": bool(q.get("cite_required")),
         "dropped": [], "widened": []}
    for g in q.get("facts", []):
        key = frozenset(g)
        if key in DROP_GROUPS:
            n["dropped"].append(DROP_GROUPS[key])
        elif set(g) & CORE_REFUSAL and not any(ch.isdigit() for alt in g for ch in alt):
            n["facts"].append(sorted(set(g) | NEW_REFUSAL_PHRASES))
            n["widened"].append("拒答：接受任何明確的拒答措辭（含「超出範圍」）")
        elif set(g) == CORRECTION_WORDS:
            n["facts"].append(CORRECTION_GENERIC)
            n["widened"].append("更正：接受「不正確／並非／有誤」等")
        else:
            extra = sorted({v for alt in g for v in _rounded_variants(alt)} - set(g))
            n["facts"].append(list(g) + extra)
            if extra:
                n["widened"].append(f"數字：{g[0]} 也接受四捨五入到一位小數（{', '.join(extra)}）")
    ft = q.get("forbid_text")
    if ft == E.NO_CITE:
        n["forbid_cite"] = True
    elif ft:
        n["forbid_text"] = list(ft)
    if not (n["facts"] or n["cite_any"] or n["forbid_numbers"] or n["forbid_cite"] or n["forbid_text"]):
        n["facts"] = [OUT_OF_SCOPE_REFUSAL]          # 例如 out_of_scope：原本只看「沒呼叫工具」，中立版改看「有沒有拒答」
        n["widened"].append("原本只檢查「不呼叫工具」，改為檢查答案有明確拒答")
    return n


def has_cite(answer: str, system: str) -> bool:
    return bool(V1_CITE.search(answer)) if system == "v1" else bool(E.CITE_PATTERN.search(answer))


def has_fake_cite_marks(answer: str, system: str) -> bool:
    if system == "v1":
        return bool(V1_CITE.search(answer))          # v1 的 SQL 答案一律附「📎 資料來源：公開資訊觀測站」，那是真實來源，不算捏造
    return any(m in answer for m in E.NO_CITE)


def check_neutral(n: dict, answer: str, system: str) -> list:
    """回傳失敗原因清單（空 = 通過）。只看答案文字。"""
    a, failures = E._norm(answer), []
    if not answer.strip():
        return ["答案是空的"]
    for g in n["facts"]:
        if not any(E._norm(alt) in a for alt in g):
            failures.append(f"答案缺少：{g[:3]}")
    for t in n["forbid_text"]:
        if t in answer:
            failures.append(f"答案出現不該出現的內容：{t}")
    if n["forbid_cite"] and has_fake_cite_marks(answer, system):
        failures.append("沒有文件依據卻出現來源標註（疑似捏造引用）")
    if n["cite_any"]:
        if not has_cite(answer, system):
            failures.append("答案沒有標註來源")
        if len(answer) < MIN_CITED_CHARS:
            failures.append("答案過短，沒有實質內容")
    if n["forbid_numbers"] and loop.NUMBER_PATTERN.search(answer):
        failures.append("查不到資料卻出現數字（疑似捏造）")
    return failures


# ---------------------------------------------------------------- 用量計量器
class UsageMeter:
    """掛在 Gemini 的 generate_content 上，統計「所有」呼叫（不管是 v1 還是 v2、哪個執行緒）。"""

    def __init__(self, target=None):
        if target is None:
            from google.genai import models
            target = models.Models
        self._cls, self._orig, self._lock = target, None, threading.Lock()
        self.totals = {"calls": 0, "prompt": 0, "cached": 0, "output": 0, "thinking": 0}

    def __enter__(self):
        orig, meter = self._cls.generate_content, self

        def wrapper(inst, *args, **kwargs):
            resp = orig(inst, *args, **kwargs)
            meter._record(resp)
            return resp

        self._orig, self._cls.generate_content = orig, wrapper
        return self

    def __exit__(self, *exc):
        self._cls.generate_content = self._orig

    def _record(self, resp):
        um = getattr(resp, "usage_metadata", None)
        with self._lock:
            self.totals["calls"] += 1
            if um:
                self.totals["prompt"] += getattr(um, "prompt_token_count", 0) or 0
                self.totals["cached"] += getattr(um, "cached_content_token_count", 0) or 0
                self.totals["output"] += getattr(um, "candidates_token_count", 0) or 0
                self.totals["thinking"] += getattr(um, "thoughts_token_count", 0) or 0

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self.totals)


def usage_delta(before: dict, after: dict) -> dict:
    return {k: after[k] - before[k] for k in after}


# ---------------------------------------------------------------- 執行器
INFRA_PATTERN = re.compile(r"ServerError|ClientError|APIError|429|503|500|RESOURCE_EXHAUSTED|UNAVAILABLE|DEADLINE|[Tt]imeout|ConnectError|ReadError")


class _ErrorCapture(logging.Handler):
    """v1 把所有例外吞掉後換成一句「暫時失敗」，這裡把日誌抓下來，才分得出是 Gemini 暫時性錯誤還是程式錯誤"""

    def __init__(self):
        super().__init__(level=logging.ERROR)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def run_v1(agent, question: str) -> dict:
    cap = _ErrorCapture()
    logging.getLogger().addHandler(cap)
    try:
        r = agent.ask(question)
        answer, extra = r.answer, {"route": r.route, "sql": r.sql}
    except Exception as e:
        cap.messages.append(f"{type(e).__name__}: {e}")
        answer, extra = "", {"route": "", "sql": ""}
    finally:
        logging.getLogger().removeHandler(cap)
    failed = bool(cap.messages) or "暫時失敗" in answer
    infra = failed and any(INFRA_PATTERN.search(m) for m in cap.messages)
    return {"answer": answer, "infra_error": infra, "internal_error": failed and not infra,
            "errors": cap.messages[:3], **extra}


def run_v2(client, model, question: str) -> dict:
    res = loop.run_agent(question, client, model)
    return {"answer": res.answer, "infra_error": res.stop_reason == "error", "internal_error": False,
            "stop_reason": res.stop_reason, "tools": [c["name"] for c in res.tool_calls],
            "own_usage": res.usage, "empty_retries": res.empty_retries, "errors": []}


# ---------------------------------------------------------------- 統計與輸出
def summarize(records: list) -> dict:
    out = {}
    for key in sorted({(r["set"], r["system"]) for r in records}):
        rs = [r for r in records if (r["set"], r["system"]) == key]
        valid = [r for r in rs if not r["infra_error"]]
        n = len(valid)
        out[f"{key[0]}|{key[1]}"] = {
            "runs": len(rs), "infra_errors": len(rs) - n, "internal_errors": sum(r["internal_error"] for r in rs),
            "passed": sum(r["passed"] for r in valid), "valid": n,
            "pass_rate": round(sum(r["passed"] for r in valid) / n, 3) if n else None,
            "avg_seconds": round(sum(r["seconds"] for r in rs) / len(rs), 1) if rs else None,
            "avg_calls": round(sum(r["usage"]["calls"] for r in rs) / len(rs), 1) if rs else None,
            "avg_cost": round(sum(r["cost"] for r in rs) / len(rs), 4) if rs else None,
        }
    return out


def print_report(records: list, nqs: dict) -> None:
    summ = summarize(records)
    print("\n" + "=" * 70 + "\nv1 vs v2 同題對照（中立評分：只看答案，不看工具呼叫）\n" + "=" * 70)
    for set_name in [s for s in SETS if any(r["set"] == s for r in records)]:
        print(f"\n【{set_name}】{SETS[set_name][1]}")
        print(f"  {'系統':<4} {'通過':>9} {'通過率':>7} {'基礎設施失敗':>12} {'v1內部錯誤':>10} {'平均秒':>7} {'平均呼叫':>8} {'平均成本':>9}")
        for system in ("v1", "v2"):
            s = summ.get(f"{set_name}|{system}")
            if not s:
                continue
            rate = f"{s['pass_rate']:.0%}" if s["pass_rate"] is not None else "n/a"
            print(f"  {system:<4} {s['passed']:>4}/{s['valid']:<4} {rate:>7} {s['infra_errors']:>12} {s['internal_errors']:>10} "
                  f"{s['avg_seconds']:>7} {s['avg_calls']:>8} ${s['avg_cost']:>8.4f}")
        diffs = []
        for q in nqs[set_name]:
            cell = {}
            for system in ("v1", "v2"):
                rs = [r for r in records if r["set"] == set_name and r["system"] == system and r["id"] == q["id"] and not r["infra_error"]]
                cell[system] = (sum(r["passed"] for r in rs), len(rs))
            if cell["v1"][1] and cell["v2"][1] and cell["v1"][0] / cell["v1"][1] != cell["v2"][0] / cell["v2"][1]:
                diffs.append(f"    {q['id']:<18} v1 {cell['v1'][0]}/{cell['v1'][1]}   v2 {cell['v2'][0]}/{cell['v2'][1]}")
        print("  兩邊結果不同的題目：" + ("無" if not diffs else "\n" + "\n".join(diffs)))
    print("\n被刪除／放寬的評分條件（兩邊同樣適用）：")
    seen = set()
    for set_name, qs in nqs.items():
        for q in qs:
            for kind, items in (("刪除", q["dropped"]), ("放寬", q["widened"])):
                for it in items:
                    if (q["id"], it) not in seen:
                        seen.add((q["id"], it))
                        print(f"  [{kind}] {q['id']}：{it}")
    print("\n注意：v1 有 router 與 Text-to-SQL，單次呼叫數與延遲的差異來自架構，不代表誰「較省」；"
          "文件類問題的答案內容正確性未自動評分，需人工抽查；通過率差距在題數很少時不具統計意義。")


def build_plan(only_sets, only_ids):
    plan = {}
    for name in only_sets:
        qs = [neutralize(q) for q in SETS[name][0]()]
        plan[name] = [q for q in qs if not only_ids or q["id"] in only_ids]
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="*", choices=list(SETS), default=list(SETS))
    ap.add_argument("--systems", nargs="*", choices=["v1", "v2"], default=["v1", "v2"])
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--only", nargs="*", help="只跑指定題目 id")
    ap.add_argument("--sleep", type=float, default=0.5)
    ap.add_argument("--max-cost", type=float, default=2.0, help="累計估計費用（美元）超過就停止")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rescore", help="用已存的結果檔重新評分（不呼叫 Gemini）")
    ap.add_argument("--out")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(name)s: %(message)s")
    E.check_api_version()
    plan = build_plan(args.sets, args.only)
    n_q = sum(len(v) for v in plan.values())
    total_runs = n_q * len(args.systems) * args.runs
    est = total_runs * 0.004
    print(f"題目 {n_q} 題 × 系統 {len(args.systems)} × 每題 {args.runs} 次 = {total_runs} 次執行；"
          f"預估費用約 ${est:.2f}（以先前量到的單次約 $0.002～0.005 估算，上限 ${args.max_cost:.2f}）")
    for name, qs in plan.items():
        print(f"  {name}：{len(qs)} 題｜{SETS[name][1]}")
    if args.dry_run:
        for qs in plan.values():
            for q in qs:
                for it in q["dropped"]:
                    print(f"  [刪除] {q['id']}：{it}")
                for it in q["widened"]:
                    print(f"  [放寬] {q['id']}：{it}")
        return

    if args.rescore:
        saved = json.loads(Path(args.rescore).read_text(encoding="utf-8"))
        byid = {q["id"]: q for qs in plan.values() for q in qs}
        for r in saved["records"]:
            q = byid.get(r["id"])
            if q is None:
                continue
            r["failures"] = [] if r["infra_error"] else check_neutral(q, r["answer"], r["system"])
            r["passed"] = not r["failures"] and not r["infra_error"]
        print_report(saved["records"], {k: v for k, v in plan.items() if any(r["set"] == k for r in saved["records"])})
        return

    tools.check_tools_in_sync()
    client, model = loop.get_client()
    agent = None
    if "v1" in args.systems:
        sys.path.insert(0, str(ROOT / "src"))
        from agent import Agent                      # v1：src/agent.py
        agent = Agent()
    records, stopped = [], False
    with UsageMeter() as meter:
        try:
            for r in range(1, args.runs + 1):
                order = args.systems if r % 2 else list(reversed(args.systems))      # 輪流先後，避免順序效應
                for set_name, qs in plan.items():
                    for q in qs:
                        for system in order:
                            before, t0 = meter.snapshot(), time.perf_counter()
                            out = run_v1(agent, q["question"]) if system == "v1" else run_v2(client, model, q["question"])
                            seconds = time.perf_counter() - t0
                            usage = usage_delta(before, meter.snapshot())
                            failures = [] if out["infra_error"] else check_neutral(q, out["answer"], system)
                            rec = {"set": set_name, "system": system, "id": q["id"], "run": r, "question": q["question"],
                                   "passed": not failures and not out["infra_error"], "failures": failures,
                                   "seconds": round(seconds, 1), "usage": usage, "cost": E.estimate_cost(usage), **out}
                            if system == "v2" and out["own_usage"].get("calls") != usage["calls"]:
                                rec["meter_mismatch"] = True
                                logger.warning("計量器與 v2 自己的統計不一致：%s vs %s", usage["calls"], out["own_usage"].get("calls"))
                            records.append(rec)
                            mark = "⚠️" if out["infra_error"] else ("✅" if rec["passed"] else "❌")
                            print(f"  {mark} {set_name}/{q['id']} {system} #{r}" + ("" if rec["passed"] else f"  {failures or out['errors']}"))
                            time.sleep(args.sleep)
                            if E.estimate_cost(meter.snapshot()) > args.max_cost:
                                print(f"\n⛔ 累計估計費用超過上限 ${args.max_cost:.2f}，停止（已完成的結果會保存）")
                                stopped = True
                                raise KeyboardInterrupt
        except KeyboardInterrupt:
            print("\n中斷：保存目前已完成的結果")
        total = meter.snapshot()
    print_report(records, plan)
    print(f"\nGemini 呼叫 {total['calls']} 次｜輸入 {total['prompt']:,}（快取 {total['cached']:,}）｜"
          f"輸出 {total['output']:,}＋思考 {total['thinking']:,}｜估計總費用 ${E.estimate_cost(total):.3f}" + ("（因超過上限而提前停止）" if stopped else ""))
    path = Path(args.out) if args.out else ROOT / "eval_results" / f"compare_{datetime.now():%Y%m%d_%H%M}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({"runs": args.runs, "sets": {k: SETS[k][1] for k in plan}, "neutral_questions": plan,
                                "records": records, "summary": summarize(records), "total_usage": total}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"完整結果（含每次的答案）已存到：{path}")


if __name__ == "__main__":
    main()
