from types import SimpleNamespace as NS

from agent_v2 import equiv
from agent_v2.loop import AgentResult


def fake_result(tool_names, answer="ok", stop="done"):
    return AgentResult(answer, stop, [{"name": n, "args": {}, "error": None, "result": {}} for n in tool_names])


def check(q, res):
    return {"passed": res.answer == "ok", "infra_error": res.stop_reason == "error", "failures": [] if res.answer == "ok" else ["x"]}


PLAN = {"s": [{"id": "q1", "question": "Q1"}, {"id": "q2", "question": "Q2"}]}


def run(loop_fn, graph_fn, runs=2, **kw):
    return equiv.run_equiv(PLAN, {"loop": loop_fn, "graph": graph_fn}, runs, check=check, **kw)


def test_identical_implementations_have_zero_mismatch():
    fn = lambda q: fake_result(["a", "b"] if q == "Q1" else ["c"])
    recs, stopped = run(fn, fn)
    m = equiv.seq_mismatch_rates(recs)
    assert not stopped and len(recs) == 8 and m["cross"] == (0, 4) and m["within_loop"] == (0, 2) and m["within_graph"] == (0, 2)


def test_different_tool_sequence_is_counted():
    recs, _ = run(lambda q: fake_result(["a"]), lambda q: fake_result(["b"]))
    assert equiv.seq_mismatch_rates(recs)["cross"] == (4, 4)


def test_noise_baseline_counts_within_impl_variation():
    state = {}
    def flaky(q):                                      # 同一題每次呼叫在 a、b 之間交替
        state[q] = state.get(q, 0) + 1
        return fake_result(["a"] if state[q] % 2 else ["b"])
    recs, _ = run(flaky, lambda q: fake_result(["a"]), runs=4)
    d, n = equiv.seq_mismatch_rates(recs)["within_loop"]
    assert n > 0 and d > 0


def test_infra_errors_excluded_from_rates_and_exceptions_recorded():
    def boom(q): raise RuntimeError("503")
    recs, _ = run(boom, lambda q: fake_result(["a"]), runs=1)
    assert all(r["infra_error"] for r in recs if r["impl"] == "loop")
    s = equiv.summarize(recs)
    assert s["s|loop"]["valid"] == 0 and s["s|graph"]["valid"] == 2
    assert equiv.seq_mismatch_rates(recs)["cross"] == (0, 0)


def test_order_alternates_between_runs():
    seen = []
    run(lambda q: seen.append("loop") or fake_result([]), lambda q: seen.append("graph") or fake_result([]), runs=2)
    assert seen[:2] == ["loop", "graph"] and seen[4:6] == ["graph", "loop"] or seen[4:8] == ["graph", "loop", "graph", "loop"]


def test_cost_cap_stops_run():
    class M:
        n = 0
        def snapshot(self):
            self.n += 1
            return {"calls": self.n, "prompt": self.n * 10_000_000, "cached": 0, "output": 0, "thinking": 0}
    recs, stopped = run(lambda q: fake_result([]), lambda q: fake_result([]), meter=M(), max_cost=1.0)
    assert stopped and len(recs) < 8


def test_report_prints_without_error():
    fn = lambda q: fake_result(["a"])
    recs, _ = run(fn, fn)
    lines = []
    equiv.print_report(recs, out=lines.append)
    assert any("graph vs loop" in l for l in lines)
