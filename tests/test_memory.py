from types import SimpleNamespace as NS

from agent_v2 import memory
from agent_v2.loop import AgentResult


def res(answer="好的答案", stop="done", leak=False, tool_errors=()):
    return AgentResult(answer, stop, [{"name": "t", "args": {}, "error": e, "result": {}} for e in tool_errors], leak_blocked=leak)


def test_to_contents_alternates_user_and_model():
    cs = memory.to_contents([{"q": "Q1", "a": "A1"}, {"q": "Q2", "a": "A2"}])
    assert [c.role for c in cs] == ["user", "model", "user", "model"]
    assert cs[0].parts[0].text == "Q1" and cs[3].parts[0].text == "A2"
    assert memory.to_contents(None) == [] and memory.to_contents([]) == []


def test_trim_keeps_latest_turns_by_count():
    h = [{"q": f"q{i}", "a": f"a{i}"} for i in range(10)]
    kept = memory.trim_history(h, max_turns=3)
    assert [t["q"] for t in kept] == ["q7", "q8", "q9"]


def test_trim_drops_oldest_whole_turns_by_chars_never_truncates():
    h = [{"q": "x", "a": "a" * 100}, {"q": "y", "a": "b" * 100}, {"q": "z", "a": "c" * 100}]
    kept = memory.trim_history(h, max_turns=4, max_chars=250)
    assert [t["q"] for t in kept] == ["y", "z"] and kept[0]["a"] == "b" * 100


def test_trim_keeps_newest_turn_even_if_it_alone_exceeds_limit():
    kept = memory.trim_history([{"q": "q", "a": "a" * 10_000}], max_chars=100)
    assert len(kept) == 1


def test_trim_does_not_mutate_input():
    h = [{"q": f"q{i}", "a": "a"} for i in range(6)]
    memory.trim_history(h, max_turns=2)
    assert len(h) == 6


def test_should_remember_only_clean_done_turns():
    assert memory.should_remember(res())
    assert not memory.should_remember(res(stop="error"))
    assert not memory.should_remember(res(stop="max_steps"))
    assert not memory.should_remember(res(stop="ungrounded"))
    assert not memory.should_remember(res(stop="empty"))
    assert not memory.should_remember(res(leak=True))
    assert not memory.should_remember(res(answer="  "))


def test_transient_tool_errors_are_not_remembered_but_other_errors_are():
    for e in ("service_error", "rate_limited", "api_unavailable"):
        assert not memory.should_remember(res(tool_errors=[e]))
    assert memory.should_remember(res(tool_errors=["not_found"]))       # 查無此公司是正常的結果
    assert memory.should_remember(res(tool_errors=[None, "duplicate_call"]))


def test_remember_returns_new_list_and_strips():
    h = [{"q": "q0", "a": "a0"}]
    new = memory.remember(h, "q1", res(answer="  a1  "))
    assert new == [{"q": "q0", "a": "a0"}, {"q": "q1", "a": "a1"}] and h == [{"q": "q0", "a": "a0"}]
    assert memory.remember(h, "q2", res(stop="error")) == h and memory.remember(None, "q", res(stop="error")) == []


def test_remember_applies_limits():
    h = []
    for i in range(7):
        h = memory.remember(h, f"q{i}", res(answer=f"a{i}"))
    assert [t["q"] for t in h] == ["q3", "q4", "q5", "q6"]
