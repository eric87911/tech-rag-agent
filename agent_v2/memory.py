"""多輪對話記憶：只存「使用者問題＋最終答案」的純文字，不存工具呼叫與工具結果。

為什麼不存工具紀錄：工具回傳很大（拖貴）、思考簽章只在同一輪內有意義、舊的查詢結果下一輪就過期了。
這裡全是純函式（沒有 I/O、不碰 Gemini），所以可以完整離線測試。
"""
from google.genai import types

MAX_TURNS = 4            # 最多保留最近幾輪
MAX_HISTORY_CHARS = 6000  # 保留的問答總字數上限；超過就丟最舊的整輪

# 工具層回報的「暫時性」錯誤：這種輪次的答案多半是「系統暫時無法…」，記住只會污染後面的對話
TRANSIENT_TOOL_ERRORS = {"api_unavailable", "service_error", "rate_limited"}


def to_contents(history: list) -> list:
    """[{'q','a'},...] → Gemini 的對話紀錄（user／model 交替的純文字）"""
    contents = []
    for turn in history or []:
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=turn["q"])]))
        contents.append(types.Content(role="model", parts=[types.Part.from_text(text=turn["a"])]))
    return contents


def trim_history(history: list, max_turns: int = MAX_TURNS, max_chars: int = MAX_HISTORY_CHARS) -> list:
    """丟最舊的整輪（不截斷單一答案：截半的答案會改變意思）。至少保留最新一輪，即使它自己就超過字數上限。"""
    kept = list(history)[-max_turns:] if max_turns > 0 else []
    while len(kept) > 1 and sum(len(t["q"]) + len(t["a"]) for t in kept) > max_chars:
        kept = kept[1:]
    return kept


def should_remember(result) -> bool:
    """這一輪要不要寫進記憶。只記「正常完成、沒被攔截、沒遇到暫時性故障」的輪次。
    被攔截的輪次（洩漏、捏造、步數用完、空回應、Gemini 出錯）不記：否則拒絕訊息會留在上下文裡被模型模仿，攻擊文字也會一直留著。"""
    if result.stop_reason != "done" or result.leak_blocked or not (result.answer or "").strip():
        return False
    return not any(c.get("error") in TRANSIENT_TOOL_ERRORS for c in result.tool_calls)


def remember(history: list, question: str, result, **limits) -> list:
    """回傳新的 history（不修改傳入的 list）"""
    if not should_remember(result):
        return list(history or [])
    return trim_history(list(history or []) + [{"q": question, "a": result.answer.strip()}], **limits)
