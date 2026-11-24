"""v2 介面用的純函式：把一輪的結果整理成要顯示的文字。

沒有 streamlit、沒有 I/O、不碰 Gemini，所以介面的「要顯示什麼」可以完整離線測試；
app_v2.py 只負責把這裡整理好的東西畫出來。
"""
import json

STOP_LABELS = {
    "done": "完成",
    "max_steps": "步數用完",
    "empty": "空白回覆",
    "error": "系統錯誤",
    "ungrounded": "數字來源無法確認",
}
MAX_VALUE_CHARS = 40       # 單一參數值最多顯示幾字
MAX_PREVIEW_CHARS = 400    # 工具回傳內容最多顯示幾字（完整內容很大，介面只看開頭）


def _short(value, limit: int) -> str:
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "…"


def format_args(args: dict) -> str:
    """{'code': '2330', 'item': 'revenue'} → "code=2330, item=revenue"（依參數名稱排序，輸出固定）"""
    return ", ".join(f"{k}={_short(v, MAX_VALUE_CHARS)}" for k, v in sorted((args or {}).items()))


def tool_line(call: dict) -> str:
    """一次工具呼叫的一行摘要；有錯誤時附上錯誤代碼"""
    base = f"{call.get('name', '?')}({format_args(call.get('args'))})"
    err = call.get("error")
    return f"❌ {base} → {err}" if err else f"✅ {base}"


def result_preview(call: dict) -> str:
    """工具回傳內容的開頭（JSON 字串，中文不跳脫）；序列化失敗就退回 str()"""
    try:
        text = json.dumps(call.get("result"), ensure_ascii=False)
    except (TypeError, ValueError):
        text = str(call.get("result"))
    return _short(text, MAX_PREVIEW_CHARS)


def usage_line(usage: dict) -> str:
    """token 用量；沒有資料回空字串"""
    if not usage or not usage.get("calls"):
        return ""
    parts = [f"Gemini 呼叫 {usage['calls']} 次", f"輸入 {usage.get('prompt', 0)}", f"輸出 {usage.get('output', 0)}",
             f"思考 {usage.get('thinking', 0)}"]
    if usage.get("cached"):
        parts.append(f"快取 {usage['cached']}")
    return "｜".join(parts) + " tokens"


def badges(turn, elapsed: float | None = None) -> list:
    """這一輪的狀態標籤：結果、守衛有沒有出手、有沒有寫入記憶"""
    out = [f"結果：{STOP_LABELS.get(turn.stop_reason, turn.stop_reason)}"]
    if turn.guard_triggered:
        out.append("守衛：已要求補查資料")
    if turn.leak_blocked:
        out.append("守衛：已攔截洩漏")
    out.append(f"記憶：{'已記住' if turn.remembered else '未寫入'}（共 {turn.history_len} 輪）")
    if elapsed is not None:
        out.append(f"耗時 {elapsed:.1f} 秒")
    return out


def build_details(turn, res=None, elapsed: float | None = None) -> dict:
    """整理成可存進 session 的純資料。res 是內層的完整結果（有工具參數與回傳）；拿不到就只列工具名稱。"""
    if res is not None:
        tools = [{"line": tool_line(c), "preview": result_preview(c)} for c in res.tool_calls]
        usage = usage_line(res.usage)
    else:
        tools = [{"line": f"✅ {name}", "preview": ""} for name in turn.tools]
        usage = usage_line(turn.usage)
    return {"badges": badges(turn, elapsed), "tools": tools, "usage": usage}


def history_to_messages(history: list) -> list:
    """記憶裡的 [{'q','a'}] → 介面的訊息列表（接續舊對話時還原畫面；沒有當時的運作細節）"""
    messages = []
    for turn in history or []:
        messages.append({"role": "user", "content": turn["q"]})
        messages.append({"role": "assistant", "content": turn["a"]})
    return messages
