"""對話迴圈：模型要求工具 → 我們執行 → 結果還給模型 → 重複，直到模型給出最終答案。

模型只負責「決定要呼叫什麼」，執行永遠由我們的程式做。
迴圈有三道保險：步數上限、重複呼叫偵測、任何例外都轉成誠實的失敗訊息。
"""
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import types

from agent_v2 import tools

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
MAX_STEPS = 6   # 模型最多「呼叫工具再思考」幾輪；一般問題 2~3 輪就夠

SYSTEM_PROMPT = """你是台股財報問答助理。今天是 {today}。
資料涵蓋 52 家台股公司 2021Q1～{latest} 的季度財報。「最新一季」指 {latest}，「去年」「今年」請依今天日期換算。

規則：
1. 所有數字都必須來自工具回傳，不得憑記憶回答。查不到就明說查不到。
2. 使用者提到公司名稱時，先用 search_company 取得代號。若回傳多家公司，列出選項請使用者確認，不要自己挑一家。
3. 描述金額時優先使用工具回傳的 value_display（已換算成兆元、億元）；需要精確數字時再附上原值與單位（原值單位為新台幣千元，EPS 為元）。
4. 毛利率、營業利益率、營業費用率一律用 get_ratio 取得，不要自己拿兩個數字相除。其他比率或成長率需要計算時，列出用到的數字與算式。
5. 排名結果若有 missing_companies，必須說明：排名只涵蓋有資料的 N 家（共 M 家），並點名缺資料的公司。比率結果若有 skipped_periods，必須說明哪些期間缺資料。
6. 你目前沒有文件檢索工具，無法回答經營策略、原因、看法。問題同時包含「數字」與「原因」時（例如「毛利率為什麼上升」），必須先用工具查出數字部分並回答（例如毛利率的走勢），再明確說明原因部分目前無法回答。不可以因為其中一半答不了就整題放棄，也不要請使用者自己去計算。
7. 不提供投資建議，不預測股價。
8. 工具回傳錯誤時，如實告知使用者原因，不要編造資料。"""


@dataclass
class AgentResult:
    answer: str
    stop_reason: str                 # done / max_steps / empty / error
    tool_calls: list = field(default_factory=list)   # [{"name","args","error"}]，之後做評估與追蹤要用


def get_client():
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("讀不到 GEMINI_API_KEY，請檢查 .env")
    return genai.Client(api_key=api_key), os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def _latest_period() -> str:
    try:
        return httpx.get(f"{tools.API_BASE_URL}/health", timeout=tools.HTTP_TIMEOUT).json()["latest_period"]
    except Exception:
        return "2025Q4"


def run_agent(question: str, client, model: str, max_steps: int = MAX_STEPS) -> AgentResult:
    system = SYSTEM_PROMPT.format(today=date.today().isoformat(), latest=_latest_period())
    config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=0,
        tools=tools.GEMINI_TOOLS,
        # 關掉 SDK 的自動執行：我們自己控制每一步（才能記錄、限制、攔截）
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=question)])]
    log, seen = [], set()

    for step in range(1, max_steps + 1):
        try:
            response = client.models.generate_content(model=model, contents=contents, config=config)
        except Exception as e:
            logger.error(f"第 {step} 輪呼叫 Gemini 失敗：{type(e).__name__}: {e}")
            return AgentResult("抱歉，系統暫時無法處理這個問題，請稍後再試。", "error", log)

        calls = response.function_calls
        if not calls:                                   # 沒有要呼叫工具 → 這就是最終答案
            text = (response.text or "").strip()
            if not text:
                logger.warning("模型回傳空白答案")
                return AgentResult("抱歉，這次沒有產生有效的回答，請換個方式再問一次。", "empty", log)
            logger.info(f"第 {step} 輪完成，共呼叫工具 {len(log)} 次")
            return AgentResult(text, "done", log)

        # 重要：必須把模型這一輪「原封不動」放回對話紀錄（內含它的函式呼叫與思考簽章），不能自己重組
        contents.append(response.candidates[0].content)

        # 模型可能一輪同時要求多個工具；每一個都要回覆，且放在同一個 user 訊息裡
        parts = []
        for call in calls:
            args = dict(call.args or {})
            signature = call.name + json.dumps(args, sort_keys=True, ensure_ascii=False)
            if signature in seen:
                result = {"error": "duplicate_call", "message": "已用相同參數查詢過，請直接使用先前的結果回答"}
            else:
                seen.add(signature)
                result = tools.execute_tool(call.name, args)
            log.append({"name": call.name, "args": args, "error": result.get("error") if isinstance(result, dict) else None})
            parts.append(types.Part.from_function_response(name=call.name, response={"result": result}))
        contents.append(types.Content(role="user", parts=parts))

    logger.warning(f"超過 {max_steps} 輪仍未得到答案，中止")
    return AgentResult("這個問題需要的查詢步驟過多，我沒能在限制內完成。請把問題拆小一點再問。", "max_steps", log)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    tools.check_tools_in_sync()
    client, model = get_client()
    q = " ".join(sys.argv[1:]) or "台積電 2025Q4 的營收是多少？"
    result = run_agent(q, client, model)
    print(f"\n問題：{q}\n")
    for i, c in enumerate(result.tool_calls, 1):
        print(f"  工具 {i}: {c['name']}({c['args']})" + (f" ❌{c['error']}" if c["error"] else ""))
    print(f"\n[{result.stop_reason}]\n{result.answer}")
