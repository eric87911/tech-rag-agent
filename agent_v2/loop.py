"""對話迴圈：模型要求工具 → 我們執行 → 結果還給模型 → 重複，直到模型給出最終答案。

模型只負責「決定要呼叫什麼」，執行永遠由我們的程式做。
迴圈有三道保險：步數上限、重複呼叫偵測、任何例外都轉成誠實的失敗訊息。
"""
import json
import logging
import os
import re
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
1. 所有數字都必須來自工具回傳，不得憑記憶回答。即使你覺得自己知道答案，只要答案裡有百分比、金額、EPS 或排名，就必須先呼叫工具。查不到就明說查不到。
2. 使用者提到公司名稱時，先用 search_company 取得代號。若回傳多家公司，列出選項請使用者確認，不要自己挑一家。
3. 描述金額時優先使用工具回傳的 value_display（已換算成兆元、億元）；需要精確數字時再附上原值與單位（原值單位為新台幣千元，EPS 為元）。
4. 毛利率、營業利益率、營業費用率一律用 get_ratio 取得，不要自己拿兩個數字相除。說明最高、最低、最新或變化幅度時，直接引用回傳的 summary，不要自己從數列中比較。其他比率或成長率需要計算時，列出用到的數字與算式。
5. 排名結果若有 missing_companies，必須說明：排名只涵蓋有資料的 N 家（共 M 家），並點名缺資料的公司。比率結果若有 skipped_periods，必須說明哪些期間缺資料。
6. 「為什麼、怎麼看、展望、管理層怎麼說」這類需要文字依據的問題，依序做：
   (a) 問題問的是某個財報指標（營收、獲利、利潤率、EPS、成本、費用等）為什麼變好、變差、成長或下滑時，先用財報工具查出該指標的數字，確認問題的前提是否成立（是不是真的如此變化）。公司不在文件庫範圍內也一樣要做這一步，數字照樣回答。問題完全不涉及財報數字（例如對產業趨勢的看法）就跳過這一步。
   (b) 再用 search_documents 查原因，問特定公司時要填 company。只能根據 passages 回答，不可以用自己的知識補充；引用的句子後面標註（來源：該段的 source_label，照抄）。
   (c) passages 是空的或有 note（例如文件庫沒有該公司）時，回答數字的部分，並明確說原因的部分文件庫中沒有資料，不要用自己的知識補，也不要引用其他公司的段落。
   段落只能用來說明它所屬的公司。逐字稿的中文是第三方翻譯，數字有出入時以英文原文為準。
7. 不提供投資建議，不預測股價。
8. 工具回傳錯誤時，如實告知使用者原因，不要編造資料。"""


# 答案裡出現「帶單位的數字」（百分比、元、兆、億、千元）→ 視為在報告數字
NUMBER_PATTERN = re.compile(r"\d[\d,]*(?:\.\d+)?\s*(?:%|％|兆|億|千元|元)")
NUDGE = ("你的回答包含數字或股票代號，但這次對話還沒有呼叫任何工具。所有數字與公司代號都必須來自工具回傳，"
         "請先呼叫適當的工具（例如 search_company），再根據工具結果回答。")
UNGROUNDED_ANSWER = "抱歉，我無法確認這個答案的數字來源，所以不提供。請換個方式再問一次。"

# Gemini 有時會「正常結束但內容是空的」（parts=None），在 temperature 0 下會固定重現；重試時稍微提高 temperature
EMPTY_RETRY_TEMPERATURES = (0.3, 0.7)

# 沒呼叫過 search_documents，卻斷言「文件庫沒有…」：這是沒檢查就下的結論
NO_DOCS_CLAIM = re.compile(r"文件庫(中|裡)?[^。\n]{0,6}(沒有|無|查無|未)")
NUDGE_DOCS = ("你說文件庫中沒有相關資料，但這次對話還沒有呼叫過 search_documents。"
              "請先呼叫它確認（問特定公司時要填 company），再根據回傳結果回答。")


def claims_no_docs_without_search(text: str, tool_calls: list) -> bool:
    return bool(NO_DOCS_CLAIM.search(text)) and not any(c["name"] == "search_documents" for c in tool_calls)


# 答案裡出現括號包起來的股票代號，例如 長榮海運 (2603)
CODE_PATTERN = re.compile(r"[（(]\d{4,6}[)）]")


def has_unsourced_numbers(text: str, tool_calls: list) -> bool:
    """還沒呼叫過任何工具，答案卻在報告數字或股票代號 → 這些只可能來自模型的記憶，不可信"""
    return not tool_calls and bool(NUMBER_PATTERN.search(text) or CODE_PATTERN.search(text))


LEAK_WINDOW = 25   # 答案只要和系統提示有「連續 25 個字」相同，就視為洩漏
LEAK_REFUSAL = "抱歉，我無法提供系統內部的設定內容。我可以協助查詢台股公司的財報數字，請告訴我想查哪一家公司、哪一期。"


# 內部名稱（工具與回傳欄位）：使用者看不到也不需要知道。改寫式洩漏換得掉句子，但很難不提到這些名稱
INTERNAL_IDENTIFIERS = ("search_company", "get_financials", "get_ranking", "get_ratio", "search_documents",
                        "missing_companies", "skipped_periods", "value_display")
LEAK_MIN_IDENTIFIERS = 2   # 答案同時出現 2 個以上不同的內部名稱 → 視為洩漏（1 個先放過，避免誤殺）


def leaks_system_prompt(answer: str, system: str) -> bool:
    """程式端檢查：不靠模型自己守規矩。兩層：
    1) 去掉空白後，答案含有系統提示中連續 LEAK_WINDOW 個字的片段（抓逐字複述）
    2) 答案同時提到多個內部工具／欄位名稱（抓改寫、條列式洩漏）"""
    a, sp = re.sub(r"\s+", "", answer), re.sub(r"\s+", "", system)
    if any(sp[i:i + LEAK_WINDOW] in a for i in range(len(sp) - LEAK_WINDOW + 1)):
        return True
    return sum(1 for name in INTERNAL_IDENTIFIERS if name in answer) >= LEAK_MIN_IDENTIFIERS


@dataclass
class AgentResult:
    answer: str
    stop_reason: str                 # done / max_steps / empty / error / ungrounded
    tool_calls: list = field(default_factory=list)   # [{"name","args","error","result"}]，評估與追蹤要用
    guard_triggered: bool = False    # 「沒查資料就報數字」的保險是否出手過
    leak_blocked: bool = False       # 答案與系統提示大量重複，已被替換成拒絕訊息
    empty_retries: int = 0           # Gemini 回傳空白後，重試了幾次


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
    def make_config(temperature: float):
        return types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            tools=tools.GEMINI_TOOLS,
            # 關掉 SDK 的自動執行：我們自己控制每一步（才能記錄、限制、攔截）
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

    contents = [types.Content(role="user", parts=[types.Part.from_text(text=question)])]
    log, seen = [], set()
    nudged = False
    docs_nudged = False
    empty_retries = 0

    for step in range(1, max_steps + 1):
        try:
            response = client.models.generate_content(model=model, contents=contents, config=make_config(0))
            for temp in EMPTY_RETRY_TEMPERATURES:
                if response.function_calls or (response.text or "").strip():
                    break
                empty_retries += 1
                cand = response.candidates[0] if getattr(response, "candidates", None) else None
                logger.warning("模型回傳空白答案（finish_reason=%s，content=%s），以 temperature=%s 重試",
                               getattr(cand, "finish_reason", None), str(getattr(cand, "content", None))[:120], temp)
                response = client.models.generate_content(model=model, contents=contents, config=make_config(temp))
        except Exception as e:
            logger.error(f"第 {step} 輪呼叫 Gemini 失敗：{type(e).__name__}: {e}")
            return AgentResult("抱歉，系統暫時無法處理這個問題，請稍後再試。", "error", log, guard_triggered=nudged, empty_retries=empty_retries)

        calls = response.function_calls
        if not calls:                                   # 沒有要呼叫工具 → 這就是最終答案
            text = (response.text or "").strip()
            if not text:
                cand = response.candidates[0] if getattr(response, "candidates", None) else None
                logger.warning("模型回傳空白答案：finish_reason=%s｜finish_message=%s｜prompt_feedback=%s｜候選數=%s",
                               getattr(cand, "finish_reason", None), getattr(cand, "finish_message", None),
                               getattr(response, "prompt_feedback", None), len(getattr(response, "candidates", None) or []))
                logger.warning("空白回覆的候選內容：%s", str(getattr(cand, "content", None))[:400])
                return AgentResult("抱歉，這次沒有產生有效的回答，請換個方式再問一次。", "empty", log, guard_triggered=nudged, empty_retries=empty_retries)
            if has_unsourced_numbers(text, log):          # 保險：沒查資料就報數字，攔下來
                logger.warning("答案含數字但沒有呼叫任何工具，疑似憑記憶捏造")
                if nudged:                                # 已經提醒過一次還是這樣 → 放棄這個答案
                    return AgentResult(UNGROUNDED_ANSWER, "ungrounded", log, guard_triggered=nudged, empty_retries=empty_retries)
                nudged = True
                contents.append(response.candidates[0].content)
                contents.append(types.Content(role="user", parts=[types.Part.from_text(text=NUDGE)]))
                continue
            if claims_no_docs_without_search(text, log) and not docs_nudged:    # 保險 3：沒檢查就說文件庫沒有
                logger.warning("答案說文件庫沒有資料，但沒有呼叫過 search_documents，提醒後重試")
                docs_nudged = True
                contents.append(response.candidates[0].content)
                contents.append(types.Content(role="user", parts=[types.Part.from_text(text=NUDGE_DOCS)]))
                continue
            if leaks_system_prompt(text, system):          # 保險 2：答案洩漏系統提示 → 換成拒絕訊息
                logger.warning("答案與系統提示高度重複，已攔截")
                return AgentResult(LEAK_REFUSAL, "done", log, guard_triggered=nudged, empty_retries=empty_retries, leak_blocked=True)
            logger.info(f"第 {step} 輪完成，共呼叫工具 {len(log)} 次")
            return AgentResult(text, "done", log, guard_triggered=nudged, empty_retries=empty_retries)

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
            log.append({"name": call.name, "args": args, "error": result.get("error") if isinstance(result, dict) else None,
                    "result": result})
            parts.append(types.Part.from_function_response(name=call.name, response={"result": result}))
        contents.append(types.Content(role="user", parts=parts))

    logger.warning(f"超過 {max_steps} 輪仍未得到答案，中止")
    return AgentResult("這個問題需要的查詢步驟過多，我沒能在限制內完成。請把問題拆小一點再問。", "max_steps", log, guard_triggered=nudged, empty_retries=empty_retries)


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
