import json
import logging
from enum import Enum

from pydantic import BaseModel, Field
from google.genai import types

import text_to_sql as t2s

logger = logging.getLogger(__name__)


# ---------- 1. 定義 Router 的輸出格式 ----------
class Route(str, Enum):
    SQL = "sql"                      # 數字題：查財報資料庫
    RAG = "rag"                      # 觀點題：查年報、法說會文件
    BOTH = "both"                    # 數字 + 原因：兩條路都走
    OUT_OF_SCOPE = "out_of_scope"    # 和台股財報、產業無關


class RouteDecision(BaseModel):
    route: Route
    reason: str = Field(description="一句話說明為什麼這樣判斷")
    sql_question: str = Field(default="", description="交給資料庫的子問題（route 為 sql 或 both 時填寫）")
    rag_question: str = Field(default="", description="交給文件檢索的子問題（route 為 rag 或 both 時填寫）")


# ---------- 2. Prompt：說明兩條路各自能回答什麼 ----------
ROUTER_RULES = """你是問答系統的路由器，負責判斷使用者的問題該交給哪個資料來源。

資料來源一：財報資料庫（sql）
- 內容：52 家台股公司 2021Q1～2025Q4 的季度財報數字
- 可查科目：營收、銷貨成本、毛利、營業費用、營業利益、EPS、現金、應收帳款、存貨、流動資產、不動產廠房設備、應付帳款、長期負債、保留盈餘
- 適合：查詢、比較、排名、計算比率（例如毛利率）

資料來源二：文件庫（rag）
- 內容：台積電 2025 年報（致股東報告書、營運概況）、台積電 2026Q1 法說會逐字稿
- 適合：經營策略、產業看法、技術進展、擴廠計畫、原因解釋、管理層說法

判斷規則：
- sql：只問數字
- rag：只問看法、原因、策略、計畫
- both：同時需要數字和原因（例如「為什麼毛利率上升」需要毛利率數字，也需要原因），此時要把問題拆成 sql_question 和 rag_question 兩個子問題
- out_of_scope：和台股公司財報、半導體產業都無關
- 拆解子問題時，保留使用者原本的時間說法（例如「去年」「最新一季」「近幾季」），不要自行換算成具體年份；時間換算由下游系統處理。

範例：
問題：台積電 2025Q4 的營收是多少？ → sql
問題：台積電怎麼看 AI 需求？ → rag
問題：台積電毛利率為什麼一直上升？ → both（sql_question：台積電 2025 年各季毛利率；rag_question：台積電毛利率上升的原因）
問題：今天天氣如何？ → out_of_scope"""

SAFE_DEFAULT = RouteDecision(route=Route.BOTH, reason="路由判斷失敗，兩條路都走以免遺漏")


# ---------- 3. 呼叫 Gemini 做判斷 ----------
def route_question(question: str, client, model: str) -> RouteDecision:
    try:
        response = client.models.generate_content(
            model=model,
            contents=f"使用者問題：{question}",
            config=types.GenerateContentConfig(
                system_instruction=ROUTER_RULES,
                temperature=0,
                response_mime_type="application/json",
                response_schema=RouteDecision,      # 強制輸出符合這個格式
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
        decision = response.parsed
        if decision is None:                       # SDK 沒解析成功時，自己再解析一次
            decision = RouteDecision(**json.loads(response.text))
    except Exception as e:
        logger.error(f"路由判斷失敗，改用安全預設值（both）：{type(e).__name__}: {e}")
        decision = SAFE_DEFAULT.model_copy(update={"sql_question": question,
                                                   "rag_question": question})
        return decision

    # 防呆：該填的子問題沒填，就用原始問題補上
    if decision.route in (Route.SQL, Route.BOTH) and not decision.sql_question:
        decision.sql_question = question
    if decision.route in (Route.RAG, Route.BOTH) and not decision.rag_question:
        decision.rag_question = question
    return decision


# ---------- 4. 測試集：用準確率衡量 Router ----------
TEST_SET = [
    
    # ----- 沒出現在 Prompt 範例中的困難題 -----
    ("鴻海 2025 年第三季賺多少錢？", "sql"),
    ("聯電的營收是不是比去年差？為什麼？", "both"),
    ("2奈米量產會不會拉低台積電的毛利率？", "rag"),
    ("聯發科怎麼看 AI 需求？", "out_of_scope"),
    ("輝達最新一季的營收", "out_of_scope"),
]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    client, model = t2s.get_client()

    correct = 0
    for question, expected in TEST_SET:
        d = route_question(question, client, model)
        ok = d.route.value == expected
        correct += ok
        mark = "✅" if ok else "❌"
        print(f"\n{mark} {question}\n   判斷：{d.route.value}（預期 {expected}）｜{d.reason}")
        if d.sql_question:
            print(f"   → SQL 子問題：{d.sql_question}")
        if d.rag_question:
            print(f"   → RAG 子問題：{d.rag_question}")

    print(f"\n準確率：{correct}/{len(TEST_SET)} = {correct / len(TEST_SET):.0%}")