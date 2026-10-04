import os
import re
import sqlite3
import logging
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
DB_PATH = ROOT / "data" / "finance.db"


# ---------- 1. 從資料庫動態產生「資料說明書」 ----------
def build_schema_context(db_path: Path = DB_PATH) -> str:
    with sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True) as conn:
        items = conn.execute(
            "SELECT DISTINCT item, display_name, unit FROM v_financials ORDER BY item").fetchall()
        companies = conn.execute(
            "SELECT code, company_name FROM companies ORDER BY code").fetchall()
        min_p, max_p = conn.execute(
            "SELECT MIN(period), MAX(period) FROM v_financials").fetchone()

    item_lines = "\n".join(f"  - {i}：{d}（單位 {u}）" for i, d, u in items)
    company_lines = "\n".join(f"  - {c}：{n}" for c, n in companies)

    return f"""## 資料表：v_financials（長表，一列 = 一家公司、一季、一個科目）
欄位：
  - code TEXT：股票代號，例如 '2330'
  - company_name TEXT：公司全名
  - industry TEXT：產業別
  - year INTEGER：年度，例如 2025
  - quarter INTEGER：季別 1~4
  - period TEXT：年季，格式 'YYYYQn'，例如 '2025Q4'
  - item TEXT：科目代碼（見下方清單）
  - display_name TEXT：科目中文名稱
  - value REAL：數值（單季數字，非累計）
  - unit TEXT：'TWD_thousand' 代表新台幣千元，'TWD' 代表新台幣元

## 可用科目（item）
{item_lines}

## 可用公司（code：名稱）
{company_lines}

## 資料期間
{min_p} 到 {max_p}。「最新一季」「去年」等相對時間，一律以 {max_p} 為基準，不要用今天的日期。"""


# ---------- 2. System Prompt：規則寫清楚 ----------
SYSTEM_RULES = """你是台股財報資料庫的 SQLite 專家。請把使用者的問題轉成一條 SQLite 查詢。

規則：
1. 只能查詢 v_financials，只能使用 SELECT。
2. 篩選公司一律用 code（請依公司清單把「台積電」這類簡稱對應到代號），不要用 company_name LIKE。
3. 比率（如毛利率）用 CASE WHEN 把不同 item 轉成欄位再相除，並乘以 100.0 避免整數除法。
4. 只輸出 SQL 本身，不要任何解釋，不要 markdown。
5. 如果問題需要的科目或公司不在清單內，只輸出：CANNOT_ANSWER
6. SELECT 的欄位一定要包含 company_name、period；如果有查詢 value，也要一併查出 display_name 和 unit。

範例：
問題：台積電 2025 年各季毛利率
SQL：
SELECT period,
       ROUND(100.0 * SUM(CASE WHEN item = 'gross_profit' THEN value END)
                   / SUM(CASE WHEN item = 'revenue' THEN value END), 1) AS gross_margin_pct
FROM v_financials
WHERE code = '2330' AND year = 2025
GROUP BY period
ORDER BY period;"""


# ---------- 3. 呼叫 Gemini 產生 SQL ----------
def get_client():
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("讀不到 GEMINI_API_KEY，請檢查 .env")
    return genai.Client(api_key=api_key), os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def clean_sql(text: str) -> str:
    # LLM 常常不聽話地包上 ```sql ... ```，這裡把它剝掉
    text = re.sub(r"```(?:sql)?", "", text, flags=re.IGNORECASE)
    return text.strip()


def generate_sql(question: str, schema: str, client, model: str) -> str:
    prompt = f"{schema}\n\n問題：{question}\nSQL："
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_RULES,
                temperature=0,  # 寫 SQL 要穩定，不要創意
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
    except Exception as e:
        logger.error(f"Gemini 呼叫失敗：{type(e).__name__}: {e}")
        raise
    if not response.text:
        raise RuntimeError("Gemini 回傳空白（可能被安全過濾擋下）")
    return clean_sql(response.text)


if __name__ == "__main__":
    schema = build_schema_context()
    logger.info(f"資料說明書長度：{len(schema)} 字元")
    client, model = get_client()

    test_questions = [
        "台積電最新一季的營收是多少？",
        "聯發科 2024 年每一季的 EPS",
        "2025Q4 毛利率最高的前五家公司",
        "台積電的本益比是多少？",  # 刻意問資料庫沒有的東西
    ]
    for q in test_questions:
        print(f"\n❓ {q}")
        print(generate_sql(q, schema, client, model))