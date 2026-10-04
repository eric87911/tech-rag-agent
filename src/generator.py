import logging
import re

from google.genai import types

import text_to_sql as t2s
from sql_agent import ask_database, SQLResult

logger = logging.getLogger(__name__)


# ---------- 1. 數字格式化：讓程式算，不讓 LLM 算 ----------
def format_amount(value, unit: str) -> str:
    if value is None:
        return "無資料"
    if unit == "TWD":                      # EPS 這類以「元」為單位的數字
        return f"{value:,.2f} 元"
    if unit == "TWD_thousand":
        twd = value * 1000                 # 千元 → 元
        sign = "-" if twd < 0 else ""
        twd = abs(twd)
        if twd >= 1e12:
            return f"新台幣 {sign}{twd / 1e12:,.2f} 兆元"
        if twd >= 1e8:
            return f"新台幣 {sign}{twd / 1e8:,.1f} 億元"
        return f"新台幣 {sign}{twd / 1e4:,.0f} 萬元"
    return f"{value:,}" if isinstance(value, (int, float)) else str(value)


def result_to_table(result: SQLResult) -> str:
    """把查詢結果轉成 Markdown 表格；有 value + unit 時改成格式化後的金額"""
    cols = list(result.columns)
    has_amount = "value" in cols and "unit" in cols
    if has_amount:
        vi, ui = cols.index("value"), cols.index("unit")
        show_cols = [c for c in cols if c not in ("value", "unit")] + ["金額"]
    else:
        show_cols = cols

    lines = ["| " + " | ".join(show_cols) + " |",
             "|" + "---|" * len(show_cols)]
    for row in result.rows:
        if has_amount:
            cells = [str(v) for i, v in enumerate(row) if i not in (vi, ui)]
            cells.append(format_amount(row[vi], row[ui]))
        else:
            cells = [format_amount(v, "") for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---------- 2. 非 ok 狀態：用固定範本，不呼叫 LLM ----------
TEMPLATES = {
    "empty": "查無符合條件的資料。可能是該公司在這個期間沒有公布這項數字，或期間超出資料範圍。",
    "cannot_answer": "目前的財報資料庫沒有這項資料。可查詢的項目包括營收、毛利、營業利益、EPS 及主要資產負債表科目。",
    "unsafe": "這個請求無法執行。本系統只提供財報資料的查詢。",
    "failed": "查詢過程發生錯誤，請換個方式描述您的問題。",
}


# ---------- 3. ok 狀態：交給 LLM 寫成回答 ----------
ANSWER_RULES = """你是專業的台股財報分析助理。請根據「查詢結果」回答使用者的問題。

規則：
1. 只能使用查詢結果裡的數字，不可以自行計算新的數字，也不可以使用你自己的知識補充數字。
2. 金額請直接照抄表格中的格式（例如「新台幣 1.05 兆元」），不要換算。
3. 用繁體中文回答，先用一句話直接回答問題，需要時再補充 1～2 句趨勢描述。
4. 不要提到 SQL、資料表或欄位名稱等技術細節。"""


def generate_answer(result: SQLResult, client, model: str) -> str:
    if result.status != "ok":
        return TEMPLATES.get(result.status, TEMPLATES["failed"])

    table = result_to_table(result)
    prompt = f"使用者問題：{result.question}\n\n查詢結果：\n{table}"
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=ANSWER_RULES,
                temperature=0.2,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
        answer = (response.text or "").strip()
    except Exception as e:
        logger.error(f"生成回答失敗：{type(e).__name__}: {e}")
        # 退路：LLM 失敗時，至少把表格直接給使用者
        answer = "（自動摘要暫時無法使用，以下為查詢結果）\n\n" + table

    # 來源由程式附加，保證不會被 LLM 編造
    if "period" in result.columns:
        periods = sorted({str(r[result.columns.index("period")]) for r in result.rows})
    else:
        # 保險：LLM 沒查出 period 欄位時，從 SQL 條件中找出期間
        periods = sorted(set(re.findall(r"\d{4}Q[1-4]", result.sql)))
    source = "公開資訊觀測站 財務報表"
    if periods:
        source += f"（{periods[0]}" + (f"～{periods[-1]}" if len(periods) > 1 else "") + "）"
    return f"{answer}\n\n📎 資料來源：{source}"


def answer_question(question: str, schema: str, client, model: str) -> dict:
    """Path A 的完整流程：問題 → SQL Agent → Generator"""
    result = ask_database(question, schema, client, model)
    answer = generate_answer(result, client, model)
    return {"answer": answer, "sql": result.sql, "status": result.status,
            "attempts": result.attempts}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    schema = t2s.build_schema_context()
    client, model = t2s.get_client()

    for q in ["台積電 2025 年各季的營收",
              "聯發科 2025Q4 的 EPS 是多少？",
              "2025Q4 毛利率最高的前三家公司",
              "台積電的本益比是多少？"]:
        out = answer_question(q, schema, client, model)
        print(f"\n{'=' * 60}\n❓ {q}\n[狀態 {out['status']}｜嘗試 {out['attempts']} 次]\n")
        print(out["answer"])