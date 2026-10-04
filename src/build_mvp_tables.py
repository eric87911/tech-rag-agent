import sqlite3
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
RAW_DB = ROOT / "data" / "finance_raw.db"
CLEAN_DB = ROOT / "data" / "finance.db"

# 資料上限：2026 年的資料來源（舊版 MOPS）不穩定，Q1 缺漏、Q2 未驗證
# TODO：上游改用 FinMind API 後再移除這個限制
MAX_PERIOD = "2025Q4"

# 白名單：只開放「已用台積電公告數字驗證過」的科目給 Agent
VERIFIED_ITEMS = [
    # 損益表（已核對台積電 2025 四季公告）
    "revenue", "cogs", "gross_profit", "operating_expenses", "operating_income", "eps",
    # 資產負債表（存量，未經單季還原，風險低）
    "cash_equivalents", "accounts_receivable", "inventory", "current_assets",
    "ppe", "accounts_payable", "long_term_debt", "retained_earnings",
]

# 人工修正：原始 company_profiles 是 Gemini 產生的，有幻覺錯誤
NAME_FIXES = {"2618": "長榮航空股份有限公司"}


def build_companies(conn):
    if not RAW_DB.exists():
        raise FileNotFoundError(f"找不到 {RAW_DB}")
    # ATTACH：讓一條 SQL 同時讀兩個資料庫檔案，原始庫一樣用唯讀模式
    conn.execute(f"ATTACH DATABASE '{RAW_DB.as_uri()}?mode=ro' AS raw")
    conn.execute("DROP TABLE IF EXISTS companies")
    conn.execute("""
        CREATE TABLE companies AS
        SELECT p.code, p.company_name, p.industry
        FROM raw.company_profiles p
        WHERE p.code IN (SELECT DISTINCT code FROM financials)
    """)
    for code, name in NAME_FIXES.items():
        conn.execute("UPDATE companies SET company_name = ? WHERE code = ?", (name, code))
    conn.commit()
    conn.execute("DETACH DATABASE raw")

    # 檢查：每家公司都要有名稱，否則 LLM 無法把「台積電」對應到代號
    missing = conn.execute("""
        SELECT DISTINCT code FROM financials
        WHERE code NOT IN (SELECT code FROM companies)
    """).fetchall()
    if missing:
        logger.warning(f"以下公司沒有名稱資料：{[m[0] for m in missing]}")
    n = conn.execute("SELECT COUNT(*) FROM companies").fetchone()[0]
    logger.info(f"companies 表：{n} 家公司")


def build_view(conn):
    placeholders = ", ".join(f"'{i}'" for i in VERIFIED_ITEMS)
    conn.execute("DROP VIEW IF EXISTS v_financials")
    conn.execute(f"""
        CREATE VIEW v_financials AS
        SELECT f.code, c.company_name, c.industry,
               f.year, f.quarter, f.period,
               f.item, f.display_name, f.value, f.unit
        FROM financials f
        JOIN companies c ON f.code = c.code
        WHERE f.item IN ({placeholders})
          AND f.period <= '{MAX_PERIOD}'
    """)
    n = conn.execute("SELECT COUNT(*) FROM v_financials").fetchone()[0]
    items = conn.execute("SELECT COUNT(DISTINCT item) FROM v_financials").fetchone()[0]
    logger.info(f"v_financials 視圖：{n:,} 筆，{items} 個科目")
    assert items == len(VERIFIED_ITEMS), "白名單中有科目在資料裡找不到，請檢查拼字"


def main():
    with sqlite3.connect(CLEAN_DB.as_uri(), uri=True) as conn:
        build_companies(conn)
        build_view(conn)
    logger.info("✅ MVP 資料層建置完成")


if __name__ == "__main__":
    main()