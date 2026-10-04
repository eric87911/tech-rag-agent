import re
import sqlite3
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
DB_PATH = ROOT / "data" / "finance.db"
MAX_ROWS = 200

# 禁止出現的關鍵字（用 \b 比對完整單字，避免誤殺 'created_at' 這類欄位名）
FORBIDDEN = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE",
             "REPLACE", "ATTACH", "DETACH", "PRAGMA", "VACUUM"]


class UnsafeSQLError(Exception):
    """SQL 沒通過安全檢查"""


def validate_sql(sql: str) -> str:
    """第 2 層防線：程式檢查。通過就回傳整理過的 SQL，否則丟出例外。"""
    sql = sql.strip().rstrip(";").strip()
    if not sql:
        raise UnsafeSQLError("SQL 是空的")
    if ";" in sql:
        raise UnsafeSQLError("只允許一條 SQL 指令")
    if not re.match(r"^(SELECT|WITH)\b", sql, re.IGNORECASE):
        raise UnsafeSQLError("只允許 SELECT 查詢")
    for word in FORBIDDEN:
        if re.search(rf"\b{word}\b", sql, re.IGNORECASE):
            raise UnsafeSQLError(f"包含禁止的關鍵字：{word}")
    return sql


def execute_sql(sql: str, db_path: Path = DB_PATH) -> dict:
    """第 3、4 層防線：唯讀連線 + 筆數上限。"""
    sql = validate_sql(sql)
    with sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True) as conn:
        cur = conn.execute(sql)
        columns = [d[0] for d in cur.description]
        rows = cur.fetchmany(MAX_ROWS + 1)   # 多取一筆，用來判斷有沒有被截斷
    truncated = len(rows) > MAX_ROWS
    if truncated:
        logger.warning(f"結果超過 {MAX_ROWS} 筆，已截斷")
    return {"columns": columns, "rows": rows[:MAX_ROWS], "truncated": truncated}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    tests = [
        # 正常查詢
        "SELECT company_name, period, value, unit FROM v_financials "
        "WHERE code = '2330' AND item = 'revenue' AND period = '2025Q4';",
        # 各種攻擊與意外
        "DROP TABLE financials",
        "SELECT 1; DELETE FROM financials",
        "WITH x AS (SELECT 1) DELETE FROM financials",
        "CANNOT_ANSWER",
        "SELECT * FROM no_such_table",
    ]
    for sql in tests:
        print(f"\n▶ {sql[:70]}")
        try:
            result = execute_sql(sql)
            print(f"  ✅ 欄位 {result['columns']}")
            for row in result["rows"][:3]:
                print(f"     {row}")
        except UnsafeSQLError as e:
            print(f"  🛡️ 擋下：{e}")
        except sqlite3.Error as e:
            print(f"  ❌ 資料庫錯誤：{type(e).__name__}: {e}")