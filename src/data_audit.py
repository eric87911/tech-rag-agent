import sqlite3
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "data" / "finance_raw.db"


def connect_readonly(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise FileNotFoundError(f"找不到資料庫：{path}，請確認已複製到 data 資料夾")
    # 唯讀模式：稽核程式不應該修改任何資料
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def run_check(conn, title, sql):
    logger.info(f"=== {title} ===")
    try:
        rows = conn.execute(sql).fetchall()
        for row in rows:
            print("   ", row)
    except sqlite3.Error as e:
        logger.error(f"查詢失敗：{e}")


def main():
    conn = connect_readonly(DB_PATH)

    run_check(conn, "檢查 1：日期格式有幾種？",
        "SELECT length(date) AS len, COUNT(*) FROM final_report_31 GROUP BY len")

    run_check(conn, "檢查 2：period 欄位有多少空值？",
        "SELECT COUNT(*) FROM final_report_31 WHERE period IS NULL")

    run_check(conn, "檢查 3：台積電 2025 各季營收（Q4 是否異常？）",
        """SELECT date, value FROM final_report_31
           WHERE code = '2330' AND standard_name = 'revenue' AND date LIKE '2025%'
           ORDER BY date""")

    run_check(conn, "檢查 4：NVDA 的 EPS 合理嗎？",
        """SELECT date, period, value FROM final_report_31
           WHERE code = 'NVDA' AND standard_name = 'eps'
           ORDER BY date DESC LIMIT 4""")

    conn.close()


if __name__ == "__main__":
    main()