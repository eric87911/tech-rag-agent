"""資料存取層：整個 API 唯一碰得到資料庫的地方。

設計原則：
1. 一律以唯讀模式連線（mode=ro）
2. 每次請求開一條新連線，用完就關（SQLite 連線預設不能跨執行緒共用）
3. SQL 的參數一律用 ? 佔位符傳入，不用字串拼接
"""
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "finance.db"
MAX_ROWS = 200


class DatabaseError(Exception):
    """資料庫層的錯誤，上層只需要處理這一種例外，不用認識 sqlite3"""


@contextmanager
def get_connection(db_path: Path = DB_PATH):
    # 注意：`with sqlite3.connect()` 只會處理交易，不會關閉連線，所以自己用 try/finally 關
    conn = sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row          # 讓查詢結果可以用欄位名稱取值
    try:
        yield conn
    finally:
        conn.close()


def fetch_all(sql: str, params: tuple = ()) -> tuple[list[dict], bool]:
    """執行查詢，回傳 (資料列, 是否被截斷)。params 只放「值」，不放欄位或表名。"""
    try:
        with get_connection() as conn:
            rows = conn.execute(sql, params).fetchmany(MAX_ROWS + 1)   # 多取一筆判斷有沒有被截斷
    except sqlite3.Error as e:
        logger.error(f"資料庫查詢失敗：{type(e).__name__}: {e}")
        raise DatabaseError("資料庫查詢失敗") from e

    truncated = len(rows) > MAX_ROWS
    if truncated:
        logger.warning(f"結果超過 {MAX_ROWS} 筆，已截斷")
    return [dict(r) for r in rows[:MAX_ROWS]], truncated


def check_database(db_path: Path = DB_PATH) -> None:
    """啟動時自我檢查：檔案存在、確實唯讀、資料表有資料。任何一項不過就拒絕啟動。"""
    if not db_path.exists():
        raise RuntimeError(f"找不到資料庫：{db_path}")

    with get_connection(db_path) as conn:
        # 不信任設定，直接試寫一次：寫得進去就代表唯讀失效
        try:
            conn.execute("CREATE TABLE _write_probe (x INTEGER)")
        except sqlite3.OperationalError:
            logger.info("✅ 唯讀驗證通過：資料庫拒絕寫入")
        else:
            raise RuntimeError("資料庫連線可以寫入，唯讀設定失效，拒絕啟動")

        n = conn.execute("SELECT COUNT(*) FROM v_financials").fetchone()[0]
        if n == 0:
            raise RuntimeError("v_financials 是空的")
        logger.info(f"✅ 資料庫檢查通過：v_financials 共 {n:,} 筆")
