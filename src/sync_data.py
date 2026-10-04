import os
import sqlite3
import logging
from pathlib import Path

from dotenv import load_dotenv

import clean_financials
import build_mvp_tables

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
RAW_DB = ROOT / "data" / "finance_raw.db"


def get_source_path() -> Path:
    load_dotenv(ROOT / ".env")
    src = os.getenv("SOURCE_DB")
    if not src:
        raise RuntimeError("請在 .env 設定 SOURCE_DB（上游 ETL 專案的資料庫路徑）")
    path = Path(src)
    if not path.exists():
        raise FileNotFoundError(f"找不到上游資料庫：{path}")
    return path


def snapshot(source: Path):
    """用 SQLite 的 backup API 複製一份快照。
    比直接複製檔案安全：就算上游 ETL 正在寫入，也能拿到一致的版本。"""
    with sqlite3.connect(f"{source.as_uri()}?mode=ro", uri=True) as src, \
         sqlite3.connect(RAW_DB) as dst:
        src.backup(dst)
        latest = dst.execute(
            "SELECT MAX(date) FROM final_report_31 WHERE length(date) = 11").fetchone()[0]
    logger.info(f"已從上游複製快照，台股最新資料日期：{latest}")


def main():
    source = get_source_path()
    logger.info(f"上游資料庫：{source}")
    snapshot(source)
    clean_financials.main()
    build_mvp_tables.main()
    logger.info("🎊 資料同步完成")


if __name__ == "__main__":
    main()