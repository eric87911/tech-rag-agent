import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

db_path = Path(os.getenv("SOURCE_DB"))
upstream_dir = db_path.parent
print(f"上游資料夾：{upstream_dir}\n")

# 第 ① ② 關：檔案有沒有下載、清洗出來
for folder in ["from_web", "row_data"]:
    files = list((upstream_dir / folder).glob("*2026Q*.csv"))
    print(f"[{folder}] 2026 年的檔案：{len(files)} 個")
    for f in files[:3]:
        print(f"    例如 {f.name}")

# 第 ③ ④ 關：資料有沒有進資料庫（唯讀模式）
with sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True) as conn:
    for table in ["all_financials", "final_report_31"]:
        n = conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE date LIKE '2026%' AND length(date) = 11"
        ).fetchone()[0]
        print(f"[{table}] 2026 年的資料：{n:,} 筆")