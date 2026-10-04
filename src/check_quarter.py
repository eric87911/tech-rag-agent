import os
import csv
import sys
import sqlite3
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")
db_path = Path(os.getenv("SOURCE_DB"))
upstream = db_path.parent

# 用法：python src\check_quarter.py 2330 2026Q1
code = sys.argv[1] if len(sys.argv) > 1 else "2330"
period = sys.argv[2] if len(sys.argv) > 2 else "2026Q1"
year = period[:4]
fname = f"mops_{code}_{period}.csv"
print(f"檢查 {code} {period}\n")

# ① 原始下載：找出含「會計項目」的標題列
raw = upstream / "from_web" / fname
if raw.exists():
    print(f"① from_web：存在，{raw.stat().st_size:,} bytes")
    with open(raw, encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if "會計項目" in "".join(row):
                print(f"   標題列：{row[:5]}")
                break
        else:
            print("   ⚠️ 找不到含「會計項目」的標題列")
else:
    print("① from_web：❌ 檔案不存在")

# ② 清洗後：date 欄位是什麼？
clean = upstream / "row_data" / fname
if clean.exists():
    with open(clean, encoding="utf-8-sig") as f:
        dates = Counter(r["date"] for r in csv.DictReader(f))
    print(f"② row_data：date 欄位值 {dict(dates)}")
else:
    print("② row_data：❌ 檔案不存在")

# ③ ④ 資料庫（唯讀）
with sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True) as conn:
    for table in ["all_financials", "final_report_31"]:
        rows = conn.execute(
            f"SELECT date, COUNT(*) FROM {table} WHERE code = ? AND date LIKE ? GROUP BY date",
            (code, f"{year}%")).fetchall()
        print(f"③④ {table} {year} 年的日期：{rows}")
    rows = conn.execute("SELECT date, COUNT(*) FROM all_financials WHERE code = ? "
                        "AND date NOT LIKE '20%' GROUP BY date", (code,)).fetchall()
    print(f"   不是 20 開頭的日期（可能是格式錯誤）：{rows}")

# 總覽：所有公司這一季的日期標籤長什麼樣子？
labels = Counter()
for p in (upstream / "row_data").glob(f"mops_*_{period}.csv"):
    with open(p, encoding="utf-8-sig") as f:
        first = next(csv.DictReader(f), None)
    labels[first["date"] if first else "（空檔案）"] += 1
print(f"\n所有公司 {period} 的日期標籤：{dict(labels)}")