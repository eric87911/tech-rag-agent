import sqlite3
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
RAW_DB = ROOT / "data" / "finance_raw.db"
CLEAN_DB = ROOT / "data" / "finance.db"


# ---------- 步驟 1：讀取原始資料（唯讀） ----------
def load_raw_tw():
    if not RAW_DB.exists():
        raise FileNotFoundError(f"找不到 {RAW_DB}")
    with sqlite3.connect(f"file:{RAW_DB}?mode=ro", uri=True) as conn:
        # 台股的日期是 11 個字元的「2025年12月31日」格式
        df = pd.read_sql("SELECT code, date, standard_name AS item, value "
                         "FROM final_report_31 WHERE length(date) = 11", conn)
        structure = pd.read_sql("SELECT item, display_name, category FROM report_structure", conn)
    logger.info(f"讀入台股原始資料 {len(df):,} 筆，{df['code'].nunique()} 家公司")
    return df, structure


# ---------- 步驟 2：解析日期 → 年、季 ----------
def parse_period(df):
    dt = pd.to_datetime(df["date"], format="%Y年%m月%d日", errors="coerce")
    bad = dt.isna().sum()
    if bad:
        logger.warning(f"有 {bad} 筆日期無法解析，將被丟棄")
    df = df.assign(year=dt.dt.year, quarter=dt.dt.quarter).dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    df["quarter"] = df["quarter"].astype(int)
    return df


# ---------- 步驟 3：依報表類型還原單季數字 ----------
def to_single_quarter(df, structure):
    # 用 category 的第一個字母分類（原始資料有一筆打錯字「C 現現金流量表」，用字母判斷比較穩）
    kind = structure.set_index("item")["category"].str[0].map(
        {"A": "income", "B": "balance", "C": "cashflow"})
    df["kind"] = df["item"].map(kind)

    # 轉成寬表：每一列是 (公司, 年度, 科目)，欄位是 Q1~Q4
    wide = df.pivot_table(index=["code", "year", "item", "kind"], columns="quarter",
                          values="value", aggfunc="first").reset_index()
    for q in [1, 2, 3, 4]:
        if q not in wide.columns:
            wide[q] = float("nan")

    out = wide.copy()
    inc = wide["kind"] == "income"
    cf = (wide["kind"] == "cashflow") & ~wide["item"].isin(["beginning_cash", "ending_cash"])

    # 損益表：Q4 = 全年 - Q1 - Q2 - Q3（任何一季缺值，結果就是 NaN）
    out.loc[inc, 4] = wide.loc[inc, 4] - wide.loc[inc, 1] - wide.loc[inc, 2] - wide.loc[inc, 3]

    # 現金流量表：單季 = 本季累計 - 上季累計（用 wide 的原始值相減，不能用 out）
    for q in [4, 3, 2]:
        out.loc[cf, q] = wide.loc[cf, q] - wide.loc[cf, q - 1]

    # 期初現金：原始值是「年初」現金，單季期初 = 上一季的期末現金
    begin = out["item"] == "beginning_cash"
    end_cash = wide[wide["item"] == "ending_cash"].set_index(["code", "year"])
    keys = pd.MultiIndex.from_frame(out.loc[begin, ["code", "year"]])
    for q in [2, 3, 4]:
        out.loc[begin, q] = end_cash[q - 1].reindex(keys).to_numpy()

    # 資產負債表（存量）：本來就是季末時點數字，不處理

    # 轉回長表
    long = out.melt(id_vars=["code", "year", "item", "kind"], value_vars=[1, 2, 3, 4],
                    var_name="quarter", value_name="value").dropna(subset=["value"])
    long["quarter"] = long["quarter"].astype(int)
    return long


# ---------- 步驟 4：補上給 LLM 看的欄位 ----------
def add_metadata(df, structure):
    df["period"] = df["year"].astype(str) + "Q" + df["quarter"].astype(str)
    df["market"] = "TW"
    df["unit"] = df["item"].map(lambda x: "TWD" if x == "eps" else "TWD_thousand")
    df = df.merge(structure[["item", "display_name"]], on="item", how="left")
    cols = ["code", "market", "year", "quarter", "period", "item", "display_name", "value", "unit"]
    return df[cols].sort_values(["code", "year", "quarter", "item"])


# ---------- 步驟 5：用公司公告的真實數字驗證 ----------
def validate(df):
    def get(code, period, item):
        s = df[(df.code == code) & (df.period == period) & (df.item == item)]["value"]
        return s.iloc[0] if len(s) else None

    rev = get("2330", "2025Q4", "revenue")   # 台積電公告：1,046,090 百萬元
    eps = get("2330", "2025Q4", "eps")       # 台積電公告：19.50 元
    assert rev is not None and eps is not None, "找不到台積電 2025Q4 資料"
    logger.info(f"驗證：台積電 2025Q4 營收 = {rev:,.0f} 千元，EPS = {eps:.2f}")
    assert 1.0e9 < rev < 1.1e9, "Q4 營收不在合理範圍，請檢查還原邏輯"
    dups = df.duplicated(["code", "period", "item"]).sum()
    assert dups == 0, f"有 {dups} 筆重複資料（同公司、同季、同科目）"


def main():
    raw, structure = load_raw_tw()
    df = parse_period(raw)
    df = to_single_quarter(df, structure)
    df = add_metadata(df, structure)
    validate(df)

    with sqlite3.connect(CLEAN_DB) as conn:
        df.to_sql("financials", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_fin ON financials(code, period, item)")
    logger.info(f"✅ 已寫入 {CLEAN_DB.name}：financials 表 {len(df):,} 筆")


if __name__ == "__main__":
    main()