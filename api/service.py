"""商業邏輯層：規則放這裡，不放在端點（main.py），也不放在資料存取（db.py）。

目前包含：
1. 公司搜尋（代號 / 全名 / 暱稱）
2. 科目的性質分類（流量 flow / 存量 stock）
"""
import logging

from api import db

logger = logging.getLogger(__name__)

# 暱稱表：只收「不是全名子字串」的說法（例如 台積電 不是 台灣積體電路製造 的子字串）
# 資料庫裡已有的簡稱（鴻海、廣達、華碩…）本來就搜得到，不用重複列
ALIASES: dict[str, str] = {
    "台積電": "2330", "台積": "2330", "tsmc": "2330",
    "聯電": "2303", "umc": "2303",
    "富士康": "2317", "foxconn": "2317",
    "聯發科": "2454", "mediatek": "2454",
    "日月光": "3711", "ase": "3711",
    "台哥大": "3045",
    "中鋼": "2002",
    "台塑": "1301",
    "台化": "1326",
    "台泥": "1101",
    "台塑化": "6505",
    "7-11": "2912", "小七": "2912",
}

# 流量：一段期間內「累積發生」的數字（損益表）；存量：某一天「當下的餘額」（資產負債表）
FLOW_ITEMS = {"revenue", "cogs", "gross_profit", "operating_expenses", "operating_income", "eps"}


def item_kind(item: str) -> str:
    return "flow" if item in FLOW_ITEMS else "stock"


def short_name(full_name: str) -> str:
    return full_name.replace("股份有限公司", "")


def search_companies(q: str) -> list[dict]:
    """依代號、全名子字串、暱稱搜尋。可能回傳多家（例如「長榮」），由呼叫端決定怎麼處理。"""
    key = q.strip().casefold()
    if not key:
        return []

    rows, _ = db.fetch_all("SELECT code, company_name, industry FROM companies ORDER BY code")
    alias_codes = {code for alias, code in ALIASES.items() if key in alias.casefold()}

    result = []
    for r in rows:
        if r["code"] == key or key in r["company_name"].casefold() or r["code"] in alias_codes:
            result.append({**r, "short_name": short_name(r["company_name"])})
    logger.info(f"公司搜尋 q={q!r} → {len(result)} 筆")
    return result


# ORDER BY 的方向不能用 ? 佔位符傳，所以用固定對照表，只允許這兩個字面值
ORDER_SQL = {"desc": "DESC", "asc": "ASC"}


def get_ranking(item: str, period: str, top_n: int, order: str, industry: str | None) -> dict | None:
    """某科目某期間的排名。回傳排名＋「沒資料的公司」，避免呼叫端把排名誤當成全體。
    該期間該科目完全沒資料時回傳 None。"""
    ind_sql, ind_params = ("", ())
    if industry:
        ind_sql, ind_params = " AND industry = ?", (industry,)

    # 1) 排名本體（排序方向走對照表，數值走 ?；同值時以代號排序，結果才穩定）
    rows, _ = db.fetch_all(
        "SELECT code, company_name, display_name, value, unit FROM v_financials "
        f"WHERE item = ? AND period = ? AND value IS NOT NULL{ind_sql} "
        f"ORDER BY value {ORDER_SQL[order]}, code LIMIT ?",
        (item, period, *ind_params, top_n))
    if not rows:
        return None

    ranked_count = db.fetch_all(
        f"SELECT COUNT(*) AS n FROM v_financials WHERE item = ? AND period = ? AND value IS NOT NULL{ind_sql}",
        (item, period, *ind_params))[0][0]["n"]

    # 2) 這個範圍內「沒有資料」的公司
    missing, _ = db.fetch_all(
        "SELECT code, company_name FROM companies WHERE 1=1"
        f"{ind_sql} AND code NOT IN "
        "(SELECT code FROM v_financials WHERE item = ? AND period = ? AND value IS NOT NULL) "
        "ORDER BY code",
        (*ind_params, item, period))

    return {
        "display_name": rows[0]["display_name"],
        "ranked_count": ranked_count,
        "total_companies": ranked_count + len(missing),
        "data": [{"rank": i, "code": r["code"], "company_name": r["company_name"],
                  "value": r["value"], "unit": r["unit"],
                  "value_display": format_value(r["value"], r["unit"])} for i, r in enumerate(rows, 1)],
        "missing_companies": missing,
    }


# ---------- 單位顯示：千元 → 兆 / 億，由程式換算，不交給模型心算 ----------
def format_value(value: float, unit: str) -> str:
    if unit == "TWD":                                   # EPS 是「元」
        return f"{value:,.2f} 元"
    v = abs(value)
    sign = "-" if value < 0 else ""
    if v >= 1e9:                                        # 1 兆元 = 1e9 千元
        return f"{sign}{v / 1e9:,.2f} 兆元"
    if v >= 1e5:                                        # 1 億元 = 1e5 千元
        return f"{sign}{v / 1e5:,.1f} 億元"
    return f"{value:,.0f} 千元"


# ---------- 比率：由「分子科目 ÷ 營收」算出，缺任一邊就列入 skipped，不默默略過 ----------
RATIOS = {
    "gross_margin": ("毛利率", "gross_profit", "revenue"),
    "operating_margin": ("營業利益率", "operating_income", "revenue"),
    "opex_ratio": ("營業費用率", "operating_expenses", "revenue"),
}


def compute_ratios(code: str, metric: str, period_from: str | None, period_to: str | None) -> dict | None:
    label, num_item, den_item = RATIOS[metric]
    sql = ("SELECT company_name, period, item, value FROM v_financials "
           "WHERE code = ? AND item IN (?, ?)")
    params: list = [code, num_item, den_item]
    if period_from:
        sql += " AND period >= ?"
        params.append(period_from)
    if period_to:
        sql += " AND period <= ?"
        params.append(period_to)
    rows, _ = db.fetch_all(sql, tuple(params))
    if not rows:
        return None

    by_period: dict[str, dict] = {}
    for r in rows:
        by_period.setdefault(r["period"], {})[r["item"]] = r["value"]

    # 資料庫在這段範圍內「應該有」的期間；這家公司整期都沒資料的，也要列入 skipped，不能默默消失
    p_sql, p_params = "SELECT DISTINCT period FROM v_financials WHERE 1=1", []
    if period_from:
        p_sql += " AND period >= ?"
        p_params.append(period_from)
    if period_to:
        p_sql += " AND period <= ?"
        p_params.append(period_to)
    expected = {r["period"] for r in db.fetch_all(p_sql, tuple(p_params))[0]}

    data, skipped = [], []
    for period in sorted(expected | set(by_period)):
        n, d = by_period.get(period, {}).get(num_item), by_period.get(period, {}).get(den_item)
        if period not in by_period:
            skipped.append({"period": period, "reason": "該期間沒有這家公司的財報資料"})
        elif n is None or d is None:
            missing = num_item if n is None else den_item
            skipped.append({"period": period, "reason": f"缺少 {missing}"})
        elif d == 0:
            skipped.append({"period": period, "reason": "營收為 0，無法計算"})
        else:
            data.append({"period": period, "ratio_pct": round(n / d * 100, 2),
                         "numerator": n, "denominator": d})
    summary = None
    if data:
        pick = lambda x: {"period": x["period"], "ratio_pct": x["ratio_pct"]}
        highest = max(data, key=lambda x: x["ratio_pct"])      # 最高 / 最低由程式找，模型不要自己掃數列
        lowest = min(data, key=lambda x: x["ratio_pct"])
        summary = {"first": pick(data[0]), "latest": pick(data[-1]),
                   "highest": pick(highest), "lowest": pick(lowest),
                   "change_pp": round(data[-1]["ratio_pct"] - data[0]["ratio_pct"], 2)}
    return {"company_name": rows[0]["company_name"], "display_name": label,
            "formula": f"{num_item} ÷ {den_item}", "data": data, "summary": summary,
            "skipped_periods": skipped}
