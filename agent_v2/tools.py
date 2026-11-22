"""工具層：把 API 的端點包成 Gemini 看得懂的「工具」。

設計原則：
1. AI 只能透過 HTTP 呼叫 API，不 import 資料庫模組（N-tier：AI 在最外層，碰不到資料）
2. 工具的 description 就是給模型看的說明書，決定模型何時、怎麼用它
3. 工具執行失敗時「回傳錯誤內容給模型」而不是丟例外，讓模型有機會自己修正或誠實回報
"""
import logging
import os
import time

import httpx
from google.genai import types

logger = logging.getLogger(__name__)

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
HTTP_TIMEOUT = 10  # 秒

ITEMS = [
    "accounts_payable", "accounts_receivable", "cash_equivalents", "cogs", "current_assets",
    "eps", "gross_profit", "inventory", "long_term_debt", "operating_expenses",
    "operating_income", "ppe", "retained_earnings", "revenue",
]
RATIO_METRICS = ["gross_margin", "operating_margin", "opex_ratio"]
ITEM_HELP = ("財報科目代碼：revenue=營收, cogs=銷貨成本, gross_profit=毛利, operating_expenses=營業費用, "
             "operating_income=營業利益, eps=每股盈餘, cash_equivalents=現金, accounts_receivable=應收帳款, "
             "inventory=存貨, current_assets=流動資產, ppe=不動產廠房設備, accounts_payable=應付帳款, "
             "long_term_debt=長期負債, retained_earnings=保留盈餘")

S = types.Schema


def _str(desc, enum=None):
    return S(type="STRING", description=desc, enum=enum)


# ---------- 1. 工具宣告（給模型看的） ----------
TOOL_DECLARATIONS = [
    types.FunctionDeclaration(
        name="search_company",
        description=("用公司名稱、暱稱或股票代號查詢公司，回傳股票代號。"
                     "其他工具都需要股票代號，所以只要使用者提到公司名稱，就要先呼叫這個工具。"
                     "可能回傳多家公司（例如「長榮」有海運和航空），此時請向使用者確認是哪一家，不要自己猜。"),
        parameters=S(type="OBJECT", properties={"q": _str("公司名稱、暱稱或代號，例如 台積電、2330")},
                     required=["q"]),
    ),
    types.FunctionDeclaration(
        name="get_financials",
        description=("查詢「某一家公司」某個科目在一段期間的季度數字。"
                     "數值單位是新台幣千元（TWD_thousand），EPS 例外為新台幣元。"
                     "沒有指定期間會回傳全部期間。資料範圍 2021Q1～2025Q4。"),
        parameters=S(type="OBJECT", properties={
            "code": _str("股票代號，例如 2330（先用 search_company 取得）"),
            "item": _str(ITEM_HELP, enum=ITEMS),
            "period_from": _str("起始期間，格式 2025Q1，可省略"),
            "period_to": _str("結束期間，格式 2025Q4，可省略"),
        }, required=["code", "item"]),
    ),
    types.FunctionDeclaration(
        name="get_ranking",
        description=("查詢「某個期間」某科目的公司排名（例如營收前五名）。"
                     "回傳結果包含 ranked_count（參與排名的公司數）與 missing_companies（該期間沒資料的公司）。"
                     "若有公司缺資料，回答時必須說明排名只涵蓋有資料的公司。"),
        parameters=S(type="OBJECT", properties={
            "item": _str(ITEM_HELP, enum=ITEMS),
            "period": _str("單一期間，格式 2025Q4"),
            "top_n": S(type="INTEGER", description="取前幾名，1~20，預設 10"),
            "order": _str("desc=由大到小（找最高），asc=由小到大（找最低）", enum=["desc", "asc"]),
            "industry": _str("只在某產業內排名，可省略。可用值：半導體與硬體製造、工業、能源與材料、"
                             "消費性產品與零售、航運、物流與交通、資訊科技與軟體服務、醫療保健與生物技術"),
        }, required=["item", "period"]),
    ),
]
TOOL_DECLARATIONS.append(
    types.FunctionDeclaration(
        name="get_ratio",
        description=("計算「某一家公司」的財務比率（毛利率、營業利益率、營業費用率）在一段期間的走勢。"
                     "比率由程式計算，需要比率時一律用這個工具，不要自己拿兩個數字相除。"
                     "回傳的 ratio_pct 是百分比。若 skipped_periods 不是空的，回答時必須說明哪些期間缺資料。"
                     "資料範圍 2021Q1～2025Q4，沒有指定期間會回傳全部期間。"),
        parameters=S(type="OBJECT", properties={
            "code": _str("股票代號，例如 2330（先用 search_company 取得）"),
            "metric": _str("gross_margin=毛利率, operating_margin=營業利益率, opex_ratio=營業費用率",
                           enum=RATIO_METRICS),
            "period_from": _str("起始期間，格式 2024Q1，可省略"),
            "period_to": _str("結束期間，格式 2025Q4，可省略"),
        }, required=["code", "metric"]),
    )
)
GEMINI_TOOLS = [types.Tool(function_declarations=TOOL_DECLARATIONS)]

# ---------- 2. 工具名稱 → (HTTP 路徑, 允許的參數)，白名單 ----------
_ROUTES = {
    "search_company": ("/companies", {"q"}),
    "get_financials": ("/financials", {"code", "item", "period_from", "period_to"}),
    "get_ranking": ("/rankings", {"item", "period", "top_n", "order", "industry"}),
    "get_ratio": ("/ratios", {"code", "metric", "period_from", "period_to"}),
}


def _clean_args(args: dict, allowed: set) -> dict:
    """只留白名單內的參數，丟掉空值，並處理 Gemini 把整數回成浮點數的問題（5.0 → 5）"""
    out = {}
    for k, v in (args or {}).items():
        if k not in allowed or v is None or v == "":
            continue
        if isinstance(v, float) and v.is_integer():
            v = int(v)
        out[k] = v
    return out


# ---------- 3. 工具執行 ----------
def execute_tool(name: str, args: dict) -> dict:
    """執行工具並回傳 dict。任何失敗都包成 {"error": ...} 交給模型，不丟例外。"""
    if name not in _ROUTES:
        logger.warning(f"模型呼叫了不存在的工具：{name}")
        return {"error": "unknown_tool", "message": f"沒有名為 {name} 的工具"}

    path, allowed = _ROUTES[name]
    params = _clean_args(args, allowed)
    start = time.perf_counter()
    try:
        resp = httpx.get(f"{API_BASE_URL}{path}", params=params, timeout=HTTP_TIMEOUT)
    except httpx.HTTPError as e:
        logger.error(f"工具 {name} 無法連線到 API：{type(e).__name__}: {e}")
        return {"error": "api_unavailable", "message": "資料服務暫時無法連線"}
    ms = (time.perf_counter() - start) * 1000
    logger.info(f"工具 {name}({params}) → HTTP {resp.status_code}，{ms:.0f}ms")

    if resp.status_code == 200:
        return resp.json()
    # 404 = 查無資料；422 = 參數有誤；503 = 服務問題。把原因原樣交給模型，由模型決定怎麼回應
    detail = resp.json().get("detail", "") if resp.headers.get("content-type", "").startswith("application/json") else ""
    return {"error": {404: "not_found", 422: "invalid_arguments", 503: "service_error"}.get(resp.status_code, "error"),
            "message": str(detail)[:300]}


# ---------- 4. 啟動時檢查：工具宣告的科目清單是否和 API 同步 ----------
def check_tools_in_sync() -> None:
    """啟動時檢查：工具宣告裡的列舉值，是否和 API 實際接受的一致。"""
    try:
        resp = httpx.get(f"{API_BASE_URL}/items", timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        schemas = httpx.get(f"{API_BASE_URL}/openapi.json", timeout=HTTP_TIMEOUT).json()["components"]["schemas"]
    except httpx.HTTPError as e:
        raise RuntimeError(f"連不上 API（{API_BASE_URL}）：{type(e).__name__}。"
                           "請先在另一個終端機執行：uvicorn api.main:app") from e
    items = {x["item"] for x in resp.json()}
    if items != set(ITEMS):
        raise RuntimeError(f"工具宣告與 API 的科目清單不一致：差異 {items ^ set(ITEMS)}")
    metrics = set(schemas["RatioMetric"]["enum"])
    if metrics != set(RATIO_METRICS):
        raise RuntimeError(f"工具宣告與 API 的比率種類不一致：差異 {metrics ^ set(RATIO_METRICS)}")
    logger.info("✅ 工具宣告與 API 一致（科目、比率種類）")
