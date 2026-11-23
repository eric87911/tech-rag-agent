"""FastAPI 入口：端點只負責「收請求、驗證、呼叫下層、回傳」，不寫 SQL 細節。"""
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query

from api import db, documents, security, service
from api.schemas import (PERIOD_PATTERN, CompaniesResponse, DocumentSearchResponse, FinancialsResponse,
                         HealthResponse, ItemInfo, Item, Order, PeriodsResponse,
                         RankingsResponse, RatioMetric, RatiosResponse)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    security.check_config()      # 沒設金鑰就直接拒絕啟動（預設拒絕）
    db.check_database()          # 檢查沒過就直接拒絕啟動
    documents.check_documents()  # 文件檢索是選用功能：失敗只警告，不阻止啟動
    yield


app = FastAPI(title="Tech Finance API", version="2.0", lifespan=lifespan)

# 除了 /health（健康檢查要用，不含敏感資料）之外，每個端點都要驗證金鑰並受限流。
# tests/test_security.py 會檢查：之後新增的端點如果忘了加，測試會失敗。
PROTECTED = [Depends(security.rate_limit_general)]
DOCS_PROTECTED = [Depends(security.rate_limit_docs)]   # 文件檢索每次都花錢，另外限得更緊


@app.get("/financials", dependencies=PROTECTED, response_model=FinancialsResponse)
def get_financials(
    code: str = Query(..., pattern=r"^\d{4,6}$", description="股票代號，例如 2330"),
    item: Item = Query(..., description="財報科目，例如 revenue"),
    period_from: str | None = Query(None, pattern=PERIOD_PATTERN, description="起始期間，例如 2024Q1"),
    period_to: str | None = Query(None, pattern=PERIOD_PATTERN, description="結束期間，例如 2025Q4"),
):
    if period_from and period_to and period_from > period_to:
        raise HTTPException(status_code=422, detail="period_from 不能晚於 period_to")

    # SQL 的結構是固定的；使用者給的值全部走 ? 佔位符
    sql = ("SELECT code, company_name, period, item, display_name, value, unit "
           "FROM v_financials WHERE code = ? AND item = ?")
    params: list = [code, item.value]
    if period_from:
        sql += " AND period >= ?"
        params.append(period_from)
    if period_to:
        sql += " AND period <= ?"
        params.append(period_to)
    sql += " ORDER BY period"

    try:
        rows, truncated = db.fetch_all(sql, tuple(params))
    except db.DatabaseError:
        raise HTTPException(status_code=503, detail="資料庫暫時無法查詢")

    if not rows:
        raise HTTPException(status_code=404, detail=f"查無資料：{code} / {item.value}")
    for r in rows:
        r["value_display"] = service.format_value(r["value"], r["unit"]) if r["value"] is not None else None
    return FinancialsResponse(count=len(rows), truncated=truncated, data=rows)


@app.get("/health", response_model=HealthResponse)
def health():
    """服務存活檢查（之後 systemd / nginx / 監控都會打這支）"""
    try:
        rows, _ = db.fetch_all("SELECT MAX(period) AS latest FROM v_financials")
    except db.DatabaseError:
        raise HTTPException(status_code=503, detail="資料庫無法連線")
    return HealthResponse(status="ok", latest_period=rows[0]["latest"])


@app.get("/companies", dependencies=PROTECTED, response_model=CompaniesResponse)
def search_companies(q: str = Query(..., min_length=1, max_length=20, description="代號、公司名或暱稱，例如 台積電")):
    """把使用者的說法對應成股票代號。可能回傳多家（例如「長榮」），呼叫端要自己處理歧義。"""
    try:
        data = service.search_companies(q)
    except db.DatabaseError:
        raise HTTPException(status_code=503, detail="資料庫暫時無法查詢")
    if not data:
        raise HTTPException(status_code=404, detail=f"找不到公司：{q}")
    return CompaniesResponse(count=len(data), data=data)


@app.get("/items", dependencies=PROTECTED, response_model=list[ItemInfo])
def list_items():
    try:
        rows, _ = db.fetch_all("SELECT DISTINCT item, display_name, unit FROM v_financials ORDER BY item")
    except db.DatabaseError:
        raise HTTPException(status_code=503, detail="資料庫暫時無法查詢")
    return [{**r, "kind": service.item_kind(r["item"])} for r in rows]


@app.get("/periods", dependencies=PROTECTED, response_model=PeriodsResponse)
def list_periods():
    try:
        rows, _ = db.fetch_all("SELECT DISTINCT period FROM v_financials ORDER BY period")
    except db.DatabaseError:
        raise HTTPException(status_code=503, detail="資料庫暫時無法查詢")
    periods = [r["period"] for r in rows]
    return PeriodsResponse(earliest=periods[0], latest=periods[-1], periods=periods)


@app.get("/rankings", dependencies=PROTECTED, response_model=RankingsResponse)
def get_rankings(
    item: Item = Query(..., description="財報科目，例如 revenue"),
    period: str = Query(..., pattern=PERIOD_PATTERN, description="期間，例如 2025Q4"),
    top_n: int = Query(10, ge=1, le=20, description="取前幾名（1~20）"),
    order: Order = Query(Order.desc, description="desc=由大到小，asc=由小到大"),
    industry: str | None = Query(None, max_length=20, description="只在某產業內排名，例如 半導體與硬體製造"),
):
    try:
        result = service.get_ranking(item.value, period, top_n, order.value, industry)
    except db.DatabaseError:
        raise HTTPException(status_code=503, detail="資料庫暫時無法查詢")
    if result is None:
        raise HTTPException(status_code=404, detail=f"查無資料：{item.value} / {period}"
                                                    + (f" / {industry}" if industry else ""))
    return RankingsResponse(item=item.value, period=period, order=order.value, industry=industry, **result)


@app.get("/ratios", dependencies=PROTECTED, response_model=RatiosResponse)
def get_ratios(
    code: str = Query(..., pattern=r"^\d{4,6}$", description="股票代號，例如 2330"),
    metric: RatioMetric = Query(..., description="gross_margin=毛利率, operating_margin=營業利益率, opex_ratio=營業費用率"),
    period_from: str | None = Query(None, pattern=PERIOD_PATTERN),
    period_to: str | None = Query(None, pattern=PERIOD_PATTERN),
):
    """由程式計算比率（模型不心算）。缺資料的期間列在 skipped_periods。"""
    if period_from and period_to and period_from > period_to:
        raise HTTPException(status_code=422, detail="period_from 不能晚於 period_to")
    try:
        result = service.compute_ratios(code, metric.value, period_from, period_to)
    except db.DatabaseError:
        raise HTTPException(status_code=503, detail="資料庫暫時無法查詢")
    if result is None:
        raise HTTPException(status_code=404, detail=f"查無資料：{code} / {metric.value}")
    return RatiosResponse(code=code, metric=metric.value, **result)


@app.get("/documents/search", dependencies=DOCS_PROTECTED, response_model=DocumentSearchResponse)
def search_documents(
    q: str = Query(..., min_length=2, max_length=200, description="要在文件中找的問題或關鍵概念"),
    top_k: int = Query(5, ge=1, le=8, description="最多回傳幾段"),
    company: str | None = Query(None, min_length=1, max_length=30, description="所問的公司（名稱、暱稱或代號）；文件庫沒有該公司時直接回空"),
):
    """在法說會逐字稿、年報等文件中找相關段落。找不到相關內容時回 200 與空串列（不是錯誤）。"""
    try:
        passages, coverage, note = documents.search(q, top_k, company)
    except documents.DocumentsUnavailable:
        raise HTTPException(status_code=503, detail="文件檢索暫時無法使用")
    return DocumentSearchResponse(query=q, count=len(passages), passages=passages, coverage=coverage, note=note)
