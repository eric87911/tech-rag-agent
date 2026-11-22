"""API 的輸入輸出格式。

輸入：用 Enum 把「可查的科目」鎖成白名單，格式不對 FastAPI 自動回 422，
      根本進不到資料庫那一層。
輸出：固定回傳結構，讓 AI（或任何呼叫端）永遠知道會拿到什麼欄位。
"""
from enum import Enum

from pydantic import BaseModel, Field

# 期間格式：4 位數年 + Q + 1~4，例如 2025Q4
PERIOD_PATTERN = r"^\d{4}Q[1-4]$"


class Item(str, Enum):
    """可查詢的 14 個科目（與 build_mvp_tables.py 的 VERIFIED_ITEMS 一致）"""
    accounts_payable = "accounts_payable"
    accounts_receivable = "accounts_receivable"
    cash_equivalents = "cash_equivalents"
    cogs = "cogs"
    current_assets = "current_assets"
    eps = "eps"
    gross_profit = "gross_profit"
    inventory = "inventory"
    long_term_debt = "long_term_debt"
    operating_expenses = "operating_expenses"
    operating_income = "operating_income"
    ppe = "ppe"
    retained_earnings = "retained_earnings"
    revenue = "revenue"


class FinancialRecord(BaseModel):
    code: str
    company_name: str
    period: str
    item: str
    display_name: str
    value: float | None
    unit: str = Field(description="TWD_thousand（新台幣千元）；EPS 為 TWD")
    value_display: str | None = Field(default=None, description="易讀版數字，例如 1.05 兆元")


class FinancialsResponse(BaseModel):
    count: int
    truncated: bool = Field(description="True 代表結果超過上限被截斷，還有更多資料")
    data: list[FinancialRecord]


class Company(BaseModel):
    code: str
    company_name: str
    short_name: str
    industry: str


class CompaniesResponse(BaseModel):
    count: int
    data: list[Company]


class ItemInfo(BaseModel):
    item: str
    display_name: str
    unit: str
    kind: str = Field(description="flow=流量（損益表，一段期間的累積）；stock=存量（資產負債表，某時點餘額）")


class PeriodsResponse(BaseModel):
    earliest: str
    latest: str
    periods: list[str]


class HealthResponse(BaseModel):
    status: str
    latest_period: str


class Order(str, Enum):
    desc = "desc"   # 由大到小（預設，找「最高」）
    asc = "asc"     # 由小到大（找「最低」）


class RankingEntry(BaseModel):
    rank: int
    code: str
    company_name: str
    value: float
    unit: str
    value_display: str | None = None


class MissingCompany(BaseModel):
    code: str
    company_name: str


class RankingsResponse(BaseModel):
    item: str
    display_name: str
    period: str
    order: str
    industry: str | None
    ranked_count: int = Field(description="該期間該科目「有資料」而參與排名的公司數")
    total_companies: int = Field(description="範圍內的公司總數（有指定產業時只算該產業）")
    data: list[RankingEntry]
    missing_companies: list[MissingCompany] = Field(
        description="該期間該科目沒有資料、因此沒參與排名的公司。排名結果不代表他們的名次")


class RatioMetric(str, Enum):
    gross_margin = "gross_margin"          # 毛利率 = 毛利 ÷ 營收
    operating_margin = "operating_margin"  # 營業利益率 = 營業利益 ÷ 營收
    opex_ratio = "opex_ratio"              # 營業費用率 = 營業費用 ÷ 營收


class RatioPoint(BaseModel):
    period: str
    ratio_pct: float = Field(description="百分比，已四捨五入到小數 2 位")
    numerator: float
    denominator: float


class SkippedPeriod(BaseModel):
    period: str
    reason: str


class RatioExtreme(BaseModel):
    period: str
    ratio_pct: float


class RatioSummary(BaseModel):
    first: RatioExtreme
    latest: RatioExtreme
    highest: RatioExtreme
    lowest: RatioExtreme
    change_pp: float = Field(description="最新一期減第一期，單位是百分點（percentage points）")


class RatiosResponse(BaseModel):
    code: str
    company_name: str
    metric: str
    display_name: str
    formula: str
    data: list[RatioPoint]
    summary: RatioSummary | None = Field(description="最高、最低、首尾變化；由程式計算，沒有資料時為 None")
    skipped_periods: list[SkippedPeriod] = Field(description="無法計算的期間與原因（缺資料不會被默默略過）")


class Passage(BaseModel):
    rank: int
    score: float = Field(description="相似度，越接近 1 越相關")
    source_label: str
    company: str
    doc_type: str
    period: str
    location: str
    text: str


class DocumentSearchResponse(BaseModel):
    query: str
    count: int
    passages: list[Passage]
    coverage: list[str] = Field(description="文件庫目前涵蓋的文件；count 為 0 時，用它告訴使用者「哪些文件有查過」")
    note: str | None = Field(default=None, description="例如文件庫沒有所問公司的文件時的說明")
