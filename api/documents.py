"""文件檢索（資料存取層）：把 v1 的向量檢索包成 API 可以呼叫的函式。

設計重點：
- 向量庫是「選用」功能：它壞了只影響 /documents/search，不能拖垮財報 API，所以 v1 的模組在第一次用到時才載入。
- 沒找到相關段落不是錯誤（回傳空串列）；基礎設施壞掉才是錯誤（DocumentsUnavailable → API 回 503）。
"""
import logging
import sys
import threading
from pathlib import Path

from api import db, service

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = str(ROOT / "src")

COMPANY_NAMES = {"TSMC": "台積電"}
# 向量庫的 metadata 只存 company="TSMC"，沒有股票代號；新增其他公司的文件時，這裡要一起補（之後可改成建庫時直接存 code）
DOC_COMPANY_CODES = {"TSMC": "2330"}
CHAPTER_NAMES = {"letter": "致股東報告書", "operations": "營運概況"}


class DocumentsUnavailable(Exception):
    """向量庫、嵌入服務或 v1 模組無法使用。細節只寫進 log，不回傳給呼叫端。"""


_lock = threading.Lock()
_ctx: dict = {}          # 初始化成功後才會有內容；失敗不快取，下一次請求會重試


def _label(meta: dict, with_location: bool = True) -> str:
    """把 metadata 轉成人看得懂的來源名稱（邏輯同 v1 的 source_label，複製是為了不去 import 很重的 rag_generator）"""
    company = COMPANY_NAMES.get(meta["company"], meta["company"])
    if meta["doc_type"] == "transcript":
        base = f"{company} {meta['period']} 法說會逐字稿（第三方整理）"
        extra = meta["location"].split(" - ")[0]
    else:
        chapter = CHAPTER_NAMES.get(Path(meta["source"]).stem.split("_")[-1], "")
        base = f"{company} {meta['period']} 年報 {chapter}".strip()
        extra = meta["location"]
    return f"{base}｜{extra}" if with_location else base


def _init() -> None:
    with _lock:
        if _ctx:
            return
        try:
            if SRC_DIR not in sys.path:
                sys.path.append(SRC_DIR)        # v1 的模組是「平放」的 import 方式（from vector_store import ...）
            from retriever import retrieve
            from vector_store import get_collection, get_embed_client
            collection = get_collection()
            client, model = get_embed_client()
            metas = collection.get(include=["metadatas"])["metadatas"]
            coverage = sorted({_label(m, with_location=False) for m in metas})
            doc_companies = sorted({m["company"] for m in metas})
        except Exception as e:
            logger.error("文件檢索初始化失敗：%s: %s", type(e).__name__, e, exc_info=True)   # 完整堆疊：部署後只能靠日誌排錯
            raise DocumentsUnavailable("init failed") from e
        _ctx.update(retrieve=retrieve, collection=collection, client=client, model=model,
                    coverage=coverage, doc_companies=doc_companies)
        logger.info(f"✅ 文件檢索就緒：{collection.count()} 段，涵蓋 {coverage}")


def check_documents() -> bool:
    """啟動時暖機。失敗只記警告，不阻止 API 啟動（財報查詢不需要它）。"""
    try:
        _init()
        return True
    except DocumentsUnavailable:
        logger.warning("⚠️ 文件檢索無法使用，/documents/search 會回 503；財報端點不受影響")
        return False


def _match_company(company: str) -> list[str]:
    """把使用者說的公司（台積電 / tsmc / 2330）對應到「文件庫裡有文件的公司」。對不上回傳空串列。"""
    try:
        codes = {c["code"] for c in service.search_companies(company)}
    except db.DatabaseError as e:
        raise DocumentsUnavailable("company lookup failed") from e
    return [c for c in _ctx["doc_companies"] if DOC_COMPANY_CODES.get(c) in codes]


def search(query: str, top_k: int, company: str | None = None) -> tuple[list[dict], list[str], str | None]:
    """回傳 (段落, 文件庫涵蓋範圍, 說明)。指定 company 但文件庫沒有該公司 → 直接回空，不做檢索
    （向量相似度只看語意，分不出「問的是哪家公司」，不先擋就會撈到別家公司的段落）。"""
    _init()
    where, note = None, None
    if company:
        matched = _match_company(company)
        if not matched:
            names = "、".join(COMPANY_NAMES.get(c, c) for c in _ctx["doc_companies"])
            logger.info(f"文件庫沒有 {company!r} 的文件，不檢索")
            return [], _ctx["coverage"], f"文件庫沒有「{company}」的文件，目前只涵蓋：{names}"
        where = {"company": matched[0]} if len(matched) == 1 else {"company": {"$in": matched}}
    try:
        hits = _ctx["retrieve"](query, _ctx["client"], _ctx["model"], _ctx["collection"], top_k=top_k, where=where)
        passages = [{
            "rank": i,
            "score": round(h.score, 3),
            "source_label": _label(h.metadata),
            "company": COMPANY_NAMES.get(h.metadata["company"], h.metadata["company"]),
            "doc_type": h.metadata["doc_type"],
            "period": h.metadata["period"],
            "location": h.metadata["location"],
            "text": h.text.split("\n", 1)[-1],        # 去掉切塊時加的標題行，避免重複
        } for i, h in enumerate(hits, start=1)]
    except Exception as e:
        logger.error(f"文件檢索失敗：{type(e).__name__}: {e}", exc_info=True)
        raise DocumentsUnavailable("search failed") from e
    return passages, _ctx["coverage"], note
