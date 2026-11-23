"""API 金鑰驗證與限流。

設計重點：
- 預設「拒絕」：沒設 API_KEY 就不啟動（除非明確設 AUTH_DISABLED=1，只給本機開發用）。
  忘了設環境變數時，服務是「起不來」而不是「裸奔」。
- 金鑰可用逗號分隔設多把：輪替時「新舊並存 → 客戶端換完 → 移除舊的」，不必停機。
- 比對用 secrets.compare_digest（固定時間），避免有人靠回應時間一個字元一個字元猜。
- 限流依「驗證過的金鑰」計算，不依 IP：在容器或反向代理後面，所有請求看起來都來自同一個 IP。
  這也讓沒通過驗證的請求不會在記憶體裡建立任何狀態。
- 限流狀態存在記憶體：單一 worker 沒問題；開多個 worker 時實際上限會乘上 worker 數，
  要精確就得改用 Redis 之類的共用儲存。
- 金鑰本身絕不寫進日誌，日誌只記雜湊前 8 碼當作「是哪把金鑰」的代號。
"""
import hashlib
import logging
import os
import secrets
import threading
import time
from collections import deque
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, HTTPException, Request
from fastapi.security import APIKeyHeader

load_dotenv(Path(__file__).resolve().parent.parent / ".env")   # 本機直接跑 uvicorn 時讀 .env；容器裡沒有這個檔，不影響

logger = logging.getLogger(__name__)

MIN_KEY_LENGTH = 16
_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _auth_disabled() -> bool:
    return os.getenv("AUTH_DISABLED") == "1"


def _configured_keys() -> list[str]:
    return [k.strip() for k in os.getenv("API_KEY", "").split(",") if k.strip()]


def check_config() -> None:
    """啟動時呼叫：設定不安全就直接拒絕啟動。"""
    if _auth_disabled():
        logger.warning("⚠️ AUTH_DISABLED=1：API 沒有任何驗證，只能用在本機開發。")
        return
    keys = _configured_keys()
    if not keys:
        raise RuntimeError("未設定 API_KEY，拒絕啟動。請在 .env 加上 API_KEY=（可用 "
                           "python -c \"import secrets; print(secrets.token_urlsafe(32))\" 產生）；"
                           "本機開發若要略過驗證，請明確設定 AUTH_DISABLED=1。")
    if any(len(k) < MIN_KEY_LENGTH for k in keys):
        raise RuntimeError(f"API_KEY 太短（至少 {MIN_KEY_LENGTH} 字元），拒絕啟動。")
    logger.info("✅ API 驗證已啟用（%d 把金鑰）", len(keys))


def _client_id(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()[:8]


def require_api_key(request: Request, key: str | None = Depends(_header)) -> str:
    """驗證通過就回傳「金鑰代號」（給限流用），否則 401。缺少與錯誤用同一則訊息，不洩漏差別。"""
    if _auth_disabled():
        return "anonymous"
    if key:
        # 每把都比對、不提前結束，讓耗時不隨「第幾把才對」而變
        matches = [secrets.compare_digest(key.encode(), k.encode()) for k in _configured_keys()]
        if any(matches):
            return _client_id(key)
    ip = request.client.host if request.client else "?"
    logger.warning("API 金鑰驗證失敗：%s %s（來源 %s）", request.method, request.url.path, ip)
    raise HTTPException(status_code=401, detail="缺少或無效的 API 金鑰")


class SlidingWindowLimiter:
    """滑動視窗限流：每個 client 在 window 秒內最多 limit 次。"""

    def __init__(self, name: str, limit_env: str, default_limit: int, window: float = 60.0, clock=time.monotonic):
        self.name, self.limit_env, self.default_limit = name, limit_env, default_limit
        self.window, self._clock = window, clock
        self._hits: dict[str, deque] = {}
        self._lock = threading.Lock()

    def _limit(self) -> int:
        try:
            return int(os.getenv(self.limit_env, self.default_limit))
        except ValueError:
            return self.default_limit

    def check(self, client_id: str) -> None:
        limit, now = self._limit(), self._clock()
        with self._lock:
            hits = self._hits.setdefault(client_id, deque())
            while hits and now - hits[0] >= self.window:
                hits.popleft()
            if len(hits) >= limit:
                retry_after = max(1, int(self.window - (now - hits[0])) + 1)
                logger.warning("限流（%s）：client=%s 超過每 %.0f 秒 %d 次", self.name, client_id, self.window, limit)
                raise HTTPException(status_code=429, detail="請求太頻繁，請稍後再試",
                                    headers={"Retry-After": str(retry_after)})
            hits.append(now)


_general = SlidingWindowLimiter("general", "RATE_LIMIT_PER_MIN", 120)
_docs = SlidingWindowLimiter("documents", "RATE_LIMIT_DOCS_PER_MIN", 20)   # 每次都會呼叫 Gemini（花錢），限得更緊


def rate_limit_general(client_id: str = Depends(require_api_key)) -> str:
    _general.check(client_id)
    return client_id


def rate_limit_docs(client_id: str = Depends(rate_limit_general)) -> str:
    _docs.check(client_id)
    return client_id
