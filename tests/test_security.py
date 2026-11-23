"""API 驗證與限流的測試。不連資料庫、不呼叫 Gemini，幾秒內跑完。"""
import logging

import pytest
from fastapi import Depends, FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from api import security
from api.security import SlidingWindowLimiter

KEY = "k" * 32
KEY2 = "j" * 32


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.setenv("API_KEY", KEY)
    for name in ("AUTH_DISABLED", "RATE_LIMIT_PER_MIN", "RATE_LIMIT_DOCS_PER_MIN"):
        monkeypatch.delenv(name, raising=False)
    security._general._hits.clear()
    security._docs._hits.clear()


@pytest.fixture
def client():
    app = FastAPI()

    @app.get("/ping", dependencies=[Depends(security.rate_limit_general)])
    def ping():
        return {"ok": True}

    @app.get("/costly", dependencies=[Depends(security.rate_limit_docs)])
    def costly():
        return {"ok": True}

    return TestClient(app)


def auth(key=KEY):
    return {"X-API-Key": key}


# ---------- 驗證 ----------
def test_missing_key_is_401(client):
    assert client.get("/ping").status_code == 401


def test_wrong_key_is_401_with_same_message_as_missing(client):
    missing = client.get("/ping")
    wrong = client.get("/ping", headers=auth("x" * 32))
    assert wrong.status_code == 401
    assert wrong.json() == missing.json()          # 不洩漏「是缺少還是錯誤」


def test_correct_key_is_200(client):
    assert client.get("/ping", headers=auth()).status_code == 200


def test_multiple_keys_for_rotation(client, monkeypatch):
    monkeypatch.setenv("API_KEY", f"{KEY},{KEY2}")
    assert client.get("/ping", headers=auth(KEY)).status_code == 200
    assert client.get("/ping", headers=auth(KEY2)).status_code == 200
    assert client.get("/ping", headers=auth("z" * 32)).status_code == 401


def test_failed_attempt_never_logs_the_key(client, caplog):
    secret_attempt = "wrong-key-value-123456"
    with caplog.at_level(logging.INFO):
        client.get("/ping", headers=auth(secret_attempt))
    assert "驗證失敗" in caplog.text
    assert secret_attempt not in caplog.text


# ---------- 預設拒絕 ----------
def test_refuses_to_start_without_key(monkeypatch):
    monkeypatch.delenv("API_KEY")
    with pytest.raises(RuntimeError, match="API_KEY"):
        security.check_config()


def test_refuses_short_key(monkeypatch):
    monkeypatch.setenv("API_KEY", "short")
    with pytest.raises(RuntimeError, match="太短"):
        security.check_config()


def test_explicit_opt_out_for_local_dev(client, monkeypatch):
    monkeypatch.delenv("API_KEY")
    monkeypatch.setenv("AUTH_DISABLED", "1")
    security.check_config()                          # 不丟例外
    assert client.get("/ping").status_code == 200


# ---------- 限流 ----------
def test_rate_limit_returns_429_with_retry_after(client, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MIN", "2")
    assert client.get("/ping", headers=auth()).status_code == 200
    assert client.get("/ping", headers=auth()).status_code == 200
    r = client.get("/ping", headers=auth())
    assert r.status_code == 429
    assert int(r.headers["Retry-After"]) >= 1


def test_rate_limit_is_per_key(client, monkeypatch):
    monkeypatch.setenv("API_KEY", f"{KEY},{KEY2}")
    monkeypatch.setenv("RATE_LIMIT_PER_MIN", "1")
    assert client.get("/ping", headers=auth(KEY)).status_code == 200
    assert client.get("/ping", headers=auth(KEY)).status_code == 429
    assert client.get("/ping", headers=auth(KEY2)).status_code == 200     # 另一把不受影響


def test_unauthenticated_requests_do_not_use_up_the_quota(client, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MIN", "1")
    for _ in range(5):
        assert client.get("/ping").status_code == 401
    assert client.get("/ping", headers=auth()).status_code == 200


def test_costly_endpoint_has_its_own_tighter_limit(client, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_DOCS_PER_MIN", "1")
    assert client.get("/costly", headers=auth()).status_code == 200
    assert client.get("/costly", headers=auth()).status_code == 429
    assert client.get("/ping", headers=auth()).status_code == 200         # 一般端點不受影響


def test_window_slides_forward():
    now = [0.0]
    limiter = SlidingWindowLimiter("t", "TEST_LIMIT_UNSET", 2, window=60, clock=lambda: now[0])
    limiter.check("a"); limiter.check("a")
    with pytest.raises(Exception):
        limiter.check("a")
    now[0] = 61.0                                    # 一分鐘後，舊的紀錄過期
    limiter.check("a")


# ---------- 結構：新增端點忘了加驗證，這裡會失敗 ----------
OPEN_ROUTES = {"/health"}


def _all_calls(dependant):
    calls = []
    for d in dependant.dependencies:
        calls.append(d.call)
        calls += _all_calls(d)
    return calls


def test_every_route_except_health_requires_auth_and_rate_limit():
    from api.main import app
    unprotected = [r.path for r in app.routes
                   if isinstance(r, APIRoute) and r.path not in OPEN_ROUTES
                   and security.rate_limit_general not in _all_calls(r.dependant)]
    assert unprotected == [], f"這些端點沒有驗證與限流：{unprotected}"


def test_documents_search_has_the_tighter_limit():
    from api.main import app
    route = next(r for r in app.routes if isinstance(r, APIRoute) and r.path == "/documents/search")
    assert security.rate_limit_docs in _all_calls(route.dependant)


def test_health_stays_open_for_container_healthcheck():
    from api.main import app
    route = next(r for r in app.routes if isinstance(r, APIRoute) and r.path == "/health")
    assert security.require_api_key not in _all_calls(route.dependant)


def test_real_app_rejects_before_validating_parameters():
    from api.main import app
    r = TestClient(app).get("/financials")          # 沒帶金鑰、也沒帶必要參數
    assert r.status_code == 401                      # 先驗證身分，不洩漏「還缺哪些參數」
