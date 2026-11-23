FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 先裝依賴、再複製程式碼：依賴檔沒變時這一層會被快取，改程式碼不必重裝（build 從分鐘級變秒級）
COPY requirements-api.txt requirements.txt ./
RUN pip install -r requirements-api.txt -c requirements.txt

# 只複製程式碼。資料庫、向量庫、金鑰一律在執行時才掛進來／注入
COPY api ./api
COPY agent_v2 ./agent_v2
COPY src ./src

# 不用 root 執行：萬一服務被攻破，攻擊者拿到的是低權限帳號
RUN useradd --create-home --uid 10001 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# 容器「活著」不等於「能服務」：直接問 /health（資料庫打得開才會回 200）
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"

# 正式環境不加 --reload（那是開發用，會監看檔案並重啟）
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
