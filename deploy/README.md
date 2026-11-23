# 部署手冊：Ubuntu（systemd + Caddy）

在 Ubuntu 26.04（arm64，OrbStack 虛擬機）上實際跑過的流程。架構：

```
使用者 ──HTTPS──▶ Caddy :443 ──▶ uvicorn 127.0.0.1:8000（systemd 服務，低權限帳號）
```

| 項目 | 位置 | 原因 |
|---|---|---|
| 程式碼、資料 | `/opt/tech-rag/app/`（root 擁有，服務帳號唯讀）| 服務被攻破也改不了程式碼；只有 `data/chroma/` 可寫 |
| Python 環境 | `/opt/tech-rag/venv/`（uv 安裝 Python 3.13）| 與容器版本一致，不依賴系統 Python |
| 金鑰 | `/etc/tech-rag/env`（root、權限 600）| 不在程式目錄、不進 git；由 systemd 讀入 |
| 執行身分 | 系統帳號 `techrag`（不能登入）| 最小權限 |

## 步驟

```bash
# 1. 帳號與目錄
sudo useradd --system --home-dir /opt/tech-rag --create-home --shell /usr/sbin/nologin techrag
sudo mkdir -p /opt/tech-rag/app /etc/tech-rag && sudo chown -R techrag:techrag /opt/tech-rag

# 2. 工具（uv 裝成系統層級，techrag 才執行得到）
sudo apt-get update -qq && sudo apt-get install -y rsync caddy
curl -LsSf https://astral.sh/uv/install.sh | sudo env UV_UNMANAGED_INSTALL=/usr/local/bin sh

# 3. 程式與資料（SRC 為專案所在資料夾；不要複製 .env）
SRC=/path/to/tech_rag_agent
sudo rsync -a --chown=techrag:techrag --exclude='__pycache__' \
  "$SRC"/api "$SRC"/agent_v2 "$SRC"/src "$SRC"/requirements-api.txt "$SRC"/requirements.txt /opt/tech-rag/app/
sudo mkdir -p /opt/tech-rag/app/data
sudo rsync -a --chown=techrag:techrag "$SRC"/data/finance.db "$SRC"/data/chroma /opt/tech-rag/app/data/
A=/opt/tech-rag/app   # 程式碼與資料庫改為 root 擁有（chroma/ 維持 techrag，因為要寫入）
sudo chown -R root:root $A/api $A/agent_v2 $A/src $A/requirements-api.txt $A/requirements.txt $A/data/finance.db
sudo chmod -R u=rwX,go=rX $A/api $A/agent_v2 $A/src $A/requirements-api.txt $A/requirements.txt $A/data/finance.db

# 4. venv（要先 cd 進 techrag 可讀的目錄，見「踩過的坑」）
sudo -u techrag -H sh -c '
  cd /opt/tech-rag \
  && export UV_PYTHON_INSTALL_DIR=/opt/tech-rag/python UV_CACHE_DIR=/opt/tech-rag/.cache \
  && uv venv /opt/tech-rag/venv --python 3.13 \
  && uv pip install --python /opt/tech-rag/venv/bin/python \
       -r /opt/tech-rag/app/requirements-api.txt -c /opt/tech-rag/app/requirements.txt'

# 5. 環境變數檔（GEMINI_API_KEY、API_KEY 等；API_KEY 沒設服務會拒絕啟動）
grep -E '^(GEMINI_API_KEY|API_KEY|GEMINI_MODEL|EMBED_MODEL)=' "$SRC"/.env | sudo tee /etc/tech-rag/env >/dev/null
sudo chown root:root /etc/tech-rag/env && sudo chmod 600 /etc/tech-rag/env

# 6. systemd 服務與強化設定
sudo cp deploy/tech-rag-api.service /etc/systemd/system/
sudo mkdir -p /etc/systemd/system/tech-rag-api.service.d
sudo cp deploy/hardening.conf /etc/systemd/system/tech-rag-api.service.d/
sudo systemctl daemon-reload && sudo systemctl enable --now tech-rag-api

# 7. Caddy（HTTPS 反向代理）
sudo cp deploy/Caddyfile /etc/caddy/Caddyfile
sudo caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile && sudo systemctl reload caddy
```

## 驗證

```bash
systemctl is-active tech-rag-api; systemctl is-enabled tech-rag-api
sudo ss -ltn | grep -c ':443 '                                   # 應為 1（reload 失敗時 caddy 仍顯示 active，不能只看 active）
KEY=$(sudo grep '^API_KEY=' /etc/tech-rag/env | cut -d= -f2-)
curl -sk https://<網域>/health                                    # 200
curl -sk -o /dev/null -w "%{http_code}\n" https://<網域>/companies?q=TSMC                       # 401
curl -sk -o /dev/null -w "%{http_code}\n" -H "X-API-Key: $KEY" https://<網域>/companies?q=TSMC # 200
sudo systemctl kill -s KILL tech-rag-api; sleep 6; systemctl is-active tech-rag-api            # 自動重啟
systemd-analyze security tech-rag-api.service --no-pager | tail -2                              # 4.9 OK
```

## 從另一台機器驗證（內部 CA）

`tls internal` 的憑證由 Caddy 自己的 CA 簽發，客戶端預設不信任。向執行中的 Caddy 取得根憑證（比猜檔案路徑可靠），複製到客戶端後用 `--cacert` 驗證：

```bash
# 在伺服器上：從 Caddy 的管理介面取得目前使用的根憑證
curl -s localhost:2019/pki/ca/local | python3 -c 'import sys,json; print(json.load(sys.stdin)["root_certificate"])' > root.crt
# 在客戶端：不加 -k，驗證憑證鏈
curl --cacert root.crt https://<網域>/health
```

已實測：客戶端（Mac）驗證通過；API 的驗證失敗日誌記到的是客戶端真實 IP（Caddy 轉送的標頭被 uvicorn 採信），不是 `127.0.0.1`。
複製憑證若出錯（例如檔案寫入被系統權限視窗打斷），會得到 `self signed certificate in certificate chain`；重新複製並比對指紋（`openssl x509 -noout -fingerprint -sha256`）即可。

## 更新程式

重跑步驟 3（rsync 與 chown），然後 `sudo systemctl restart tech-rag-api`。看日誌：`sudo journalctl -u tech-rag-api -f`。

## 踩過的坑

- **`uv` 報 `failed to open file /home/…/uv.toml: Permission denied`**：uv 會從當前目錄往上找設定檔，而服務帳號讀不了你的家目錄。在 `sudo -u techrag sh -c 'cd /opt/tech-rag && …'` 裡先 `cd`。
- **SQLite／ChromaDB 放在掛載的共用資料夾會 `disk I/O error`**（檔案鎖不支援）。資料一律放在機器自己的磁碟（這裡是 btrfs）。
- **Ubuntu 的 apt 只有舊版 Caddy（2.6.2）**，新版的日誌過濾簡寫語法不能用。要新版請用 Caddy 官方 apt 來源。
- **Caddy 存取日誌會把自訂標頭（`X-API-Key`）明文寫進去**，要在設定中刪除；已用測試金鑰實測日誌中不出現。
- **`systemctl reload caddy` 失敗時服務仍是 `active`**（舊設定繼續跑），要另外確認 443 有在聽。
- **架構**：Mac（Apple Silicon）與這台虛擬機是 arm64；搬到一般雲端主機（多為 x86_64）時，Docker image 要用 `--platform linux/amd64` 重 build。

## 尚未驗證

- 公開網域上的 Let's Encrypt 自動憑證（這裡用內部 CA）。
- 重新開機後服務自動啟動（只確認了 `enabled`）。
- 防火牆設定（雲端主機上需只開 22／80／443）。
- `/openapi.json` 與 `/docs` 仍然公開。
