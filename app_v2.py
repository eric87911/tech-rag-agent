"""v2 介面：多輪對話 + 工具呼叫 / 守衛 / 記憶的可視化。

    streamlit run app_v2.py

前置：資料 API 要先在跑（uvicorn api.main:app），.env 要有 GEMINI_API_KEY（與 API_KEY，若 API 有開驗證）。
對話存在 SQLite（預設 data/chat_memory.db，可用環境變數 CHAT_DB 改），關掉瀏覽器後用編號可以接續。
"""
import logging
import os
import sys
import time
from pathlib import Path

import httpx
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from agent_v2 import tools                                   # noqa: E402
from agent_v2 import ui_format as F                          # noqa: E402
from agent_v2.chat import Chat, open_sqlite_saver            # noqa: E402
from agent_v2.loop import get_client                         # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("app_v2")

st.set_page_config(page_title="科技產業財報 AI 助理 v2", page_icon="📈", layout="wide")

DB_PATH = os.getenv("CHAT_DB", str(ROOT / "data" / "chat_memory.db"))
EXAMPLES = ["台積電 2025Q4 的營收是多少？",
            "那毛利率呢？",
            "2025Q4 毛利率最高的前五家公司",
            "台積電怎麼看 AI 需求？"]


# ---------- 1. 昂貴的東西只建立一次（整個伺服器共用） ----------
@st.cache_resource(show_spinner="🔧 系統初始化中…")
def load_runtime():
    tools.check_tools_in_sync()                  # 工具宣告與路由表不一致就在這裡失敗，而不是問到一半
    client, model = get_client()
    saver = open_sqlite_saver(DB_PATH)           # 連線在整個伺服器生命週期內都開著（Streamlit 不會通知關閉）
    return client, model, saver


try:
    client, model, saver = load_runtime()
except Exception as e:
    logger.error("初始化失敗：%s: %s", type(e).__name__, e)
    st.error(f"系統初始化失敗：{type(e).__name__}: {e}")
    st.info("請確認：1. `.env` 裡有 GEMINI_API_KEY　2. 資料 API 已啟動（`uvicorn api.main:app`）　"
            "3. 已建立向量資料庫（`python src/vector_store.py`）")
    st.stop()


def api_status() -> tuple[bool, str]:
    """資料 API 有沒有在跑。每次重畫都查（很快），API 掛掉時讓使用者一眼看到原因，而不是問了才得到錯誤。"""
    try:
        r = httpx.get(f"{tools.API_BASE_URL}/health", timeout=3)
        r.raise_for_status()
        return True, r.json().get("latest_period", "")
    except Exception as e:
        logger.warning("資料 API 無法連線：%s: %s", type(e).__name__, e)
        return False, ""


# ---------- 2. 每個瀏覽器分頁各自的狀態 ----------
if "chat" not in st.session_state:
    holder = {}                                                  # 接住內層的完整結果（含工具參數與回傳）
    st.session_state.holder = holder
    st.session_state.chat = Chat(client, model, checkpointer=saver, on_result=lambda res: holder.__setitem__("res", res))
    st.session_state.thread_id = st.session_state.chat.new_thread()
    st.session_state.messages = []

chat: Chat = st.session_state.chat


def switch_thread(thread_id: str):
    """切換到某個對話：從 SQLite 還原記憶與畫面"""
    st.session_state.thread_id = thread_id
    st.session_state.messages = F.history_to_messages(chat.history(thread_id))


def show_details(details: dict):
    with st.expander("🔍 這一輪的運作細節"):
        st.markdown("　".join(f"`{b}`" for b in details["badges"]))
        if details["tools"]:
            st.markdown("**工具呼叫：**")
            for t in details["tools"]:
                st.markdown(t["line"])
                if t["preview"]:
                    st.code(t["preview"], language="json")
        else:
            st.markdown("**工具呼叫：** 無")
        if details["usage"]:
            st.caption(details["usage"])


# ---------- 3. 側邊欄 ----------
ok, latest = api_status()
with st.sidebar:
    st.title("📈 科技產業財報 AI 助理 v2")
    st.markdown("手寫 **函式呼叫 Agent**（LangGraph）+ 多輪記憶。答案先經過防護層才交給你；"
                "每一輪用了哪些工具、守衛有沒有出手、有沒有寫入記憶，都可以展開看。")
    if ok:
        st.success(f"資料 API 正常（最新一季 {latest}）")
    else:
        st.error(f"資料 API 無法連線（{tools.API_BASE_URL}），請先啟動它")

    st.subheader("🧵 對話")
    st.caption(f"目前對話編號：`{st.session_state.thread_id}`")
    st.caption(f"對話存在 `{Path(DB_PATH).name}`（含你的提問與答案）")
    resume = st.text_input("接續舊對話（輸入編號）", key="resume_id", placeholder="例如 de30282a")
    if st.button("接續", use_container_width=True) and resume.strip():
        if not chat.history(resume.strip()):
            st.toast("找不到這個編號的紀錄，會當成新對話開始")    # 打錯編號時不要默默變成空對話
        switch_thread(resume.strip())
        st.rerun()
    c1, c2 = st.columns(2)
    if c1.button("新對話", use_container_width=True):
        switch_thread(chat.new_thread())
        st.rerun()
    if c2.button("刪除此對話", use_container_width=True):
        try:
            chat.forget(st.session_state.thread_id)
        except Exception as e:
            logger.error("刪除對話失敗：%s: %s", type(e).__name__, e)
            st.error("刪除失敗，請看終端機的日誌")
        else:
            switch_thread(chat.new_thread())
            st.rerun()

    with st.expander("🧠 目前記住的內容"):
        remembered = chat.history(st.session_state.thread_id)
        if not remembered:
            st.caption("還沒有記住任何一輪。")
        for i, turn in enumerate(remembered, 1):
            a = turn["a"]
            st.markdown(f"**{i}. 問：** {turn['q']}")
            st.caption(f"答：{a[:80]}{'…' if len(a) > 80 else ''}")
        st.caption("只記純文字問答（最多 4 輪）；工具結果不記。被攔截、出錯的輪次不寫入。")

    st.subheader("💡 試試看")
    st.caption("先問第 1 題，再問第 2 題，看它能不能接上文。")
    for q in EXAMPLES:
        if st.button(q, use_container_width=True):
            st.session_state.pending = q


# ---------- 4. 對話區 ----------
st.header("💬 問問台股與台積電")
if not st.session_state.messages:
    st.caption("可以問財報數字、排名、比率，也可以問經營策略與產業看法；追問時可以省略公司名稱。")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("details"):
            show_details(msg["details"])


# ---------- 5. 處理新問題 ----------
question = st.chat_input("輸入你的問題…", disabled=not ok) or st.session_state.pop("pending", None)

if question and ok:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        st.session_state.holder.pop("res", None)
        start = time.perf_counter()
        with st.spinner("思考中…（查資料、檢查答案）"):
            try:
                turn = chat.ask(question, st.session_state.thread_id)
            except Exception as e:
                logger.error("這一輪失敗：%s: %s", type(e).__name__, e)
                st.session_state.messages.pop()           # 這一輪沒寫進記憶，畫面也不留著，兩邊保持一致
                st.error(f"處理問題時發生錯誤：{type(e).__name__}。這一輪沒有寫入記憶，請再問一次。")
                st.stop()
        elapsed = time.perf_counter() - start

        st.markdown(turn.answer)
        details = F.build_details(turn, st.session_state.holder.get("res"), elapsed)
        show_details(details)

    st.session_state.messages.append({"role": "assistant", "content": turn.answer, "details": details})
    st.rerun()      # 重畫一次，讓側邊欄的「目前記住的內容」反映這一輪
