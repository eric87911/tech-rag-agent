import sys
import logging
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))    # 讓 app.py 可以 import src 裡的模組
from agent import Agent

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

st.set_page_config(page_title="科技產業財報 AI 助理", page_icon="📈", layout="wide")

ROUTE_LABELS = {"sql": "📊 財報資料庫", "rag": "📚 年報與法說會",
                "both": "📊 + 📚 兩者並用", "out_of_scope": "🚫 超出範圍"}
EXAMPLES = ["台積電 2025 年各季的營收",
            "2025Q4 毛利率最高的前五家公司",
            "台積電怎麼看 AI 需求？",
            "台積電毛利率為什麼一直上升？"]


# ---------- 1. Agent 只建立一次（整個伺服器共用） ----------
@st.cache_resource(show_spinner="🔧 系統初始化中…")
def load_agent() -> Agent:
    return Agent()


try:
    agent = load_agent()
except Exception as e:
    st.error(f"系統初始化失敗：{type(e).__name__}: {e}")
    st.info("請確認：1. `.env` 裡有 GEMINI_API_KEY　2. 已執行 `python src/sync_data.py`　"
            "3. 已執行 `python src/vector_store.py` 建立向量資料庫")
    st.stop()


# ---------- 2. 對話紀錄存在 session_state（每次重跑都還在） ----------
if "messages" not in st.session_state:
    st.session_state.messages = []


def show_details(details: dict):
    """顯示系統運作細節：路由、耗時、SQL"""
    with st.expander("🔍 系統運作細節"):
        st.markdown(f"**路由：** {ROUTE_LABELS.get(details['route'], details['route'])}")
        st.markdown(f"**判斷理由：** {details['reason']}")
        timing = "｜".join(f"{k} {v:.1f} 秒" for k, v in details["timings"].items())
        st.markdown(f"**耗時：** {timing}")
        if details.get("sql"):
            st.markdown("**執行的 SQL：**")
            st.code(details["sql"], language="sql")


# ---------- 3. 側邊欄 ----------
with st.sidebar:
    st.title("📈 科技產業財報 AI 助理")
    st.markdown("結合 **Text-to-SQL** 與 **RAG** 的混合架構問答系統，"
                "自動判斷問題類型，從財報資料庫或年報、法說會中找答案。")
    st.subheader("📊 資料範圍")
    st.markdown("- **財報數字**：52 家台股公司，2021Q1～2025Q4\n"
                "- **文件**：台積電 2025 年報（致股東報告書、營運概況）、"
                "2026Q1 法說會逐字稿（第三方整理）")
    st.subheader("💡 試試看")
    for q in EXAMPLES:
        if st.button(q, use_container_width=True):
            st.session_state.pending = q
    st.divider()
    if st.button("🗑️ 清除對話", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------- 4. 對話區：先把歷史訊息畫出來 ----------
st.header("💬 問問台股與台積電")
if not st.session_state.messages:
    st.caption("可以問財報數字（例如營收、毛利率、EPS），也可以問經營策略與產業看法。")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("details"):
            show_details(msg["details"])


# ---------- 5. 處理新問題（來自輸入框或範例按鈕） ----------
question = st.chat_input("輸入你的問題…") or st.session_state.pop("pending", None)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.status("思考中…", expanded=True) as status:
            try:
                resp = agent.ask(question, on_step=st.write)    # 每個階段的訊息直接寫在進度框裡
                status.update(label=f"完成（{resp.timings.get('總計', 0):.1f} 秒）",
                              state="complete", expanded=False)
            except Exception as e:
                status.update(label="發生錯誤", state="error")
                st.error(f"處理問題時發生錯誤：{type(e).__name__}: {e}")
                st.stop()

        st.markdown(resp.answer)
        details = {"route": resp.route, "reason": resp.route_reason,
                   "timings": resp.timings, "sql": resp.sql}
        show_details(details)

    st.session_state.messages.append(
        {"role": "assistant", "content": resp.answer, "details": details})