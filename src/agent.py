import sys
import time
import logging
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

import text_to_sql as t2s
from generator import answer_question
from rag_generator import answer_from_docs
from router import route_question, Route
from vector_store import get_collection, get_embed_client

logger = logging.getLogger(__name__)

OUT_OF_SCOPE_MSG = ("這個問題超出了我能回答的範圍。我可以回答：\n"
                    "- 📊 52 家台股公司 2021～2025 年的季度財報數字（營收、毛利、EPS 等）\n"
                    "- 📝 台積電的經營策略與產業看法（2025 年報、2026Q1 法說會）")


@dataclass
class AgentResponse:
    question: str
    route: str = ""
    route_reason: str = ""
    answer: str = ""
    sql: str = ""                                   # Path A 實際執行的 SQL（方便除錯與展示）
    timings: dict = field(default_factory=dict)     # 每個階段花的秒數


class Agent:
    def __init__(self):
        """資源只初始化一次，之後每次提問都重複使用"""
        t0 = time.perf_counter()
        self.gen_client, self.gen_model = t2s.get_client()
        self.embed_client, self.embed_model = get_embed_client()
        self.schema = t2s.build_schema_context()
        self.collection = get_collection()
        logger.info(f"Agent 初始化完成（{time.perf_counter() - t0:.1f} 秒）")

    # ---------- 兩條路徑，各自包好錯誤處理 ----------
    def _run_sql(self, question: str) -> dict:
        try:
            return answer_question(question, self.schema, self.gen_client, self.gen_model)
        except Exception as e:
            logger.error(f"Path A 失敗：{type(e).__name__}: {e}")
            return {"answer": "（數據查詢暫時失敗，請稍後再試）", "sql": "", "status": "error"}

    def _run_rag(self, question: str) -> dict:
        try:
            return answer_from_docs(question, self.gen_client, self.gen_model,
                                    self.embed_client, self.embed_model, self.collection)
        except Exception as e:
            logger.error(f"Path B 失敗：{type(e).__name__}: {e}")
            return {"answer": "（文件檢索暫時失敗，請稍後再試）", "sources": [], "status": "error"}

    @staticmethod
    def _timed(func, *args):
        t0 = time.perf_counter()
        result = func(*args)
        return result, time.perf_counter() - t0

    # ---------- 主流程 ----------
    def ask(self, question: str) -> AgentResponse:
        resp = AgentResponse(question=question)

        decision, resp.timings["路由判斷"] = self._timed(
            route_question, question, self.gen_client, self.gen_model)
        resp.route, resp.route_reason = decision.route.value, decision.reason
        logger.info(f"路由：{resp.route}｜{resp.route_reason}")

        if decision.route == Route.OUT_OF_SCOPE:
            resp.answer = OUT_OF_SCOPE_MSG

        elif decision.route == Route.SQL:
            out, resp.timings["數據查詢"] = self._timed(self._run_sql, decision.sql_question)
            resp.answer, resp.sql = out["answer"], out.get("sql", "")

        elif decision.route == Route.RAG:
            out, resp.timings["文件檢索"] = self._timed(self._run_rag, decision.rag_question)
            resp.answer = out["answer"]

        else:  # BOTH：兩條路同時執行
            with ThreadPoolExecutor(max_workers=2) as pool:
                fa = pool.submit(self._timed, self._run_sql, decision.sql_question)
                fb = pool.submit(self._timed, self._run_rag, decision.rag_question)
                (sql_out, resp.timings["數據查詢"]), (rag_out, resp.timings["文件檢索"]) = \
                    fa.result(), fb.result()
            resp.sql = sql_out.get("sql", "")
            resp.answer = (f"### 📊 數據\n\n{sql_out['answer']}\n\n"
                           f"### 📝 管理層觀點\n\n{rag_out['answer']}")

        resp.timings["總計"] = sum(resp.timings.values()) if decision.route != Route.BOTH else \
            resp.timings["路由判斷"] + max(resp.timings["數據查詢"], resp.timings["文件檢索"])
        return resp


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    for name in ["httpx", "vector_store", "generator", "sql_agent", "rag_generator"]:
        logging.getLogger(name).setLevel(logging.WARNING)

    agent = Agent()
    # 可以直接帶問題執行：python src\agent.py "台積電怎麼看 AI 需求？"
    questions = sys.argv[1:] or ["台積電 2025 年各季的營收",
                                 "台積電怎麼看 AI 需求？",
                                 "台積電毛利率為什麼一直上升？",
                                 "推薦一部好看的電影"]
    for q in questions:
        r = agent.ask(q)
        timing = "｜".join(f"{k} {v:.1f}s" for k, v in r.timings.items())
        print(f"\n{'=' * 60}\n❓ {q}\n🧭 路由：{r.route}（{r.route_reason}）\n⏱️ {timing}\n")
        print(r.answer)