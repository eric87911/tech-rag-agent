import re
import logging
from pathlib import Path

from google.genai import types

import text_to_sql as t2s
from retriever import retrieve, Hit
from vector_store import get_collection, get_embed_client

logger = logging.getLogger(__name__)

COMPANY_NAMES = {"TSMC": "台積電"}
CHAPTER_NAMES = {"letter": "致股東報告書", "operations": "營運概況"}


# ---------- 1. 把 metadata 轉成人看得懂的來源名稱 ----------
def source_label(meta: dict) -> str:
    company = COMPANY_NAMES.get(meta["company"], meta["company"])
    if meta["doc_type"] == "transcript":
        speaker = meta["location"].split(" - ")[0]          # 只留人名
        return f"{company} {meta['period']} 法說會逐字稿（第三方整理）｜{speaker}"
    # 年報：從檔名最後一段判斷章節，例如 ..._ch1_letter.pdf → 致股東報告書
    chapter_key = Path(meta["source"]).stem.split("_")[-1]
    chapter = CHAPTER_NAMES.get(chapter_key, "")
    return f"{company} {meta['period']} 年報 {chapter}｜{meta['location']}"


def build_context(hits: list[Hit]) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        body = h.text.split("\n", 1)[-1]      # 去掉切塊時加的標題，避免重複
        blocks.append(f"[{i}] {source_label(h.metadata)}\n{body}")
    return "\n\n".join(blocks)


# ---------- 2. Prompt ----------
RAG_RULES = """你是專業的台股產業分析助理。請只根據「參考資料」回答使用者的問題。

規則：
1. 只能使用參考資料中的內容，不可以使用你自己的知識補充。
2. 每一句陳述的後面都要標註來源編號，例如「AI 需求非常強勁 [1]」；同一句有多個來源時寫成 [1][3]。
3. 如果參考資料只能回答問題的一部分，要明確說出哪一部分資料中沒有提到。
4. 參考資料是英文時，請翻譯成繁體中文再引用。
5. 用繁體中文回答，先用一兩句話直接回答，再分點補充重點，總長度不超過 300 字。
6. 數字、百分比、年份必須和參考資料完全一致，不可以改寫。逐字稿的中文是第三方翻譯，可能有誤；中英文對數字的描述不一致時，以英文原文為準，並在中文後面附上英文原文，例如「CAGR 朝向 50 幾 % 的高段（原文：higher 50s）」。
7. 說明「資料中沒有提到」的句子，不要標註來源編號。"""

NO_RESULT = "在目前的文件（台積電年報、法說會逐字稿）中，沒有找到和這個問題相關的內容。"


# ---------- 3. 引用驗證：由程式決定最後列出哪些來源 ----------
def verify_citations(answer: str, n_sources: int) -> tuple[str, list[int]]:
    cited = []
    def check(m):
        n = int(m.group(1))
        if 1 <= n <= n_sources:
            if n not in cited:
                cited.append(n)
            return m.group(0)
        logger.warning(f"移除不存在的引用編號 [{n}]")
        return ""                              # 不存在的編號直接刪掉
    answer = re.sub(r"\[(\d+)\]", check, answer)
    return answer, sorted(cited)


def generate_rag_answer(question: str, hits: list[Hit], client, model: str) -> dict:
    if not hits:
        return {"answer": NO_RESULT, "sources": [], "status": "no_result"}

    prompt = f"使用者問題：{question}\n\n參考資料：\n{build_context(hits)}"
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=RAG_RULES,
                temperature=0.2,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
        answer = (response.text or "").strip()
    except Exception as e:
        # 退路：LLM 失敗時，直接列出找到的段落摘要
        logger.error(f"生成回答失敗：{type(e).__name__}: {e}")
        previews = [f"[{i}] {h.text.split(chr(10), 1)[-1][:120]}…" for i, h in enumerate(hits, 1)]
        answer = "（自動摘要暫時無法使用，以下為相關段落）\n\n" + "\n\n".join(previews)

    answer, cited = verify_citations(answer, len(hits))
    if not cited:
        logger.warning("回答中沒有任何引用編號，列出所有參考資料")
        cited = list(range(1, len(hits) + 1))

    sources = [f"[{n}] {source_label(hits[n - 1].metadata)}" for n in cited]
    full = answer + "\n\n📎 資料來源：\n" + "\n".join(sources)
    return {"answer": full, "sources": sources, "status": "ok"}


def answer_from_docs(question: str, gen_client, gen_model: str,
                     embed_client, embed_model: str, collection) -> dict:
    """Path B 的完整流程：問題 → 檢索 → 生成附引用的回答"""
    hits = retrieve(question, embed_client, embed_model, collection)
    logger.info(f"檢索到 {len(hits)} 段（分數 {[round(h.score, 3) for h in hits]}）")
    return generate_rag_answer(question, hits, gen_client, gen_model)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    for name in ["httpx", "vector_store"]:
        logging.getLogger(name).setLevel(logging.WARNING)

    gen_client, gen_model = t2s.get_client()
    embed_client, embed_model = get_embed_client()
    collection = get_collection()

    for q in ["台積電怎麼看 AI 需求？",
              "台積電在美國亞利桑那州的擴廠計畫",
              "台積電員工餐廳好吃嗎？",
              "今天台北天氣如何？"]:
        out = answer_from_docs(q, gen_client, gen_model, embed_client, embed_model, collection)
        print(f"\n{'=' * 60}\n❓ {q}\n")
        print(out["answer"])