import logging
from dataclasses import dataclass

from vector_store import get_collection, get_embed_client, embed_texts

logger = logging.getLogger(__name__)

TOP_K = 5             # 最後要給 LLM 的段落數
CANDIDATES = 15       # 先多抓一些候選，再挑選
PER_SOURCE_CAP = 2    # 每份文件最多選幾段（確保來源多樣性）
MIN_SCORE = 0.65      # 根據實驗：相關問題最低 0.696，無關問題最高 0.609，取中間偏下


@dataclass
class Hit:
    text: str
    metadata: dict
    score: float      # 相似度，越接近 1 越相關


def retrieve(question: str, client, model: str, collection,
             top_k: int = TOP_K, per_source_cap: int = PER_SOURCE_CAP,
             min_score: float = MIN_SCORE, where: dict = None) -> list[Hit]:
    q_vec = embed_texts([question], client, model, "RETRIEVAL_QUERY")[0]
    raw = collection.query(query_embeddings=[q_vec], n_results=CANDIDATES, where=where)

    candidates = [Hit(text=d, metadata=m, score=1 - dist)
                  for d, m, dist in zip(raw["documents"][0], raw["metadatas"][0],
                                        raw["distances"][0])]

    # 1. 相關性門檻：太不相關的直接丟掉
    if min_score is not None:
        candidates = [h for h in candidates if h.score >= min_score]

    # 2. 來源多樣性：依分數由高到低挑，每份文件最多 per_source_cap 段
    selected, per_source = [], {}
    for h in candidates:                      # ChromaDB 回傳時已經依相似度排序
        src = h.metadata["source"]
        if per_source.get(src, 0) >= per_source_cap:
            continue
        selected.append(h)
        per_source[src] = per_source.get(src, 0) + 1
        if len(selected) == top_k:
            break
    return selected


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("vector_store").setLevel(logging.WARNING)

    client, model = get_embed_client()
    collection = get_collection()

    # 實驗：相關問題 vs 無關問題，觀察相似度分布，用來決定門檻
    experiments = {
        "相關": ["台積電怎麼看 AI 需求？",
                 "2奈米製程的進度如何？",
                 "台積電在美國亞利桑那州的擴廠計畫",
                 "先進封裝 CoWoS 的產能狀況"],
        "無關": ["台積電員工餐廳好吃嗎？",
                 "今天台北天氣如何？",
                 "推薦一部好看的電影"],
    }
    for group, questions in experiments.items():
        print(f"\n{'#' * 20} {group}問題 {'#' * 20}")
        for q in questions:
            hits = retrieve(q, client, model, collection)
            print(f"\n❓ {q}")
            for h in hits:
                print(f"   {h.score:.3f}｜{h.metadata['source'][:35]:35}｜{h.metadata['location'][:25]}")