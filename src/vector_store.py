import os
import time
import logging
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types

from chunker import build_chunks
from doc_loader import load_all

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
CHROMA_DIR = ROOT / "data" / "chroma"
COLLECTION = "tsmc_docs"
EMBED_DIM = 768
BATCH_SIZE = 50
MAX_RETRIES = 3


def get_embed_client():
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("讀不到 GEMINI_API_KEY，請檢查 .env")
    return genai.Client(api_key=api_key), os.getenv("EMBED_MODEL", "gemini-embedding-001")


def embed_texts(texts: list[str], client, model: str, task_type: str) -> list[list[float]]:
    """把文字轉成向量。task_type：存文件用 RETRIEVAL_DOCUMENT，查詢用 RETRIEVAL_QUERY"""
    vectors = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                result = client.models.embed_content(
                    model=model,
                    contents=batch,
                    config=types.EmbedContentConfig(task_type=task_type,
                                                    output_dimensionality=EMBED_DIM),
                )
                vectors.extend(e.values for e in result.embeddings)
                break
            except Exception as e:
                # 429 = 額度用完：等一下再試；其他錯誤直接丟出
                if "429" in str(e) and attempt < MAX_RETRIES:
                    wait = 20 * attempt
                    logger.warning(f"撞到額度上限，{wait} 秒後重試（第 {attempt} 次）")
                    time.sleep(wait)
                else:
                    raise
        logger.info(f"已向量化 {min(start + BATCH_SIZE, len(texts))}/{len(texts)} 塊")
    if len(vectors) != len(texts):
        raise RuntimeError(f"向量數量 {len(vectors)} 和文字數量 {len(texts)} 不一致")
    return vectors


def get_collection():
    """取得已建立好的向量資料庫（給 B4 檢索用）"""
    db = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return db.get_collection(COLLECTION)


def build_index():
    chunks = build_chunks(load_all())
    if not chunks:
        raise RuntimeError("沒有任何文件段落，請確認 data/docs 裡有檔案")

    client, model = get_embed_client()
    vectors = embed_texts([c.text for c in chunks], client, model, "RETRIEVAL_DOCUMENT")

    # 完整重建：先刪掉舊的資料表，避免已刪除的文件還殘留在資料庫裡
    # 用 ChromaDB 自己的刪除功能，而不是直接刪資料夾（Windows 上檔案被占用時會刪不掉）
    db = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if COLLECTION in [c.name for c in db.list_collections()]:
        db.delete_collection(COLLECTION)
    collection = db.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    collection.add(
        ids=[c.chunk_id for c in chunks],
        embeddings=vectors,
        documents=[c.text for c in chunks],
        metadatas=[c.metadata for c in chunks],
    )
    logger.info(f"✅ 向量資料庫建立完成：{collection.count()} 塊，存放於 {CHROMA_DIR}")
    return collection


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    collection = build_index()

    # 冒煙測試（Smoke Test）：問一個問題，看看找回來的段落合不合理
    client, model = get_embed_client()
    question = "台積電怎麼看 AI 需求？"
    q_vec = embed_texts([question], client, model, "RETRIEVAL_QUERY")[0]
    hits = collection.query(query_embeddings=[q_vec], n_results=3)

    print(f"\n❓ {question}")
    for doc, meta, dist in zip(hits["documents"][0], hits["metadatas"][0], hits["distances"][0]):
        print(f"\n[相似度 {1 - dist:.3f}] {meta['source']}｜{meta['location']}")
        print(doc[:200].replace("\n", " ") + "…")