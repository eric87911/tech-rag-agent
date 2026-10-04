import re
import logging
from pathlib import Path
from dataclasses import dataclass

import pymupdf

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
DOCS_DIR = ROOT / "data" / "docs"

CJK = r"\u4e00-\u9fff\u3000-\u303f\uff00-\uffef"   # 中文字與全形標點


@dataclass
class Section:
    """一段讀取出來的文字，以及它的來源資訊（之後做引用來源要用）"""
    source: str      # 檔名
    company: str     # TSMC
    period: str      # 2025 或 2026Q1
    doc_type: str    # annual / transcript
    location: str    # PDF 的頁碼，或逐字稿的發言者
    text: str


# ---------- 1. 從檔名解析來源資訊 ----------
def parse_filename(path: Path) -> dict:
    """TSMC_2026Q1_transcript.txt → company=TSMC, period=2026Q1, doc_type=transcript"""
    m = re.match(r"^([A-Za-z]+)_(\d{4}(?:Q[1-4])?)_([a-z]+)", path.stem)
    if not m:
        raise ValueError(f"檔名格式不符（應為 公司_期間_類型_xxx）：{path.name}")
    return {"company": m.group(1).upper(), "period": m.group(2), "doc_type": m.group(3)}


# ---------- 2. 文字清理 ----------
def clean_text(text: str) -> str:
    text = re.sub(r"[\x00-\x08\x0b-\x1f]", "", text)      # 去掉控制字元（例如「●\x01」）
    # 中文字之間被插入的空格（PDF 左右對齊造成），例如「本 公 司」→「本公司」
    text = re.sub(rf"(?<=[{CJK}]) +(?=[{CJK}])", "", text)
    # 換行接中文：直接接起來；換行接英文：補一個空格
    text = re.sub(rf"(?<=[{CJK}])\n(?=[{CJK}])", "", text)
    text = re.sub(r"[ \t]*\n[ \t]*", " ", text)
    return re.sub(r" {2,}", " ", text).strip()


# ---------- 3. 讀取逐字稿 ----------
def load_transcript(path: Path, meta: dict) -> list[Section]:
    lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines()]

    # 跳過開頭的第三方摘要，只保留「完整原文」之後的逐字稿本文
    if "完整原文" in lines:
        lines = lines[lines.index("完整原文") + 1:]
    else:
        logger.warning(f"{path.name} 找不到「完整原文」標記，將讀取整份檔案")

    lines = [l for l in lines if l]     # 去掉空行
    sections, speaker, paragraphs = [], None, []

    def flush():
        if speaker and paragraphs:
            sections.append(Section(source=path.name, location=speaker,
                                    text="\n\n".join(paragraphs), **meta))

    i = 0
    while i < len(lines):
        # 發言者名稱的特徵：同一行連續出現兩次
        if i + 1 < len(lines) and lines[i] == lines[i + 1]:
            flush()
            speaker, paragraphs = lines[i], []
            i += 2
            continue
        paragraphs.append(lines[i])
        i += 1
    flush()
    return sections


# ---------- 4. 讀取多欄 PDF ----------
def detect_columns(x_starts: list[float], min_gap: float = 100) -> list[float]:
    """從文字區塊的左邊界 X 座標，找出每一欄的起點"""
    columns = []
    for x in sorted(x_starts):
        if not columns or x - columns[-1] > min_gap:
            columns.append(x)
    return columns


PARAGRAPH_GAP = 8    # 上下兩行距離超過 8pt 視為新段落（同段落的行距約 4～5pt）
SENTENCE_END = "。！？：」"


def merge_blocks(blocks, column_of) -> str:
    """PDF 的每一行常常是獨立的 block，這裡把同一段落的行接回來"""
    paragraphs, current, prev = [], "", None
    for b in blocks:
        line = clean_text(b[4])
        if prev is None:
            current = line
        else:
            same_column = column_of(b) == column_of(prev)
            if same_column:
                new_paragraph = b[1] - prev[3] > PARAGRAPH_GAP
            else:
                # 換欄時看不到行距，改看上一行是否以句號等結尾
                new_paragraph = current.endswith(tuple(SENTENCE_END))
            if new_paragraph:
                paragraphs.append(current)
                current = line
            else:
                current = clean_text(current + "\n" + line)
        prev = b
    if current:
        paragraphs.append(current)
    return "\n\n".join(paragraphs)


def load_pdf(path: Path, meta: dict) -> list[Section]:
    sections = []
    with pymupdf.open(path) as doc:
        for page_no, page in enumerate(doc, start=1):
            # 每個 block：(x0, y0, x1, y1, 文字, 編號, 類型)，類型 0 是文字、1 是圖片
            blocks = [b for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]
            # 過濾只有數字的區塊（頁碼，例如「006」）
            blocks = [b for b in blocks if not re.fullmatch(r"[\d\s]+", b[4])]
            if not blocks:
                continue

            columns = detect_columns([b[0] for b in blocks])

            def column_of(block):
                # 屬於「起點 ≤ 這個區塊左邊界」的最右邊那一欄（容許 5pt 誤差）
                return max(i for i, c in enumerate(columns) if c <= block[0] + 5)

            # 關鍵：先依欄位、再由上往下排序，避免左右欄文字交錯
            blocks.sort(key=lambda b: (column_of(b), b[1]))
            text = merge_blocks(blocks, column_of)
            sections.append(Section(source=path.name, location=f"第 {page_no} 頁",
                                    text=text, **meta))
    return sections


# ---------- 5. 讀取整個資料夾 ----------
def load_all(docs_dir: Path = DOCS_DIR) -> list[Section]:
    if not docs_dir.exists():
        raise FileNotFoundError(f"找不到文件資料夾：{docs_dir}")

    loaders = {".txt": load_transcript, ".pdf": load_pdf}
    all_sections = []
    for path in sorted(docs_dir.iterdir()):
        loader = loaders.get(path.suffix.lower())
        if loader is None:
            logger.info(f"略過不支援的檔案：{path.name}")
            continue
        try:
            meta = parse_filename(path)
            sections = loader(path, meta)
        except Exception as e:
            # 一份文件失敗，不影響其他文件
            logger.error(f"讀取失敗 {path.name}：{type(e).__name__}: {e}")
            continue
        chars = sum(len(s.text) for s in sections)
        logger.info(f"✅ {path.name}：{len(sections)} 段，共 {chars:,} 字")
        all_sections.extend(sections)
    return all_sections


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    sections = load_all()

    # 品質檢查：還有沒有殘留「中 文 字 間 的 空 格」
    leftover = sum(len(re.findall(rf"[{CJK}] [{CJK}]", s.text)) for s in sections)
    print(f"\n總共 {len(sections)} 段；殘留的中文空格：{leftover} 處")

    # 每份文件各印出一段預覽，用肉眼確認順序正確
    shown = set()
    for s in sections:
        if s.source in shown:
            continue
        shown.add(s.source)
        print(f"\n{'=' * 60}\n📄 {s.source}｜{s.doc_type}｜{s.period}｜{s.location}")
        print(s.text[:300] + "…")