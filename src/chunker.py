import re
import logging
from dataclasses import dataclass

from doc_loader import Section, load_all, CJK

logger = logging.getLogger(__name__)

MAX_CHARS = {"annual": 600, "transcript": 1500}   # 逐字稿中英雙份，上限放寬
OVERLAP_MAX = 200     # 重疊段落最多 200 字，太長就不重疊
MIN_CHARS = 50        # 太短的塊（例如「(以外語發言)」）直接丟掉
QUESTION_MIN = 100    # 分析師發言超過 100 字才算「提問」（排除「謝謝」「沒錯」這類回應）

DOC_NAMES = {"annual": "年報", "transcript": "法說會逐字稿"}
COMPANY_NAMES = {"TSMC": "台積電"}


@dataclass
class Chunk:
    chunk_id: str      # 唯一編號，例如 TSMC_2025_annual_ch1_letter.pdf::3
    text: str          # 實際拿去向量化的文字（含標題）
    metadata: dict     # 來源資訊，之後做引用用


# ---------- 共用工具 ----------
def is_chinese(text: str) -> bool:
    cjk = len(re.findall(f"[{CJK}]", text))
    return cjk / max(len(text), 1) > 0.3


def pair_paragraphs(text: str) -> list[str]:
    """把段落切成最小單位；逐字稿的「英文 + 中文翻譯」綁成一組，不可拆開"""
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    units, i = [], 0
    while i < len(paragraphs):
        if (i + 1 < len(paragraphs)
                and not is_chinese(paragraphs[i]) and is_chinese(paragraphs[i + 1])):
            units.append(paragraphs[i] + "\n" + paragraphs[i + 1])
            i += 2
        else:
            units.append(paragraphs[i])
            i += 1
    return units


def split_long(text: str, max_chars: int) -> list[str]:
    """單一段落太長時，改用句號切"""
    sentences = re.split(r"(?<=[。！？.!?])", text)
    pieces, current = [], ""
    for s in sentences:
        if current and len(current) + len(s) > max_chars:
            pieces.append(current)
            current = ""
        current += s
    if current:
        pieces.append(current)
    return pieces


def pack(units: list[str], max_chars: int) -> list[list[str]]:
    """把單位依序疊成塊；相鄰兩塊保留一個單位的重疊"""
    chunks, current = [], []
    for u in units:
        if current and sum(len(x) for x in current) + len(u) > max_chars:
            chunks.append(current)
            last = current[-1]
            current = [last] if len(last) <= OVERLAP_MAX else []
        current.append(u)
    if current:
        chunks.append(current)
    return chunks


# ---------- 法說會 Q&A：以「一位分析師的完整問答」為一組 ----------
def role_of(speaker: str) -> str:
    if "Analyst" in speaker:
        return "analyst"
    if "Investor Relations" in speaker:
        return "host"
    if speaker.startswith("Operator"):
        return "operator"
    return "management"


def short_name(speaker: str) -> str:
    return speaker.split(" - ")[0]       # "C.C. Wei - Chairman & CEO" → "C.C. Wei"


def group_transcript(sections: list[Section]) -> list[tuple[Section, list]]:
    """回傳 (Section, turns)：簡報段落 turns 為空；Q&A 組的 turns 是 [(發言者, 角色, 內容)]"""
    qa_start = next((i for i, s in enumerate(sections)
                     if role_of(s.location) == "analyst"), len(sections))
    result = [(s, []) for s in sections[:qa_start]]      # 前半段簡報：維持以發言者為單位

    groups = []
    for s in sections[qa_start:]:
        role = role_of(s.location)
        if role == "operator":
            continue                                     # 總機只是串場
        name = short_name(s.location)
        # 換了一位分析師，才開始新的一組（同一位分析師的追問、道謝都留在同一組）
        if role == "analyst" and (not groups or groups[-1]["analyst"] != name):
            groups.append({"analyst": name, "turns": []})
        if groups:
            groups[-1]["turns"].append((name, role, s.text))

    base = sections[0]
    for g in groups:
        answerers = []
        for name, role, _ in g["turns"]:
            if role != "analyst" and name not in answerers:
                answerers.append(name)
        if not answerers:
            continue                                     # 只有提問、沒有回答的不要
        location = f"Q&A｜{g['analyst']} 提問，{'、'.join(answerers)} 回答"
        sec = Section(source=base.source, company=base.company, period=base.period,
                      doc_type=base.doc_type, location=location, text="")
        result.append((sec, g["turns"]))
    return result


def chunk_qa(turns: list, max_chars: int) -> list[str]:
    """Q&A 切塊：每段發言標上發言者；被拆開時，後面的塊補上「最近一次的提問」"""
    units = []     # 每個單位記錄：文字、當時最近一次的提問、發言者
    latest_question = ""
    for name, role, text in turns:
        if role == "analyst" and len(text) >= QUESTION_MIN:
            zh = [p for p in text.split("\n\n") if is_chinese(p)]
            latest_question = (zh[0] if zh else text)[:150]
        for j, u in enumerate(pair_paragraphs(text)):
            for piece in (split_long(u, max_chars) if len(u) > max_chars else [u]):
                label = f"▍{name}\n" if j == 0 else ""
                units.append({"text": label + piece, "question": latest_question,
                              "speaker": name})

    info = {u["text"]: u for u in units}
    results = []
    for i, chunk in enumerate(pack([u["text"] for u in units], max_chars)):
        first = info[chunk[0]]
        # 塊的開頭如果是某人發言的中段，補上發言者名稱，避免分不清是誰說的
        if not chunk[0].startswith("▍"):
            chunk = [f"▍{first['speaker']}（續）\n{chunk[0]}"] + chunk[1:]
        body = "\n\n".join(chunk)
        q = first["question"]
        # 不是第一塊、而且這塊沒有包含提問本身 → 補上原始問題，讓這塊能獨立被理解
        if i > 0 and q and q not in body:
            body = f"（承上，問題：{q}）\n\n{body}"
        results.append(body)
    return results


# ---------- 一般文件 ----------
def make_header(section: Section) -> str:
    company = COMPANY_NAMES.get(section.company, section.company)
    doc = DOC_NAMES.get(section.doc_type, section.doc_type)
    return f"【{company} {section.period} {doc}｜{section.location}】"


def chunk_section(section: Section) -> list[str]:
    max_chars = MAX_CHARS.get(section.doc_type, 600)
    units = []
    for u in pair_paragraphs(section.text):
        units.extend(split_long(u, max_chars) if len(u) > max_chars else [u])
    return ["\n\n".join(c) for c in pack(units, max_chars)]


def build_chunks(sections: list[Section]) -> list[Chunk]:
    # 逐字稿先依 Q&A 分組；其他文件維持原樣
    items = [(s, []) for s in sections if s.doc_type != "transcript"]
    by_source = {}
    for s in sections:
        if s.doc_type == "transcript":
            by_source.setdefault(s.source, []).append(s)
    for secs in by_source.values():
        items.extend(group_transcript(secs))

    result, counters = [], {}
    for s, turns in items:
        header = make_header(s)
        max_chars = MAX_CHARS.get(s.doc_type, 600)
        bodies = chunk_qa(turns, max_chars) if turns else chunk_section(s)
        for body in bodies:
            if len(body) < MIN_CHARS:
                continue
            n = counters.get(s.source, 0)
            counters[s.source] = n + 1
            result.append(Chunk(
                chunk_id=f"{s.source}::{n}",
                text=f"{header}\n{body}",
                metadata={"source": s.source, "company": s.company, "period": s.period,
                          "doc_type": s.doc_type, "location": s.location},
            ))
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    chunks = build_chunks(load_all())

    print(f"\n共 {len(chunks)} 塊")
    for src in sorted({c.metadata["source"] for c in chunks}):
        lens = [len(c.text) for c in chunks if c.metadata["source"] == src]
        print(f"  {src}：{len(lens)} 塊，長度 最短 {min(lens)}／平均 {sum(lens)//len(lens)}／最長 {max(lens)}")

    # 檢查 Q&A 分組結果
    qa = sorted({c.metadata["location"] for c in chunks if c.metadata["location"].startswith("Q&A")})
    print(f"\nQ&A 共 {len(qa)} 組：")
    for loc in qa:
        n = sum(1 for c in chunks if c.metadata["location"] == loc)
        print(f"  {loc}（{n} 塊）")