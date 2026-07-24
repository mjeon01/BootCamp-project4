"""Core RAG utilities for the BUFS international student insurance chatbot."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import faiss
import fitz
import numpy as np
from openai import OpenAI


PDF_NAME = "Health_Insurance_info.pdf"
DEFAULT_CHAT_MODEL = "gpt-5.6-luna"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

UNSUPPORTED_KO = (
    "제공된 Health_Insurance_info.pdf에서 이 질문에 대한 내용을 확인할 수 없습니다. "
    "국민건강보험공단(033-811-2000)에 직접 확인해 주세요."
)
UNSUPPORTED_EN = (
    "I cannot verify the answer from Health_Insurance_info.pdf. "
    "Please contact the National Health Insurance Service at 033-811-2000."
)

CHANGE_NOTICE_KO = (
    "보험료와 제도는 변경될 수 있습니다. 최신 내용은 국민건강보험공단 홈페이지 "
    "또는 외국인 전용 번호(033-811-2000)로 재확인하세요."
)
CHANGE_NOTICE_EN = (
    "Premiums and policies may change. Reconfirm the latest information on the NHIS website "
    "or through the foreigners' helpline (033-811-2000)."
)

SCHOOL_CONFLICT_KO = (
    "문서 표기 주의: 영어 4페이지의 제목은 BUFS이지만 본문에는 PNU와 PNU Group "
    "Insurance라는 표현이 남아 있습니다. 챗봇은 이를 임의로 수정하지 않습니다."
)
SCHOOL_CONFLICT_EN = (
    "Document wording note: the English title on page 4 says BUFS, while the body still contains "
    "\"PNU\" and \"PNU Group Insurance.\" The chatbot does not silently correct this conflict."
)


@dataclass(frozen=True)
class Chunk:
    text: str
    page: int
    language: str
    source: str = PDF_NAME
    chunk_id: str = ""


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float


def normalize_pdf_text(text: str) -> str:
    """Repair common extraction artifacts while preserving useful line boundaries."""
    text = text.replace("\u3000", " ").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf_pages(pdf_path: str | Path) -> list[dict[str, object]]:
    """Load the only approved knowledge source with page and language metadata."""
    pdf_path = Path(pdf_path)
    if pdf_path.name != PDF_NAME:
        raise ValueError(f"Only {PDF_NAME} may be used as the knowledge source.")
    if not pdf_path.exists():
        raise FileNotFoundError(f"Missing required PDF: {pdf_path}")

    document = fitz.open(pdf_path)
    pages: list[dict[str, object]] = []
    try:
        for index, page in enumerate(document, start=1):
            text = normalize_pdf_text(page.get_text("text"))
            pages.append(
                {
                    "page": index,
                    "language": "ko" if index <= 3 else "en",
                    "source": PDF_NAME,
                    "text": text,
                }
            )
    finally:
        document.close()
    return pages


def _split_long_block(block: str, max_chars: int, overlap: int) -> Iterable[str]:
    if len(block) <= max_chars:
        yield block
        return
    start = 0
    while start < len(block):
        end = min(start + max_chars, len(block))
        if end < len(block):
            boundary = max(
                block.rfind("\n", start, end),
                block.rfind(". ", start, end),
                block.rfind("다. ", start, end),
            )
            if boundary > start + max_chars // 2:
                end = boundary + 1
        yield block[start:end].strip()
        if end >= len(block):
            break
        start = max(0, end - overlap)


def build_chunks(
    pages: Sequence[dict[str, object]],
    max_chars: int = 900,
    overlap: int = 140,
) -> list[Chunk]:
    """Create page-bounded chunks so every retrieved passage has an exact citation."""
    chunks: list[Chunk] = []
    for page_data in pages:
        text = str(page_data["text"])
        raw_blocks = [part.strip() for part in re.split(r"\n(?=\S)", text) if part.strip()]
        buffer = ""
        page_chunks: list[str] = []
        for block in raw_blocks:
            candidate = f"{buffer}\n{block}".strip() if buffer else block
            if len(candidate) <= max_chars:
                buffer = candidate
                continue
            if buffer:
                page_chunks.extend(_split_long_block(buffer, max_chars, overlap))
            buffer = block
        if buffer:
            page_chunks.extend(_split_long_block(buffer, max_chars, overlap))

        for local_index, chunk_text in enumerate(page_chunks, start=1):
            digest = hashlib.sha1(
                f"{page_data['page']}:{local_index}:{chunk_text}".encode("utf-8")
            ).hexdigest()[:12]
            chunks.append(
                Chunk(
                    text=chunk_text,
                    page=int(page_data["page"]),
                    language=str(page_data["language"]),
                    source=str(page_data["source"]),
                    chunk_id=f"p{page_data['page']}-{local_index}-{digest}",
                )
            )
    return chunks


def create_openai_client(api_key: str) -> OpenAI:
    if not api_key or not api_key.strip():
        raise ValueError("OPENAI_API_KEY is required.")
    return OpenAI(api_key=api_key.strip())


def embed_texts(
    client: OpenAI,
    texts: Sequence[str],
    model: str = DEFAULT_EMBEDDING_MODEL,
) -> np.ndarray:
    response = client.embeddings.create(model=model, input=list(texts))
    vectors = np.asarray([item.embedding for item in response.data], dtype="float32")
    faiss.normalize_L2(vectors)
    return vectors


def create_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError("A non-empty 2D embedding matrix is required.")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def retrieve(
    query: str,
    client: OpenAI,
    index: faiss.IndexFlatIP,
    chunks: Sequence[Chunk],
    language: str,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    top_k: int = 5,
) -> list[SearchHit]:
    query_vector = embed_texts(client, [query], embedding_model)
    candidate_count = min(len(chunks), max(top_k * 3, top_k))
    scores, indices = index.search(query_vector, candidate_count)
    preferred = [
        SearchHit(chunks[int(idx)], float(score))
        for score, idx in zip(scores[0], indices[0])
        if idx >= 0 and chunks[int(idx)].language == language
    ]
    fallback = [
        SearchHit(chunks[int(idx)], float(score))
        for score, idx in zip(scores[0], indices[0])
        if idx >= 0 and chunks[int(idx)].language != language
    ]
    return (preferred + fallback)[:top_k]


def detect_language(text: str, selected: str = "auto") -> str:
    if selected in {"ko", "en"}:
        return selected
    return "ko" if re.search(r"[가-힣]", text) else "en"


def _context_text(hits: Sequence[SearchHit]) -> str:
    sections = []
    for hit in hits:
        sections.append(
            f"[SOURCE: {hit.chunk.source}, PAGE: {hit.chunk.page}, "
            f"LANGUAGE: {hit.chunk.language}]\n{hit.chunk.text}"
        )
    return "\n\n".join(sections)


def build_rag_prompt(
    question: str,
    hits: Sequence[SearchHit],
    language: str,
) -> str:
    allowed_pages = sorted({hit.chunk.page for hit in hits})
    school_terms = (
        "학교",
        "대학",
        "부산외대",
        "부산외국어대학교",
        "bufs",
        "pnu",
        "university",
        "school",
    )
    conflict_rule = (
        f"5. Explicitly state this wording conflict: {SCHOOL_CONFLICT_KO if language == 'ko' else SCHOOL_CONFLICT_EN}"
        if any(term in question.lower() for term in school_terms)
        else "5. Do not discuss the BUFS/PNU wording conflict unless the question is about the university or school name."
    )
    if language == "ko":
        format_instruction = """반드시 한국어로 다음 형식을 사용하세요.
핵심 답변: 2~4문장
절차:
1. 필요한 순서
주의사항:
- 문서에 적힌 주의사항
PDF 페이지: Health_Insurance_info.pdf p. X[, p. Y]"""
        unsupported = f"근거가 없으면 정확히 다음 한 문장만 출력하세요:\n{UNSUPPORTED_KO}"
    else:
        format_instruction = """Answer in English using exactly this structure.
Key answer: 2-4 sentences
Steps:
1. Required sequence
Important notes:
- Notes stated in the document
PDF pages: Health_Insurance_info.pdf p. X[, p. Y]"""
        unsupported = f"If the evidence is insufficient, output exactly this sentence:\n{UNSUPPORTED_EN}"

    return f"""You are a document-grounded assistant for international students at Busan University of Foreign Studies.

Rules:
1. Use only the excerpts below. Do not add general knowledge, assumptions, or updated facts from memory.
2. If the excerpts do not directly support the answer, use the required unsupported sentence.
3. Cite only these retrieved page numbers: {allowed_pages}.
4. Preserve numbers, dates, phone numbers, addresses, and procedures exactly as written.
{conflict_rule}
6. Do not claim that an old premium, policy, contact, or procedure is current. Say that the PDF states it and advise NHIS reconfirmation.
7. Keep the response concise and practical.

{format_instruction}

{unsupported}

Question:
{question}

Approved excerpts:
{_context_text(hits)}
"""


def answer_question(
    question: str,
    hits: Sequence[SearchHit],
    client: OpenAI,
    language: str,
    model: str = DEFAULT_CHAT_MODEL,
) -> str:
    prompt = build_rag_prompt(question, hits, language)
    response = client.responses.create(
        model=model,
        reasoning={"effort": "low"},
        max_output_tokens=900,
        input=prompt,
    )
    answer = response.output_text.strip()
    if not answer:
        return UNSUPPORTED_KO if language == "ko" else UNSUPPORTED_EN
    return answer


def should_show_change_notice(answer: str, language: str) -> bool:
    if answer in {UNSUPPORTED_KO, UNSUPPORTED_EN}:
        return False
    keywords = (
        ("보험료", "납부", "자격", "가입", "비자", "연락처", "공단", "제도")
        if language == "ko"
        else ("premium", "payment", "eligibility", "enroll", "visa", "contact", "nhis", "policy")
    )
    lowered = answer.lower()
    return any(keyword in lowered for keyword in keywords)


def extract_cited_pages(answer: str, hits: Sequence[SearchHit]) -> list[SearchHit]:
    """Keep UI source cards aligned with page numbers cited in the generated answer."""
    cited = {
        int(value)
        for value in re.findall(r"(?:p\.?|페이지[:\s]*)\s*(\d+)", answer, flags=re.IGNORECASE)
    }
    if not cited:
        return list(hits)
    return [hit for hit in hits if hit.chunk.page in cited]


def model_name() -> str:
    return os.getenv("OPENAI_MODEL", DEFAULT_CHAT_MODEL)
