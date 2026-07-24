"""Streamlit entry point for the BUFS international student insurance RAG chatbot."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag_core import (
    CHANGE_NOTICE_EN,
    CHANGE_NOTICE_KO,
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    PDF_NAME,
    SCHOOL_CONFLICT_EN,
    SCHOOL_CONFLICT_KO,
    UNSUPPORTED_EN,
    UNSUPPORTED_KO,
    answer_question,
    build_chunks,
    create_faiss_index,
    create_openai_client,
    detect_language,
    embed_texts,
    extract_cited_pages,
    load_pdf_pages,
    model_name,
    retrieve,
    should_show_change_notice,
)


load_dotenv(Path(__file__).with_name(".env"))

st.set_page_config(
    page_title="BUFS Health Insurance Guide",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root {
        --kakao-yellow: #FEE500;
        --kakao-ink: #191919;
        --telegram-blue: #229ED9;
        --telegram-deep: #168AC1;
        --chat-bg: #EEF4F8;
        --line: #DCE7EE;
        --muted: #6D7E8A;
      }

      html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo",
                     "Segoe UI", sans-serif;
      }
      .stApp {
        background:
          radial-gradient(circle at 18% 0%, rgba(34,158,217,.10), transparent 28rem),
          linear-gradient(180deg, #F8FBFD 0%, var(--chat-bg) 100%);
      }
      [data-testid="stHeader"] {background: transparent;}
      .block-container {
        max-width: 900px;
        padding-top: 1.15rem;
        padding-bottom: 7rem;
      }

      [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #FFFFFF 0%, #F4F9FC 100%);
        border-right: 1px solid #DDE9F0;
        box-shadow: 10px 0 30px rgba(50, 88, 110, .05);
      }
      [data-testid="stSidebar"] .block-container {padding-top: 1.3rem;}

      .side-profile {
        display: flex;
        align-items: center;
        gap: .8rem;
        padding: .85rem;
        margin-bottom: 1.1rem;
        border: 1px solid #DCE9F0;
        border-radius: 18px;
        background: #FFFFFF;
        box-shadow: 0 8px 24px rgba(30, 86, 116, .07);
      }
      .side-avatar, .profile-avatar, .welcome-avatar {
        display: grid;
        place-items: center;
        flex: 0 0 auto;
        font-weight: 800;
      }
      .side-avatar {
        width: 42px; height: 42px;
        border-radius: 14px;
        background: var(--kakao-yellow);
        color: var(--kakao-ink);
        font-size: 1.25rem;
      }
      .side-profile strong {display: block; color: #18313F; font-size: .95rem;}
      .side-profile small {color: var(--muted); font-size: .77rem;}

      .messenger-header {
        display: flex;
        align-items: center;
        gap: 1rem;
        min-height: 92px;
        padding: 1.1rem 1.25rem;
        border-radius: 24px 24px 16px 16px;
        background: linear-gradient(135deg, var(--telegram-blue), #54B8E6);
        color: white;
        box-shadow: 0 14px 34px rgba(34, 158, 217, .24);
      }
      .profile-avatar {
        position: relative;
        width: 58px; height: 58px;
        border-radius: 20px;
        background: var(--kakao-yellow);
        color: var(--kakao-ink);
        font-size: 1.55rem;
        box-shadow: inset 0 -3px 0 rgba(0,0,0,.08), 0 8px 18px rgba(0,0,0,.12);
      }
      .online-dot {
        position: absolute;
        right: -2px; bottom: -2px;
        width: 15px; height: 15px;
        border: 3px solid #FFFFFF;
        border-radius: 50%;
        background: #38D67A;
      }
      .profile-copy {min-width: 0; flex: 1;}
      .profile-copy h1 {
        margin: 0 0 .2rem;
        color: white;
        font-size: 1.52rem;
        letter-spacing: -.03em;
      }
      .profile-copy p {
        margin: 0;
        color: rgba(255,255,255,.88);
        font-size: .87rem;
      }
      .header-badge {
        padding: .45rem .7rem;
        border: 1px solid rgba(255,255,255,.35);
        border-radius: 999px;
        background: rgba(255,255,255,.16);
        color: white;
        font-size: .72rem;
        font-weight: 800;
        letter-spacing: .06em;
      }
      .status-strip {
        display: flex;
        flex-wrap: wrap;
        gap: .45rem;
        padding: .75rem .25rem 1.05rem;
      }
      .status-pill {
        display: inline-flex;
        align-items: center;
        gap: .3rem;
        padding: .38rem .62rem;
        border: 1px solid #D8E6ED;
        border-radius: 999px;
        background: rgba(255,255,255,.8);
        color: #49616F;
        font-size: .75rem;
        font-weight: 650;
      }

      .welcome-row {
        display: flex;
        align-items: flex-start;
        gap: .65rem;
        margin: .4rem 0 1.35rem;
      }
      .welcome-avatar {
        width: 38px; height: 38px;
        border-radius: 14px;
        background: var(--telegram-blue);
        color: white;
        font-size: .72rem;
        box-shadow: 0 6px 14px rgba(34,158,217,.2);
      }
      .welcome-bubble {
        max-width: 76%;
        padding: .82rem 1rem;
        border: 1px solid #DCE8EF;
        border-radius: 7px 18px 18px 18px;
        background: white;
        color: #243A47;
        box-shadow: 0 7px 20px rgba(35, 74, 96, .07);
        line-height: 1.55;
        font-size: .92rem;
      }
      .welcome-bubble strong {color: var(--telegram-deep);}
      .welcome-bubble small {display:block; margin-top:.35rem; color:#7B8B95;}

      .quick-label {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin: .3rem 0 .6rem;
      }
      .quick-label strong {color: #294453; font-size: .96rem;}
      .quick-label span {color: #8798A2; font-size: .74rem;}

      div[data-testid="stButton"] > button {
        min-height: 3.05rem;
        justify-content: flex-start;
        padding: .65rem .85rem;
        border: 1px solid #D7E4EB;
        border-radius: 15px;
        background: rgba(255,255,255,.92);
        color: #29414F;
        font-weight: 700;
        box-shadow: 0 5px 14px rgba(39, 77, 98, .055);
        transition: transform .16s ease, border-color .16s ease, background .16s ease;
      }
      div[data-testid="stButton"] > button:hover {
        transform: translateY(-2px);
        border-color: var(--kakao-yellow);
        background: #FFFDEA;
        color: #111111;
      }
      div[data-testid="stButton"] > button:focus {
        box-shadow: 0 0 0 3px rgba(34,158,217,.16);
      }

      [data-testid="stChatMessage"] {
        gap: .65rem;
        padding: .55rem .2rem;
        background: transparent;
      }
      [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] {
        max-width: 78%;
        padding: .72rem .95rem;
        border: 1px solid #DCE8EF;
        border-radius: 7px 18px 18px 18px;
        background: #FFFFFF;
        box-shadow: 0 7px 20px rgba(35, 74, 96, .07);
      }
      [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        flex-direction: row-reverse;
      }
      [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
        border-color: #F0DA00;
        border-radius: 18px 7px 18px 18px;
        background: var(--kakao-yellow);
        color: var(--kakao-ink);
        box-shadow: 0 7px 18px rgba(170, 148, 0, .12);
      }
      [data-testid="stChatMessage"] [data-testid="stChatMessageAvatarUser"] {
        background: var(--kakao-yellow);
        color: #1A1A1A;
      }
      [data-testid="stChatMessage"] [data-testid="stChatMessageAvatarAssistant"] {
        background: var(--telegram-blue);
        color: white;
      }

      [data-testid="stChatInput"] {
        border: 1px solid #CFE0E9;
        border-radius: 999px;
        background: #FFFFFF;
        box-shadow: 0 12px 32px rgba(31, 76, 101, .16);
      }
      [data-testid="stChatInput"] textarea {
        min-height: 52px;
        padding-left: .45rem;
      }
      [data-testid="stChatInput"] button {
        border-radius: 50%;
        background: var(--telegram-blue);
        color: white;
      }

      .source-card {
        border: 1px solid #D9E6ED;
        border-left: 4px solid var(--telegram-blue);
        border-radius: 13px;
        padding: .82rem .9rem;
        margin: .5rem 0;
        background: #F7FBFD;
      }
      .source-card strong {color: #197FB0;}
      .source-card small {color: #526B79; line-height: 1.5;}
      [data-testid="stExpander"] {
        border: 1px solid #DCE8EF;
        border-radius: 13px;
        background: rgba(255,255,255,.7);
      }
      [data-testid="stAlert"] {border-radius: 14px;}

      .footer-note {
        margin-top: 1.6rem;
        padding: .9rem 1rem;
        border: 1px solid #D8E5EC;
        border-radius: 16px;
        background: rgba(255,255,255,.72);
        color: #687B87;
        font-size: .78rem;
        line-height: 1.55;
      }
      .footer-note strong {color: #35505E;}

      @media (max-width: 700px) {
        .block-container {padding-top: .65rem; padding-left: .75rem; padding-right: .75rem;}
        .messenger-header {min-height: 82px; padding: .9rem; border-radius: 20px 20px 14px 14px;}
        .profile-avatar {width: 49px; height: 49px; border-radius: 17px;}
        .profile-copy h1 {font-size: 1.18rem;}
        .profile-copy p {font-size: .72rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
        .header-badge {display: none;}
        .welcome-bubble, [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] {max-width: 88%;}
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_secret(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, os.getenv(name, default))
        if not value and name == "OPENAI_API_KEY":
            value = st.secrets.get("model_api_key", os.getenv("model_api_key", default))
        return str(value)
    except FileNotFoundError:
        return os.getenv(name, default)


@st.cache_resource(show_spinner=False)
def build_store(api_key: str, pdf_signature: int):
    del pdf_signature
    client = create_openai_client(api_key)
    pages = load_pdf_pages(Path(__file__).with_name(PDF_NAME))
    chunks = build_chunks(pages)
    vectors = embed_texts(client, [chunk.text for chunk in chunks], DEFAULT_EMBEDDING_MODEL)
    return client, chunks, create_faiss_index(vectors)


def render_sources(hits):
    pages_seen = set()
    with st.expander("근거 문서 / Sources", expanded=False):
        for hit in hits:
            key = hit.chunk.page
            if key in pages_seen:
                continue
            pages_seen.add(key)
            excerpt = hit.chunk.text.replace("\n", " ")
            if len(excerpt) > 280:
                excerpt = excerpt[:277].rstrip() + "..."
            st.markdown(
                f"""
                <div class="source-card">
                  <strong>{hit.chunk.source} · p. {hit.chunk.page}</strong><br/>
                  <small>{excerpt}</small>
                </div>
                """,
                unsafe_allow_html=True,
            )


api_key = get_secret("OPENAI_API_KEY")
chat_model = get_secret("OPENAI_MODEL", DEFAULT_CHAT_MODEL) or model_name()
pdf_path = Path(__file__).with_name(PDF_NAME)

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.markdown(
        """
        <div class="side-profile">
          <div class="side-avatar">💬</div>
          <div>
            <strong>BUFS 보험 도우미</strong>
            <small>Document-grounded assistant</small>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("#### 채팅 설정")
    language_label = st.radio(
        "답변 언어 / Answer language",
        ("자동 감지 / Auto", "한국어", "English"),
        index=0,
    )
    selected_language = {
        "자동 감지 / Auto": "auto",
        "한국어": "ko",
        "English": "en",
    }[language_label]
    st.markdown(
        f"""
        <div style="padding:.72rem .8rem;border-radius:14px;background:#EAF6FC;
                    color:#41606F;font-size:.78rem;line-height:1.65;">
          <b style="color:#168AC1;">● 문서 연결됨</b><br/>
          Model&nbsp; <code>{chat_model}</code><br/>
          Source&nbsp; <code>{PDF_NAME}</code><br/>
          Pages&nbsp; 7
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("🧹 대화 내용 지우기", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.warning(SCHOOL_CONFLICT_KO)
    if not api_key:
        st.error(
            "OPENAI_API_KEY가 설정되지 않았습니다. 로컬에서는 환경변수, "
            "배포 환경에서는 Streamlit Secrets에 등록하세요."
        )

st.markdown(
    """
    <div class="messenger-header">
      <div class="profile-avatar">🏥<span class="online-dot"></span></div>
      <div class="profile-copy">
        <h1>외국인 유학생 건강보험 도우미</h1>
        <p>BUFS International Student Support · 지금 상담 가능</p>
      </div>
      <div class="header-badge">PDF RAG</div>
    </div>
    <div class="status-strip">
      <span class="status-pill">🟢 Online</span>
      <span class="status-pill">📄 PDF 7 pages</span>
      <span class="status-pill">🌐 한국어 · English</span>
      <span class="status-pill">🔒 문서 근거 답변</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if not st.session_state.messages:
    st.markdown(
        """
        <div class="welcome-row">
          <div class="welcome-avatar">AI</div>
          <div class="welcome-bubble">
            안녕하세요! <strong>BUFS 건강보험 도우미</strong>예요. 👋<br/>
            보험 가입, 보험료 납부, 병원 이용, 자격확인서 발급을 물어보세요.
            <small>Health_Insurance_info.pdf 안의 내용만 근거로 답변합니다.</small>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="quick-label">
      <strong>빠른 질문</strong>
      <span>버튼을 누르면 바로 질문해요</span>
    </div>
    """,
    unsafe_allow_html=True,
)
example_questions = [
    ("💳 보험료 납부 방법", "국민건강보험료는 어떻게 납부하나요?"),
    ("🏥 병원에서 이용하기", "병원에서 건강보험을 어떻게 사용하나요?"),
    ("📄 자격확인서 발급", "자격득실확인서는 어떻게 발급받나요?"),
    ("🌐 NHIS certificate", "How can I get an NHIS certificate?"),
]
cols = st.columns(2)
selected_example = None
for index, (label, example) in enumerate(example_questions):
    if cols[index % 2].button(label, use_container_width=True, key=f"example-{index}"):
        selected_example = example

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("notice"):
            st.info(message["notice"])
        if message.get("hits"):
            render_sources(message["hits"])

typed_question = st.chat_input(
    "건강보험에 대해 질문하세요 / Ask about health insurance",
    disabled=not api_key or not pdf_path.exists(),
)
question = selected_example or typed_question

if question:
    language = detect_language(question, selected_language)
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("PDF 근거를 찾고 있습니다... / Searching the PDF..."):
                signature = pdf_path.stat().st_mtime_ns
                client, chunks, index = build_store(api_key, signature)
                hits = retrieve(
                    question,
                    client,
                    index,
                    chunks,
                    language=language,
                    top_k=5,
                )
                answer = answer_question(
                    question,
                    hits,
                    client,
                    language=language,
                    model=chat_model,
                )

            if not hits:
                answer = UNSUPPORTED_KO if language == "ko" else UNSUPPORTED_EN
            st.markdown(answer)
            notice = None
            if should_show_change_notice(answer, language):
                notice = CHANGE_NOTICE_KO if language == "ko" else CHANGE_NOTICE_EN
                st.info(notice)
            if answer not in {UNSUPPORTED_KO, UNSUPPORTED_EN}:
                hits = extract_cited_pages(answer, hits)
                render_sources(hits)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "notice": notice,
                    "hits": hits if answer not in {UNSUPPORTED_KO, UNSUPPORTED_EN} else [],
                }
            )
        except Exception as exc:
            st.error(
                "답변을 생성하지 못했습니다. API 키, 모델 접근 권한, 네트워크 상태를 확인하세요. "
                f"오류 유형: {type(exc).__name__}"
            )

st.markdown(
    f"""
    <div class="footer-note">
      <strong>안내</strong> · 이 서비스는 행정·의료 상담을 대체하지 않습니다.
      보험료와 제도는 변경될 수 있으므로 국민건강보험공단에 재확인하세요.<br/>
      <span>{SCHOOL_CONFLICT_EN}</span>
    </div>
    """,
    unsafe_allow_html=True,
)
