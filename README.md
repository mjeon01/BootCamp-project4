# BUFS 외국인 유학생 건강보험 RAG 챗봇

`Health_Insurance_info.pdf` 한 개만 근거로 부산외국어대학교 외국인 유학생의
건강보험 질문에 한국어 또는 영어로 답하는 Streamlit 서비스입니다.

UI는 카카오톡의 노란 말풍선과 텔레그램의 블루 메신저 레이아웃을 조합했으며,
정상 답변마다 실제 근거 PDF 페이지를 표시합니다.

## 주요 기능

- PyMuPDF로 7페이지 PDF를 읽고 페이지·언어 메타데이터 보존
- OpenAI `text-embedding-3-small` 임베딩과 FAISS 유사도 검색
- OpenAI Responses API와 `gpt-5.6-luna` 사용
- 질문 언어 자동 감지 또는 한국어·영어 직접 선택
- `핵심 답변 → 절차 → 주의사항 → PDF 페이지` 형식
- 문서 밖 질문 거절 및 변경 가능 정보 NHIS 재확인 안내
- 영어 4페이지의 BUFS/PNU 표기 충돌을 임의 수정하지 않고 안내
- 메신저형 채팅 UI, 빠른 질문 버튼, 대화 초기화, 페이지별 출처 카드

## 프로젝트 파일

| 파일 | 역할 | GitHub 업로드 |
|---|---|---|
| `app.py` | Streamlit 화면, 채팅 상태, 사용자 입력 및 RAG 호출 | 필수 |
| `rag_core.py` | PDF 처리, 청킹, 임베딩, FAISS 검색, 프롬프트와 답변 생성 | 필수 |
| `Health_Insurance_info.pdf` | 챗봇이 사용하는 유일한 지식 문서 | 필수 |
| `requirements.txt` | 배포 환경에서 설치할 Python 패키지 | 필수 |
| `README.md` | 프로젝트 구조, 실행·배포 방법 | 권장 |
| `NHIS_RAG_Chatbot.ipynb` | 전체 RAG 처리 과정과 한국어·영어 테스트 결과 | 제출물 |
| `프로젝트_보고서_20201487_전문기.docx` | 7페이지 프로젝트 결과 보고서 | 제출물 |
| `.gitignore` | `.env`, 캐시 등 비공개·불필요 파일 제외 | 필수 |
| `.env` | 로컬 OpenAI API 키 | **업로드 금지** |

## 코드 구성

### `app.py` — 화면과 사용자 흐름

| 구성 | 역할 |
|---|---|
| `load_dotenv()` | 로컬 `.env`에서 API 키를 읽음 |
| `st.set_page_config()` | 앱 제목, 아이콘, 화면 너비 설정 |
| CSS 블록 | 카카오톡·텔레그램 혼합형 헤더, 말풍선, 버튼, 입력창 디자인 |
| `get_secret()` | 로컬 환경변수와 Streamlit Secrets에서 설정값 조회 |
| `build_store()` | PDF 청크를 임베딩하고 FAISS 인덱스를 캐시에 저장 |
| `render_sources()` | 답변에 인용된 PDF 페이지와 발췌문을 출처 카드로 표시 |
| Sidebar | 언어 선택, 모델·PDF 상태, 대화 초기화, 문서 표기 경고 |
| 빠른 질문 버튼 | 보험료, 병원 이용, 자격확인서 대표 질문 입력 |
| `st.session_state.messages` | 대화 기록 유지 |
| 질문 처리 블록 | 언어 감지 → 검색 → 답변 생성 → 변경 안내 → 출처 표시 |
| 예외 처리 | API 키, 모델 권한, 네트워크 오류를 사용자에게 안내 |

### `rag_core.py` — RAG 핵심 로직

| 코드 | 역할 |
|---|---|
| `Chunk` | 텍스트, PDF 페이지, 언어, 출처, 청크 ID 저장 |
| `SearchHit` | 검색된 청크와 유사도 점수 저장 |
| `normalize_pdf_text()` | PDF 추출 과정에서 생긴 공백·줄바꿈 정리 |
| `load_pdf_pages()` | PDF 7페이지를 읽고 한국어·영어 메타데이터 부여 |
| `_split_long_block()` | 긴 텍스트를 최대 글자 수와 중첩 기준으로 분리 |
| `build_chunks()` | 페이지 경계를 넘지 않는 15개 청크 생성 |
| `create_openai_client()` | OpenAI API 클라이언트 생성 |
| `embed_texts()` | 텍스트 임베딩 생성 및 L2 정규화 |
| `create_faiss_index()` | 코사인 유사도 검색용 FAISS 인덱스 생성 |
| `retrieve()` | 질문과 가까운 청크를 검색하고 질문 언어를 우선 배치 |
| `detect_language()` | 한글 포함 여부 또는 사용자 선택으로 답변 언어 결정 |
| `build_rag_prompt()` | 문서 근거 제한, 답변 형식, 페이지 인용 규칙 구성 |
| `answer_question()` | `gpt-5.6-luna` Responses API로 최종 답변 생성 |
| `should_show_change_notice()` | 보험료·자격·절차 관련 답변의 재확인 안내 여부 판단 |
| `extract_cited_pages()` | 모델 답변에 실제 인용된 페이지만 출처 카드로 필터링 |

## 질문 처리 흐름

```text
사용자 질문
   ↓
답변 언어 감지
   ↓
질문 임베딩 생성
   ↓
FAISS에서 관련 PDF 청크 검색
   ↓
검색된 발췌문만 프롬프트에 포함
   ↓
gpt-5.6-luna 답변 생성
   ↓
변경 가능 정보 안내 + 실제 인용 페이지 표시
```

PDF에서 직접 근거를 찾지 못하면 일반 지식으로 답하지 않고 확인 불가 메시지를
출력합니다.

## 로컬 실행

Python 3.11 또는 3.12 환경을 권장합니다.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
printf 'OPENAI_API_KEY="your-api-key"\n' > .env
streamlit run app.py
```

기본 모델은 `gpt-5.6-luna`입니다. 필요한 경우 `.env`에 아래 설정을 추가할 수 있습니다.

```env
OPENAI_MODEL="gpt-5.6-luna"
```

## Streamlit Community Cloud 배포

1. `.env`를 제외한 프로젝트 파일을 GitHub 공개 저장소에 업로드합니다.
2. Streamlit Community Cloud에서 저장소와 `app.py`를 선택합니다.
3. App settings → Secrets에 다음 값을 등록합니다.

```toml
OPENAI_API_KEY = "sk-..."
OPENAI_MODEL = "gpt-5.6-luna"
```

4. 재부팅 후 한국어·영어 대표 질문과 출처 페이지를 확인합니다.

API 키는 코드, 노트북 출력, Git 커밋에 포함하지 마세요. `.gitignore`가 `.env`와
Streamlit 로컬 Secrets 파일을 제외하도록 설정되어 있습니다.

## 지식 범위와 한계

- 지식 소스는 `Health_Insurance_info.pdf` 하나뿐입니다.
- 보험료, 자격, 납부 절차와 연락처는 변경될 수 있으므로 NHIS에 재확인해야 합니다.
- 영어 4페이지 제목은 BUFS이지만 본문 일부에는 PNU가 남아 있습니다.
- OpenAI API 장애, 할당량 또는 모델 접근 권한에 따라 답변 생성이 실패할 수 있습니다.
