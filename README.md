# ROI-RAG (`project_poc`) | Standalone Inference Repository

> **Redundancy- and Diversity-Oriented Retrieval-Augmented Generation (ROI-RAG)** 단독 추론 및 인덱싱을 위해 경량화·최적화된 GitHub 저장소입니다.

`rag_poc` 레포지토리에서 Self-RAG 및 실험용 벤치마크 평가 코드를 제외하고, 오직 **ROI-RAG의 오프라인 인덱싱(Offline Indexing)**과 **온라인 질의응답 추론(Online Inference)**을 빠르고 간편하게 수행할 수 있도록 모듈화된 프로젝트 구조입니다.

---

## 📁 디렉토리 및 파일 구조

```
project_poc/
├── config.py             # ROI-RAG 핵심 환경 설정 및 파라미터 관리
├── embeddings.py         # SentenceTransformers 기반 로컬 임베딩 래퍼
├── entropy.py            # Pairwise Cosine Similarity 및 RE/DE 엔트로피 연산
├── llm_client.py         # Gemini / Ollama 백엔드 통합 LLM 추론 클라이언트
├── indexer.py            # ROI-RAG 오프라인 인덱싱 (청킹 -> 이웃 탐색 -> EU 생성 -> FAISS 저장)
├── roi_rag.py            # ROI-RAG 온라인 추론 파이프라인 (검색 + 하이브리드 컨텍스트 답변 생성)
├── build_index.py        # CLI 기반 문서 인덱스 생성 도구
├── run_inference.py      # CLI 기반 질의응답 및 대화형 추론 실행 도구
├── requirements.txt      # Python 의존성 패키지 목록
├── .env.example          # API 키 및 백엔드 설정 환경변수 템플릿
├── tests/                # 단위 테스트 모듈 (`test_roi_rag.py`)
└── app/                  # FastAPI 웹/REST API 서비스 및 웹 UI
    ├── main.py           # REST API 엔드포인트 (`/api/build-index`, `/api/query` 등)
    └── templates/
        └── index.html    # 다크 모드 기반의 프리미엄 ROI-RAG Studio 웹 인터페이스
```

---

## 🚀 시작하기 (Installation)

### 1. 가상환경 생성 및 의존성 설치

```bash
# 가상환경 생성 및 활성화 (Windows PowerShell 기준)
python -m venv .venv
.\.venv\Scripts\activate

# 필수 패키지 설치
pip install -r requirements.txt
```

### 2. 환경변수(`.env`) 설정

`.env.example` 파일을 복사하여 `.env` 파일을 생성하고 LLM 백엔드 및 API 키를 입력합니다.

```env
# LLM 백엔드 선택 ("gemini" 또는 "ollama")
LLM_BACKEND=gemini

# Gemini API 설정 (LLM_BACKEND=gemini 일 때)
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.1-flash-lite

# Ollama 설정 (LLM_BACKEND=ollama 일 때)
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen2.5:14b
```

---

## 🛠 사용 방법 (Usage)

### 1. CLI로 오프라인 인덱스 생성 (`build_index.py`)

문서 파일(`.txt`) 또는 텍스트 문자열을 전달하여 FAISS 벡터 인덱스(`data/faiss_index.bin`) 및 메타데이터(`data/roi_rag_index.json`)를 생성합니다.

```bash
# 텍스트 문서 파일(.txt) 인덱싱
python build_index.py --file path/to/sample.txt

# 또는 텍스트 직접 입력 인덱싱
python build_index.py --text "엔비디아는 AI 반도체 선두 주자로 최신 아키텍처를 발표하였습니다."
```

### 2. CLI로 질의응답 추론 실행 (`run_inference.py`)

```bash
# 단발성 질문 실행
python run_inference.py --query "엔비디아의 핵심 경쟁력은 무엇인가요?"

# 대화형 대화 모드 (Interactive CLI Mode) 실행
python run_inference.py --interactive
```

### 3. FastAPI REST API 및 웹 UI 구동 (`app/main.py`)

브라우저에서 GUI로 문서를 업로드하고 Evidence Unit(EU) 구성 내역을 확인하며 질의응답을 진행할 수 있습니다.

```bash
uvicorn app.main:app --port 8000 --reload
```
- 브라우저에서 `http://localhost:8000` 에 접속하여 **ROI-RAG Studio** 웹 UI를 사용합니다.
- API Swagger 문서: `http://localhost:8000/docs`

---

## 📐 ROI-RAG 핵심 동작 원리

1. **후보 이웃 탐색 (Candidate Neighborhood Construction)**:
   - 로컬 임베딩(`all-MiniLM-L6-v2`)으로 각 청크의 상위 $K$개 의미적 이웃을 탐색합니다.
2. **엔트로피 기반 탐욕적 EU 생성 (Entropy-Guided EU Construction)**:
   - 중복 엔트로피(RE: $\le \Theta_{RE}$)를 유지하면서 다양성 엔트로피(DE)가 최대가 되는 청크들을 묶어 **Evidence Unit (EU)**을 구성합니다.
3. **적응형 요약 (Adaptive Summarization)**:
   - **LOW Regime** ($\text{RE} < \tau_{low}$): 중복이 적으므로 LLM 요약 없이 원문 유지
   - **MEDIUM Regime** ($\tau_{low} \le \text{RE} < \tau_{high}$): 팩트 및 엔티티를 보존하는 요약 생성
   - **HIGH Regime** ($\text{RE} \ge \tau_{high}$): 높은 중복 정보를 압축하는 고밀도 단락 생성
4. **하이브리드 컨텍스트 질의 추론 (Grounded Hybrid Retrieval)**:
   - 질의 시 FAISS로 Top-$K$ EU를 검색한 후, **[EU 요약문 + 대표 원문 단락]**을 하이브리드로 LLM에 제공하여 환각(Hallucination) 없는 답변을 생성합니다.

---

## 🧪 단위 테스트 실행

```bash
python -m unittest discover tests
```
