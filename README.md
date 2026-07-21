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
        └── index.html    # 다크 모드 기반의 프리미엄 PROJECT Studio 웹 인터페이스
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

### 2. CLI 또는 GUI로 질의응답 및 웹 브라우저 실행 (`run_inference.py`)

```bash
# PROJECT Studio 웹 GUI 서버 시작 및 브라우저 자동 실행 (기본 동작 또는 --gui)
python run_inference.py --gui
# 또는 인자 없이 실행하면 자동으로 GUI 모드가 실행됩니다:
python run_inference.py

# 단발성 CLI 질문 실행
python run_inference.py --query "엔비디아의 핵심 경쟁력은 무엇인가요?"

# 대화형 CLI 대화 모드 (Interactive Mode) 실행
python run_inference.py --interactive
```

### 3. 웹 GUI 바로 실행 전용 스크립트 (`run_gui.py` 또는 `run_gui.bat`)

FastAPI(`app/main.py`)에 내장된 다크 모드 프리미엄 **PROJECT Studio** 웹 UI를 즉시 브라우저로 띄울 수 있습니다. 기본적으로 외부 기기 접속까지 모두 허용하도록 개방 모드(`--host 0.0.0.0`)로 구동됩니다.

```bash
# 전용 파이썬 스크립트로 실행 (서버 구동 + 브라우저 자동 오픈 + 외부망 개방)
python run_gui.py --host 0.0.0.0

# 또는 Windows에서 실행할 경우, 더블클릭만으로 실행되는 배치 파일 이용
run_gui.bat

# 기존 uvicorn 직접 구동 방식
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🌐 접속 주소(URL) 및 외부 접속(LTE/5G/회사) 완벽 가이드

서버가 실행된 후, 사용자님의 환경에 따라 다음 3가지 주소 중 하나로 접속하실 수 있습니다.

### 📍 1. 내 PC 모니터에서 볼 때 (Local Access)
- **접속 URL:** `http://127.0.0.1:8000` (또는 `http://localhost:8000`)
- **설명:** 서버를 켠 컴퓨터 내에서 가장 빠르고 직접적으로 접속하는 주소입니다.

### 📍 2. 같은 집/회사 Wi-Fi에 연결된 스마트폰·노트북에서 볼 때 (LAN Access)
- **접속 URL:** `http://192.168.0.22:8000` (현재 PC의 로컬 LAN IP)
- **설명:** 동일한 Wi-Fi(공유기) 망에 연결된 다른 기기에서 접속할 때 사용합니다.
- **⚡ 원클릭 방화벽 해결 (`1단계_방화벽포트개방.bat`):**
  - 만약 Wi-Fi 연결에서 "페이지가 작동하지 않습니다"가 뜨면 윈도우 방화벽이 막고 있는 것입니다.
  - 프로젝트 폴더의 `1단계_방화벽포트개방.bat` 파일을 **우클릭 $\rightarrow$ '관리자 권한으로 실행'** 하시면 8000번 포트가 1초 만에 자동 허용됩니다.

### 📍 3. 외부 환경(LTE/5G 데이터, 회사, 외부 인터넷)에서 볼 때 (External Access)
공유기 설정이나 복잡한 포트포워딩 없이도 전 세계 어디서든 **PROJECT Studio**에 접속할 수 있도록 전용 터널을 지원합니다.

- **🚀 추천 (포트포워딩 없는 커스텀 고정 주소):**
  - **접속 URL:** `https://project-studio-rag.loca.lt`
  - **실행 방법:** 프로젝트 폴더의 **`2단계_외부접속터널실행.bat`** 을 더블 클릭(또는 터미널에서 `npx -y localtunnel --port 8000 --subdomain project-studio-rag`)해 두시면 외부(회사/LTE)에서 언제든 위 주소로 접속됩니다.
  - *(최초 접속 시 `loca.lt` 스팸 방지 안내 창이 나타날 수 있습니다. 화면 중앙의 파란색 `Click to Continue` 버튼을 클릭하시거나 접속 IP를 입력하시면 즉시 UI로 연결됩니다)*

- **🏠 대안 (공유기 포트포워딩 사용 시):**
  - **접속 URL:** `http://219.241.82.8:8000` (외부 공인 IP 주소)
  - **설명:** 공유기 관리자 페이지(`192.168.0.1` 등)에서 포트포워딩(`외부 8000` $\rightarrow$ `내부 192.168.0.22:8000`)이 등록되어 있어야만 접속이 가능합니다.

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
