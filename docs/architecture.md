# ROI-RAG Project Architecture

이 문서는 `project_poc` (ROI-RAG) 프로젝트의 시스템 아키텍처와 주요 컴포넌트들의 역할을 정리한 문서입니다.

## 1. System Overview (시스템 개요)
ROI-RAG(Redundancy- and Diversity-Oriented RAG)는 단순 텍스트 검색을 넘어, 정보의 중복도(Redundancy)와 다양성(Diversity)을 계산하여 최적의 근거 단위(Evidence Unit, EU)를 구성하고 이를 기반으로 LLM이 답변을 생성하는 고도화된 RAG 시스템입니다.

### 1.1 C&C View (Component & Connector Architecture)
아래는 시스템이 런타임에 어떻게 상호작용하는지 보여주는 C&C(Component and Connector) 뷰입니다.

```text
               [ 👤 User Browser ]
                        │
                        ▼ (Upload Document / Ask Question)
     ┌─────────────────────────────────────────────────────────┐
     │                   Web Application                       │
     │         [ 🌐 FastAPI Server (app/main.py) ]             │
     └─────────┬─────────────────────────────────────┬─────────┘
               │                                     │
    (1. Build Index)                                 │ (4. Query)
               ▼                                     ▼
     ┌─────────────────────────────────────────────────────────┐
     │                 ROI-RAG Core System                     │
     │                                                         │
     │ [ ⚙️ Indexer ](indexer.py)    [ 🧠 Pipeline ](roi_rag.py)│
     │  - 200단어 Segment 쪼개기        - 사용자 질문 벡터화       │
     │  - 엔트로피 기반 EU 조립         - FAISS Top-K 검색       │
     │  - LLM 기반 EU 요약 생성         - 하이브리드 컨텍스트 조립 │
     │         │                                     │         │
     │         ├─────(Calculate RE / DE)──────┐      │         │
     │         ▼                              ▼      │         │
     │ [ 📊 Entropy Engine ]     [ 🧮 Embedding Model ]        │
     │    (entropy.py)              (embeddings.py)            │
     │                                        │      │         │
     │                                        │      ▼         │
     │                           [ 🤖 LLM Client ](llm_client) │
     │                            - 요약+원문 기반 답변 생성   │
     └─────────┬──────────────────────────────┬──────┬─────────┘
               │                              │      │
               │ (2. Save Centroid Vectors)   │      ▼ (5. Generate)
               │ (3. Save EU Summaries)       │  [ Ollama 생성 / Gemini 평가 ]
               ▼                              ▼
     ┌─────────────────────────────────────────────────────────┐
     │                     Data Storage                        │
     │  [ 🗄️ FAISS DB ](faiss)      [ 📄 JSON Store ](json)    │
     └─────────────────────────────────────────────────────────┘
```

## 2. Core Components (주요 구성 요소)

### 2.1 Data Processing & Indexing (데이터 처리 및 색인)
- **`indexer.py`**: 문서 전처리, 청킹(Chunking), 임베딩 생성 및 FAISS 벡터 DB 색인을 담당합니다. 텍스트를 Segment로 쪼개고, 의미적 유사도에 따라 Evidence Unit(EU)으로 묶은 뒤 요약본을 생성합니다.
- **`entropy.py`**: 문장 간의 의미적 중복도(RE, Redundancy Entropy)와 다양성(DE, Diversity Entropy)을 수학적으로 계산하여, Indexer가 최적의 EU를 구성할 수 있도록 돕습니다.
- **`embeddings.py`**: 텍스트를 벡터로 변환하는 임베딩 모델(기본값: `all-MiniLM-L6-v2`)을 로드하고 관리합니다.

### 2.2 Retrieval & Generation Pipeline (검색 및 추론 파이프라인)
- **`roi_rag.py`**: 실제 사용자의 질문이 들어왔을 때 실행되는 메인 파이프라인입니다.
  1. 사용자 질문 벡터화
  2. FAISS 기반 Top-K Evidence Unit 검색
  3. 요약본과 원문이 결합된 하이브리드 컨텍스트(Hybrid Context) 조립
  4. 엄격한 규칙이 적용된 프롬프트 생성 후 LLM 호출
- **`llm_client.py`**: LLM(대형 언어 모델)과의 통신을 담당합니다. 일반 RAG 생성은 로컬 `Ollama`의 `llama2:7b`를 사용하며, RAGAS 평가는 Gemini API(`gemini-2.5-flash-lite`)를 사용합니다.

### 2.3 Web Backend & GUI (웹 서버 및 사용자 인터페이스)
- **`app/main.py`**: `FastAPI`를 기반으로 구동되는 비동기 웹 서버입니다. 문서 업로드, 인덱싱, 질의응답을 처리하는 REST API 엔드포인트를 제공합니다.
- **`app/templates/index.html`**: 사용자 친화적인 다크 모드 기반의 프리미엄 웹 GUI 프론트엔드 코드입니다.
- **`run_gui.py` / `run_gui.bat`**: FastAPI 서버를 실행하고 사용자의 브라우저를 자동으로 띄워주는 실행 스크립트입니다.

### 2.4 CI/CD & Deployment (자동 배포 및 외부 접속)
- **`.github/workflows/deploy.yml` & `deploy.bat`**: GitHub Actions와 Self-hosted Runner를 활용해, `main` 브랜치에 코드가 머지될 때마다 로컬 Windows PC의 서버를 자동으로 업데이트하고 재시작하는 CI/CD 파이프라인입니다.
- **`2단계_외부접속터널실행.bat`**: `localtunnel`을 활용해 로컬망에 있는 서버를 외부 인터넷 환경에서도 접속 가능하도록 터널링을 열어줍니다.

## 3. Configuration (설정 관리)
- **`config.py`**: 임베딩 청크 크기, 검색할 문서 수(K), 엔트로피 임계값(Threshold) 등 시스템 전반의 동작을 제어하는 핵심 파라미터들이 정의되어 있습니다.
- **`.env`**: Ollama 모델과 Gemini API 실행 설정(`OLLAMA_MODEL`, `GEMINI_API_KEY`, `GEMINI_MODEL` 등)을 관리합니다.

## 4. Evaluation (평가 - 향후 도입 예정)
현재 파이프라인(`roi_rag.py`)은 생성된 답변(`answer`)과 참조한 문서(`retrieved_contexts`)를 분리하여 반환합니다. Faithfulness 평가는 이 값을 한 번의 Gemini API 호출에 전달하며, 평가 결과 캐시는 사용하지 않습니다.
