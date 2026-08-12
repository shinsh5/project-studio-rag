# PROJECT Studio ROI-RAG

로컬 문서를 인덱싱하고 검색 근거에 기반해 답변하는 독립형 ROI-RAG 웹 애플리케이션입니다.

- 일반 RAG 답변과 인덱스 요약: 로컬 Ollama `llama2:7b`
- 임베딩과 검색: SentenceTransformers + FAISS
- RAGAS faithfulness 평가: `codex exec` (평가할 때마다 새로 실행, 캐시 없음)
- 웹 서비스: FastAPI + 내장 웹 UI
- Gemini API 키는 사용하지 않습니다.

## 실행 구조

| 용도 | 실행 엔진 | 기본 설정 |
|---|---|---|
| 문서 임베딩 | SentenceTransformers | `all-MiniLM-L6-v2` |
| RAG 답변 및 EU 요약 | Ollama | `llama2:7b` |
| RAGAS faithfulness 평가 | Codex CLI | `codex exec` |
| 벡터 검색 | FAISS | `data/faiss_index.bin` |
| 웹 서버 | FastAPI/Uvicorn | `0.0.0.0:8000` 권장 |

## 요구 사항

이 저장소는 Windows 실행을 기준으로 구성되어 있습니다.

- Git
- 64비트 Python `3.12.x`
- [Poetry](https://python-poetry.org/) `2.x`
- [Ollama for Windows](https://docs.ollama.com/windows)
- `llama2:7b` 모델용 최소 약 8GB RAM과 약 4GB 디스크 공간
- 설치 및 로그인이 완료된 Codex CLI

문서 인덱싱과 질의응답은 로컬 Ollama에서 실행됩니다. RAGAS faithfulness 평가만 Codex CLI 로그인을 사용합니다.

## 빠른 시작

### 1. 저장소 받기

PowerShell에서 실행합니다.

```powershell
git clone https://github.com/shinsh5/project-studio-rag.git
Set-Location project-studio-rag
```

비공개 저장소라면 GitHub 인증 권한이 필요합니다.

### 2. Poetry 설치

Poetry가 이미 설치되어 있다면 이 단계는 건너뜁니다.

```powershell
python -m pip install --user pipx
python -m pipx ensurepath
python -m pipx install poetry
```

`ensurepath` 실행 후에는 VS Code와 터미널을 완전히 종료했다가 다시 실행합니다.

```powershell
poetry --version
```

`poetry`가 계속 인식되지 않는 Windows 환경에서는 다음 경로로 확인할 수 있습니다.

```powershell
& "$env:USERPROFILE\.local\bin\poetry.exe" --version
```

### 3. 프로젝트 의존성 설치

저장소에 포함된 `poetry.lock`을 사용해 동일한 버전으로 설치합니다. `poetry.toml` 설정에 따라 가상환경은 프로젝트의 `.venv` 폴더에 생성됩니다.

```powershell
poetry install --no-root --sync
```

Poetry 명령의 PATH 문제가 있다면 다음 명령을 대신 사용합니다.

```powershell
& "$env:USERPROFILE\.local\bin\poetry.exe" install --no-root --sync
```

### 4. Ollama 및 Llama 2 준비

[Ollama Windows 설치 프로그램](https://docs.ollama.com/windows)을 설치한 후 새 터미널에서 실행합니다.

```powershell
ollama pull llama2:7b
ollama list
```

Ollama는 기본적으로 `http://127.0.0.1:11434`에서 실행됩니다. 모델 정보는 [Ollama llama2 페이지](https://ollama.com/library/llama2)에서 확인할 수 있습니다.

### 5. 환경 파일 생성

```powershell
Copy-Item .env.example .env
```

기본 `.env` 설정은 다음과 같습니다.

```env
LLM_BACKEND=ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=llama2:7b

# RAGAS faithfulness evaluation only
CODEX_MODEL=gpt-5.6-luna
CODEX_TIMEOUT_SECONDS=300

EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
CHUNKING_STRATEGY=roi_rag
```

`.env`는 Git에서 제외됩니다. 토큰이나 로그인 파일을 저장소에 커밋하지 마세요.

### 6. 웹 서버 실행

로컬, LAN, Tailscale 접속을 모두 허용하려면 반드시 `--host 0.0.0.0`을 지정합니다.

```powershell
poetry run python run_gui.py --host 0.0.0.0 --port 8000
```

Poetry가 PATH에서 인식되지 않아도 프로젝트 가상환경을 직접 실행할 수 있습니다.

```powershell
& ".\.venv\Scripts\python.exe" ".\run_gui.py" --host 0.0.0.0 --port 8000
```

서버를 중지하려면 실행 터미널에서 `Ctrl+C`를 누릅니다.

## 접속 주소

| 환경 | 주소 | 조건 |
|---|---|---|
| 서버 PC | `http://127.0.0.1:8000` | 서버 실행 중 |
| 같은 LAN/Wi-Fi | `http://<서버-LAN-IP>:8000` | `--host 0.0.0.0`, 방화벽 허용 |
| Tailscale | `http://<서버-Tailscale-IP>:8000` | 양쪽 기기가 같은 Tailnet에 연결 |
| MagicDNS | `http://<Tailscale-장치명>:8000` | Tailnet에서 MagicDNS 활성화 |

LAN 또는 Tailscale 접속이 Windows 방화벽에 막히면 `1단계_방화벽포트개방.bat`을 관리자 권한으로 실행합니다.

이 서비스에는 자체 사용자 인증이 없습니다. 인터넷에 직접 공개하지 말고 Tailscale ACL 등으로 접근 대상을 제한하세요.

## 사용 방법

### 웹 UI

서버 실행 후 `http://127.0.0.1:8000`에 접속합니다.

- 텍스트 직접 입력 또는 UTF-8 텍스트 파일 업로드
- ROI-RAG 인덱스 생성
- 인덱스 기반 질의응답
- Codex CLI 기반 RAGAS faithfulness 평가

저장소의 `data/` 폴더에는 기본 인덱스가 포함되어 있습니다. 자신의 문서를 사용하려면 웹 UI 또는 CLI로 새 인덱스를 생성하세요.

### CLI 인덱싱

```powershell
poetry run python build_index.py --file ".\documents\sample.txt"
```

텍스트를 직접 전달할 수도 있습니다.

```powershell
poetry run python build_index.py --text "인덱싱할 문서 내용"
```

인덱스를 다시 만들면 `data/`의 기존 인덱스 파일이 갱신됩니다.

### CLI 질의응답

단일 질문:

```powershell
poetry run python run_inference.py --query "문서의 핵심 내용은 무엇인가요?"
```

대화형 모드:

```powershell
poetry run python run_inference.py --interactive
```

### RAGAS 평가

답변 생성에는 Ollama `llama2:7b`를 사용하고, faithfulness 평가에는 로그인된 Codex CLI를 사용합니다.

```powershell
poetry run python evaluate_ragas.py
```

Faithfulness claim boundaries are deterministic: the application splits the same answer
into the same sentence/semicolon-based claim IDs on every run, without a claim cache.
Codex judges only those fixed IDs and cannot add, remove, or rewrite claims. Verdicts are
still freshly generated on every evaluation, so the claim denominator is reproducible
while an individual Codex verdict can remain nondeterministic.

프로젝트 루트에 `eval_dataset.json`이 있으면 해당 데이터셋을 사용하고, 없으면 내장 예제 한 건을 평가합니다.

## REST API

서버 실행 후 Swagger UI에서 전체 스키마를 확인할 수 있습니다.

```text
http://127.0.0.1:8000/docs
```

주요 엔드포인트:

- `GET /api/current-index`: 현재 인덱스 정보
- `POST /api/build-index`: 텍스트 인덱싱
- `POST /api/upload-file`: UTF-8 텍스트 파일 업로드
- `POST /api/query`: RAG 질의응답
- `POST /api/evaluate-single`: 한 번의 `codex exec` 호출로 RAGAS faithfulness 평가
- `POST /api/evaluate-single-stream`: 단계별 진행 상태와 최종 결과를 NDJSON으로 스트리밍

## 테스트

```powershell
poetry run python -m unittest discover -s tests -t .
poetry check --lock
```

## 문제 해결

### `poetry`가 인식되지 않음

VS Code를 완전히 종료하고 다시 실행하거나 다음 전체 경로를 사용합니다.

```powershell
& "$env:USERPROFILE\.local\bin\poetry.exe" install --no-root --sync
```

### `ModuleNotFoundError` 발생

시스템 Python이 아니라 Poetry 가상환경으로 실행하고 의존성을 다시 동기화합니다.

```powershell
poetry install --no-root --sync
poetry run python run_gui.py --host 0.0.0.0 --port 8000
```

### Ollama 모델을 찾을 수 없음

```powershell
ollama pull llama2:7b
ollama list
```

Ollama가 실행 중인지 `http://127.0.0.1:11434`와 작업 표시줄의 Ollama 아이콘을 확인합니다.

### RAGAS 평가가 느리거나 실패함

```powershell
codex login status
codex exec --ephemeral "Reply with exactly: OK"
```

평가는 답변 생성에 실제로 사용된 전체 컨텍스트를 한 번의 읽기 전용 `codex exec` 호출에 전달합니다. GUI는 입력 확인, 컨텍스트 준비, Codex 판정, 결과 검증, 점수 계산 단계와 경과 시간을 실시간으로 표시합니다. 결과 캐시는 사용하지 않으므로 평가 버튼을 누를 때마다 새로 계산합니다. 필요하면 `.env`의 `CODEX_TIMEOUT_SECONDS`를 조정하세요.

### 외부 기기에서 접속할 수 없음

1. 서버가 `--host 0.0.0.0`으로 실행됐는지 확인합니다.
2. Windows 방화벽에서 TCP 8000을 허용합니다.
3. Tailscale을 사용한다면 양쪽 기기의 로그인 상태와 ACL을 확인합니다.
4. 브라우저 주소에 `http://`와 `:8000`을 모두 입력합니다.

### 8000번 포트가 이미 사용 중임

```powershell
poetry run python run_gui.py --host 0.0.0.0 --port 8001
```

접속 주소도 `http://127.0.0.1:8001`로 변경합니다.

## 주요 파일

```text
project-studio-rag/
├── app/                     # FastAPI API 및 웹 UI
├── data/                    # FAISS 인덱스와 ROI-RAG 메타데이터
├── docs/architecture.md     # 아키텍처 설명
├── tests/                   # 단위 테스트
├── config.py                # 환경변수와 ROI-RAG 설정
├── embeddings.py            # 로컬 임베딩
├── entropy.py               # RE/DE 엔트로피 계산
├── evaluate_ragas.py        # Codex exec 기반 RAGAS faithfulness 평가
├── indexer.py               # 청킹, Evidence Unit 생성, 인덱싱
├── llm_client.py            # Ollama 및 codex exec 어댑터
├── roi_rag.py               # 검색 및 답변 파이프라인
├── run_gui.py               # 웹 서버 실행
├── pyproject.toml           # Poetry 프로젝트 설정
└── poetry.lock              # 고정 의존성 버전
```

## 배포 업데이트

Windows self-hosted 환경에서는 다음 스크립트가 Git 업데이트, Poetry 의존성 동기화, 서버 재시작을 순서대로 수행합니다.

```powershell
.\deploy_poetry.bat
```

운영 경로와 방화벽, Tailscale 정책은 각 서버 환경에 맞게 별도로 설정해야 합니다.
