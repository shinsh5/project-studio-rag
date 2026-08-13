# RAGAS 평가 엔진: Codex CLI → Gemini API 전환

## 배경

RAGAS faithfulness 평가를 Codex CLI(`codex exec`, 모델 `gpt-5.6-luna`)로 실행할 때
평가할 때마다 결과가 달라지는 비일관성 문제가 있었다. Codex exec CLI는
temperature/seed 같은 샘플링 파라미터를 노출하지 않아 코드에서 제어할 방법이
없었던 것이 근본 원인이다. (생성용 Ollama는 `OLLAMA_TEMPERATURE=0.0`,
`OLLAMA_SEED=42`로 이미 고정되어 있었다.)

Gemini API로 교체하면 `temperature=0`을 직접 지정할 수 있어 판정 일관성을
높일 수 있다. 평가자 모델은 비용 절감을 위해 `gemini-2.5-flash-lite`
($0.10/$0.40 per 1M input/output tokens)를 선택했다.

## 변경 범위

RAGAS 평가는 `ragas.evaluate()` 같은 RAGAS 라이브러리 API를 쓰지 않고,
`evaluate_ragas.py`가 claim 추출·프롬프트 생성·검증을 전부 자체 구현한 뒤
LLM judge 호출 지점만 외부 CLI/API에 위임하는 구조였다. 따라서 교체 범위는
그 judge 호출 어댑터 하나로 좁혀졌고, claim 추출이나 점수 계산 로직은
전혀 건드리지 않았다.

### `config.py`, `.env.example`
- 제거: `CODEX_CLI_PATH`, `CODEX_MODEL`, `CODEX_TIMEOUT_SECONDS`
- 추가: `GEMINI_API_KEY`, `GEMINI_MODEL`(기본값 `gemini-2.5-flash-lite`),
  `GEMINI_TEMPERATURE`(기본값 `0.0`), `GEMINI_TIMEOUT_SECONDS`(기본값 `300`)

### `llm_client.py`
- 제거: `_codex_command`, `_run_codex`, `codex_generate`,
  `codex_generate_structured` (subprocess 기반 Codex CLI 어댑터 전체)
- 추가: `gemini_generate_structured(prompt, schema) -> dict`
  - `google-genai` SDK로 `response_mime_type="application/json"` +
    `response_schema`를 사용해 구조화 출력 강제
  - `_prepare_response_schema()`가 pydantic이 생성한 JSON Schema를 Gemini
    호환 형태로 정제:
    - `$ref`/`$defs`를 인라인으로 펼침 (Gemini `Schema`는 참조를 지원하지 않음)
    - `title`/`pattern`/`minLength`/`maxLength`/`additionalProperties` 제거
      (Gemini `Schema`가 인식하지 못하는 키워드)
    - 정수/숫자/불리언 타입의 `enum` 제거 (Gemini `Schema.enum`은 문자열만
      허용하여, `verdict: Literal[0, 1]`이 만든 정수 enum을 그대로 넘기면
      SDK 요청 단계에서 `validation errors ... Input should be a valid string`
      오류로 실패했음)
  - 스키마에서 빠진 제약(`pattern`, `verdict`의 0/1 범위 등)은
    `evaluate_ragas.py`가 응답을 pydantic 모델로 재검증하는 기존 단계에서
    그대로 강제되므로 검증 손실은 없음

### `evaluate_ragas.py`
- `llm_client.codex_generate_structured` 호출을
  `llm_client.gemini_generate_structured`로 교체 (한 곳)
- `CodexFaithfulnessOutput` → `GeminiFaithfulnessOutput`,
  진행 단계명 `starting_codex`/`running_codex` →
  `starting_gemini`/`running_gemini` 등 이름 교체
- 타임아웃 참조를 `config.CODEX_TIMEOUT_SECONDS` → `config.GEMINI_TIMEOUT_SECONDS`로 변경

### `app/main.py`, `app/templates/index.html`
- 응답 필드(`ragas_backend`, `judge_model`)와 에러 메시지, 진행 상황 UI
  라벨의 "codex"/"Codex" 문구를 "gemini"/"Gemini"로 교체

### `requirements.txt`, `pyproject.toml`
- `google-genai` 의존성 추가 (설치 및 동작 확인 완료)

### `deploy_poetry.bat`
- 배포 스크립트가 Codex CLI 설치 여부를 확인하고 없으면 배포를 실패시키던
  체크를 제거 (더 이상 Codex CLI를 사용하지 않으므로 이 체크가 있으면
  항상 실패함)

### 테스트
- `tests/test_llm_client.py`: Codex subprocess mock 테스트를 Gemini SDK
  mock 테스트로 교체. 스키마 정제 로직(`$ref` 인라인, 불필요 키워드 제거,
  비문자열 enum 제거) 단위 테스트 추가
- `tests/test_evaluate_ragas.py`, `tests/test_ragas_stream.py`: mock 대상
  경로와 진행 단계 문자열을 Gemini 기준으로 교체
- 전체 37개 테스트 통과 확인 (`python -m unittest discover -s tests -t .`)

### 문서
- `README.md`, `docs/architecture.md`: 요구 사항, 실행 구조 표, 환경변수
  예시, 문제 해결 섹션 등에서 Codex 관련 서술을 Gemini 기준으로 갱신

## 마이그레이션 체크리스트 (커밋 후 실행 전 확인)

- [ ] `.env`에 `GEMINI_API_KEY` 설정 ([Google AI Studio](https://aistudio.google.com/apikey)에서 발급)
- [ ] `poetry install --no-root --sync` 또는 `pip install -r requirements.txt`로
      `google-genai` 설치
- [ ] 동일 질문을 여러 번 평가해 faithfulness 점수 분산이 줄었는지 확인
- [ ] API 키가 대화/스크린샷 등으로 외부에 노출된 적이 있다면 재발급 검토

## 사후 검증: 분산이 완전히 사라지지는 않았다 (2026-08-12)

전환 후 실측한 결과, **한 세션 안에서는 일관되지만 세션이 바뀌면 여전히
달라진다.** 생성 쪽은 완전히 결정적임을 먼저 확인했다 — 동일 질문 2회 실행 시
답변이 sha256까지 일치한다. 따라서 변동은 전부 판정자 쪽 원인이다.

동일 질문·동일 답변에 대한 측정값:

| 시점 | faithfulness | claim 수 |
|---|---|---|
| 세션 A | 0.900 | 10 |
| 세션 B (연속 4회) | 0.750 | 4 |
| GUI 실행 | 0.667 | 3~9 |

분자가 아니라 **분모가 요동친다.** 원인은 verdict 판정이 아니라
`metric._create_statements()`의 답변 분해다. `temperature=0`으로도 Gemini의
statement 분해는 세션 간 재현되지 않는다.

특히 답변에 웹 스크래핑 잔여물(영상 캡션·타임스탬프 등)이 섞이면 문장 경계가
모호해져 분해가 크게 흔들린다. 코퍼스 세그먼트의 20%가 이런 오염을 포함한다.

**실무적 함의:** 전략 간 0.05 수준의 차이는 유의미한 개선으로 해석할 수 없다.
비교 실험에서는 문항당 3회 이상 측정해 중앙값을 쓰고, 코퍼스 정제를 선행해야
한다. 상세 측정은 `docs/faithfulness-baseline.md` 참조.
