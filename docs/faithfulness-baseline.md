# ROI-RAG Faithfulness 기준선 측정 (MULTINEWS_QA 50문항)

Small-to-Big 결합 효과를 측정하려면 먼저 ROI-RAG 단독 기준선이 필요하다.
이 문서는 그 기준선과, 측정 과정에서 드러난 **지표 자체의 한계**를 기록한다.

## 1. 측정 조건

| 항목 | 값 |
|---|---|
| 생성/요약 모델 | `qwen2.5:7b` (temperature 0, seed 42, num_ctx 8192) |
| 판정 모델 | `gemini-2.5-flash-lite` (temperature 0), RAGAS `Faithfulness` |
| 청킹 전략 | `roi_rag` (CHUNK_SIZE 200 / OVERLAP 50) |
| 인덱스 | 세그먼트 169, EU 59 (LOW 24 / MEDIUM 8 / HIGH 27) |
| 코퍼스 | `data/multinews_corpus.txt` (**정제 전** 원본) |
| 문항 | `data/multinews_clean_dataset.py`의 `MULTINEWS_QA` 50개 |
| 응답 캐시 | 비활성 (`use_cache=False`) |

생성은 완전히 결정적임을 확인했다. 동일 질문 2회 실행 시 답변이 바이트 단위로
일치한다(sha256 동일). 따라서 이후 관찰되는 점수 변동은 전부 판정자 쪽 원인이다.

## 2. 결과 요약

```
성공 49 / 에러 1
평균 0.9001   중앙값 1.0000

1.000  ##################################### 37
0.889  # 1
0.857  ## 2
0.750  ## 2
0.667  #### 4
0.333  # 1
0.000  ## 2
```

**49문항 중 37개(76%)가 만점이다.** 지표가 상단에 포화되어 있어, 이 상태로는
Small-to-Big의 개선분을 검출하기 어렵다.

에러 1건(#25)은 Gemini가 구조화 출력 파싱에 실패한 것으로 시스템 문제가 아니다.

## 3. 1.0 미만 문항 분류

만점을 못 받은 12문항을 원인별로 나누면 **두 부류 모두 STB로 해결되지 않는다.**

### 3.1 과잉 거부형 (5건) — #9, #14, #36, #39, #43

모델이 근거를 받고도 "정보가 없다"고 답한 경우. RAGAS는 이 거부 문장을
"컨텍스트로 뒷받침되지 않는 statement"로 판정해 감점한다.

가장 명확한 사례가 **#43 (0.000)** 이다.

- 질문: *What caused the chain link fence at Fleming park in Augusta to become electrified?*
- 답변: *"The provided information does not contain details about a chain link fence at Fleming park in Augusta becoming electrified."*
- 그러나 컨텍스트에 포함된 **seg 143에 정답이 그대로 있었다**:
  > "A 12-year-old boy died Monday after being electrocuted when he touched a fence
  > that had been **electrified by a live wire** at a Fleming Athletic Complex ball field"

검색은 성공했고 생성이 실패했다. 원인으로 두 가지가 겹친다.

1. **세그먼트가 기사 경계를 가로지른다.** seg 143의 실제 내용은
   `"...reconstruct the form of a penis from A 12-year-old boy died Monday after being
   electrocuted..."` 로, 음경 이식 기사 끝과 감전 사고 기사 시작이 구분자 없이
   이어져 있다. 200단어 고정 윈도우 청킹의 부작용이다.
2. **질문과 원문의 표현 차이.** 질문은 "chain link fence at Fleming park",
   원문은 "a fence ... at a Fleming Athletic Complex ball field"이다.

### 3.2 추론 확장형 (7건) — #2, #6, #10, #22, #29, #32, #47

컨텍스트에 없는 판단·계산·시제를 답변에 덧붙인 경우. 이쪽이 faithfulness가
본래 잡아야 할 진짜 실패다.

| # | 점수 | 감점된 statement |
|---|---|---|
| 47 | 0.750 | "As of now, Walker is approximately 14 years old." (생년월일에서 나이를 **계산**) |
| 10 | 0.667 | "Officer Crystal Almeida **remains** in critical condition." (시제를 현재로 확장) |
| 6 | 0.857 | "Alex Chu is **not willing** to conduct archaeological investigations." (판단 추가) |
| 29 | 0.889 | "The U.S. Supreme Court **will find** the health care law constitutional." (예측) |

## 4. 지표 자체의 문제

### 4.1 적절한 거부가 처벌된다

프롬프트 규칙 3은 *"근거에서 찾을 수 없으면 문서에 없다고 명시하라"* 이다.
모델이 이 지시를 따르면 RAGAS는 그 문장을 unsupported로 채점한다.
**환각하는 모델과 올바르게 거부하는 모델이 똑같이 0.0을 받는다.**

`retrieved_contexts`와 대조하는 방식의 구조적 한계이며, ROI-RAG나 STB 어느
쪽으로도 개선되지 않는다.

### 4.2 판정이 세션 간 재현되지 않는다

`docs/gemini-migration.md`는 Codex CLI의 비일관성을 Gemini `temperature=0`으로
해결했다고 기록한다. 그러나 **세션 간에는 여전히 재현되지 않는다.**

동일 질문·동일 답변(sha256 일치 확인)에 대해:

| 측정 시점 | 점수 | claim 수 |
|---|---|---|
| 세션 A | 0.900 | 10개 |
| 세션 B (4회 연속) | 0.750 | 4개 |
| GUI 실행 | 0.667 | 3~9개 |

한 세션 안에서는 4회 연속 동일했으나 세션이 바뀌면 달라진다. 원인은 판정이
아니라 **`_create_statements`의 답변 분해**다. 분모가 10 → 4 → 3으로 요동친다.

즉 문제가 해결된 것이 아니라 범위가 좁혀졌을 뿐이다. **0.05 수준의 차이를
전략 간 개선으로 해석해서는 안 된다.**

### 4.3 코퍼스 오염이 분해 불안정을 유발한다

세그먼트 169개 중 **34개(20%)** 에 웹 스크래핑 잔여물이 섞여 있다.

| 유형 | 세그먼트 수 |
|---|---|
| 영상 플레이리스트 (`1:11 See ...`, `Pause 1:58`) | 23 |
| 공유/embed 위젯 (`toggle caption`, `autoplay`) | 10 |
| 날씨 위젯 (`Winds S at 10 to 20 mph`) | 4 |
| JavaScript / legacy Twitter 안내 | 2 |
| 아카이브 크롤러 안내 | 2 |

이 텍스트가 답변에 옮겨지면 RAGAS가 문장 경계를 일관되게 잡지 못한다.
GUI 샘플 `HGTV Renovation Lawsuit`이 0.667~0.9를 오간 원인이 이것이다.
해당 답변의 claim 10개 중 6개가 동영상 UI 텍스트였다.

> 참고: 이 문항은 원문 자체가 `"...favor the television show but not` 에서
> 잘려 있어 **정답이 코퍼스에 존재하지 않는다.** 평가 문항으로 부적합하다.

## 5. STB 비교에 쓸 수 있는 문항

**현재 데이터로는 적합한 문항을 찾지 못했다.** 1.0 미만 12문항 중

- 과잉 거부형 5건 → 생성 문제. 근거는 이미 전달되고 있어 STB로 개선 여지 없음
- 추론 확장형 7건 → 생성 문제. 컨텍스트 부족이 아니라 모델이 덧붙인 것

즉 **검색 누락 때문에 감점된 문항이 하나도 없다.** ROI-RAG의 검색은 이
코퍼스·이 문항 세트에서 충분히 동작하고 있으며, STB가 메울 공백이 측정되지
않는다.

보조 실험으로, EU 경계를 가로지르는 문항 3개를 검색 시뮬레이션으로 선별해
GUI에 추가했다(`🔬 Cross-EU Hard Samples`, 커밋 `61e25e3`). 인접 세그먼트
168쌍 중 **141쌍(84%)** 이 서로 다른 EU로 분리되는 성질을 이용한 것이다.
다만 이들도 아직 faithfulness 실측으로 검증되지 않았다.

## 6. 다음 단계 제안

측정 도구를 먼저 고치지 않으면 STB 비교 결과를 신뢰할 수 없다.

1. **코퍼스 정제** — 문단 단위 규칙 4개 + 구절 제거 3개로 2,204자(1.4%)만
   제거하면 오염 34개 세그먼트가 정리된다. 정상 타임스탬프(`8:11 a.m.` 등)는
   보존된다. 인덱스 재빌드 약 70초.
2. **기사 경계 인식 청킹** — #43의 근본 원인. 현재 `segment_text()`는 코퍼스
   전체를 단어 윈도우로 자르므로 서로 다른 기사가 한 세그먼트에 섞인다.
3. **거부 문장을 채점에서 제외** — "정보 없음" 류 statement는 faithfulness
   분모에서 빼거나 별도 지표(거부율)로 분리한다.
4. **반복 측정** — 세션 간 변동이 크므로 문항당 3회 이상 측정해 중앙값을 쓴다.
5. **지표 추가** — faithfulness는 76%가 만점이라 변별력이 없다. STB가 본래
   우위를 갖는 `ContextRecall`, 정확도를 보는 `AnswerCorrectness`를 함께 잰다.

---

*측정일: 2026-08-12 / 원본 결과: `sweep_results.json` (문항별 답변·실패 claim·검색 EU 포함)*
