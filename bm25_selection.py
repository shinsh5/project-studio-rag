"""
BM25 기반 문장 단위 Evidence Selection.

검색된 근거의 원문 스니펫은 프롬프트 토큰의 82%를 차지하지만, 세그먼트 하나에
평균 9.5문장이 들어 있고 질문에 실제로 필요한 것은 보통 1~2문장이다. 이 모듈은
각 세그먼트를 문장으로 나눈 뒤 질의와의 BM25 점수 상위 N개만 남겨 프롬프트를
압축한다.

밀집 임베딩 대신 BM25를 쓰는 이유:
- 문장 임베딩이 인덱스에 없어 질의 시점에 수십 개를 인코딩해야 한다 (BM25는 비용 0)
- 뉴스 코퍼스라 고유명사 일치가 곧 정답인 질문이 많다
- 스크래핑 잔여물(영상 캡션·날씨 위젯)은 질의어와 겹치지 않아 자연히 탈락한다

EU 요약문에는 적용하지 않는다. 이미 압축된 텍스트라 더 잘라내면 정보 손실이 크다.
"""
import math
import re

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_TOKEN = re.compile(r"[a-z0-9]+")

# BM25 표준 파라미터. k1은 단어 빈도 포화 지점, b는 길이 정규화 강도.
K1 = 1.5
B = 0.75

# 후보 풀이 세그먼트 하나 분량(평균 9.5문장)이라 IDF 통계가 작다. 이 규모에서는
# "from" 같은 불용어도 IDF가 0.8을 넘고, 길이 정규화까지 겹치면 짧고 무관한 문장이
# 상위로 올라온다. 질의에서 불용어를 걷어내야 고유명사 신호가 살아난다.
_STOPWORDS = frozenset("""
a an the and or but if then than that this these those there here
is are was were be been being am do does did doing done
have has had having will would shall should can could may might must
i you he she it we they me him her us them my your his its our their
of in on at to for from by with about into over after before between
under above during through against out up down off again further
what which who whom whose when where why how
as so no not nor only own same too very s t just don now
""".split())


def split_sentences(text: str) -> list[str]:
    """Split into sentences, dropping fragments too short to carry a fact."""
    parts = [s.strip() for s in _SENTENCE_BOUNDARY.split(text)]
    return [s for s in parts if len(s.split()) >= 3]


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _bm25_scores(query_tokens: list[str], docs: list[list[str]]) -> list[float]:
    """
    Score each doc against the query. IDF is computed over the candidate pool
    itself, so a name appearing in one of a few dozen sentences scores high
    without needing a precomputed corpus-wide statistic.
    """
    n = len(docs)
    if n == 0:
        return []

    lengths = [len(d) for d in docs]
    avg_len = sum(lengths) / n if n else 0.0

    doc_freq: dict[str, int] = {}
    for doc in docs:
        for term in set(doc):
            doc_freq[term] = doc_freq.get(term, 0) + 1

    scores = [0.0] * n
    for term in set(query_tokens):
        df = doc_freq.get(term, 0)
        if df == 0:
            continue
        idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
        for i, doc in enumerate(docs):
            freq = doc.count(term)
            if freq == 0:
                continue
            denom = freq + K1 * (1.0 - B + B * (lengths[i] / avg_len if avg_len else 1.0))
            scores[i] += idf * (freq * (K1 + 1.0)) / denom
    return scores


def select_sentences(query: str, text: str, top_n: int) -> str:
    """
    Keep the top_n sentences of one raw passage by BM25 relevance to the query,
    restored to their original order so the passage still reads sequentially.

    Returns the text unchanged when it already has at most top_n sentences, or
    when no sentence shares a term with the query -- dropping everything would
    remove evidence the retriever deliberately selected.
    """
    if top_n <= 0:
        return text

    sentences = split_sentences(text)
    if len(sentences) <= top_n:
        return text

    query_tokens = [t for t in _tokenize(query) if t not in _STOPWORDS]
    scores = _bm25_scores(query_tokens, [_tokenize(s) for s in sentences])
    if not any(score > 0 for score in scores):
        return " ".join(sentences[:top_n])

    ranked = sorted(range(len(sentences)), key=lambda i: -scores[i])[:top_n]
    return " ".join(sentences[i] for i in sorted(ranked))
