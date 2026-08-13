"""
ROI-RAG 온라인 추론 파이프라인 모듈.
질의(Query) 임베딩, FAISS Top-K Evidence Unit 검색, 대표 원문 단락 하이브리드 컨텍스트 구성 및
LLM 기반 최종 답변 추론을 담당합니다.
"""
import hashlib
import json
import os
import threading
import time
from collections import OrderedDict, Counter
import numpy as np
import bm25_selection
import config
import llm_client
from indexer import load_roi_rag_index
from embeddings import get_embedding_model


_cached_index_data = None
_cached_index_manager = None
_cached_index_mtime = None
_cached_index_strategy = None
_response_cache: OrderedDict[str, str] = OrderedDict()
_response_cache_lock = threading.RLock()
_response_generation_lock = threading.Lock()


def _response_cache_key(prompt: str, index_version: str) -> str:
    """Build a stable key from every input that can affect Llama generation."""
    payload = {
        "prompt": prompt,
        "index_version": index_version,
        "model": config.OLLAMA_MODEL,
        "options": {
            "temperature": config.OLLAMA_TEMPERATURE,
            "seed": config.OLLAMA_SEED,
            "top_k": config.OLLAMA_TOP_K,
            "top_p": config.OLLAMA_TOP_P,
            "num_ctx": config.OLLAMA_NUM_CTX,
            "num_predict": config.OLLAMA_NUM_PREDICT,
            "num_batch": config.OLLAMA_NUM_BATCH,
        },
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _get_cached_response(cache_key: str) -> str | None:
    """Return and refresh one in-memory LRU response-cache entry."""
    with _response_cache_lock:
        answer = _response_cache.pop(cache_key, None)
        if answer is not None:
            _response_cache[cache_key] = answer
        return answer


def _store_cached_response(cache_key: str, answer: str) -> None:
    """Store one response while enforcing the configured LRU size limit."""
    with _response_cache_lock:
        _response_cache.pop(cache_key, None)
        _response_cache[cache_key] = answer
        max_size = max(1, config.LLM_RESPONSE_CACHE_MAX_SIZE)
        while len(_response_cache) > max_size:
            _response_cache.popitem(last=False)


def clear_response_cache() -> None:
    """Clear Llama answers only; RAGAS evaluations are never cached."""
    with _response_cache_lock:
        _response_cache.clear()

def _rank_segments_by_query(
    segment_indices: list[int],
    segment_embeddings: np.ndarray,
    query_emb: np.ndarray,
) -> list[int]:
    """
    Reorders one EU's segment indices by cosine similarity to the query, so that
    prompt truncation keeps the segments most relevant to the question instead of
    whichever ones the greedy EU construction happened to add first.
    Falls back to the stored order when embeddings are unavailable.
    """
    if len(segment_embeddings) == 0:
        return segment_indices

    valid = [idx for idx in segment_indices if idx < len(segment_embeddings)]
    if not valid:
        return segment_indices

    mat = segment_embeddings[valid].astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    mat = mat / norms

    q_norm = np.linalg.norm(query_emb)
    q_vec = (query_emb / q_norm if q_norm > 0 else query_emb).astype(np.float32)

    similarities = mat @ q_vec
    return [valid[i] for i in np.argsort(-similarities)]


def get_roi_rag_pipeline():
    """
    Returns a callable pipeline function for executing ROI-RAG queries against the built index.
    """
    global _cached_index_data, _cached_index_manager, _cached_index_mtime
    embedder = get_embedding_model()

    if config.LLM_BACKEND.lower() == "ollama":
        llm_client.warmup_ollama_model()

    def run_pipeline(
        query: str,
        k: int = config.RETRIEVAL_K,
        use_cache: bool = config.LLM_RESPONSE_CACHE_DEFAULT,
        use_bm25: bool | None = None,
    ) -> dict:
        global _cached_index_data, _cached_index_manager, _cached_index_mtime, _cached_index_strategy

        index_path = config.get_index_path()
        current_strategy = config.CHUNKING_STRATEGY
        if use_bm25 is None:
            use_bm25 = config.BM25_EVIDENCE_SELECTION

        if not os.path.exists(index_path):
            return {
                "answer": "Index has not been built yet. Please index some documents first using build_index.py.",
                "retrieved_contexts": [],
                "raw_contexts": [],
                "prompt": "",
                "latency_ms": 0,
                "api_calls": 0,
                "tokens_used": 0,
                "usage": {},
                "cache_hit": False
            }

        try:
            current_mtime = os.path.getmtime(index_path)
            if (
                _cached_index_data is None
                or _cached_index_manager is None
                or _cached_index_mtime != current_mtime
                or _cached_index_strategy != current_strategy
            ):
                _cached_index_data, _cached_index_manager = load_roi_rag_index()
                _cached_index_mtime = current_mtime
                _cached_index_strategy = current_strategy
                print(
                    f"[ROIRAG] Index loaded/cached in memory "
                    f"(strategy={current_strategy}, mtime={current_mtime})."
                )
            index_data, index_manager = _cached_index_data, _cached_index_manager
        except Exception as e:
            print(f"[ROIRAG] Error loading index: {e}")
            return {
                "answer": f"Error loading ROI-RAG index: {e}",
                "retrieved_contexts": [],
                "raw_contexts": [],
                "prompt": "",
                "latency_ms": 0,
                "api_calls": 0,
                "tokens_used": 0,
                "usage": {},
                "cache_hit": False
            }

        start_time = time.time()

        # 1. Query Embedding
        query_emb = embedder.embed_query_np(query)

        # 2. FAISS Top-K EU Lookup
        eu_indices, similarity_scores = index_manager.search(query_emb, k=k)

        segments = index_data["segments"]
        evidence_units = index_data["evidence_units"]
        segment_embeddings = index_data.get(
            "segment_embeddings", np.empty((0, 384), dtype=np.float32)
        )

        retrieved_eus = []
        context_parts = []
        parent_candidates = 0
        parent_duplicates_removed = 0
        used_parent_ids: set[int] = set()
        bm25_sentences_before = 0
        bm25_sentences_after = 0

        is_stb = index_data.get("chunking_strategy") == "small_to_big"
        parent_chunks = index_data.get("parent_chunks", [])
        leaf_to_parent = index_data.get("leaf_to_parent", [])

        def _trim(raw_text: str) -> str:
            """Apply BM25 and collect the compression metrics shown in React Flow."""
            nonlocal bm25_sentences_before, bm25_sentences_after
            before = len(bm25_selection.split_sentences(raw_text))
            bm25_sentences_before += before
            if not use_bm25:
                bm25_sentences_after += before
                return raw_text
            selected = bm25_selection.select_sentences(
                query, raw_text, config.BM25_SENTENCES_PER_SEGMENT
            )
            bm25_sentences_after += len(bm25_selection.split_sentences(selected))
            return selected

        for idx, score in zip(eu_indices, similarity_scores):
            if idx >= len(evidence_units):
                continue
            eu = evidence_units[idx]
            retrieved_eus.append(eu)

            if is_stb and parent_chunks and leaf_to_parent:
                # EU 내 각 segment가 속한 parent를 집계
                seg_indices = [s_idx for s_idx in eu["segment_indices"] if s_idx < len(leaf_to_parent)]
                parent_ids = [leaf_to_parent[s_idx] for s_idx in seg_indices]
                parent_counter = Counter(parent_ids)

                # AUTOMERGE_THRESHOLD 이상인 parent 청크를 컨텍스트로 사용
                selected_parents = [
                    pid for pid, cnt in parent_counter.items()
                    if cnt / len(seg_indices) >= config.AUTOMERGE_THRESHOLD
                ]

                # A parent can be reached from several retrieved EUs. Keep its
                # full text only once while preserving every EU summary.
                had_selected_parents = bool(selected_parents)
                valid_parent_ids = [
                    pid for pid in sorted(selected_parents)
                    if 0 <= pid < len(parent_chunks)
                ]
                parent_candidates += len(valid_parent_ids)
                selected_parents = [
                    pid for pid in valid_parent_ids if pid not in used_parent_ids
                ]
                parent_duplicates_removed += (
                    len(valid_parent_ids) - len(selected_parents)
                )
                used_parent_ids.update(selected_parents)

                if selected_parents:
                    parent_texts = "\n\n".join([
                        f"[Parent Chunk #{pid}]\n{_trim(parent_chunks[pid])}"
                        for pid in sorted(selected_parents)
                        if pid < len(parent_chunks)
                    ])
                    eu_text = (
                        f"Evidence Unit #{eu['eu_id']} (Similarity: {score:.4f}, Redundancy: {eu['regime']}, "
                        f"RE: {eu['re']}, DE: {eu['de']})\n"
                        f"Summary: {eu['summary']}\n"
                        f"Extended Context (Small-to-Big):\n{parent_texts}"
                    )
                elif not had_selected_parents:
                    # threshold 미달 시 leaf 원문 그대로
                    supporting_segs = [segments[s_idx] for s_idx in eu["segment_indices"] if s_idx < len(segments)]
                    raw_text = "\n".join([f"- {_trim(seg)}" for seg in supporting_segs])
                    eu_text = (
                        f"Evidence Unit #{eu['eu_id']} (Similarity: {score:.4f}, Redundancy: {eu['regime']}, "
                        f"RE: {eu['re']}, DE: {eu['de']})\n"
                        f"Summary: {eu['summary']}\n"
                        f"Top Original Snippets:\n{raw_text}"
                    )
                else:
                    # All selected parents were already emitted by an earlier EU.
                    eu_text = (
                        f"Evidence Unit #{eu['eu_id']} (Similarity: {score:.4f}, Redundancy: {eu['regime']}, "
                        f"RE: {eu['re']}, DE: {eu['de']})\n"
                        f"Summary: {eu['summary']}\n"
                        "Extended Context (Small-to-Big): parent text already included above."
                    )
            else:
                # Hybrid Context Strategy: Provide the condensed summary AND top-3 representative raw snippets,
                # picking the snippets most similar to the query rather than the first three stored.
                ranked_indices = _rank_segments_by_query(
                    eu["segment_indices"], segment_embeddings, query_emb
                )
                supporting_segs = [segments[s_idx] for s_idx in ranked_indices[:3] if s_idx < len(segments)]
                representative_segments = "\n".join([f"- {_trim(seg)}" for seg in supporting_segs]) if supporting_segs else ""
                eu_text = (
                    f"Evidence Unit #{eu['eu_id']} (Similarity: {score:.4f}, Redundancy: {eu['regime']}, "
                    f"RE: {eu['re']}, DE: {eu['de']})\n"
                    f"Summary: {eu['summary']}\n"
                    f"Top Original Snippets:\n{representative_segments}"
                )
            context_parts.append(eu_text)

        context_str = "\n\n".join(context_parts)

        # 3. Grounded Prompt Assembly
        prompt = (
            "You are a helpful and strict assistant. Answer the user's query using ONLY the retrieved Evidence Units below. "
            "Each Evidence Unit contains a condensed summary of facts and representative original snippets.\n"
            "CRITICAL RULES:\n"
            "1. Your answer must be fully grounded in the provided evidence.\n"
            "2. Do NOT use any outside knowledge.\n"
            "3. If the specific details cannot be found in the evidence, clearly state that the information is not present in the provided documents.\n"
            "4. Do NOT hallucinate. Keep the answer highly focused and precise.\n\n"
            f"=== RETRIEVED EVIDENCE ===\n{context_str}\n==========================\n\n"
            f"User Query: {query}\n\n"
            "Answer:"
        )

        # 4. Generation (Llama response cache only; RAGAS is always fresh)
        api_calls = 0
        tokens_used = 0
        usage: dict = {}
        cache_hit = False
        bm25_tag = (
            f"bm25:{config.BM25_SENTENCES_PER_SEGMENT}" if use_bm25 else "bm25:off"
        )
        index_version = (
            f"{current_strategy}:{current_mtime}:{os.path.getsize(index_path)}:{bm25_tag}"
        )
        cache_key = _response_cache_key(prompt, index_version)

        try:
            with _response_generation_lock:
                answer = _get_cached_response(cache_key) if use_cache else None
                cache_hit = answer is not None
                if answer is None:
                    api_calls = 1
                    answer, usage = llm_client.generate_with_usage(prompt)
                    # Ollama reports exact counts; fall back to the character
                    # estimate only when it omits them.
                    tokens_used = usage.get("total_tokens") or (
                        len(prompt) + len(answer)
                    ) // 4
                    if use_cache:
                        _store_cached_response(cache_key, answer)
        except Exception as e:
            answer = f"[LLM Error: {e}]\n\nFallback Evidence Summaries:\n" + "\n".join([eu["summary"] for eu in retrieved_eus])

        latency_ms = int((time.time() - start_time) * 1000)

        return {
            "answer": answer,
            "retrieved_contexts": context_parts,
            "raw_contexts": retrieved_eus,
            "prompt": prompt,
            "latency_ms": latency_ms,
            "api_calls": api_calls,
            "tokens_used": tokens_used,
            "usage": usage,
            "bm25_evidence_selection": use_bm25,
            "pipeline_metrics": {
                "embedding_dimension": int(query_emb.shape[-1]),
                "retrieved_eus": len(retrieved_eus),
                "small_to_big_enabled": bool(is_stb and parent_chunks and leaf_to_parent),
                "expanded_parents": len(used_parent_ids),
                "parent_candidates": parent_candidates,
                "unique_parents": len(used_parent_ids),
                "parent_duplicates_removed": parent_duplicates_removed,
                "bm25_enabled": bool(use_bm25),
                "bm25_sentences_before": bm25_sentences_before,
                "bm25_sentences_after": bm25_sentences_after,
                "prompt_chars": len(prompt),
                "context_chars": len(context_str),
                "latency_ms": latency_ms,
                "cache_hit": cache_hit,
            },
            "cache_hit": cache_hit
        }

    return run_pipeline
