"""
ROI-RAG 온라인 추론 파이프라인 모듈.
질의(Query) 임베딩, FAISS Top-K Evidence Unit 검색, 대표 원문 단락 하이브리드 컨텍스트 구성 및
LLM 기반 최종 답변 추론을 담당합니다.
"""
import time
import config
import llm_client
from indexer import load_roi_rag_index
from embeddings import get_embedding_model

def get_roi_rag_pipeline():
    """
    Returns a callable pipeline function for executing ROI-RAG queries against the built index.
    """
    try:
        index_data, index_manager = load_roi_rag_index()
    except FileNotFoundError:
        print("[ROIRAG] Warning: Index not found. Please run build_index first.")
        index_data, index_manager = None, None

    embedder = get_embedding_model()

    def run_pipeline(query: str, k: int = config.RETRIEVAL_K) -> dict:
        nonlocal index_data, index_manager

        if index_data is None or index_manager is None:
            try:
                index_data, index_manager = load_roi_rag_index()
            except FileNotFoundError:
                return {
                    "answer": "Index has not been built yet. Please index some documents first using build_index.py.",
                    "retrieved_contexts": [],
                    "raw_contexts": [],
                    "prompt": "",
                    "latency_ms": 0,
                    "api_calls": 0,
                    "tokens_used": 0
                }

        start_time = time.time()

        # 1. Query Embedding
        query_emb = embedder.embed_query_np(query)

        # 2. FAISS Top-K EU Lookup
        eu_indices, similarity_scores = index_manager.search(query_emb, k=k)

        segments = index_data["segments"]
        evidence_units = index_data["evidence_units"]

        retrieved_eus = []
        context_parts = []

        for idx, score in zip(eu_indices, similarity_scores):
            if idx >= len(evidence_units):
                continue
            eu = evidence_units[idx]
            retrieved_eus.append(eu)

            supporting_segs = [segments[s_idx] for s_idx in eu["segment_indices"] if s_idx < len(segments)]
            # Hybrid Context Strategy: Provide the condensed summary AND top-3 representative raw snippets
            representative_segments = "\n".join([f"- {seg}" for seg in supporting_segs[:3]]) if supporting_segs else ""

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

        # 4. Generation
        api_calls = 0
        tokens_used = 0

        try:
            api_calls += 1
            answer = llm_client.generate(prompt)
            tokens_used += (len(prompt) + len(answer)) // 4  # Approximate token count by character length
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
            "tokens_used": tokens_used
        }

    return run_pipeline
