"""
ROI-RAG 전용 오프라인 인덱서 모듈.
텍스트 청킹, 후보 이웃 탐색, 엔트로피 기반 탐욕적(Greedy) Evidence Unit(EU) 구성,
적응형 LLM 요약 및 FAISS 벡터 인덱싱을 담당합니다.
"""
import os
import json
import time
import numpy as np
import faiss
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
import llm_client
from embeddings import get_embedding_model
from entropy import calculate_redundancy_entropy, calculate_diversity_entropy

def segment_text(text: str) -> list[str]:
    """
    Splits text into chunks of roughly CHUNK_SIZE words, with CHUNK_OVERLAP overlap.
    """
    words = text.split()
    if not words:
        return []
    
    segments = []
    step = config.CHUNK_SIZE - config.CHUNK_OVERLAP
    if step <= 0:
        step = config.CHUNK_SIZE
        
    for i in range(0, len(words), step):
        chunk_words = words[i:i + config.CHUNK_SIZE]
        chunk_text = " ".join(chunk_words)
        segments.append(chunk_text)
        if i + config.CHUNK_SIZE >= len(words):
            break
            
    return segments

def build_candidate_neighborhoods(embeddings: np.ndarray, k: int = config.NEIGHBORHOOD_K) -> tuple[list[list[int]], np.ndarray]:
    """
    For each segment, finds the indices of its top-K nearest semantic neighbors (including itself).
    Returns (neighborhoods, precomputed_similarity_matrix).
    """
    n = len(embeddings)
    if n == 0:
        return [], np.empty((0, 0), dtype=np.float32)

    k = min(k, n)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    norm_embeddings = embeddings / norms

    sim_matrix = np.dot(norm_embeddings, norm_embeddings.T)
    sim_matrix = np.clip(sim_matrix, 0.0, 1.0).astype(np.float32)

    neighborhoods = []
    for i in range(n):
        nearest_indices = np.argsort(-sim_matrix[i])[:k]
        neighborhoods.append(nearest_indices.tolist())

    return neighborhoods, sim_matrix

def _entropy_from_sim_submatrix(global_sim: np.ndarray, indices: list[int]) -> tuple[float, float]:
    """
    Extract a submatrix from the precomputed global similarity matrix and compute RE/DE.
    Avoids recomputing O(m² * d) dot products.
    """
    idx = np.array(indices)
    sub = global_sim[np.ix_(idx, idx)]
    re = calculate_redundancy_entropy(sub)
    de = calculate_diversity_entropy(sub)
    return re, de

def summarize_eu_mock(segments: list[str], regime: str) -> str:
    """
    Rule-based mock summarization fallback when LLM fails or API key is not set.
    """
    if regime == "LOW" or not segments:
        return "\n\n".join(segments)
    elif regime == "MEDIUM":
        return f"[Partial Summary Fallback]\n" + "\n\n".join(segments[:max(1, len(segments)//2)])
    else:
        return f"[Aggressive Summary Fallback]\n" + segments[0]

def summarize_eu_with_llm(segments: list[str], regime: str) -> str:
    """
    Performs adaptive summarization based on redundancy regime using configured LLM.
    """
    if regime == "LOW" or not segments:
        return "\n\n".join(segments)

    context_text = "\n---\n".join([f"Segment {i+1}: {seg}" for i, seg in enumerate(segments)])

    if regime == "MEDIUM":
        prompt = (
            f"Please synthesize the following related pieces of text into a concise, bulleted summary "
            f"of the key unique facts. Eliminate duplicate details and keep it brief.\n"
            f"CRITICAL INSTRUCTION: You MUST preserve all proper nouns, entities (names of people, places, organizations), dates, and specific numbers from the text. Do NOT omit them.\n\n"
            f"Context:\n{context_text}"
        )
    else:  # HIGH regime
        prompt = (
            f"The following pieces of text contain highly redundant and repetitive information. "
            f"Write a short and highly dense paragraph summarizing the core unique message. "
            f"Do not include wordy explanations.\n"
            f"CRITICAL INSTRUCTION: Even though you are summarizing, you MUST preserve all proper nouns, entities (names of people, places, organizations), dates, and specific numbers from the text. Do NOT omit them. It is crucial to keep specific details intact.\n\n"
            f"Context:\n{context_text}"
        )

    try:
        return llm_client.generate(prompt)
    except Exception as e:
        print(f"[Indexer] LLM summarization error ({e}). Falling back to rule-based summary.")
        return summarize_eu_mock(segments, regime)

class FAISSIndexManager:
    """
    Manages the FAISS vector index and backup NumPy cosine index for Evidence Units (EUs).
    """
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = None
        self.has_faiss = True
        self.eu_embeddings = []
        
    def build_index(self, embeddings: np.ndarray):
        """
        Builds FAISS index (Inner Product / Cosine Similarity on normalized vectors).
        embeddings: shape (n_eus, dimension)
        """
        self.eu_embeddings = embeddings.astype(np.float32)
        n_eus = len(embeddings)
        if n_eus == 0:
            return
            
        norms = np.linalg.norm(self.eu_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        norm_embeddings = self.eu_embeddings / norms
        
        try:
            self.index = faiss.IndexFlatIP(self.dimension)
            self.index.add(norm_embeddings)
            self.has_faiss = True
            print(f"[FAISSIndexManager] FAISS index successfully built with {n_eus} vectors.")
        except Exception as e:
            print(f"[FAISSIndexManager] Failed to initialize FAISS index: {e}. Falling back to NumPy search.")
            self.has_faiss = False
            self.index = None
            
    def search(self, query_embedding: np.ndarray, k: int = config.RETRIEVAL_K) -> tuple[list[int], list[float]]:
        """
        Searches the index for top-K nearest neighbors.
        Returns: (indices, similarity_scores)
        """
        n_eus = len(self.eu_embeddings)
        if n_eus == 0:
            return [], []
            
        k = min(k, n_eus)
        
        q_norm = np.linalg.norm(query_embedding)
        q_vec = query_embedding / q_norm if q_norm > 0 else query_embedding
        q_vec = q_vec.reshape(1, -1).astype(np.float32)
        
        if self.has_faiss and self.index is not None:
            try:
                scores, indices = self.index.search(q_vec, k)
                return indices[0].tolist(), scores[0].tolist()
            except Exception as e:
                print(f"[FAISSIndexManager] FAISS search failed ({e}). Falling back to NumPy.")
                
        # NumPy fallback search
        norms = np.linalg.norm(self.eu_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        norm_embeddings = self.eu_embeddings / norms
        
        similarities = np.dot(norm_embeddings, q_vec.T).flatten()
        top_k_indices = np.argsort(-similarities)[:k]
        top_k_scores = similarities[top_k_indices]
        
        return top_k_indices.tolist(), top_k_scores.tolist()

    def save(self, filepath: str):
        if self.has_faiss and self.index is not None:
            try:
                faiss.write_index(self.index, filepath)
                print(f"[FAISSIndexManager] Saved FAISS index to {filepath}")
            except Exception as e:
                print(f"[FAISSIndexManager] Could not save FAISS index: {e}")

    def load(self, filepath: str):
        if os.path.exists(filepath):
            try:
                self.index = faiss.read_index(filepath)
                self.has_faiss = True
                print(f"[FAISSIndexManager] Loaded FAISS index from {filepath}")
            except Exception as e:
                print(f"[FAISSIndexManager] Could not load FAISS index: {e}. Fallback to NumPy search.")
                self.has_faiss = False
        else:
            self.has_faiss = False

def build_roi_rag_index(text: str) -> dict:
    """
    Complete offline ROI-RAG pipeline:
    1. Text Chunking
    2. Segment Embedding
    3. Candidate Neighborhood Construction
    4. Greedy Entropy-Guided EU Construction
    5. Adaptive Summarization
    6. FAISS Vector Indexing & Save
    """
    t0 = time.time()
    print("======================================================")
    print("Starting ROI-RAG Offline Index Building...")
    print("======================================================")

    # 1. Chunking
    segments = segment_text(text)
    if not segments:
        print("[Indexer] No segments found.")
        return {"segments": [], "evidence_units": []}

    if config.MAX_SEGMENTS and len(segments) > config.MAX_SEGMENTS:
        print(f"[Indexer] MAX_SEGMENTS={config.MAX_SEGMENTS}: truncating from {len(segments)} segments.")
        segments = segments[:config.MAX_SEGMENTS]

    print(f"[Indexer] Created {len(segments)} text segments.")

    # 2. Embedding
    embedder = get_embedding_model()
    segment_embeddings = embedder.embed_documents_np(segments)
    print(f"[Indexer] Embedding completed in {time.time()-t0:.1f}s")

    # 3. Neighborhood Construction
    neighborhoods, global_sim = build_candidate_neighborhoods(segment_embeddings)

    # 4. Greedy EU Construction
    t1 = time.time()
    unassigned = set(range(len(segments)))
    evidence_units = []

    for seed_idx in range(len(segments)):
        if seed_idx not in unassigned:
            continue

        eu_indices = [seed_idx]
        unassigned.remove(seed_idx)

        candidates = [idx for idx in neighborhoods[seed_idx] if idx in unassigned and idx != seed_idx]

        while len(eu_indices) < config.MAX_EU_SIZE and candidates:
            best_candidate = None
            best_de = -1

            for cand in candidates:
                test_indices = eu_indices + [cand]
                re, de = _entropy_from_sim_submatrix(global_sim, test_indices)

                if re <= config.THETA_RE and de > best_de:
                    best_de = de
                    best_candidate = cand

            if best_candidate is not None:
                eu_indices.append(best_candidate)
                unassigned.remove(best_candidate)
                candidates = [idx for idx in candidates if idx in unassigned and idx != best_candidate]
            else:
                break

        final_re, final_de = _entropy_from_sim_submatrix(global_sim, eu_indices)

        if final_re < config.TAU_LOW:
            regime = "LOW"
        elif final_re < config.TAU_HIGH:
            regime = "MEDIUM"
        else:
            regime = "HIGH"

        evidence_units.append({
            "eu_id": len(evidence_units),
            "segment_indices": eu_indices,
            "re": round(final_re, 4),
            "de": round(final_de, 4),
            "regime": regime,
            "summary": ""
        })

    print(f"[Indexer] Constructed {len(evidence_units)} Evidence Units in {time.time()-t1:.1f}s")

    # 5. Adaptive Summarization
    t2 = time.time()
    print(f"[Indexer] Summarizing {len(evidence_units)} EUs with {config.LLM_SUMMARIZATION_WORKERS} workers...")

    def _summarize(eu):
        eu_segments = [segments[idx] for idx in eu["segment_indices"]]
        return eu["eu_id"], summarize_eu_with_llm(eu_segments, eu["regime"])

    with ThreadPoolExecutor(max_workers=config.LLM_SUMMARIZATION_WORKERS) as pool:
        futures = {pool.submit(_summarize, eu): eu for eu in evidence_units}
        for future in as_completed(futures):
            eu_id, summary = future.result()
            evidence_units[eu_id]["summary"] = summary

    print(f"[Indexer] Summarization completed in {time.time()-t2:.1f}s")

    # 6. FAISS Indexing (centroid embedding of each EU's segments)
    eu_embeddings = np.array(
        [np.mean(segment_embeddings[eu["segment_indices"]], axis=0) for eu in evidence_units],
        dtype=np.float32
    )

    dimension = segment_embeddings.shape[1]
    roi_index_manager = FAISSIndexManager(dimension=dimension)
    roi_index_manager.build_index(eu_embeddings)
    roi_index_manager.save(config.FAISS_INDEX_PATH)

    # Save embeddings & metadata
    np.save(config.SEGMENT_EMBEDDINGS_PATH, segment_embeddings)
    np.save(config.EU_EMBEDDINGS_PATH, eu_embeddings)

    index_data = {
        "segments": segments,
        "evidence_units": evidence_units,
    }

    with open(config.INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index_data, f, ensure_ascii=False)

    print(f"[Indexer] Index successfully saved to {config.DATA_DIR}. Total build time: {time.time()-t0:.1f}s")
    return {**index_data, "eu_embeddings": eu_embeddings, "segment_embeddings": segment_embeddings}

def _load_embeddings(npy_path: str, json_key: str, index_data: dict) -> np.ndarray:
    if os.path.exists(npy_path):
        return np.load(npy_path)
    arr = index_data.get(json_key, [])
    return np.array(arr, dtype=np.float32)

def load_roi_rag_index() -> tuple[dict, FAISSIndexManager]:
    """
    Loads saved index metadata and initializes FAISS manager for Evidence Units.
    """
    if not os.path.exists(config.INDEX_PATH):
        raise FileNotFoundError("ROI-RAG index metadata file not found. Please build the index first.")

    with open(config.INDEX_PATH, "r", encoding="utf-8") as f:
        index_data = json.load(f)

    eu_embeddings = _load_embeddings(config.EU_EMBEDDINGS_PATH, "eu_embeddings", index_data)
    dimension = eu_embeddings.shape[1] if len(eu_embeddings) > 0 else 384

    index_manager = FAISSIndexManager(dimension=dimension)
    index_manager.load(config.FAISS_INDEX_PATH)
    index_manager.eu_embeddings = eu_embeddings

    return index_data, index_manager
