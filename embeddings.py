"""
SentenceTransformer 기반 로컬 임베딩 모듈.
NumPy 배열 형태 및 리스트 형태의 임베딩 연산을 모두 지원합니다.
"""
import numpy as np
from sentence_transformers import SentenceTransformer
import config

class LocalEmbeddingModel:
    """
    SentenceTransformers Wrapper for local fast embeddings.
    """
    def __init__(self, model_name: str = config.EMBEDDING_MODEL_NAME):
        print(f"[Embeddings] Loading local embedding model: {model_name}...")
        self._st_model = SentenceTransformer(model_name)
        self.model = model_name
        print("[Embeddings] Embedding model loaded successfully.")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self._st_model.encode(
            texts, show_progress_bar=True, batch_size=config.EMBEDDING_BATCH_SIZE
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        embedding = self._st_model.encode(text, show_progress_bar=False)
        return embedding.tolist()

    # NumPy interfaces optimized for FAISS and pairwise similarity calculations
    def embed_documents_np(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        embeddings = self._st_model.encode(
            texts, show_progress_bar=True, batch_size=config.EMBEDDING_BATCH_SIZE
        )
        return np.array(embeddings, dtype=np.float32)

    def embed_query_np(self, text: str) -> np.ndarray:
        embedding = self._st_model.encode(text, show_progress_bar=False)
        return np.array(embedding, dtype=np.float32)

# Singleton instance for global reuse
_embedding_model_instance = None

def get_embedding_model() -> LocalEmbeddingModel:
    global _embedding_model_instance
    if _embedding_model_instance is None:
        _embedding_model_instance = LocalEmbeddingModel()
    return _embedding_model_instance
