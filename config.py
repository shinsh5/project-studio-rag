"""Runtime configuration for the standalone ROI-RAG service."""

import os

from dotenv import load_dotenv


load_dotenv()

# Internal generation and indexing summaries always use local Ollama.
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2:7b")
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.0"))
OLLAMA_SEED = int(os.getenv("OLLAMA_SEED", "42"))
OLLAMA_TOP_K = int(os.getenv("OLLAMA_TOP_K", "1"))
OLLAMA_TOP_P = float(os.getenv("OLLAMA_TOP_P", "1.0"))
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "160"))
OLLAMA_NUM_BATCH = int(os.getenv("OLLAMA_NUM_BATCH", "32"))
OLLAMA_KEEP_ALIVE = int(os.getenv("OLLAMA_KEEP_ALIVE", "-1"))
OLLAMA_FRESH_RUNNER = os.getenv("OLLAMA_FRESH_RUNNER", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
LLM_RESPONSE_CACHE_DEFAULT = os.getenv(
    "LLM_RESPONSE_CACHE_DEFAULT",
    "true",
).lower() in {"1", "true", "yes", "on"}
LLM_RESPONSE_CACHE_MAX_SIZE = int(os.getenv("LLM_RESPONSE_CACHE_MAX_SIZE", "128"))
# Only RAGAS faithfulness evaluation uses the Gemini API. Generation remains on Ollama.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.0"))
GEMINI_TIMEOUT_SECONDS = int(os.getenv("GEMINI_TIMEOUT_SECONDS", "300"))

# Embedding Configuration
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

# Chunking Strategy: "roi_rag" | "small_to_big"
CHUNKING_STRATEGY = os.getenv("CHUNKING_STRATEGY", "roi_rag")

# Segmenting Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 200))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))

# Small-to-Big Configuration
STB_LEAF_SIZE = int(os.getenv("STB_LEAF_SIZE", 80))
STB_LEAF_OVERLAP = int(os.getenv("STB_LEAF_OVERLAP", 20))
STB_LEAVES_PER_PARENT = int(os.getenv("STB_LEAVES_PER_PARENT", 3))
AUTOMERGE_THRESHOLD = float(os.getenv("AUTOMERGE_THRESHOLD", 0.0))

# Candidate Neighborhood Configuration
NEIGHBORHOOD_K = int(os.getenv("NEIGHBORHOOD_K", 10))

# Entropy-Guided Evidence Unit Configuration
MAX_EU_SIZE = int(os.getenv("MAX_EU_SIZE", 6))
THETA_RE = float(os.getenv("THETA_RE", 0.95))

# Adaptive Summarization Configuration
TAU_LOW = float(os.getenv("TAU_LOW", 0.60))
TAU_HIGH = float(os.getenv("TAU_HIGH", 0.85))

# Retrieval Configuration
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", 3))

# Performance Configuration
MAX_SEGMENTS = None
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", 128))
LLM_SUMMARIZATION_WORKERS = int(os.getenv("LLM_SUMMARIZATION_WORKERS", 1))

# Directory and Storage Paths
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(WORKSPACE_DIR, "data")
INDEX_PATH = os.path.join(DATA_DIR, "roi_rag_index.json")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
SEGMENT_EMBEDDINGS_PATH = os.path.join(DATA_DIR, "segment_embeddings.npy")
EU_EMBEDDINGS_PATH = os.path.join(DATA_DIR, "eu_embeddings.npy")

os.makedirs(DATA_DIR, exist_ok=True)
