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
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "512"))
OLLAMA_NUM_BATCH = int(os.getenv("OLLAMA_NUM_BATCH", "32"))
OLLAMA_KEEP_ALIVE = int(os.getenv("OLLAMA_KEEP_ALIVE", "-1"))
OLLAMA_FRESH_RUNNER = os.getenv("OLLAMA_FRESH_RUNNER", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
# Off by default: while comparing retrieval strategies, a cache hit returns an
# answer generated under different settings and reports zero tokens, which
# silently corrupts token and latency measurements.
LLM_RESPONSE_CACHE_DEFAULT = os.getenv(
    "LLM_RESPONSE_CACHE_DEFAULT",
    "false",
).lower() in {"1", "true", "yes", "on"}
LLM_RESPONSE_CACHE_MAX_SIZE = int(os.getenv("LLM_RESPONSE_CACHE_MAX_SIZE", "128"))
# Only RAGAS faithfulness evaluation uses the Gemini API. Generation remains on Ollama.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.0"))
GEMINI_TIMEOUT_SECONDS = int(os.getenv("GEMINI_TIMEOUT_SECONDS", "300"))
# RAGAS answer relevancy generates this many candidate questions per answer and
# averages their similarity to the real question. RAGAS forces temperature 0.3
# whenever this is above 1 (see BaseRagasLLM.get_temperature), so any value > 1
# makes the score irreproducible regardless of GEMINI_TEMPERATURE.
#
# Kept at 1 because run-to-run noise otherwise swamps the effect being measured.
# Repeated scoring of identical answers showed the spread grows with answer
# length: short answers stayed fixed, but a 271-character answer ranged 0.779 to
# 0.814 at strictness 5 -- 2.7x the 0.0129 difference we were trying to detect
# between summarisation models. Strictness 1 measured a 0.0000 spread.
#
# The trade-off is single-sample bias: one generated question decides the score
# instead of an ensemble. That bias applies equally to both arms of a paired
# comparison, whereas run-to-run noise does not cancel.
RAGAS_ANSWER_RELEVANCY_STRICTNESS = int(
    os.getenv("RAGAS_ANSWER_RELEVANCY_STRICTNESS", "1")
)

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

# BM25 Sentence-Level Evidence Selection
# Orthogonal to CHUNKING_STRATEGY: this trims the raw snippets a strategy already
# selected, so it stacks on top of either roi_rag or small_to_big. EU summaries are
# left untouched. Raw snippets are 82% of the evidence tokens at ~9.5 sentences per
# segment, so keeping only the query-relevant ones is where compression pays.
BM25_EVIDENCE_SELECTION = os.getenv("BM25_EVIDENCE_SELECTION", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
BM25_SENTENCES_PER_SEGMENT = int(os.getenv("BM25_SENTENCES_PER_SEGMENT", 3))

# Performance Configuration
MAX_SEGMENTS = None
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", 128))
LLM_SUMMARIZATION_WORKERS = int(os.getenv("LLM_SUMMARIZATION_WORKERS", 1))

# Directory and Storage Paths
# Each chunking strategy keeps its own index directory (data/<strategy>/) so
# switching CHUNKING_STRATEGY never overwrites another strategy's index.
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT_DIR = os.path.join(WORKSPACE_DIR, "data")


def get_data_dir(strategy: str | None = None) -> str:
    """Return the index directory for a chunking strategy (default: current)."""
    resolved_strategy = strategy or CHUNKING_STRATEGY
    data_dir = os.path.join(DATA_ROOT_DIR, resolved_strategy)
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def get_index_path(strategy: str | None = None) -> str:
    return os.path.join(get_data_dir(strategy), "roi_rag_index.json")


def get_faiss_index_path(strategy: str | None = None) -> str:
    return os.path.join(get_data_dir(strategy), "faiss_index.bin")


def get_segment_embeddings_path(strategy: str | None = None) -> str:
    return os.path.join(get_data_dir(strategy), "segment_embeddings.npy")


def get_eu_embeddings_path(strategy: str | None = None) -> str:
    return os.path.join(get_data_dir(strategy), "eu_embeddings.npy")


def get_index_readme_path(strategy: str | None = None) -> str:
    return os.path.join(get_data_dir(strategy), "README.md")
