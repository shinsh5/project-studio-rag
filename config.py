"""
ROI-RAG 단독 추론 환경을 위한 환경변수 및 시스템 설정 모듈.
"""
import os
from dotenv import load_dotenv

# Load environment variables (.env file)
load_dotenv()

# LLM Backend: "gemini" or "ollama"
LLM_BACKEND = os.getenv("LLM_BACKEND", "gemini")

# Ollama Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")

# Gemini Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
LLM_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

# Embedding Configuration
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

# Segmenting (Chunking) Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 200))      # Target tokens/words per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))

# Candidate Neighborhood Configuration
NEIGHBORHOOD_K = int(os.getenv("NEIGHBORHOOD_K", 10))  # Number of semantic neighbors

# Entropy-Guided Evidence Unit (EU) Construction Configuration
MAX_EU_SIZE = int(os.getenv("MAX_EU_SIZE", 6))
THETA_RE = float(os.getenv("THETA_RE", 0.95))          # Max acceptable RE threshold when growing an EU

# Adaptive Summarization Configuration
# RE ranges: Low < TAU_LOW <= Medium < TAU_HIGH <= High
TAU_LOW = float(os.getenv("TAU_LOW", 0.60))            # Below this RE -> Low redundancy -> Return raw text
TAU_HIGH = float(os.getenv("TAU_HIGH", 0.85))          # Above this RE -> High redundancy -> Aggressive dense summary

# Retrieval Configuration
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", 3))         # Retrieve top-K EUs during inference

# Performance Configuration
MAX_SEGMENTS = None                                    # Cap segment count if needed (None = no limit)
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", 128))
LLM_SUMMARIZATION_WORKERS = int(os.getenv("LLM_SUMMARIZATION_WORKERS", 4))  # Parallel threads for summarization

# Directory & Storage Paths
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(WORKSPACE_DIR, "data")
INDEX_PATH = os.path.join(DATA_DIR, "roi_rag_index.json")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
SEGMENT_EMBEDDINGS_PATH = os.path.join(DATA_DIR, "segment_embeddings.npy")
EU_EMBEDDINGS_PATH = os.path.join(DATA_DIR, "eu_embeddings.npy")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)
