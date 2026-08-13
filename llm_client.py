"""LLM adapters for local generation and Gemini-based RAGAS evaluation."""

import threading

import config


_OLLAMA_GENERATION_LOCK = threading.Lock()


def generate(prompt: str) -> str:
    """Generate RAG answers and index summaries with the local Ollama model."""
    if config.LLM_BACKEND.lower() != "ollama":
        raise ValueError(
            f"Unsupported internal LLM backend: {config.LLM_BACKEND}. "
            "Set LLM_BACKEND=ollama."
        )
    return _ollama_generate(prompt)


def generate_json(prompt: str, stop=None) -> str:
    """Generate JSON with the local model for non-RAGAS callers."""
    if config.LLM_BACKEND.lower() != "ollama":
        raise ValueError(
            f"Unsupported evaluation LLM backend: {config.LLM_BACKEND}. "
            "Set LLM_BACKEND=ollama."
        )
    return _ollama_generate(prompt, json_mode=True, stop=stop)


def _ollama_client():
    import ollama

    return ollama.Client(host=config.OLLAMA_BASE_URL)


def _ollama_generate(prompt: str, json_mode: bool = False, stop=None) -> str:
    try:
        options = {
            "temperature": config.OLLAMA_TEMPERATURE,
            "seed": config.OLLAMA_SEED,
            "top_k": config.OLLAMA_TOP_K,
            "top_p": config.OLLAMA_TOP_P,
            "num_ctx": config.OLLAMA_NUM_CTX,
            "num_predict": config.OLLAMA_NUM_PREDICT,
            "num_batch": config.OLLAMA_NUM_BATCH,
        }
        if stop:
            options["stop"] = [stop] if isinstance(stop, str) else list(stop)

        request = {
            "model": config.OLLAMA_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "options": options,
            "keep_alive": config.OLLAMA_KEEP_ALIVE,
        }
        if json_mode:
            request["format"] = "json"

        with _OLLAMA_GENERATION_LOCK:
            client = _ollama_client()
            if config.OLLAMA_FRESH_RUNNER:
                client.generate(
                    model=config.OLLAMA_MODEL,
                    prompt="",
                    keep_alive=0,
                )
            response = client.chat(**request)
        return response["message"]["content"].strip()
    except Exception as exc:
        print(f"[LLM Client] Ollama Error: {exc}")
        raise


def warmup_ollama_model(model_name: str | None = None):
    """Load the configured Ollama model in a background thread."""
    if config.LLM_BACKEND.lower() != "ollama":
        return
    if config.OLLAMA_FRESH_RUNNER or config.OLLAMA_KEEP_ALIVE == 0:
        return


    model = model_name or config.OLLAMA_MODEL

    def _warmup_worker():
        print(f"\n[LLM Client] Warming up Ollama model '{model}'...")
        try:
            _ollama_client().generate(model=model, prompt="", keep_alive=-1)
            print(f"[LLM Client] Ollama model '{model}' is ready.")
        except Exception as exc:
            print(f"[LLM Client] Ollama warmup warning for '{model}': {exc}")

    threading.Thread(target=_warmup_worker, daemon=True).start()


_RAGAS_LLM = None
_RAGAS_EMBEDDINGS = None


def get_ragas_embeddings():
    """
    Return a RAGAS-compatible embeddings wrapper backed by the same local
    SentenceTransformer used for indexing, so answer-relevancy scoring costs
    no API calls and stays consistent with retrieval.
    """
    global _RAGAS_EMBEDDINGS
    if _RAGAS_EMBEDDINGS is None:
        from ragas.embeddings import BaseRagasEmbeddings

        from embeddings import get_embedding_model

        class _LocalRagasEmbeddings(BaseRagasEmbeddings):
            def __init__(self):
                self._model = get_embedding_model()

            def embed_query(self, text: str) -> list[float]:
                return self._model.embed_query(text)

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return self._model.embed_documents(texts)

            async def aembed_query(self, text: str) -> list[float]:
                return self.embed_query(text)

            async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
                return self.embed_documents(texts)

        _RAGAS_EMBEDDINGS = _LocalRagasEmbeddings()
    return _RAGAS_EMBEDDINGS


def get_ragas_llm():
    """Return a RAGAS-compatible LLM wrapper backed by the configured Gemini model."""
    global _RAGAS_LLM
    if _RAGAS_LLM is None:
        if not config.GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to your .env file."
            )
        from langchain_google_genai import ChatGoogleGenerativeAI
        from ragas.llms import LangchainLLMWrapper

        chat_model = ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            google_api_key=config.GEMINI_API_KEY,
            temperature=config.GEMINI_TEMPERATURE,
        )
        _RAGAS_LLM = LangchainLLMWrapper(chat_model)
    return _RAGAS_LLM