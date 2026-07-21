"""
통합 LLM 클라이언트 모듈.
config.LLM_BACKEND 설정에 따라 Gemini 또는 Ollama API로 답변 및 적응형 요약을 요청합니다.
"""
import config

def generate(prompt: str) -> str:
    """
    Call the configured LLM backend and return the response text.
    """
    if config.LLM_BACKEND.lower() == "ollama":
        return _ollama_generate(prompt)
    return _gemini_generate(prompt)

def _ollama_generate(prompt: str) -> str:
    import ollama
    try:
        resp = ollama.chat(
            model=config.OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0},
            keep_alive=-1,
        )
        return resp["message"]["content"].strip()
    except Exception as e:
        print(f"[LLM Client] Ollama Error: {e}")
        raise

def warmup_ollama_model(model_name: str = None):
    """
    Pre-loads the configured Ollama model into memory and sets keep_alive=-1 (permanent residency).
    Runs asynchronously in a background thread so it doesn't block startup.
    """
    if config.LLM_BACKEND.lower() != "ollama":
        return
    model = model_name or config.OLLAMA_MODEL
    import threading

    def _warmup_worker():
        import ollama
        print(f"\n[LLM Client] Warming up and permanently locking Ollama model '{model}' into RAM/VRAM (keep_alive=-1)...")
        try:
            # 1차 시도: generate endpoint에 빈 프롬프트 전달하여 모델 로드 및 상주
            ollama.generate(
                model=model,
                prompt="",
                keep_alive=-1,
            )
            print(f"[LLM Client] Ollama model '{model}' successfully locked in memory permanently.")
        except Exception as e:
            try:
                # 2차 시도: chat endpoint에 더미 메시지로 로드 시도
                ollama.chat(
                    model=model,
                    messages=[{"role": "user", "content": "hi"}],
                    keep_alive=-1,
                )
                print(f"[LLM Client] Ollama model '{model}' successfully locked via chat endpoint.")
            except Exception as e2:
                print(f"[LLM Client] Warning during model pre-warm '{model}': {e2}")

    threading.Thread(target=_warmup_worker, daemon=True).start()

def _gemini_generate(prompt: str) -> str:
    from google import genai
    if not config.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set. Please set it in config or .env file.")
        
    client = genai.Client(api_key=config.GEMINI_API_KEY)
    resp = client.models.generate_content(
        model=config.LLM_MODEL,
        contents=prompt
    )
    return resp.text.strip()
