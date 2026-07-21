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
        )
        return resp["message"]["content"].strip()
    except Exception as e:
        print(f"[LLM Client] Ollama Error: {e}")
        raise

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
