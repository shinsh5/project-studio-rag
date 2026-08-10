"""
RAGAS 평가 자동화 스크립트
Google Gemini API의 엄격한 호출 제한(15 RPM)을 회피하기 위한 커스텀 LLM 래퍼 포함.
"""
import sys
import os
import time
import asyncio
import threading
import requests
from datasets import Dataset

# Project modules
import config
from roi_rag import get_roi_rag_pipeline
from embeddings import get_embedding_model

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms.base import BaseRagasLLM
from langchain_core.outputs import LLMResult, Generation

api_lock = threading.Lock()

class RagasGoogleREST(BaseRagasLLM):
    """
    구글 Gemini REST API를 직접 호출하며, 초당 호출 횟수 제한(Rate Limit)을 우회하기 위해
    4.1초의 강제 지연시간(Sleep)을 갖는 커스텀 Ragas LLM 래퍼입니다.
    """
    def __init__(self, model_name, api_key):
        self.model_name = model_name
        self.api_key = api_key

    def generate_text(self, prompt, n: int = 1, temperature: float = 0.0, stop=None, callbacks=None):
        prompt_text = prompt.to_string()
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt_text}]}],
            "generationConfig": {"temperature": temperature}
        }
        retries = 10
        delay = 2
        for i in range(retries):
            with api_lock:
                # 15 RPM(분당 15회) 제한을 위해 호출 전 무조건 4.1초 대기
                time.sleep(4.1) 
                print(f"[RAGAS LLM-Judge] API 요청 중... (attempt {i+1})")
                try:
                    response = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=30)
                except Exception as e:
                    print(f"[RAGAS LLM-Judge] Network Error: {e}. Retrying...")
                    time.sleep(delay)
                    delay *= 2
                    continue
                
            if response.status_code == 200:
                try:
                    text = response.json()['candidates'][0]['content']['parts'][0]['text']
                    return LLMResult(generations=[[Generation(text=text)]])
                except Exception:
                    return LLMResult(generations=[[Generation(text="")]])
            elif response.status_code == 429:
                print(f"[RAGAS LLM-Judge] 429 Rate Limit 감지. {delay}초 대기 후 재시도...")
                time.sleep(delay)
                delay *= 2
            else:
                raise Exception(f"API Error {response.status_code}: {response.text}")
        
        raise Exception("Max retries exceeded for RagasGoogleREST")

    async def agenerate_text(self, prompt, n: int = 1, temperature: float = 0.0, stop=None, callbacks=None):
        # 비동기 호출을 동기 래퍼로 우회
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.generate_text(prompt, n, temperature, stop, callbacks))

    def is_finished(self, response: LLMResult) -> bool:
        return True

def to_ragas_dataset(results: list[dict]) -> Dataset:
    data = {
        "question": [r["question"] for r in results],
        "answer": [r["answer"] for r in results],
        "contexts": [r["contexts"] for r in results],
        "ground_truth": [r["ground_truth"] for r in results]
    }
    return Dataset.from_dict(data)

def main():
    api_key = config.GEMINI_API_KEY
    if not api_key:
        print("ERROR: GEMINI_API_KEY가 .env 파일에 설정되어 있지 않습니다.")
        sys.exit(1)

    print("==================================================")
    print("ROI-RAG 평가 파이프라인 시작")
    print("==================================================")

    # 1. 평가용 데이터셋 로드
    # eval_dataset.json 파일이 있으면 해당 파일에서 로드하고, 없으면 기본 샘플 1개를 사용합니다.
    dataset_path = "eval_dataset.json"
    import json
    
    if os.path.exists(dataset_path):
        print(f"[1/3] '{dataset_path}' 파일에서 평가 데이터를 불러옵니다...")
        with open(dataset_path, 'r', encoding='utf-8') as f:
            test_samples = json.load(f)
    else:
        print("[1/3] 평가 데이터 파일이 없어 기본 샘플 1개를 사용합니다.")
        test_samples = [
            {
                "question": "What is the primary function of ROI-RAG?",
                "ground_truth": "ROI-RAG focuses on evaluating redundancy and diversity to construct optimal evidence units for LLM generation."
            }
            # 여기에 직접 추가하실 수 있습니다!
            # ,
            # {
            #     "question": "두 번째 질문을 입력하세요",
            #     "ground_truth": "여기에 사람이 작성한 정답(모범 답안)을 입력하세요"
            # }
        ]

    # 2. ROI-RAG 파이프라인 로드
    print("[2/3] RAG 파이프라인 초기화 중...")
    pipeline = get_roi_rag_pipeline()
    
    # 3. RAG 추론 수행
    print(f"[3/4] 총 {len(test_samples)}개의 데이터에 대해 RAG 추론 중...")
    results = []
    for sample in test_samples:
        print(f" -> 질문: {sample['question']}")
        rag_out = pipeline(sample["question"])
        
        results.append({
            "question": sample["question"],
            "answer": rag_out["answer"],
            "contexts": rag_out["retrieved_contexts"], # 요약본+원문이 포함된 하이브리드 컨텍스트 리스트
            "ground_truth": sample["ground_truth"]
        })

    ds = to_ragas_dataset(results)

    # 4. RAGAS 설정
    print("[3/3] RAGAS 평가 시작 (Gemini API 4.1초 락 사용)...")
    eval_llm = RagasGoogleREST(model_name="gemini-3.1-flash-lite", api_key=api_key)
    eval_embeddings = get_embedding_model()
    
    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper
        ragas_embeddings = LangchainEmbeddingsWrapper(eval_embeddings)
    except Exception:
        ragas_embeddings = eval_embeddings

    # max_workers=1로 설정하여 병렬 호출 원천 차단
    from ragas.run_config import RunConfig
    rc = RunConfig(max_workers=1, timeout=600, max_retries=10)

    # 5. RAGAS 평가 수행
    try:
        metrics = [faithfulness, answer_relevancy]
        result = evaluate(
            dataset=ds, 
            metrics=metrics, 
            llm=eval_llm, 
            embeddings=ragas_embeddings, 
            run_config=rc
        )
        print("\n==================================================")
        print("Evaluation Results (Scores)")
        print("==================================================")
        print(result)
    except Exception as e:
        print("\nError during evaluation:", type(e).__name__, str(e))

if __name__ == "__main__":
    main()
