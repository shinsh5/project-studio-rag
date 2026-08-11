"""Evaluate ROI-RAG outputs with RAGAS and Codex CLI as the judge LLM."""

import asyncio
import json
import os

from datasets import Dataset
from langchain_core.outputs import Generation, LLMResult
from ragas.llms.base import BaseRagasLLM

import config
import llm_client
from embeddings import get_embedding_model
from roi_rag import get_roi_rag_pipeline


class RagasCodexCLI(BaseRagasLLM):
    """RAGAS LLM adapter backed by authenticated `codex exec` calls."""

    def generate_text(
        self,
        prompt,
        n: int = 1,
        temperature: float = 0.0,
        stop=None,
        callbacks=None,
    ):
        prompt_text = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)
        text = llm_client.codex_generate(prompt_text)
        return LLMResult(generations=[[Generation(text=text)]])

    async def agenerate_text(
        self,
        prompt,
        n: int = 1,
        temperature: float = 0.0,
        stop=None,
        callbacks=None,
    ):
        return await asyncio.to_thread(
            self.generate_text, prompt, n, temperature, stop, callbacks
        )

    def is_finished(self, response: LLMResult) -> bool:
        return True


def to_ragas_dataset(results: list[dict]) -> Dataset:
    return Dataset.from_dict(
        {
            "question": [item["question"] for item in results],
            "answer": [item["answer"] for item in results],
            "contexts": [item["contexts"] for item in results],
            "ground_truth": [item["ground_truth"] for item in results],
        }
    )


def main():
    dataset_path = "eval_dataset.json"
    if os.path.exists(dataset_path):
        with open(dataset_path, "r", encoding="utf-8") as stream:
            test_samples = json.load(stream)
    else:
        test_samples = [
            {
                "question": "What is the primary function of ROI-RAG?",
                "ground_truth": (
                    "ROI-RAG evaluates redundancy and diversity to construct "
                    "evidence units for grounded generation."
                ),
            }
        ]

    pipeline = get_roi_rag_pipeline()
    results = []
    for sample in test_samples:
        rag_output = pipeline(sample["question"])
        results.append(
            {
                "question": sample["question"],
                "answer": rag_output["answer"],
                "contexts": rag_output["retrieved_contexts"],
                "ground_truth": sample["ground_truth"],
            }
        )

    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import answer_relevancy, faithfulness
    from ragas.run_config import RunConfig

    embeddings = get_embedding_model()
    try:
        ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)
    except Exception:
        ragas_embeddings = embeddings

    result = evaluate(
        dataset=to_ragas_dataset(results),
        metrics=[faithfulness, answer_relevancy],
        llm=RagasCodexCLI(),
        embeddings=ragas_embeddings,
        run_config=RunConfig(max_workers=1, timeout=600, max_retries=2),
    )
    try:
        print(result.to_pandas().to_string())
    except Exception:
        print(result)


if __name__ == "__main__":
    main()
