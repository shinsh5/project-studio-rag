"""
ROI-RAG 전용 FastAPI 웹 및 REST API 애플리케이션.
"""
import asyncio
import os
import json
import math
import time
from fastapi import FastAPI, Form, HTTPException, Request, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import sys

# Ensure parent directory (project workspace root) is in sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import config
from indexer import build_roi_rag_index, load_roi_rag_index, load_roi_rag_index_for
from roi_rag import clear_response_cache, get_roi_rag_pipeline
from evaluate_ragas import (
    RagasStructuredOutputError,
    score_faithfulness,
    score_response_relevancy,
)

app = FastAPI(
    title="ROI-RAG Standed Inference API",
    description="Redundancy- and Diversity-Oriented RAG standalone inference service."
)

# Setup Jinja2 templates directory
templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
templates = Jinja2Templates(directory=templates_dir)

# Initialize pipeline
roi_pipeline = get_roi_rag_pipeline()

# Request Models
class QueryRequest(BaseModel):
    query: str
    use_cache: bool = config.LLM_RESPONSE_CACHE_DEFAULT
    chunking_strategy: str | None = None
    # Small-to-Big only: "automerge" | "all_segments". Applied per query, no rebuild.
    stb_retrieval_mode: str | None = None
    automerge_threshold: float | None = None

class BuildIndexRequest(BaseModel):
    text: str
    chunking_strategy: str = "roi_rag"
    stb_leaf_size: int = 80
    stb_leaf_overlap: int = 20
    stb_leaves_per_parent: int = 3
    automerge_threshold: float = 0.0

# Endpoints
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "llm_backend": config.LLM_BACKEND,
            "ragas_backend": "gemini",
            "response_cache_default": config.LLM_RESPONSE_CACHE_DEFAULT
        }
    )

def _read_index_readme(strategy: str) -> str:
    readme_path = config.get_index_readme_path(strategy)
    if not os.path.exists(readme_path):
        return ""
    with open(readme_path, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/api/current-index")
def get_current_index(strategy: str | None = None):
    requested_strategy = strategy or config.CHUNKING_STRATEGY
    try:
        index_data, _ = load_roi_rag_index_for(requested_strategy)
        return {
            "status": "success",
            "segments_count": len(index_data.get("segments", [])),
            "parent_chunks_count": len(index_data.get("parent_chunks", [])),
            "eus_count": len(index_data.get("evidence_units", [])),
            "chunking_strategy": index_data.get("chunking_strategy", requested_strategy),
            "evidence_units": index_data.get("evidence_units", []),
            "segments": index_data.get("segments", []),
            "readme": _read_index_readme(requested_strategy),
        }
    except FileNotFoundError:
        return {
            "status": "empty",
            "segments_count": 0,
            "parent_chunks_count": 0,
            "eus_count": 0,
            "chunking_strategy": requested_strategy,
            "evidence_units": [],
            "segments": [],
            "readme": "",
        }

def _apply_chunking_config(request: BuildIndexRequest):
    config.CHUNKING_STRATEGY = request.chunking_strategy
    config.STB_LEAF_SIZE = request.stb_leaf_size
    config.STB_LEAF_OVERLAP = request.stb_leaf_overlap
    config.STB_LEAVES_PER_PARENT = request.stb_leaves_per_parent
    config.AUTOMERGE_THRESHOLD = request.automerge_threshold

@app.post("/api/build-index")
def build_index(request: BuildIndexRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text input cannot be empty.")

    try:
        _apply_chunking_config(request)
        index_data = build_roi_rag_index(request.text)
        clear_response_cache()
        global roi_pipeline
        roi_pipeline = get_roi_rag_pipeline()

        return {
            "status": "success",
            "segments_count": len(index_data.get("segments", [])),
            "parent_chunks_count": len(index_data.get("parent_chunks", [])),
            "eus_count": len(index_data.get("evidence_units", [])),
            "chunking_strategy": request.chunking_strategy,
            "evidence_units": index_data.get("evidence_units", [])
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload-file")
def upload_file(
    file: UploadFile = File(...),
    chunking_strategy: str = Form("roi_rag"),
    stb_leaf_size: int = Form(80),
    stb_leaf_overlap: int = Form(20),
    stb_leaves_per_parent: int = Form(3),
    automerge_threshold: float = Form(0.0),
):
    try:
        content = file.file.read()
        text = content.decode("utf-8")

        req = BuildIndexRequest(
            text=text,
            chunking_strategy=chunking_strategy,
            stb_leaf_size=stb_leaf_size,
            stb_leaf_overlap=stb_leaf_overlap,
            stb_leaves_per_parent=stb_leaves_per_parent,
            automerge_threshold=automerge_threshold,
        )
        _apply_chunking_config(req)
        index_data = build_roi_rag_index(text, source_label=file.filename or "")
        clear_response_cache()
        global roi_pipeline
        roi_pipeline = get_roi_rag_pipeline()

        return {
            "status": "success",
            "filename": file.filename,
            "segments_count": len(index_data.get("segments", [])),
            "parent_chunks_count": len(index_data.get("parent_chunks", [])),
            "eus_count": len(index_data.get("evidence_units", [])),
            "chunking_strategy": chunking_strategy,
            "evidence_units": index_data.get("evidence_units", [])
        }
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Only UTF-8 encoded text files are supported.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query")
def execute_query(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    if request.chunking_strategy:
        config.CHUNKING_STRATEGY = request.chunking_strategy

    if request.stb_retrieval_mode and request.stb_retrieval_mode not in config.STB_RETRIEVAL_MODES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown stb_retrieval_mode '{request.stb_retrieval_mode}'. "
                f"Expected one of {list(config.STB_RETRIEVAL_MODES)}."
            ),
        )

    # AutoMerge threshold is a retrieval-time knob, so a query may override it.
    if request.automerge_threshold is not None:
        config.AUTOMERGE_THRESHOLD = request.automerge_threshold

    try:
        res = roi_pipeline(
            request.query,
            use_cache=request.use_cache,
            stb_retrieval_mode=request.stb_retrieval_mode,
        )
        return {
            "status": "success",
            "roi_rag": {
                "answer": res["answer"],
                "retrieved_contexts": res["retrieved_contexts"],
                "raw_contexts": res["raw_contexts"],
                "latency_ms": res["latency_ms"],
                "api_calls": res["api_calls"],
                "tokens_used": res["tokens_used"],
                "prompt": res["prompt"],
                "cache_hit": res["cache_hit"],
                "stb_retrieval_mode": res.get("stb_retrieval_mode")
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

class EvaluateSingleRequest(BaseModel):
    question: str
    answer: str
    contexts: list[str]


async def _score_answer_relevancy(request) -> tuple[float | None, str | None]:
    """
    Answer relevancy needs no ground truth, so it runs for ad-hoc GUI questions.
    Context recall is deliberately absent here: it requires a reference answer,
    which only batch evaluation over MULTINEWS_QA has.
    """
    try:
        return await score_response_relevancy(
            question=request.question,
            answer=request.answer,
        ), None
    except Exception as exc:
        print(f"[RAGAS] Answer relevancy failed: {type(exc).__name__}: {exc}")
        return None, f"{type(exc).__name__}: {exc}"


def _evaluation_response(eval_result, answer_relevancy=None, relevancy_error=None):
    f_score = float(eval_result.value)
    if not math.isfinite(f_score):
        raise ValueError(
            "RAGAS evaluation did not produce a finite faithfulness score."
        )
    scores = {"faithfulness": f_score}
    if answer_relevancy is not None:
        scores["answer_relevancy"] = answer_relevancy
    return {
        "status": "success",
        "scores": scores,
        "details": {
            "answer_relevancy_error": relevancy_error,
            "supported_claims": eval_result.supported_claims,
            "total_claims": eval_result.total_claims,
            "claims": [claim.model_dump() for claim in eval_result.claims],
            "contexts_evaluated": eval_result.contexts_evaluated,
            "judge_model": f"ragas.Faithfulness + gemini ({config.GEMINI_MODEL})",
        },
    }


def _stream_line(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n"

@app.post("/api/evaluate-single")
async def evaluate_single(request: EvaluateSingleRequest):
    try:
        started_at = time.perf_counter()
        eval_result = await score_faithfulness(
            question=request.question,
            answer=request.answer,
            contexts=request.contexts,
        )
        relevancy, relevancy_error = await _score_answer_relevancy(request)
        elapsed = time.perf_counter() - started_at
        print(f"[RAGAS] Faithfulness + answer relevancy finished (elapsed={elapsed:.1f}s)")
        return _evaluation_response(eval_result, relevancy, relevancy_error)
    except RagasStructuredOutputError as exc:
        raise HTTPException(
            status_code=422,
            detail=(
                "Gemini RAGAS judge returned invalid structured output. "
                f"Reason: {exc}"
            ),
        )
    except TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=(
                "Gemini RAGAS evaluation timed out after "
                f"{config.GEMINI_TIMEOUT_SECONDS} seconds."
            ),
        )
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/evaluate-single-stream")
async def evaluate_single_stream(request: EvaluateSingleRequest):
    """Stream real evaluation stages as newline-delimited JSON events."""

    async def event_stream():
        started_at = time.perf_counter()
        progress_queue: asyncio.Queue[dict] = asyncio.Queue()

        async def report_progress(event: dict):
            await progress_queue.put({"type": "progress", **event})

        async def run_evaluation():
            try:
                eval_result = await score_faithfulness(
                    question=request.question,
                    answer=request.answer,
                    contexts=request.contexts,
                    progress_callback=report_progress,
                )
                await report_progress({
                    "stage": "answer_relevancy",
                    "progress": 97,
                    "message": "answer relevancy 계산 중",
                })
                relevancy, relevancy_error = await _score_answer_relevancy(request)
                response = _evaluation_response(eval_result, relevancy, relevancy_error)
                response.update(
                    {
                        "type": "result",
                        "elapsed_seconds": round(
                            time.perf_counter() - started_at,
                            3,
                        ),
                    }
                )
                return response
            except RagasStructuredOutputError as exc:
                return {
                    "type": "error",
                    "status_code": 422,
                    "detail": (
                        "Gemini RAGAS judge returned invalid structured output. "
                        f"Reason: {exc}"
                    ),
                }
            except TimeoutError:
                return {
                    "type": "error",
                    "status_code": 504,
                    "detail": (
                        "Gemini RAGAS evaluation timed out after "
                        f"{config.GEMINI_TIMEOUT_SECONDS} seconds."
                    ),
                }
            except Exception as exc:
                import traceback

                traceback.print_exc()
                return {
                    "type": "error",
                    "status_code": 500,
                    "detail": str(exc),
                }

        evaluation_task = asyncio.create_task(run_evaluation())
        try:
            while not evaluation_task.done() or not progress_queue.empty():
                try:
                    event = await asyncio.wait_for(
                        progress_queue.get(),
                        timeout=1.0,
                    )
                    yield _stream_line(event)
                except TimeoutError:
                    yield _stream_line(
                        {
                            "type": "heartbeat",
                            "elapsed_seconds": round(
                                time.perf_counter() - started_at,
                                1,
                            ),
                        }
                    )

            final_event = await evaluation_task
            yield _stream_line(final_event)
        finally:
            if not evaluation_task.done():
                evaluation_task.cancel()

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )
