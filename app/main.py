"""
ROI-RAG 전용 FastAPI 웹 및 REST API 애플리케이션.
"""
import asyncio
import os
import json
import math
import time
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import sys

# Ensure parent directory (project workspace root) is in sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import config
from indexer import build_roi_rag_index, load_roi_rag_index
from roi_rag import clear_response_cache, get_roi_rag_pipeline
from evaluate_ragas import RagasStructuredOutputError, score_faithfulness

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
            "ragas_backend": "codex",
            "response_cache_default": config.LLM_RESPONSE_CACHE_DEFAULT
        }
    )

@app.get("/api/current-index")
def get_current_index():
    try:
        index_data, _ = load_roi_rag_index()
        return {
            "status": "success",
            "segments_count": len(index_data.get("segments", [])),
            "eus_count": len(index_data.get("evidence_units", [])),
            "chunking_strategy": index_data.get("chunking_strategy", "roi_rag"),
            "evidence_units": index_data.get("evidence_units", []),
            "segments": index_data.get("segments", [])
        }
    except FileNotFoundError:
        return {
            "status": "empty",
            "segments_count": 0,
            "eus_count": 0,
            "chunking_strategy": "roi_rag",
            "evidence_units": [],
            "segments": []
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
    chunking_strategy: str = "roi_rag",
    stb_leaf_size: int = 80,
    stb_leaf_overlap: int = 20,
    stb_leaves_per_parent: int = 3,
    automerge_threshold: float = 0.0,
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
        index_data = build_roi_rag_index(text)
        clear_response_cache()
        global roi_pipeline
        roi_pipeline = get_roi_rag_pipeline()

        return {
            "status": "success",
            "filename": file.filename,
            "segments_count": len(index_data.get("segments", [])),
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
        
    try:
        res = roi_pipeline(request.query, use_cache=request.use_cache)
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
                "cache_hit": res["cache_hit"]
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


def _evaluation_response(eval_result):
    f_score = float(eval_result.value)
    if not math.isfinite(f_score):
        raise ValueError(
            "RAGAS evaluation did not produce a finite faithfulness score."
        )
    return {
        "status": "success",
        "scores": {"faithfulness": f_score},
        "details": {
            "supported_claims": eval_result.supported_claims,
            "total_claims": eval_result.total_claims,
            "claims": [claim.model_dump() for claim in eval_result.claims],
            "contexts_evaluated": eval_result.contexts_evaluated,
            "judge_model": f"codex exec ({config.CODEX_MODEL or 'CLI default'})",
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
        elapsed = time.perf_counter() - started_at
        print(f"[RAGAS] Faithfulness finished (elapsed={elapsed:.1f}s)")
        return _evaluation_response(eval_result)
    except RagasStructuredOutputError as exc:
        raise HTTPException(
            status_code=422,
            detail=(
                "Codex RAGAS judge returned invalid structured output. "
                f"Reason: {exc}"
            ),
        )
    except TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=(
                "Codex RAGAS evaluation timed out after "
                f"{config.CODEX_TIMEOUT_SECONDS} seconds."
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
                response = _evaluation_response(eval_result)
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
                        "Codex RAGAS judge returned invalid structured output. "
                        f"Reason: {exc}"
                    ),
                }
            except TimeoutError:
                return {
                    "type": "error",
                    "status_code": 504,
                    "detail": (
                        "Codex RAGAS evaluation timed out after "
                        f"{config.CODEX_TIMEOUT_SECONDS} seconds."
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
