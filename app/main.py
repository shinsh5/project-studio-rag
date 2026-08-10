"""
ROI-RAG 전용 FastAPI 웹 및 REST API 애플리케이션.
"""
import os
import json
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import sys

# Ensure parent directory (project workspace root) is in sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import config
from indexer import build_roi_rag_index, load_roi_rag_index
from roi_rag import get_roi_rag_pipeline

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
            "has_gemini_key": bool(config.GEMINI_API_KEY)
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
        res = roi_pipeline(request.query)
        return {
            "status": "success",
            "roi_rag": {
                "answer": res["answer"],
                "retrieved_contexts": res["retrieved_contexts"],
                "raw_contexts": res["raw_contexts"],
                "latency_ms": res["latency_ms"],
                "api_calls": res["api_calls"],
                "tokens_used": res["tokens_used"],
                "prompt": res["prompt"]
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

@app.post("/api/evaluate-single")
def evaluate_single(request: EvaluateSingleRequest):
    try:
        from evaluate_ragas import RagasGoogleREST, to_ragas_dataset
        from embeddings import get_embedding_model
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy
        from ragas.run_config import RunConfig
        
        api_key = config.GEMINI_API_KEY
        if not api_key:
            raise HTTPException(status_code=400, detail="GEMINI_API_KEY is not configured.")
            
        eval_llm = RagasGoogleREST(model_name="gemini-3.1-flash-lite", api_key=api_key)
        eval_embeddings = get_embedding_model()
        
        try:
            from ragas.embeddings import LangchainEmbeddingsWrapper
            ragas_embeddings = LangchainEmbeddingsWrapper(eval_embeddings)
        except Exception:
            ragas_embeddings = eval_embeddings
            
        rc = RunConfig(max_workers=1, timeout=600, max_retries=10)
        
        # We need a ground_truth to pass to_ragas_dataset, even if it's empty.
        results = [{
            "question": request.question,
            "answer": request.answer,
            "contexts": request.contexts,
            "ground_truth": ""
        }]
        
        ds = to_ragas_dataset(results)
        metrics = [faithfulness, answer_relevancy]
        
        eval_result = evaluate(
            dataset=ds,
            metrics=metrics,
            llm=eval_llm,
            embeddings=ragas_embeddings,
            run_config=rc
        )
        
        return {
            "status": "success",
            "scores": {
                "faithfulness": eval_result.get("faithfulness", 0.0),
                "answer_relevancy": eval_result.get("answer_relevancy", 0.0)
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
