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
            "evidence_units": index_data.get("evidence_units", []),
            "segments": index_data.get("segments", [])
        }
    except FileNotFoundError:
        return {
            "status": "empty",
            "segments_count": 0,
            "eus_count": 0,
            "evidence_units": [],
            "segments": []
        }

@app.post("/api/build-index")
def build_index(request: BuildIndexRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text input cannot be empty.")
        
    try:
        index_data = build_roi_rag_index(request.text)
        global roi_pipeline
        roi_pipeline = get_roi_rag_pipeline()
        
        return {
            "status": "success",
            "segments_count": len(index_data.get("segments", [])),
            "eus_count": len(index_data.get("evidence_units", [])),
            "evidence_units": index_data.get("evidence_units", [])
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload-file")
def upload_file(file: UploadFile = File(...)):
    try:
        content = file.file.read()
        text = content.decode("utf-8")
        
        index_data = build_roi_rag_index(text)
        global roi_pipeline
        roi_pipeline = get_roi_rag_pipeline()
        
        return {
            "status": "success",
            "filename": file.filename,
            "segments_count": len(index_data.get("segments", [])),
            "eus_count": len(index_data.get("evidence_units", [])),
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
