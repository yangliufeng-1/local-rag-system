"""FastAPI 服务：提供 /ask、/index、/health 与内置页面。"""
from __future__ import annotations

from pathlib import Path
from typing import List, Literal

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .config import PROJECT_ROOT, settings
from .pipeline import RAGSystem

app = FastAPI(title="Local RAG System", version="1.0.0")

_rag: RAGSystem | None = None


def get_rag() -> RAGSystem:
    global _rag
    if _rag is None:
        _rag = RAGSystem(settings)
    return _rag


class AskRequest(BaseModel):
    question: str
    top_k_retrieve: int | None = None
    top_k_rerank: int | None = None


class IndexRequest(BaseModel):
    doc_path: str | None = None
    rebuild: bool = False


@app.get("/")
def index_page():
    return FileResponse(Path(PROJECT_ROOT) / "web" / "index.html")


@app.get("/health")
def health():
    rag = get_rag()
    return {
        "status": "ok",
        "backend": rag.llm.backend,
        "collection": settings.collection_name,
        "documents": rag._collection.count(),
    }


@app.post("/index")
def build_index(req: IndexRequest):
    return get_rag().build_index(req.doc_path, rebuild=req.rebuild)


@app.post("/ask")
def ask(req: AskRequest):
    return get_rag().ask(req.question, req.top_k_retrieve, req.top_k_rerank)
