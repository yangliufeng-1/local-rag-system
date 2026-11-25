"""RAG 主流程：分片 -> 索引 -> 召回 -> 重排 -> 生成。"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Dict, List

import chromadb

from .config import Settings, settings as default_settings
from .llm import LLMClient


class RAGSystem:
    """对视频中 Notebook 代码的工程化封装。"""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or default_settings
        self.settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.settings.chroma_dir))
        self._collection = self._client.get_or_create_collection(
            name=self.settings.collection_name
        )
        self._embed_model = None
        self._rerank_model = None
        self.llm = LLMClient(self.settings)

    # ---------- 1. 分片 ----------
    def split_into_chunks(self, doc_path: Path | str | None = None) -> List[str]:
        path = Path(doc_path or self.settings.doc_path)
        with open(path, "r", encoding="utf-8") as file:
            content = file.read()
        chunks = [c.strip() for c in content.split(self.settings.chunk_separator)]
        return [c for c in chunks if c]

    # ---------- 2. Embedding + 索引 ----------
    @property
    def embed_model(self):
        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer

            self._embed_model = SentenceTransformer(self.settings.embedding_model)
        return self._embed_model

    def embed(self, texts: List[str]) -> List[List[float]]:
        vectors = self.embed_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vectors]

    def build_index(
        self, doc_path: Path | str | None = None, rebuild: bool = False
    ) -> Dict[str, object]:
        if rebuild:
            self._client.delete_collection(self.settings.collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self.settings.collection_name
            )

        chunks = self.split_into_chunks(doc_path)
        if not chunks:
            return {"chunks": 0, "indexed": 0, "action": "skipped"}

        before = self._collection.count()
        ids = [self._chunk_id(c) for c in chunks]
        batch_size = 32
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            self._collection.upsert(
                documents=batch,
                embeddings=self.embed(batch),
                ids=ids[start : start + batch_size],
            )
        return {
            "chunks": len(chunks),
            "indexed": self._collection.count() - before,
            "total": self._collection.count(),
            "action": "rebuilt" if rebuild else "upserted",
        }

    @staticmethod
    def _chunk_id(chunk: str) -> str:
        return hashlib.md5(chunk.encode("utf-8")).hexdigest()

    # ---------- 3. 召回 ----------
    def retrieve(self, query: str, top_k: int | None = None) -> List[Dict[str, object]]:
        if self._collection.count() == 0:
            self.build_index()
        k = min(top_k or self.settings.top_k_retrieve, self._collection.count())
        if k <= 0:
            return []
        results = self._collection.query(
            query_embeddings=self.embed([query]), n_results=k
        )
        documents = (results.get("documents") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]
        ids = (results.get("ids") or [[]])[0]
        return [
            {"id": i, "text": d, "distance": float(dist)}
            for i, d, dist in zip(ids, documents, distances)
        ]

    # ---------- 4. 重排 ----------
    @property
    def rerank_model(self):
        if self._rerank_model is None:
            from sentence_transformers import CrossEncoder

            self._rerank_model = CrossEncoder(self.settings.reranker_model)
        return self._rerank_model

    def rerank(
        self, query: str, candidates: List[Dict[str, object]], top_k: int | None = None
    ) -> List[Dict[str, object]]:
        if not candidates:
            return []
        # 轻量模式：不下载 CrossEncoder，直接用向量召回距离排序。
        if self.settings.reranker_model.strip().lower() in {"none", "off", ""}:
            scored = [
                {**c, "score": -float(c.get("distance") or 0)}
                for c in candidates
            ]
            scored.sort(key=lambda x: x["score"], reverse=True)
            return scored[: top_k or self.settings.top_k_rerank]

        scores = self.rerank_model.predict(
            [(query, str(c["text"])) for c in candidates]
        )
        scored = [
            {**c, "score": float(score)}
            for c, score in zip(candidates, scores)
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[: top_k or self.settings.top_k_rerank]

    # ---------- 5. 生成 ----------
    def ask(
        self,
        question: str,
        top_k_retrieve: int | None = None,
        top_k_rerank: int | None = None,
    ) -> Dict[str, object]:
        timings: Dict[str, float] = {}
        t0 = time.perf_counter()
        candidates = self.retrieve(question, top_k_retrieve)
        t1 = time.perf_counter()
        reranked = self.rerank(question, candidates, top_k_rerank)
        t2 = time.perf_counter()
        answer = self.llm.generate(question, [str(c["text"]) for c in reranked])
        t3 = time.perf_counter()

        timings["retrieve_ms"] = round((t1 - t0) * 1000, 1)
        timings["rerank_ms"] = round((t2 - t1) * 1000, 1)
        timings["generate_ms"] = round((t3 - t2) * 1000, 1)
        timings["total_ms"] = round((t3 - t0) * 1000, 1)

        return {
            "question": question,
            "answer": answer,
            "sources": [
                {"text": c["text"], "score": c["score"], "distance": c.get("distance")}
                for c in reranked
            ],
            "backend": self.llm.backend,
            "timings_ms": timings,
        }
