"""集中配置：环境变量、路径与模型参数。"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# 国内环境下优先使用 HF 镜像；已有 HF_ENDPOINT 时不覆盖。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from dotenv import load_dotenv  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Settings:
    """RAG 系统配置，全部可用环境变量覆盖。"""

    doc_path: Path = PROJECT_ROOT / "data" / "doc.md"
    chroma_dir: Path = PROJECT_ROOT / "chroma_db"
    collection_name: str = "rag_docs"

    # 与视频一致：text2vec 做向量化，mmARCO CrossEncoder 做重排。
    embedding_model: str = "shibing624/text2vec-base-chinese"
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    chunk_separator: str = "\n\n"

    top_k_retrieve: int = 5
    top_k_rerank: int = 3

    # auto 依次尝试 gemini -> openai-compatible -> ollama -> extractive
    llm_backend: str = "auto"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:3b"

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(PROJECT_ROOT / ".env", override=False)
        return cls(
            doc_path=Path(os.getenv("RAG_DOC_PATH", str(PROJECT_ROOT / "data" / "doc.md"))),
            chroma_dir=Path(os.getenv("RAG_CHROMA_DIR", str(PROJECT_ROOT / "chroma_db"))),
            collection_name=os.getenv("RAG_COLLECTION", "rag_docs"),
            embedding_model=os.getenv("EMBEDDING_MODEL", cls.embedding_model),
            reranker_model=os.getenv("RERANKER_MODEL", cls.reranker_model),
            top_k_retrieve=int(os.getenv("TOP_K_RETRIEVE", "5")),
            top_k_rerank=int(os.getenv("TOP_K_RERANK", "3")),
            llm_backend=os.getenv("LLM_BACKEND", "auto").lower(),
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            gemini_model=os.getenv("GEMINI_MODEL", cls.gemini_model),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_base_url=os.getenv("OPENAI_BASE_URL", cls.openai_base_url),
            openai_model=os.getenv("OPENAI_MODEL", cls.openai_model),
            ollama_host=os.getenv("OLLAMA_HOST", cls.ollama_host),
            ollama_model=os.getenv("OLLAMA_MODEL", cls.ollama_model),
        )


settings = Settings.from_env()
