"""无需下载模型的轻量测试。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_system.config import Settings  # noqa: E402
from rag_system.llm import LLMClient, build_prompt  # noqa: E402
from rag_system.pipeline import RAGSystem  # noqa: E402


def test_split_chunks_by_blank_line(tmp_path: Path):
    doc = tmp_path / "doc.md"
    doc.write_text("第一段\n\n第二段\n\n第三段", encoding="utf-8")
    settings = Settings(doc_path=doc, chroma_dir=tmp_path / "chroma")
    assert RAGSystem(settings).split_into_chunks() == ["第一段", "第二段", "第三段"]


def test_prompt_keeps_video_format():
    prompt = build_prompt("问题", ["片段A", "片段B"])
    assert "用户问题: 问题" in prompt
    assert "片段A\n\n片段B" in prompt
    assert "不要编造信息" in prompt


def test_extractive_fallback(tmp_path: Path):
    settings = Settings(llm_backend="extractive", chroma_dir=tmp_path / "chroma")
    answer = LLMClient(settings).generate("问题", ["证据一", "证据二"])
    assert "离线证据模式" in answer
    assert "证据一" in answer and "证据二" in answer
