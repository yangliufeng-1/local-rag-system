"""大模型生成层：兼容视频使用的 Gemini，也支持本地/兼容接口与离线兜底。"""
from __future__ import annotations

from typing import List

import httpx

from .config import Settings

PROMPT_TEMPLATE = """你是一位知识助手，请根据用户的问题和下列片段生成准确的回答。

用户问题: {query}

相关片段:
{chunks}

请基于上述内容作答，不要编造信息。"""


def build_prompt(query: str, chunks: List[str]) -> str:
    return PROMPT_TEMPLATE.format(query=query, chunks="\n\n".join(chunks))


class LLMClient:
    """按配置选择生成后端，自动模式会逐级降级。"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._resolved: str | None = None
        self._client = None

    def _ollama_alive(self) -> bool:
        try:
            resp = httpx.get(f"{self.settings.ollama_host}/api/tags", timeout=1.5)
            return resp.status_code == 200
        except Exception:
            return False

    def resolve_backend(self) -> str:
        if self._resolved is not None:
            return self._resolved
        requested = self.settings.llm_backend
        candidates = []
        if requested == "auto":
            candidates = ["gemini", "openai", "ollama", "extractive"]
        else:
            candidates = [requested, "extractive"]

        for backend in candidates:
            if backend == "gemini" and self.settings.gemini_api_key:
                self._resolved = "gemini"
            elif backend == "openai" and self.settings.openai_api_key:
                self._resolved = "openai"
            elif backend == "ollama" and self._ollama_alive():
                self._resolved = "ollama"
            elif backend == "extractive":
                self._resolved = "extractive"
            if self._resolved:
                break
        return self._resolved or "extractive"

    @property
    def backend(self) -> str:
        return self.resolve_backend()

    def generate(self, query: str, chunks: List[str]) -> str:
        backend = self.backend
        prompt = build_prompt(query, chunks)
        if backend == "gemini":
            return self._generate_gemini(prompt)
        if backend == "openai":
            return self._generate_openai(prompt)
        if backend == "ollama":
            return self._generate_ollama(prompt)
        return self._extractive(chunks)

    def _generate_gemini(self, prompt: str) -> str:
        from google import genai

        if self._client is None:
            self._client = genai.Client(api_key=self.settings.gemini_api_key)
        response = self._client.models.generate_content(
            model=self.settings.gemini_model, contents=prompt
        )
        return response.text or ""

    def _generate_openai(self, prompt: str) -> str:
        base = self.settings.openai_base_url.rstrip("/")
        resp = httpx.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {self.settings.openai_api_key}"},
            json={
                "model": self.settings.openai_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _generate_ollama(self, prompt: str) -> str:
        resp = httpx.post(
            f"{self.settings.ollama_host}/api/chat",
            json={
                "model": self.settings.ollama_model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    @staticmethod
    def _extractive(chunks: List[str]) -> str:
        """无可用大模型时的离线兜底：只做证据摘录，不编造。"""
        if not chunks:
            return "知识库中没有检索到相关内容。"
        parts = ["【离线证据模式】当前未配置大模型，以下为检索到的原文片段：\n"]
        for i, chunk in enumerate(chunks, 1):
            parts.append(f"{i}. {chunk.strip()}")
        return "\n\n".join(parts)
