"""命令行入口：index / ask / chat / demo / serve。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import settings
from .pipeline import RAGSystem


def _rag() -> RAGSystem:
    return RAGSystem(settings)


def cmd_index(args: argparse.Namespace) -> None:
    rag = _rag()
    stats = rag.build_index(args.doc, rebuild=args.rebuild)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if args.preview:
        for i, chunk in enumerate(rag.split_into_chunks(args.doc)):
            print(f"[{i}] {chunk}\n")


def cmd_ask(args: argparse.Namespace) -> None:
    result = _rag().ask(args.question, args.top_k_retrieve, args.top_k_rerank)
    print("\n" + result["answer"] + "\n")
    print("--- 引用片段 ---")
    for i, src in enumerate(result["sources"], 1):
        print(f"[{i}] score={src['score']:.4f} {src['text'][:120]}...")
    print(f"\n后端: {result['backend']} | 耗时: {result['timings_ms']} ms")


def cmd_chat(args: argparse.Namespace) -> None:
    rag = _rag()
    print("RAG 交互问答（输入 exit / quit 退出）")
    while True:
        try:
            question = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"exit", "quit", "q"}:
            break
        if not question:
            continue
        result = rag.ask(question)
        print("\n助手:", result["answer"])
        print(f"\n[后端={result['backend']} 耗时={result['timings_ms']['total_ms']}ms]")


def cmd_demo(args: argparse.Namespace) -> None:
    """逐步展示视频里的每个环节，便于对照学习。"""
    rag = _rag()
    query = args.question or "哆啦A梦使用的3个秘密道具分别是什么？"

    print("=" * 20, "1. 分片", "=" * 20)
    chunks = rag.split_into_chunks(args.doc)
    for i, chunk in enumerate(chunks):
        print(f"[{i}] {chunk[:100]}...")

    print("\n" + "=" * 20, "2. Embedding + 写入 Chroma", "=" * 20)
    stats = rag.build_index(args.doc, rebuild=args.rebuild)
    print(stats)

    print("\n" + "=" * 20, "3. 召回 Top-5", "=" * 20)
    candidates = rag.retrieve(query, 5)
    for i, c in enumerate(candidates):
        print(f"[{i}] distance={c['distance']:.4f} {str(c['text'])[:100]}...")

    print("\n" + "=" * 20, "4. CrossEncoder 重排 Top-3", "=" * 20)
    reranked = rag.rerank(query, candidates, 3)
    for i, c in enumerate(reranked):
        print(f"[{i}] score={c['score']:.4f} {str(c['text'])[:100]}...")

    print("\n" + "=" * 20, "5. 大模型生成", "=" * 20)
    result = rag.ask(query)
    print(result["answer"])
    print(f"\n后端: {result['backend']} | 耗时: {result['timings_ms']} ms")


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run("rag_system.api:app", host=args.host, port=args.port, reload=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag", description="本地 RAG 系统：分片/索引/召回/重排/生成"
    )
    sub = parser.add_subparsers(dest="command")

    p_index = sub.add_parser("index", help="构建或更新向量索引")
    p_index.add_argument("--doc", default=str(settings.doc_path))
    p_index.add_argument("--rebuild", action="store_true", help="清空后重建")
    p_index.add_argument("--preview", action="store_true", help="打印分片内容")
    p_index.set_defaults(func=cmd_index)

    p_ask = sub.add_parser("ask", help="单轮问答")
    p_ask.add_argument("question")
    p_ask.add_argument("--top-k-retrieve", type=int, default=None)
    p_ask.add_argument("--top-k-rerank", type=int, default=None)
    p_ask.set_defaults(func=cmd_ask)

    p_chat = sub.add_parser("chat", help="交互式多轮问答")
    p_chat.set_defaults(func=cmd_chat)

    p_demo = sub.add_parser("demo", help="逐阶段演示视频中的完整流程")
    p_demo.add_argument("--doc", default=str(settings.doc_path))
    p_demo.add_argument("--question", default=None)
    p_demo.add_argument("--rebuild", action="store_true")
    p_demo.set_defaults(func=cmd_demo)

    p_serve = sub.add_parser("serve", help="启动 Web API 与页面")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
