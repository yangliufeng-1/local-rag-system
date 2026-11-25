# 本地 RAG 系统（按视频流程实现）

复现 B 站视频 [《使用Python构建RAG系统 —— 用代码还原 RAG系统的每个细节》](https://www.bilibili.com/video/BV1wc3izUEUb) 的完整链路，并做了工程化改造：持久化向量库、模型复用、命令行/Web 双入口、多生成后端与离线降级。

> 原视频代码：<https://github.com/MarkTechStation/VideoCode/tree/main/使用Python构建RAG系统/rag>
>
> 本仓库地址：<https://github.com/yangliufeng-1/local-rag-system>

## 核心链路

| 阶段 | 视频原方案 | 本实现 |
| --- | --- | --- |
| 分片 | 按空行切分 `doc.md` | 一致，支持自定义分隔符 |
| 向量化 | `shibing624/text2vec-base-chinese` | 一致，归一化向量 |
| 向量库 | `chromadb.EphemeralClient` | `PersistentClient`，索引落盘复用 |
| 召回 | Top-5 向量相似度 | 一致，可配置 |
| 重排 | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | 一致，模型只加载一次 |
| 生成 | Gemini 2.5 Flash | Gemini / OpenAI 兼容 / Ollama / 离线证据模式自动选择 |

## 目录结构

```
rag-system/
├── main.py                 # 免安装入口
├── data/doc.md            # 示例知识库（视频原文）
├── src/rag_system/
│   ├── config.py          # 配置与环境变量
│   ├── pipeline.py        # 分片/索引/召回/重排/生成
│   ├── llm.py             # 生成后端与离线兜底
│   ├── cli.py             # 命令行入口
│   └── api.py             # FastAPI 服务
├── web/index.html         # 轻量 Web 界面
├── tests/test_smoke.py    # 轻量测试
└── .env.example           # 配置模板
```

## 快速开始（Windows / Git Bash）

```bash
cd rag-system
python -m venv .venv

# 建议先装 CPU 版 PyTorch，避免下载 CUDA 依赖
.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pip install -e .
```

国内环境默认走 HF 镜像 `https://hf-mirror.com` 下载模型（约 500 MB，首次运行需要等待）。

### 1. 构建索引

```bash
python main.py index --rebuild --preview
```

### 2. 问答

```bash
python main.py ask "哆啦A梦使用的3个秘密道具分别是什么？"
python main.py chat          # 交互模式
python main.py demo          # 逐阶段打印视频里的每一步
```

### 3. Web 界面

```bash
python main.py serve
```

浏览器打开 <http://127.0.0.1:8000>。API 端点：`GET /health`、`POST /ask`、`POST /index`。

## 生成后端配置

复制 `.env.example` 为 `.env`，按需填写。`LLM_BACKEND=auto` 时按以下顺序选择：

1. **Gemini**（视频原方案）：填写 `GEMINI_API_KEY`，模型默认 `gemini-2.5-flash`
2. **OpenAI 兼容**：填写 `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL`，可接 DeepSeek、通义等
3. **Ollama 本地模型**：安装 Ollama 后执行 `ollama pull qwen2.5:3b`，无需密钥
4. **离线证据模式**：以上都不可用时，直接返回重排后的原文片段并标注，不会编造答案

切换为强制模式：`LLM_BACKEND=extractive`（或 `gemini` / `openai` / `ollama`）。

### 轻量模式（下载量从 ~880 MB 降到 ~100 MB）

如果想先跑通流程、暂不下载视频中的两个大模型，在 `.env` 中设置：

```env
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
RERANKER_MODEL=none
LLM_BACKEND=extractive
```

此时重排退化为「用向量相似度排序」，其余分片/索引/召回/证据生成流程不变；确认可用后再改回默认值即可使用视频同款模型。

## 换成自己的知识库

把文档放到 `data/` 下，然后：

```bash
python main.py index --doc data/你的文档.md --rebuild
python main.py ask "你的问题"
```

默认切片单位是空行；长段落建议先用换行分隔成语义块，效果更好。

## 说明

- 向量库默认写在 `chroma_db/`，已加入 `.gitignore`；删除该目录后下次问答会自动重建。
- `.env` 含密钥，已加入 `.gitignore`，**不要提交到 GitHub**。
- 单次问答耗时可在返回结果 `timings_ms` 中查看，便于对比召回/重排/生成各阶段。
