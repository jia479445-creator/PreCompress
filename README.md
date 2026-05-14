# precompress

LightMem 项目里"预压缩 + 核心记忆抽取"那条链路的**独立可运行版本**。

```
原始文本 / 对话消息
        │
        ▼  LLMLingua-2（默认 microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank）
压缩后的文本   ── 分块 · token 预算二次重试 · 最后截断
        │
        ▼  OpenAI 兼容 LLM + METADATA_GENERATE_PROMPT
核心记忆事实   ── [{ "source_id": <int>, "fact": "<一句话事实>" }, ...]
```

## 与原 LightMem 的对应关系

`precompress/` 里的代码**逐字 / 近逐字搬运自原 LightMem 仓库**，去依赖外的改动控制到最少：

| 本项目文件 | 抽取自 LightMem | 与原版的差异 |
|---|---|---|
| `precompress/compressor.py` | `src/lightmem/factory/pre_compressor/llmlingua_2.py` + `src/lightmem/configs/pre_compressor/llmlingua_2.py` | 把 pydantic 配置 `LlmLingua2Config` 内联到文件顶部 |
| `precompress/extractor.py` | `src/lightmem/factory/memory_manager/openai.py` + `src/lightmem/configs/memory_manager/base_config.py` | 把 `BaseMemoryManagerConfig` 内联；`flat` 与 `event` 两条分支以及 `_merge_dual_perspective_results` 均保留逐字搬运。**唯一一处行为级修改**：原版硬编码的 `max_workers = 5` 现在读 `EXTRACT_MAX_WORKERS` 环境变量（默认仍是 5） |
| `precompress/prompts.py` | `src/lightmem/memory/prompts.py` | 逐字搬运。包含全部 6 段 prompt：`METADATA_GENERATE_PROMPT`、`METADATA_GENERATE_PROMPT_locomo`、`LoCoMo_Event_Binding_factual`、`LoCoMo_Event_Binding_relational`、`LoCoMo_Cross_Event_Consolidation`、`UPDATE_PROMPT`，以及完整的 `EXTRACTION_PROMPTS` 派发表 |
| `precompress/utils.py` | `src/lightmem/memory/utils.py` | 仅搬运 `clean_response`，其他无关函数不引入 |
| `precompress/pipeline.py` | 新写（对应 `src/lightmem/memory/lightmem.py::LightMemory.add_memory` 的"预压缩 → 抽取"片段） | 极简的胶水脚本 |
| `precompress/env.py` | 新增 | `.env` 加载器 + 配置工厂 |

## 项目结构

```
standalone/
├── README.md
├── .env.example            # 配置模板 —— 复制成 .env 后编辑
├── .gitignore              # 已忽略 .env，防止 key 泄漏
├── pyproject.toml
├── requirements.txt
├── precompress/
│   ├── __init__.py         # 导入时自动加载 .env
│   ├── compressor.py
│   ├── extractor.py
│   ├── pipeline.py
│   ├── prompts.py
│   ├── utils.py
│   └── env.py              # .env 解析 + 配置工厂
├── examples/
│   └── demo.py             # CLI 入口（读 .env）
└── tests/
    ├── conftest.py
    ├── data/
    │   ├── sample_dialogue.json   # flat 模式测试数据
    │   └── locomo_dialogue.json   # event 模式测试数据
    ├── test_compressor.py        # 真实 LLMLingua-2 模型测试
    ├── test_extractor.py         # 真实 OpenAI API 测试（flat + event）
    └── test_pipeline.py          # 端到端测试
```

## 复现步骤

### 环境需求

| 项 | 推荐 |
|---|---|
| OS | Linux / macOS（Windows 走 WSL2） |
| Python | 3.9 ~ 3.12 |
| RAM | ≥ 4 GB（CPU 模式）/ ≥ 8 GB（CUDA 模式） |
| 磁盘 | ≥ 5 GB（含模型权重） |
| GPU | 可选，但强烈建议（A10 / 3060 / T4 都够） |

### 步骤 1：把项目搬到目标机器

```bash
# 假设源在 /Users/lijia/Desktop/科研/组内/LightMem-main/standalone

# 方式 A：rsync 推到服务器
rsync -avz --exclude '__pycache__' --exclude '.env' --exclude '.venv' \
    /Users/lijia/Desktop/科研/组内/LightMem-main/standalone/ \
    user@server:/root/autodl-tmp/PreCompress/

# 方式 B：先 git init 推到自己仓库，再 git clone
cd /Users/lijia/Desktop/科研/组内/LightMem-main/standalone
git init && git add -A && git commit -m "initial standalone snapshot"
git remote add origin git@your-repo:precompress.git
git push -u origin main
# 服务器端：
# git clone git@your-repo:precompress.git /root/autodl-tmp/PreCompress
```

AutoDL 用户注意：**只有 `/root/autodl-tmp/` 是持久化目录**，容器重启后其它路径会被擦除。把项目和模型权重都放在 `autodl-tmp` 下。

### 步骤 2：建虚拟环境 + 装依赖

```bash
cd /root/autodl-tmp/PreCompress     # 改成你的实际路径
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

GPU 服务器额外装匹配 CUDA 的 PyTorch（`llmlingua` 依赖里默认是 CPU 版的 torch）：

```bash
# CUDA 12.1 为例
pip install --index-url https://download.pytorch.org/whl/cu121 torch
```

国内服务器建议同时换 pip 源：

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### 步骤 3：下载 LLMLingua-2 权重

#### 国内 / AutoDL —— 用 HF 镜像

```bash
export HF_ENDPOINT=https://hf-mirror.com
# 长期生效写到 ~/.bashrc 里
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc

mkdir -p /root/autodl-tmp/models
hf download microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank \
    --local-dir /root/autodl-tmp/models/llmlingua-2-bert
```

> 老命令 `huggingface-cli download` 已弃用，新版改用 `hf download`。

#### 国外 / 能直连 HuggingFace

```bash
hf download microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank \
    --local-dir ./models/llmlingua-2-bert
```

#### 校验权重完整

```bash
ls -lh /root/autodl-tmp/models/llmlingua-2-bert/
```

应包含（缺一不可）：

```
config.json
tokenizer.json
tokenizer_config.json
vocab.txt
pytorch_model.bin         # 或 model.safetensors（约 700 MB）
```

如果 `pytorch_model.bin` 只有几 KB，说明 LFS 没拉下来，重新下一次。

### 步骤 4：配置 `.env`

```bash
cp .env.example .env
chmod 600 .env      # 含 API key，限制权限
vim .env
```

至少改这几项：

```bash
OPENAI_API_KEY=sk-你的key
OPENAI_BASE_URL=https://api.openai.com/v1     # 或中转地址
OPENAI_MODEL=gpt-4o-mini

LLMLINGUA_MODEL=/root/autodl-tmp/models/llmlingua-2-bert   # 改成你刚下的本地路径
LLMLINGUA_DEVICE=cuda                                       # CPU 服务器写 cpu
LLMLINGUA_RATE=0.5                                          # 压缩率，越小压得越狠
```

### 步骤 5：跑通验证

冷启动一次，确认模型加载、压缩、抽取链路都通：

```bash
# 只跑压缩（不需要 OPENAI_API_KEY）
python -c "
from precompress import LlmLingua2Config, LlmLingua2Compressor
cfg = LlmLingua2Config(llmlingua_config={
    'model_name': '/root/autodl-tmp/models/llmlingua-2-bert',
    'device_map': 'cuda',
    'use_llmlingua2': True,
})
c = LlmLingua2Compressor(cfg)
out = c.compress([{'role': 'user', 'content': 'My name is Alice. I am a physics teacher in Boston.'}], c.tokenizer)
print('compressed:', out[0]['content'])
"

# 跑 demo（含 LLM 抽取）
python -m examples.demo
```

`demo` 正常输出"原文 → 压缩后 → 核心记忆事实 JSON"三段内容就算复现成功。

### 步骤 6：跑测试套件（可选）

```bash
# 真模型 + 真 API 全链路
RUN_INTEGRATION=1 OPENAI_API_KEY=$OPENAI_API_KEY pytest -v

# 只验 LLMLingua-2（不调 API）
RUN_INTEGRATION=1 pytest -v tests/test_compressor.py

# 只验 OpenAI 调用（不加载模型）
OPENAI_API_KEY=$OPENAI_API_KEY pytest -v tests/test_extractor.py
```

## 三种使用方式

### 方式 A：CLI 一次性运行

```bash
python -m examples.demo                  # 内置 demo 文本
python -m examples.demo /data/input.txt  # 处理你自己的文件
```

如果 `OPENAI_API_KEY` 没填，demo 会自动降级为"只压缩，不抽取"。

### 方式 B：Python API（写在你自己的代码里）

最简单 —— 让 `.env` 决定一切：

```python
from precompress import run_from_env

result = run_from_env("我的长文本……")
print("压缩后：", result.compressed_messages[0]["content"])
print("核心事实：", result.core_memory_facts)
print(f"token：{result.tokens_before} → {result.tokens_after}")
```

调用前临时改某个参数（环境变量优先级高于 `.env`）：

```python
import os
os.environ["LLMLINGUA_RATE"] = "0.3"     # 这次压得更狠
os.environ["OPENAI_MODEL"] = "glm-4.6"

from precompress import run_from_env
result = run_from_env(chat_history)
```

完全自定义（不走 `.env`）：

```python
from precompress import (
    LlmLingua2Config,
    BaseMemoryManagerConfig,
    LlmLingua2Compressor,
    OpenaiManager,
)

compressor = LlmLingua2Compressor(LlmLingua2Config())
compressed = compressor.compress(messages, compressor.tokenizer)

manager = OpenaiManager(BaseMemoryManagerConfig(model="gpt-4o-mini"))
results = manager.meta_text_extract(
    extract_list=[[compressed]],         # 三层嵌套：[api_call][topic][message]
    messages_use="user_only",
    topic_id_mapping=[[1]],
    extraction_mode="flat",
)
facts = results[0]["cleaned_result"]
```

### 方式 C：pytest 真实集成测试

```bash
pytest                                          # 默认全 skip
RUN_INTEGRATION=1 pytest                        # 真模型
OPENAI_API_KEY=sk-... pytest                    # 真 API
RUN_INTEGRATION=1 OPENAI_API_KEY=sk-... pytest  # 端到端
```

## `.env` 配置全表

| 分类 | 变量 | 默认值 | 说明 |
|---|---|---|---|
| **OpenAI** | `OPENAI_API_KEY` | — | 留空 → 只压缩、不抽取 |
| | `OPENAI_BASE_URL` | `https://api.openai.com/v1` | 自部署 vLLM / 中转服务都改这里 |
| | `OPENAI_MODEL` | `gpt-4o-mini` | 用于抽取的对话模型 |
| | `LLM_TEMPERATURE` | `0.1` | |
| | `LLM_MAX_TOKENS` | `2000` | |
| | `LLM_TOP_P` | `0.1` | |
| **OpenRouter** | `OPENROUTER_API_KEY` | — | 设了则忽略 OPENAI_* |
| | `OPENROUTER_API_BASE` | `https://openrouter.ai/api/v1` | |
| | `OPENROUTER_SITE_URL` / `OPENROUTER_APP_NAME` | — | 可选分析头 |
| **LLMLingua-2** | `LLMLINGUA_MODEL` | `microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank` | 也接受本地目录 |
| | `LLMLINGUA_DEVICE` | `cuda` | CPU 写 `cpu` |
| | `LLMLINGUA_USE_V2` | `true` | false → 用 LLMLingua-1 |
| | `LLMLINGUA_RATE` | `0.5` | 保留的 token 比例；越小压得越狠 |
| | `LLMLINGUA_TARGET_TOKEN` | `-1` | 正整数 → 压到约这么多 token |
| | `LLMLINGUA_MAX_BATCH_SIZE` | `50` | |
| | `LLMLINGUA_MAX_FORCE_TOKEN` | `100` | |
| | `LLMLINGUA_INSTRUCTION` | 空 | 可选 instruction 文本 |
| **并发** | `EXTRACT_MAX_WORKERS` | `5` | `_extract_with_prompt` 的最大并发数；event 模式实际可达 `2×` |
| **测试** | `RUN_INTEGRATION` | 未设置 | `1` → 启用真实模型测试 |

加载优先级：**调用时传给类构造函数的参数 > 真实 OS 环境变量 > `.env` 文件 > 代码里的默认值**。

## 并发原理

在 `OpenaiManager._extract_with_prompt` 里：

```python
max_workers = min(len(extract_list), int(os.environ.get("EXTRACT_MAX_WORKERS", "5")))
```

- `extract_list` 是三层嵌套 `[api_call][topic_segment][message]`；最外层每个元素 = 一次 LLM 调用，正是并发的单位。
- 实际并发线程数 = `min(extract_list 长度, EXTRACT_MAX_WORKERS)`。
- 单次 `run_pipeline(...)` 调用产生 `len(extract_list) == 1`，所以**默认是单线程**；要真用上并发得把多组输入合到一次 `meta_text_extract` 里。
- **event 模式**会跑两次 `_extract_with_prompt`（factual + relational），所以 in-flight 请求可达 `2 × EXTRACT_MAX_WORKERS`。

并发不会自动多 key —— `OpenaiManager` 内部只持有一个 `OpenAI` 客户端，并发请求共用。`openai-python` 是线程安全的，单 key + 多线程是官方推荐写法。只有以下情形才需要多 key 池（需要给 `extractor.py` 打补丁）：

- 中转服务硬封顶"单 key 并发 N 路"
- 需要多账户分摊成本
- 单 key 经常被风控降速

## 部署到服务器

### 用 systemd 让服务长跑

如果你包了一层 FastAPI（参见 README 末尾"扩展建议"），可以这样让它常驻：

```bash
sudo tee /etc/systemd/system/precompress.service > /dev/null <<'EOF'
[Unit]
Description=precompress HTTP service
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/precompress
EnvironmentFile=/opt/precompress/.env
ExecStart=/opt/precompress/.venv/bin/uvicorn serve:app --host 0.0.0.0 --port 8000 --workers 1
Restart=always
RestartSec=5
Environment=CUDA_VISIBLE_DEVICES=0

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now precompress
sudo systemctl status precompress
journalctl -u precompress -f             # 实时日志
```

### 配合自部署 LLM（vLLM）

不想花 OpenAI 钱可以在同一台 GPU 机器跑 vLLM：

```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --port 8001 --gpu-memory-utilization 0.85
```

`.env` 改成：

```bash
OPENAI_API_KEY=任意非空字符串
OPENAI_BASE_URL=http://localhost:8001/v1
OPENAI_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
```

`OpenaiManager` 不区分 OpenAI 官方和 vLLM，接口完全兼容。

## 常见问题

**Q：首次 import 卡很久 / `OSError: We couldn't connect to 'https://huggingface.co'`**

A：HuggingFace 在国内被墙。要么换镜像（`export HF_ENDPOINT=https://hf-mirror.com`），要么先下到本地、在 `.env` 里把 `LLMLINGUA_MODEL` 改成本地绝对路径。

**Q：报错 `RuntimeError: Failed to initialize LlmLingua2Compressor: ... model_name must be one of [...] or a valid local path`**

A：`LlmLingua2Config` 校验器只放行三种官方模型名 + 任意"存在的本地目录"。你填的路径不存在、或者拼错了。`ls $LLMLINGUA_MODEL` 确认一下。

**Q：跑得很慢，单次压缩要十几秒**

A：99% 是因为在 CPU 上推理。`.env` 改 `LLMLINGUA_DEVICE=cuda`，并确保装的是 CUDA 版 torch（`python -c "import torch; print(torch.cuda.is_available())"` 应该返回 `True`）。

**Q：调用 OpenAI 兼容 API 报 429 / Rate limit**

A：把 `EXTRACT_MAX_WORKERS` 调小（默认 5 → 3 或 1）。或者升账户档位 / 切中转。

**Q：测试 `pytest` 全部 skip 是 bug 吗？**

A：不是，那是设计的。所有测试都需要真实模型 / 真实 API，没设环境变量时自动 skip。要跑就 `RUN_INTEGRATION=1 OPENAI_API_KEY=sk-... pytest`。

**Q：能不能改成同时支持多个 LLMLingua 模型轮询？**

A：当前架构是单进程持有单个 `LlmLingua2Compressor` 实例（模型加载一次 ≈ 1.5 GB 显存）。多模型需要改 `pipeline.py` 维护一个 compressor 池。需要的话告诉我。

## 扩展建议（按需）

下面这些不在当前项目里，但如果你要做生产部署可能用得上 —— 让我加哪个都行：

- `serve.py` —— FastAPI HTTP 服务（含 `/v1/compress` 和 `/healthz`）
- `Dockerfile` —— 一键容器化
- `deploy/precompress.service` —— systemd 单元文件
- 多 key 轮询补丁
- 批处理脚本（cron 跑队列）
