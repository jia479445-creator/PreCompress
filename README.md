# precompress

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

### 步骤 1：下载项目

```bash
git clone https://github.com/jia479445-creator/PreCompress.git
```

### 步骤 2：环境配置

```bash
cd /root/autodl-tmp/PreCompress      # 改成你的实际路径
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 步骤 3：下载 LLMLingua-2 权重

```bash
hf download microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank \
    --local-dir ./models/llmlingua-2-bert
```

### 步骤 4：配置 `.env`

```bash
cp .env.example .env
chmod 600 .env      # 含 API key，限制权限
vim .env
```

配置您的OPENAI_API_KEY、OPENAI_MODEL、LLMLINGUA_MODEL、LLMLINGUA_RATE、LLMLINGUA_DEVICE。

### 步骤 5：快速跑通

```bash
# 只跑压缩
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

# 跑 demo
python -m examples.demo
```

### 步骤 6：跑测试套件（可选）

```bash
RUN_INTEGRATION=1 OPENAI_API_KEY=$OPENAI_API_KEY pytest -v

RUN_INTEGRATION=1 pytest -v tests/test_compressor.py

OPENAI_API_KEY=$OPENAI_API_KEY pytest -v tests/test_extractor.py
```

## 三种使用方式

### 方式 A：CLI 一次性运行

```bash
python -m examples.demo                  # 内置 demo 文本
python -m examples.demo /data/input.txt  # 处理你自己的文件
```

### 方式 B：Python API

```python
from precompress import run_from_env

result = run_from_env("我的长文本……")
print("压缩后：", result.compressed_messages[0]["content"])
print("核心事实：", result.core_memory_facts)
print(f"token：{result.tokens_before} → {result.tokens_after}")
```

### 方式 C：pytest 真实集成测试

```bash
pytest                                         
RUN_INTEGRATION=1 pytest                        
OPENAI_API_KEY=sk-... pytest                    
RUN_INTEGRATION=1 OPENAI_API_KEY=sk-... pytest  
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

