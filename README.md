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