# PreCompress

本项目用于在把长对话送进大模型之前，先做一层预处理，尽量减少无效 token 的消耗。分为三个步骤：

1. 调用 `LLMLingua-2` 压缩输入对话
2. 把压缩后的对话送给 LLM 做一次事实抽取
3. 返回结果，或者按样本写成 JSON 文件


## 项目结构

```text
PreCompress/
├── README.md
├── .env.example            # 配置模板
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── data/
│   └── README.md           # 数据集放置说明
├── precompress/
│   ├── __init__.py         # 导入时自动加载 .env
│   ├── compressor.py       # LLMLingua-2 压缩器
│   ├── extractor.py        # OpenAI-compatible LLM 抽取器
│   ├── pipeline.py         # 压缩 + 事实抽取 的端到端流程入口
│   ├── prompts.py          # 抽取 prompt
│   ├── utils.py
│   ├── env.py              
│   └── longmemeval.py      
├── examples/
│   ├── __init__.py
│   ├── demo.py             # 单样本快速复现 demo
│   └── run_longmemeval.py  
```

## 安装

```bash
cd PreCompress
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

下载 `LLMLingua-2` 权重：

```bash
hf download microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank \
  --local-dir ./models/llmlingua-2-bert
```


## 配置

复制配置模板：

```bash
cp .env.example .env
```

修改基础配置：

```bash
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini

LLMLINGUA_MODEL=/abs/path/to/PreCompress/models/llmlingua-2-bert
LLMLINGUA_DEVICE=cpu

LONGMEMEVAL_DATA=/abs/path/to/longmemeval_s.json
```

## 快速 Demo

```bash
python -m examples.demo
```

## 完整跑 LongMemEval

运行入口：

```bash
python -m examples.run_longmemeval \
  --output-dir outputs/longmemeval \
  --mode flatten
```

两种模式：

- `flatten`：把一个样本的所有 session 拼成一次输入，再做一次压缩和抽取
- `session`：每个 session 分开跑，结果按 session 分别保存

输出：

- 每个样本写一个 JSON 到 `--output-dir`
