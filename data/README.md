# data/

Drop the LongMemEval JSON file(s) here, e.g.:

```
data/
└── longmemeval_s.json     # ~500 samples, ~50 MB
```

The actual JSON is **not committed** (see `.gitignore`). Download it from
HuggingFace:

- Dataset page: <https://huggingface.co/datasets/xiaowu0162/longmemeval>

Then point `LONGMEMEVAL_DATA` in your `.env` at the absolute path of the
JSON file. See the project root README for the full setup.
