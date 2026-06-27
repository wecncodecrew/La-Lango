# 🌍 La Lango AI

> **Low-Resource Language Translation API** — A community-driven NLP platform for translating
> regional dialects, built from scratch by students, for the world.

[![CI](https://github.com/wecncodecrew/La-Lango/actions/workflows/ci.yml/badge.svg)](https://github.com/wecncodecrew/La-Lango/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Good First Issues](https://img.shields.io/github/issues/wecncodecrew/La-Lango/good-first-issue)](https://github.com/wecncodecrew/La-Lango/issues?q=is%3Aissue+label%3Agood-first-issue)

---

## What is La Lango AI?

Hundreds of regional dialects and low-resource languages have little to no machine translation
support. Commercial AI APIs ignore them because they are not profitable.

**La Lango AI is our answer to that.**
We are an open, community-built translation platform where every language deserves a model.

> **No external AI APIs. No black boxes. Everything is implemented from scratch.**

---

## Project structure

```
la-lango-ai/
│
├── 📁 backend/           ← Translation engine + REST API (Python)
│   ├── lalango/          ← Core package: models, tokenizers, data, API
│   ├── scripts/          ← CLI tools: train, evaluate, preprocess
│   ├── tests/            ← Automated tests
│   └── experiments/      ← Jupyter notebooks (start here!)
│
├── 📁 frontend/          ← Web UI (plain HTML / CSS / JavaScript)
│   └── index.html        ← Entire UI in one file, no framework needed
│
├── 📁 languages/         ← Community language registry (add yours here!)
├── 📁 data/              ← Your datasets go here (not tracked by git)
├── 📁 docs/              ← Guides: architecture, data format, adding a language
│
├── README.md
├── CONTRIBUTING.md
├── ROADMAP.md
└── LICENSE
```

---

## How the system works

```
Browser  (frontend/index.html)
    │  HTTP fetch
    ▼
FastAPI  (backend/lalango/api/)
    │
    ▼
Tokenizer  (backend/lalango/tokenizers/)
    │  splits text into characters/subwords
    ▼
Translation Model  (backend/lalango/models/)
    │  predicts the translation
    ▼
Tokenizer  ← decodes output back to text
    │
    ▼
Browser receives translation
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/Wecncode/la-lango-ai.git
cd la-lango-ai

pip install -r backend/requirements.txt
```

### 2. Start the backend API

```bash
uvicorn backend.lalango.api.main:app --reload
# Swagger UI → http://localhost:8000/docs
```

### 3. Open the frontend

```bash
cd frontend && python -m http.server 5500
# Open → http://localhost:5500
```

### 4. Preprocess data, train, evaluate

```bash
# Preprocess a raw parallel corpus
PYTHONPATH=backend python backend/scripts/preprocess.py \
  --src data/raw/my-lang/train.src \
  --tgt data/raw/my-lang/train.tgt \
  --output data/processed/my-lang/

# Train a model
PYTHONPATH=backend python backend/scripts/train.py \
  --lang-pair konkani-english \
  --data data/processed/konkani-english/

# Evaluate
PYTHONPATH=backend python backend/scripts/evaluate.py \
  --checkpoint checkpoints/konkani-english.pt \
  --data data/processed/konkani-english/test.json
```

---

## Our learning roadmap

```
Phase 1 ── Seq2Seq LSTM                🟢 Beginner
Phase 2 ── + Bahdanau Attention        🔵 Intermediate
Phase 3 ── + BPE Tokenizer             🔵 Intermediate
Phase 4 ── Transformer from scratch    🟠 Advanced
Phase 5 ── Evaluation & Benchmarking   🔴 Research
```

---

## API at a glance

| Method | Endpoint      | What it does                 |
|--------|---------------|------------------------------|
| POST   | `/translate`  | Translate a sentence         |
| GET    | `/languages`  | List all supported languages |
| GET    | `/health`     | Check if the server is up    |

---

## Contributing

| Label                 | Who it is for                              |
|-----------------------|--------------------------------------------|
| 🟢 `good-first-issue` | First-timers, documentation, small fixes   |
| 🔵 `data`             | Data cleaning, tokenizers, preprocessing   |
| 🟠 `model`            | Building and improving translation models  |
| 🔴 `research`         | Evaluation metrics, paper writing          |

Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a PR.
Check [ROADMAP.md](ROADMAP.md) to see what is being worked on.

---

## Community

- 💬 [GitHub Discussions](https://github.com/Wecncode/la-lango-ai/discussions)
- 🐛 [Open an issue](https://github.com/Wecncode/la-lango-ai/issues)
- 📖 [docs/](docs/)

---

Made with ❤️ by the [Wecncode](https://github.com/Wecncode) community — MIT License.
