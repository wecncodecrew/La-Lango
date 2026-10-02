# La Lango

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
La-Lango/
│
├── lalango/                  # Main package
│   ├── __init__.py           
│   ├── tokenizers/
│   │   ├── __init__.py       
│   │   └── char_tokenizer.py 
│   ├── data/
│   │   ├── __init__.py       
│   │   └── dataset.py        
│   └── models/
│       ├── __init__.py       
│       └── seq2seq_lstm.py   
│
├── scripts/
│   └── train.py              
│
├── datasets/
│   └── sample_data.csv       
│
├── requirements.txt          # NEW: Dependencies
├── .gitignore                # NEW: Ignored files
└── README.md                 # NEW: Local dev instructions
```

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
git clone https://github.com/Wecncode/La-Lango.git
cd La-Lango

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

We welcome contributors of all levels!

| Label                 | Who it is for                              |
|-----------------------|--------------------------------------------|
| 🟢 `good-first-issue` | First-timers, documentation, small fixes   |
| 🔵 `data`             | Data cleaning, tokenizers, preprocessing   |
| 🟠 `model`            | Building and improving translation models  |
| 🔴 `research`         | Evaluation metrics, paper writing          |

Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a PR.
Check [ROADMAP.md](ROADMAP.md) to see what is being worked on.

---

## Local Development Setup

To work on Phase 1 of La Lango locally, follow these steps:

**1. Clone the repository**
```bash

_git clone [https://github.com/wecncodecrew/La-Lango.git](https://github.com/wecncodecrew/La-Lango.git)
cd La-Lango_

```
---

## Community

- 💬 [GitHub Discussions](https://github.com/Wecncode/La-Lango/discussions)
- 🐛 [Open an issue](https://github.com/Wecncode/La-Lango/issues)
- 📖 [docs/](docs/) — including the [System Design](docs/system_design.md) and [Project Management (WBS + Gantt)](docs/project_management.md)

---

Made with ❤️ by the [Wecncode](https://github.com/Wecncode) community — MIT License.
