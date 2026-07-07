# La Lango AI - System Design

This document is the single source of truth for **how La Lango AI is built** -
the architecture, the components, how data flows through the system, and the
step-by-step logic of the core workflows (training, inference, and API requests).

All diagrams use [Mermaid](https://mermaid.js.org/), which renders automatically
on GitHub. If you are reading this in an editor without Mermaid support, view it
on GitHub or install a Mermaid preview extension.

> For a plain-English explanation of the *models* themselves (LSTM, attention,
> transformer), see [`architecture.md`](architecture.md). This document focuses
> on the *system* as a whole.

---

## 1. System Overview

La Lango AI is a **low-resource machine translation platform** with three layers:

| Layer | Technology | Responsibility |
|-------|-----------|----------------|
| **Presentation** | Plain HTML/CSS/JS (`frontend/index.html`) | User interface for entering and viewing translations |
| **Application / API** | FastAPI (`backend/lalango/api/`) | REST endpoints, request validation, model orchestration |
| **ML Engine** | PyTorch (`backend/lalango/`) | Tokenization, translation models, training, evaluation |

The guiding principle: **everything is implemented from scratch - no external
AI APIs, no black boxes.**

```mermaid
graph TB
    subgraph Client["🖥️ Client Layer"]
        UI["Web UI<br/>frontend/index.html"]
    end

    subgraph API["⚙️ Application Layer - FastAPI"]
        Routes["Routes<br/>api/routes.py"]
        Main["App + Model Registry<br/>api/main.py"]
    end

    subgraph Engine["🧠 ML Engine Layer - PyTorch"]
        Tok["Tokenizers<br/>tokenizers/"]
        Models["Translation Models<br/>models/"]
        Eval["Evaluation<br/>evaluation/"]
    end

    subgraph Offline["🛠️ Offline Pipeline - CLI Scripts"]
        Pre["preprocess.py"]
        Train["train.py"]
        Evaluate["evaluate.py"]
    end

    subgraph Storage["💾 Storage"]
        Raw["data/raw/"]
        Proc["data/processed/"]
        Ckpt["checkpoints/"]
        Lang["languages/ registry"]
    end

    UI -->|"HTTP JSON"| Routes
    Routes --> Main
    Main -->|"load model"| Models
    Routes --> Tok
    Tok --> Models
    Models -->|"decode"| Tok

    Pre --> Raw
    Pre --> Proc
    Train --> Proc
    Train --> Ckpt
    Evaluate --> Ckpt
    Evaluate --> Eval
    Main -->|"reads at startup"| Ckpt
    Main --> Lang
```

---

## 2. Component Architecture

The backend core package `backend/lalango/` is organised into focused modules.
This mirrors the actual folder structure on disk.

```mermaid
graph LR
    subgraph lalango["backend/lalango/"]
        direction TB

        subgraph api["api/"]
            main["main.py<br/>app + registry"]
            routes["routes.py<br/>endpoints"]
        end

        subgraph tokenizers["tokenizers/"]
            char["char_tokenizer.py"]
            bpe["bpe_tokenizer.py<br/>(Phase 3)"]
        end

        subgraph data["data/"]
            dataset["dataset.py"]
            cleaner["cleaner.py"]
            splitter["splitter.py"]
        end

        subgraph models["models/"]
            seq2seq["seq2seq_lstm.py"]
            attn["attention.py<br/>(Phase 2)"]
            trans["transformer.py<br/>(Phase 4)"]
        end

        subgraph evaluation["evaluation/"]
            bleu["bleu.py"]
            chrf["chrf.py"]
            report["report.py"]
        end
    end

    routes --> main
    routes --> char
    routes --> seq2seq
    char --> seq2seq
    attn --> seq2seq
    dataset --> char
    cleaner --> dataset
    splitter --> dataset
    bleu --> report
    chrf --> report
```

### Module responsibilities

| Module | File(s) | Responsibility |
|--------|---------|----------------|
| **API** | `api/main.py`, `api/routes.py` | HTTP endpoints, request/response schemas, model registry |
| **Tokenizers** | `tokenizers/char_tokenizer.py` | Convert text ↔ integer token IDs (char-level now, BPE later) |
| **Data** | `data/dataset.py`, `cleaner.py`, `splitter.py` | Load corpora, clean pairs, split train/val/test, batch |
| **Models** | `models/seq2seq_lstm.py`, `attention.py`, `transformer.py` | The neural translation architectures |
| **Evaluation** | `evaluation/bleu.py`, `chrf.py`, `report.py` | Quality metrics and reporting |
| **Scripts** | `scripts/preprocess.py`, `train.py`, `evaluate.py` | Offline CLI pipelines |

---

## 3. Data Flow - Translation Request (Runtime)

This is what happens when a user translates a sentence in the browser.

```mermaid
sequenceDiagram
    actor User
    participant UI as Web UI
    participant API as FastAPI (routes.py)
    participant Reg as Model Registry (main.py)
    participant Tok as Tokenizer
    participant Model as Seq2Seq Model

    User->>UI: Type text, click "Translate"
    UI->>API: POST /translate {source_lang, target_lang, text}
    API->>API: Validate language pair (SUPPORTED_PAIRS)
    alt Pair not supported
        API-->>UI: 404 with helpful message
    else Pair supported
        API->>Reg: Get model for (src, tgt)
        Reg-->>API: Loaded model + tokenizers
        API->>Tok: encode(source text)
        Tok-->>API: token IDs [4, 12, 8, 2]
        API->>Model: translate(token IDs)
        Model->>Model: encode → decode (greedy)
        Model-->>API: output token IDs
        API->>Tok: decode(output IDs)
        Tok-->>API: translated text
        API-->>UI: 200 {translated_text, ...}
        UI-->>User: Display translation
    end
```

---

## 4. Data Flow - Offline Training Pipeline

This is the flow a contributor runs to produce a trained model, before it is
ever served by the API.

```mermaid
flowchart TD
    A["Raw parallel corpus<br/>data/raw/*.src + *.tgt"] --> B["preprocess.py"]
    B --> B1["cleaner.py<br/>strip, filter by length,<br/>remove empty pairs"]
    B1 --> B2["splitter.py<br/>train / val / test split"]
    B2 --> C["Processed data<br/>data/processed/&lt;lang-pair&gt;/"]

    C --> D["train.py"]
    D --> D1["Build vocab<br/>char_tokenizer"]
    D1 --> D2["Encode + batch<br/>dataset.py"]
    D2 --> D3["Training loop<br/>Seq2SeqLSTM + teacher forcing"]
    D3 --> D4{"Val loss<br/>improved?"}
    D4 -->|Yes| D5["Save best checkpoint<br/>checkpoints/&lt;lang-pair&gt;/"]
    D4 -->|No| D6["Continue / early stop"]
    D6 --> D3

    D5 --> E["evaluate.py"]
    E --> E1["Load checkpoint<br/>+ test set"]
    E1 --> E2["Greedy decode test set"]
    E2 --> E3["bleu.py + chrf.py"]
    E3 --> F["Evaluation report<br/>report.py"]
```

---

## 5. Flowchart - Model Inference (Greedy Decoding)

The core logic inside `Seq2SeqLSTM.translate()`.

```mermaid
flowchart TD
    Start(["translate(source_ids)"]) --> Enc["Encoder reads source<br/>→ context / hidden state"]
    Enc --> Init["decoder_input = &lt;SOS&gt;<br/>outputs = []"]
    Init --> Loop{"step &lt; max_len<br/>AND last token ≠ &lt;EOS&gt;?"}
    Loop -->|Yes| Step["Decoder.forward_step(<br/>decoder_input, hidden)"]
    Step --> Pick["Pick highest-probability<br/>token (argmax)"]
    Pick --> Append["Append token to outputs<br/>decoder_input = token"]
    Append --> Loop
    Loop -->|No| Decode["Tokenizer decodes<br/>output IDs → text"]
    Decode --> End(["return translation"])
```

---

## 6. API Contract

Defined in `backend/lalango/api/routes.py`.

| Method | Endpoint | Request | Response |
|--------|----------|---------|----------|
| `POST` | `/translate` | `{ source_lang, target_lang, text }` | `{ translated_text, source_lang, target_lang, model }` |
| `GET` | `/languages` | - | `{ supported_pairs: [...] }` |
| `GET` | `/health` | - | `{ status: "ok" }` |

**Error handling:** unsupported language pairs return `404` with a message
listing supported pairs and pointing to the `languages/` registry.

```mermaid
stateDiagram-v2
    [*] --> Idle
    Idle --> Validating: POST /translate
    Validating --> NotFound: pair unsupported
    Validating --> Translating: pair supported
    NotFound --> Idle: 404 response
    Translating --> Idle: 200 response
```

---

## 7. Deployment View (Current & Target)

```mermaid
graph TB
    subgraph Dev["Current - Local Development"]
        direction LR
        FE1["Static server<br/>python -m http.server 5500"]
        BE1["uvicorn<br/>:8000"]
        FE1 --> BE1
    end

    subgraph Target["Target - Hosted"]
        direction LR
        CDN["Static host / CDN<br/>(frontend)"]
        APIH["ASGI host<br/>(uvicorn/gunicorn)"]
        MODELS["Model checkpoints<br/>(mounted volume / object store)"]
        CDN --> APIH
        APIH --> MODELS
    end

    Dev -.evolves into.-> Target
```

> **Note:** hosting is not yet set up - this is the intended target. See the
> project management doc for where this sits on the timeline.

---

## 8. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **From-scratch models (no external AI APIs)** | Core mission: educational + full control for low-resource languages |
| **Character-level tokenizer first** | Simplest entry point; works for any script without pre-built vocab |
| **Phased model roadmap (LSTM → Attention → BPE → Transformer)** | Each phase builds intuition for the next; keeps contributions accessible |
| **Single-file frontend (no framework)** | Zero build step; approachable for first-time contributors |
| **Language registry in `languages/`** | Community can add languages via config + corpus without touching core code |
| **FastAPI** | Automatic OpenAPI docs, async-ready, Pydantic validation |

---

## 9. Related Documents

- [`architecture.md`](architecture.md) - how the ML models work (conceptual)
- [`data_format.md`](data_format.md) - corpus and processed data formats
- [`adding_a_language.md`](adding_a_language.md) - contributor guide for new languages
- [`evaluation_guide.md`](evaluation_guide.md) - how metrics are computed
- [`project_management.md`](project_management.md) - WBS, Gantt chart, milestones
