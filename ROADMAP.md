# 🗺️ La Lango AI — Roadmap

This file describes what we are building, in what order, and what is left to do.
If you want to contribute, this is the best place to understand where the project is headed.

---

## Phase 1 — Seq2Seq LSTM (baseline model)
**Status: 🚧 In progress**

The goal of Phase 1 is to have a working end-to-end translation pipeline —
even if the translations are not great yet.

### What needs to be built

- [ ] Character-level tokenizer (`lalango/tokenizers/char_tokenizer.py`)
- [ ] Data loader that reads parallel sentence pairs (`lalango/data/dataset.py`)
- [ ] Encoder: LSTM that reads a source sentence (`lalango/models/seq2seq_lstm.py`)
- [ ] Decoder: LSTM that generates the target sentence one word at a time
- [ ] Training loop (`scripts/train.py`)
- [ ] Greedy decoding (pick the most likely word at each step)
- [ ] Basic translate function in the API

### What "done" looks like

We can run `scripts/train.py`, train on a small parallel corpus,
and get back a (rough) translation from the API.

---

## Phase 2 — Bahdanau Attention
**Status: 📋 Planned**

Without attention, the encoder squishes an entire sentence into one vector.
For long sentences this loses a lot of information.

Attention lets the decoder "look back" at every word in the source sentence
when generating each word of the translation.

### What needs to be built

- [ ] Attention score calculation (`lalango/models/attention.py`)
- [ ] Connect attention to the decoder in `seq2seq_lstm.py`
- [ ] Attention weight visualization in the experiments notebook

---

## Phase 3 — BPE Tokenizer
**Status: 📋 Planned**

Character-level tokenization is simple but creates very long sequences.
Byte Pair Encoding (BPE) finds common subword units and uses those instead,
giving us shorter sequences and better handling of rare words.

### What needs to be built

- [ ] BPE vocabulary builder (`lalango/tokenizers/bpe_tokenizer.py`)
- [ ] Encode and decode functions
- [ ] Script to build a vocabulary from a corpus

---

## Phase 4 — Transformer
**Status: 📋 Planned**

The Transformer is the architecture behind modern translation systems.
We will build it from scratch once we have learned from the LSTM.

### What needs to be built

- [ ] Multi-head self-attention
- [ ] Positional encoding
- [ ] Encoder and Decoder stacks
- [ ] Training with learning rate warmup

---

## Phase 5 — Evaluation and Benchmarking
**Status: 📋 Planned**

We need standard ways to measure how good our translations are,
so we can compare models and track improvements over time.

### What needs to be built

- [ ] BLEU score (`lalango/evaluation/bleu.py`)
- [ ] chrF score (`lalango/evaluation/chrf.py`)
- [ ] Evaluation report generator
- [ ] Leaderboard per language pair

---

## Frontend

The web UI lives in `frontend/index.html` — a single HTML/CSS/JS file, no framework.

### Current features
- [x] Language pair selector (populated from `/languages` API)
- [x] Source text input with character counter
- [x] Translation output panel with copy button
- [x] API health status indicator
- [x] Supported languages grid
- [x] Ctrl+Enter keyboard shortcut

### Open frontend issues (good first issues!)
- [ ] Swap source ↔ target language button
- [ ] Translation history panel (last 5 translations)
- [ ] Dark / light mode toggle
- [ ] Character-by-character output animation
- [ ] Offline mode with helpful error screen

---

## Long-term ideas (open for discussion)

- Web UI for trying translations in a browser
- Model export to ONNX for fast inference
- Support for right-to-left scripts (Arabic, Urdu)
- Community dataset hosting

Have an idea? Open a [GitHub Discussion](https://github.com/wecncodecrew/La-Lango/discussions).
