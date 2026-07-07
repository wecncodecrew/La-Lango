# La Lango AI - Project Management

This document defines **what we are building, who does what, and when**. It
contains the Work Breakdown Structure (WBS), a Gantt chart, and per-phase
milestones with objectives and tasks.

It complements [`ROADMAP.md`](../ROADMAP.md) (the *what*) by adding the
*structure, sequencing, and tracking* layer on top.

> **Diagrams** use [Mermaid](https://mermaid.js.org/) and render on GitHub.
> **Dates** are planning placeholders - adjust the Gantt `section` dates to your
> actual sprint calendar. They are expressed as relative durations so they stay
> valid even if the start date shifts.

---

## 1. Goal Hierarchy

The project is organised into three levels:

- **General Task** - a phase of the project (maps to ROADMAP phases)
- **Specific Objective** - a concrete outcome within a phase
- **Task** - an actionable unit of work under an objective (PR-sized)

```mermaid
graph TD
    G["🎯 GOAL<br/>Working from-scratch translation<br/>platform for low-resource languages"]

    G --> P1["General Task 1<br/>Phase 1: Seq2Seq LSTM"]
    G --> P2["General Task 2<br/>Phase 2: Attention"]
    G --> P3["General Task 3<br/>Phase 3: BPE Tokenizer"]
    G --> P4["General Task 4<br/>Phase 4: Transformer"]
    G --> P5["General Task 5<br/>Phase 5: Evaluation"]
    G --> P6["General Task 6<br/>Platform: API, Frontend, Data, Infra"]
```

---

## 2. Work Breakdown Structure (WBS)

A hierarchical decomposition of all deliverables. IDs (e.g. `1.2.3`) are stable
references you can cite in issues and PRs.

```mermaid
graph LR
    Root["La Lango AI"]

    Root --> W1["1. Phase 1<br/>Seq2Seq LSTM"]
    Root --> W2["2. Phase 2<br/>Attention"]
    Root --> W3["3. Phase 3<br/>BPE Tokenizer"]
    Root --> W4["4. Phase 4<br/>Transformer"]
    Root --> W5["5. Phase 5<br/>Evaluation"]
    Root --> W6["6. Platform"]

    W1 --> W1a["1.1 Tokenization"]
    W1 --> W1b["1.2 Data Pipeline"]
    W1 --> W1c["1.3 Model"]
    W1 --> W1d["1.4 Training"]
    W1 --> W1e["1.5 API Integration"]

    W6 --> W6a["6.1 REST API"]
    W6 --> W6b["6.2 Frontend"]
    W6 --> W6c["6.3 Data & Languages"]
    W6 --> W6d["6.4 DevOps / CI"]
    W6 --> W6e["6.5 Documentation"]
```

### 2.1 WBS Dictionary (full decomposition)

| WBS ID | Deliverable | Key tasks | Owner (label) |
|--------|-------------|-----------|---------------|
| **1** | **Phase 1 - Seq2Seq LSTM** | | 🟠 model |
| 1.1 | Character tokenizer | encode/decode, special tokens, padding | 🔵 data |
| 1.2 | Data pipeline | `dataset.py` loader, `cleaner.py`, `splitter.py`, batching | 🔵 data |
| 1.3 | Model | `Encoder.forward`, `Decoder.forward_step`, `Seq2SeqLSTM.forward` | 🟠 model |
| 1.4 | Training loop | `train.py`, teacher forcing, checkpointing | 🟠 model |
| 1.5 | API integration | wire `translate()` into `routes.py`, greedy decode | 🟠 model |
| **2** | **Phase 2 - Attention** | | 🟠 model |
| 2.1 | Attention score calc | `attention.py` Bahdanau additive attention | 🟠 model |
| 2.2 | Decoder integration | connect attention into `seq2seq_lstm.py` | 🟠 model |
| 2.3 | Attention visualization | heatmap in experiments notebook | 🔴 research |
| **3** | **Phase 3 - BPE Tokenizer** | | 🔵 data |
| 3.1 | BPE vocab builder | `bpe_tokenizer.py` merge-rule learning | 🔵 data |
| 3.2 | Encode/decode | subword encode + decode round-trip | 🔵 data |
| 3.3 | Vocab build script | CLI to build vocab from corpus | 🔵 data |
| **4** | **Phase 4 - Transformer** | | 🟠 model |
| 4.1 | Positional encoding | `PositionalEncoding` in `transformer.py` | 🟠 model |
| 4.2 | Multi-head attention | `MultiHeadAttention` | 🟠 model |
| 4.3 | Encoder/Decoder stacks | full transformer assembly | 🟠 model |
| 4.4 | Training w/ warmup | LR schedule, integrate into `train.py` | 🟠 model |
| **5** | **Phase 5 - Evaluation** | | 🔴 research |
| 5.1 | BLEU | `bleu.py` n-gram precision + brevity penalty | 🔴 research |
| 5.2 | chrF | `chrf.py` char n-gram F-score | 🔴 research |
| 5.3 | Report generator | `report.py` side-by-side + scores | 🔴 research |
| 5.4 | Leaderboard | per-language-pair scoreboard | 🔴 research |
| **6** | **Platform (cross-cutting)** | | mixed |
| 6.1 | REST API | `/translate`, `/languages`, `/health`, model registry | 🟠 model |
| 6.2 | Frontend | swap button, history, dark mode, animations | 🟢 good-first |
| 6.3 | Data & languages | Kiswahili + additional language registrations | 🔵 data |
| 6.4 | DevOps / CI | flake8+pytest CI ✅, hosting, model artifact storage | 🟢/🟠 |
| 6.5 | Documentation | system design ✅, PM ✅, per-phase guides | 🟢 good-first |

---

## 3. Gantt Chart

The plan runs from **6 July 2026 to early September 2026 (~2 months)**. To keep
the schedule realistic within that window, Phase 1 comes first, then **Phases 2,
3, and 5 run in parallel** (they only need the Phase 1 baseline), Phase 4 follows
Phase 2, and the **Platform track runs alongside throughout**. Adjust the start
anchors (`2026-07-06`, etc.) if your kickoff date shifts.

```mermaid
%%{init: {"gantt": {"barHeight": 24, "barGap": 6, "topPadding": 60, "leftPadding": 170, "rightPadding": 60, "fontSize": 12, "sectionFontSize": 13, "gridLineStartPadding": 40}}}%%
gantt
    title La Lango AI - Delivery Timeline
    dateFormat  YYYY-MM-DD
    axisFormat  %b %d

    section Phase 1 - LSTM
    Tokenizer             :done,    p11, 2026-07-06, 3d
    Data pipeline         :done,    p12, after p11, 4d
    Model enc/dec         :active,  p13, after p12, 6d
    Training loop         :         p14, after p13, 5d
    API integration       :         p15, after p14, 4d
    M1                    :milestone, m1, after p15, 0d

    section Phase 2 - Attention
    Attention calc        :         p21, after m1, 5d
    Decoder integration   :         p22, after p21, 5d
    Attn visualization    :         p23, after p22, 3d
    M2                    :milestone, m2, after p23, 0d

    section Phase 3 - BPE
    Vocab builder         :         p31, after m1, 5d
    Encode / decode       :         p32, after p31, 4d
    Build script          :         p33, after p32, 3d
    M3                    :milestone, m3, after p33, 0d

    section Phase 5 - Eval
    BLEU metric           :         p51, after m1, 4d
    chrF metric           :         p52, after p51, 4d
    Report generator      :         p53, after p52, 4d
    M5a                   :milestone, m5a, after p53, 0d

    section Phase 4 - Transformer
    Positional encoding   :         p41, after m2, 4d
    Multi-head attention  :         p42, after p41, 6d
    Enc / Dec stacks      :         p43, after p42, 6d
    Training + warmup     :         p44, after p43, 5d
    M4                    :milestone, m4, after p44, 0d

    section Phase 5 - Board
    Leaderboard           :         p54, after m4, 5d
    M5                    :milestone, m5, after p54, 0d

    section Platform
    CI flake8 + pytest    :done,    pf1, 2026-07-06, 3d
    Kiswahili language    :active,  pf2, 2026-07-09, 5d
    Docs (design + PM)    :active,  pf4, 2026-07-09, 4d
    Frontend upgrades     :         pf3, after pf2, 10d
    Hosting / deploy      :         pf5, after m1, 8d
```

---

## 4. Milestones

Each milestone is a **verifiable, demoable outcome** - the definition of "done"
for a phase.

```mermaid
graph LR
    M1["M1 - Phase 1 complete<br/>End-to-end translation<br/>via API (rough quality)"]
    M2["M2 - Phase 2 complete<br/>Attention improves<br/>long-sentence translation"]
    M3["M3 - Phase 3 complete<br/>BPE shortens sequences,<br/>handles rare words"]
    M4["M4 - Phase 4 complete<br/>Transformer beats<br/>LSTM baseline"]
    M5["M5 - Phase 5 complete<br/>BLEU/chrF leaderboard<br/>per language pair"]
    M1 --> M2 --> M3 --> M4 --> M5
```

| ID | Milestone | Definition of Done | Verification |
|----|-----------|--------------------|--------------|
| **M1** | Phase 1 - Working baseline | `train.py` trains on a corpus; `/translate` returns a real (rough) translation | Manual demo + `test_seq2seq_lstm.py` passing |
| **M2** | Phase 2 - Attention | Attention wired into decoder; measurable BLEU gain on long sentences | Eval comparison M1 vs M2 |
| **M3** | Phase 3 - BPE | BPE encode/decode round-trips; shorter sequences than char-level | Unit tests + sequence-length comparison |
| **M4** | Phase 4 - Transformer | Transformer trains and outperforms LSTM baseline | Leaderboard entry beats M1/M2 |
| **M5** | Phase 5 - Evaluation | Automated BLEU + chrF report and per-pair leaderboard | Report artifact generated in CI or script |

---

## 5. Task Board Mapping (per phase)

Recommended issue labels ↔ WBS mapping so the board stays organised.

| Label | Meaning | Typical WBS areas |
|-------|---------|-------------------|
| 🟢 `good-first-issue` | First-timers, docs, small fixes | 6.2, 6.5 |
| 🔵 `data` | Data cleaning, tokenizers, preprocessing | 1.1, 1.2, 3.x, 6.3 |
| 🟠 `model` | Models and training | 1.3, 1.4, 1.5, 2.x, 4.x, 6.1 |
| 🔴 `research` | Metrics, evaluation, analysis | 2.3, 5.x |

---

## 6. Current Status Snapshot

> Update this table at the end of each sprint.

| Track | Status | Notes |
|-------|--------|-------|
| Phase 1 - LSTM | 🚧 In progress | Model impl in review (PR #5); tokenizer + data pipeline present |
| Phase 2 - Attention | 📋 Planned | Skeleton in `attention.py` |
| Phase 3 - BPE | 📋 Planned | `bpe_tokenizer.py` to be created |
| Phase 4 - Transformer | 📋 Planned | Skeletons in `transformer.py` |
| Phase 5 - Evaluation | 🟡 Partial | `bleu.py`/`chrf.py` exist with tests |
| Platform - CI | ✅ Done | flake8 + pytest passing |
| Platform - Kiswahili | 🚧 In progress | Language registration (PR #2) |
| Platform - Docs | 🚧 In progress | This document + system design added |
| Platform - Hosting | 📋 Planned | Not yet set up |

---

## 7. Dependencies & Critical Path

```mermaid
graph LR
    P1["Phase 1<br/>LSTM baseline"] --> P2["Phase 2<br/>Attention"]
    P1 --> P3["Phase 3<br/>BPE"]
    P1 --> P5["Phase 5<br/>Evaluation"]
    P2 --> P4["Phase 4<br/>Transformer"]
    P3 --> P4
    P4 --> LB["Leaderboard<br/>(5.4)"]
    P5 --> LB

    Data["Language data<br/>(6.3)"] --> P1
    CI["CI green<br/>(6.4)"] --> P1
```

**Critical path:** `Data + CI → Phase 1 (M1) → Phase 2 (M2) + Phase 3 (M3) →
Phase 4 (M4) → Leaderboard (M5)`.

Phase 1 is the bottleneck - everything depends on a working baseline, so it gets
priority.

---

## 8. Related Documents

- [`ROADMAP.md`](../ROADMAP.md) - phase descriptions and "done" criteria
- [`system_design.md`](system_design.md) - architecture, data flow, flowcharts
- [`CONTRIBUTING.md`](../CONTRIBUTING.md) - how to pick up and submit tasks
