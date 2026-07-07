# Architecture Guide

This document explains how the translation models in La Lango AI work,
written for someone who has taken an introductory ML or Python course.

---

## The big picture: Seq2Seq

All our models follow the same basic idea called **Sequence to Sequence (Seq2Seq)**:

```
"How are you?" ──► [ENCODER] ──► [context vector] ──► [DECODER] ──► "Koso asa?"
```

- The **Encoder** reads the source sentence word by word and builds up a summary of it.
- The **Decoder** takes that summary and generates the translation word by word.

Think of it like a human translator:
- First they read the whole sentence (encoding)
- Then they write out the translation (decoding)

---

## Phase 1: LSTM Encoder-Decoder

An LSTM (Long Short-Term Memory) is a type of neural network that is good at
reading sequences. It has a "memory" that it updates as it reads each word.

```
Source:   "How    are    you   ?"
           │      │      │     │
           ▼      ▼      ▼     ▼
Encoder: [LSTM]─[LSTM]─[LSTM]─[LSTM]
                                 │
                          hidden state
                          (the "summary")
                                 │
                                 ▼
Decoder: [LSTM]─[LSTM]─[LSTM]─[LSTM]
           │      │      │      │
           ▼      ▼      ▼      ▼
Target:  "Koso"  "asa"   "?"   <end>
```

**The problem with this approach:**
The entire source sentence is compressed into a single vector (the hidden state).
For long sentences, this loses a lot of information.
This is why we add Attention in Phase 2.

---

## Phase 2: Bahdanau Attention

Attention lets the decoder look at every word in the source sentence
when generating each word of the translation.

Instead of relying only on the final hidden state, the decoder can say:
*"To generate this next word, I should pay more attention to word 3 and word 5
of the source sentence."*

```
Decoder generating word 2 of translation:
  Looks at source word 1 ──► attention weight: 0.1
  Looks at source word 2 ──► attention weight: 0.7  ← paying most attention here
  Looks at source word 3 ──► attention weight: 0.2
  Weighted sum of source words ──► context for this decoding step
```

---

## Phase 4: Transformer

The Transformer is the architecture behind modern translation systems.
Instead of processing words one at a time (like LSTM),
it looks at all words in a sentence simultaneously using **self-attention**.

We will build this in Phase 4. By then, the concepts from Phases 1–3
will make it much easier to understand.

---

## How tokenization fits in

Before any of this can happen, we need to turn text into numbers.
This is what the **tokenizer** does.

```
"How are you?" ──► tokenizer ──► [4, 12, 8, 2]
                                    │
                            these numbers are fed
                            into the Encoder
```

In Phase 1 we use **character-level tokenization** - each character gets a number.
In Phase 3 we upgrade to **BPE** - common subword chunks get their own numbers,
which gives the model a better vocabulary to work with.

---

## Where to read the code

| Concept              | File                               |
|----------------------|------------------------------------|
| Character tokenizer  | `backend/lalango/tokenizers/char_tokenizer.py` |
| LSTM model           | `backend/lalango/models/seq2seq_lstm.py`   |
| Attention            | `backend/lalango/models/attention.py`      |
| Transformer          | `backend/lalango/models/transformer.py`    |
| Training loop        | `backend/scripts/train.py`                 |

Start with `char_tokenizer.py` - it is the simplest file and
a great entry point into the codebase.
