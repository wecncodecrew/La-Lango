# Evaluation Guide

How do we know if our translations are good?
This guide explains the metrics we use and how to run them.

---

## Why automatic metrics?

Having a human read every translation would be ideal, but it is slow and expensive.
Automatic metrics give us a fast way to compare models and track improvement over time.

They are not perfect - a high BLEU score does not always mean the translation sounds natural -
but they are a useful signal.

---

## BLEU Score

**BLEU (Bilingual Evaluation Understudy)** measures how many word sequences (n-grams)
in the predicted translation also appear in the reference (human) translation.

Score range: **0 to 100**. Higher is better.

| BLEU Score | What it roughly means                    |
|------------|------------------------------------------|
| < 10       | Almost no overlap with reference         |
| 10 – 20    | Some overlap, but translation is rough   |
| 20 – 40    | Decent translation, understandable       |
| > 40       | High quality (rare for low-resource)     |

**Important:** For low-resource languages, even a BLEU of 10–15 is a meaningful result.
Do not be discouraged by low scores early on.

---

## chrF Score

**chrF (Character n-gram F-score)** is similar to BLEU but works at the character level
instead of the word level.

This makes it better for:
- Languages with complex word forms (morphologically rich languages)
- Languages where spaces are not used to separate words

Score range: **0 to 100**. Higher is better.

---

## Running evaluation

```bash
PYTHONPATH=backend python backend/scripts/evaluate.py \
  --checkpoint checkpoints/konkani-english.pt \
  --data data/processed/konkani-english/test.json \
  --metrics bleu chrf
```

This will print a report like:

```
Language pair: konkani-english
Model:         seq2seq_lstm
Test set size: 500 sentences

BLEU:  12.4
chrF:  31.7

Example translations:
  Source:     Koso asa?
  Reference:  How are you?
  Predicted:  How are you doing?

  Source:     Hanv school ita.
  Reference:  I am going to school.
  Predicted:  I going to school.
```

---

## Interpreting results

- Compare your model against the **baseline** (the simple LSTM with no attention).
- If your BLEU score is higher than the baseline, your improvement is real.
- Always report results on the **test set**, never the training set.
  (If you evaluate on training data, you are cheating - the model has already seen it.)

---

## Where the metric code lives

- BLEU: `backend/lalango/evaluation/bleu.py`
- chrF: `backend/lalango/evaluation/chrf.py`
- Report generator: `backend/lalango/evaluation/report.py`

These are implemented from scratch with comments explaining the math.
Reading them is a great way to understand how evaluation works.
