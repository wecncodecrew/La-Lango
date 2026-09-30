# Data Format Guide

This guide explains the exact format your data must be in before you can use
the preprocessing scripts or train a model.

---

## The core idea: parallel sentences

A parallel corpus is a collection of sentences in two languages where each sentence
in language A is matched with its translation in language B.

Example (Konkani → English):

```
Konkani:  Koso asa?
English:  How are you?

Konkani:  Mhaka boro feel zata.
English:  I am feeling good.
```

Every line in the source file corresponds to the same line in the target file.
Line 1 source = Line 1 target. Line 2 source = Line 2 target. And so on.

---

## Folder structure

Organize your data like this:

```
data/
└── raw/
    └── konkani-english/
        ├── train.src        ← source language sentences (one per line)
        ├── train.tgt        ← target language sentences (one per line)
        ├── val.src          ← validation split source
        ├── val.tgt          ← validation split target
        ├── test.src         ← test split source
        └── test.tgt         ← test split target
```

If you only have one file, the `preprocess.py` script will split it automatically.

---

## File format rules

1. **One sentence per line.** No blank lines in the middle.
2. **UTF-8 encoding.** This is important for non-Latin scripts.
3. **Same number of lines** in `.src` and `.tgt` files.
4. **No headers.** Just sentences, nothing else.

### Good example (train.src)
```
Koso asa?
Mhaka boro feel zata.
Hanv school ita.
```

### Good example (train.tgt)
```
How are you?
I am feeling good.
I am going to school.
```

---

## Recommended split sizes

| Split      | Purpose                              | Recommended size  |
|------------|--------------------------------------|-------------------|
| `train`    | The model learns from this           | 80% of your data  |
| `val`      | Check progress during training       | 10% of your data  |
| `test`     | Final evaluation (use only once!)    | 10% of your data  |

For low-resource languages, even 3,000–5,000 sentence pairs in `train` is enough to start.

---

## After formatting your data

Run the preprocessing script:

```bash
PYTHONPATH=backend python backend/scripts/preprocess.py \
  --src data/raw/konkani-english/train.src \
  --tgt data/raw/konkani-english/train.tgt \
  --output data/processed/konkani-english/
```

This will:
1. Clean the text (remove extra spaces, fix unicode issues)
2. Build a vocabulary from the training set
3. Save everything in a format the training script can read

---

## CSV datasets

Some datasets ship as a single table with one sentence pair per row instead of
two `.src`/`.tgt` files. `preprocess.py` reads those directly:

```bash
PYTHONPATH=backend python backend/scripts/preprocess.py \
  --csv languages/english-kiswahili/dataset.csv \
  --output data/processed/english-kiswahili/
```

### CSV rules

1. **Header row required**, naming the two languages: `english,<target language>`
   — for example `english,spanish` or `english,kiswahili`.
2. **One pair per row**, English first, target language second. Rows with an
   empty cell are skipped.
3. **The script finds the columns itself.** It reads the `english` column and
   uses the next column as the target language, so a two-column file needs no
   extra flags. For anything else, name the columns:
   `--source-column english --target-column kiswahili`.
4. **UTF-8 encoding.** Same as the text files.
5. **Extra columns are ignored**, so a file may carry provenance columns if you
   want them.

### Example

```csv
english,kiswahili
Are you sure?,"Je, una uhakika?"
You wanted to tell me about freedom?,Ulitaka kunieleza kuhusu uhuru?
```

Quote any cell that contains a comma, and use `--delimiter "\t"` for
tab-separated files.

---

## Privacy note

Please do not commit personal or private data to this repository.
Only contribute data that is publicly available or that you have the rights to share.
