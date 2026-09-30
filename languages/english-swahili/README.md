# English → Swahili

Metadata and preparation instructions for `english_swahili_sentence_pairs_train.csv`.
This contribution registers a parallel training corpus; it does not include a trained model
or evaluation results. The accompanying `config.json` contains language and dataset metadata,
not model hyperparameters.

## Languages

English is the source language and Swahili (Kiswahili) is the target language.
Both use the Latin script and are written left to right. Swahili is a Bantu language
spoken in East Africa, including Kenya and Tanzania. Its noun classes and rich verb
morphology are relevant to tokenization. The dataset's specific dialect is unspecified.

## Dataset

| Property | Value |
| --- | --- |
| Local file | `data/raw/english-swahili/english_swahili_sentence_pairs_train.csv` |
| Format | UTF-8 CSV with a header |
| Source column | `English sentence` |
| Target column | `Swahili Translation` |
| Raw sentence pairs | 210,471 (excluding the header) |
| File size | 22,229,431 bytes |
| Supplied split | Training only |
| Dataset source | [Rogendo/English-Swahili-Sentence-Pairs on Hugging Face](https://huggingface.co/datasets/Rogendo/English-Swahili-Sentence-Pairs) |
| Dataset publisher | Rogendo |
| Contributor | Unspecified |
| Dataset license | Unspecified; the repository's code license does not establish the dataset's license |
| Domains / translation method | Unspecified |

Obtain the CSV from the linked Hugging Face dataset's **Files** tab and place it at
the local path above. The dataset page lists 210,471 training rows, matching the local
CSV count. The upstream dataset card is empty and does not specify a dataset license
or describe how the translations were collected.

## Observed quality issues

Inspection of the raw CSV found:

- 7,123 duplicate rows beyond their first occurrence.
- One empty English cell and one empty Swahili cell.
- 46 cells containing line breaks in each language column.
- Some entries are words or fragments rather than complete sentences.
- Maximum text lengths of 1,372 characters in English and 1,209 in Swahili.

The existing local `train.src` has 210,479 lines, while `train.tgt` has 210,471.
Regenerate aligned files from the CSV before training. These structural checks do not
verify translation accuracy; bilingual review and independent evaluation are still needed.

## Prepare aligned data

Run this from the repository root after obtaining the CSV. It parses quoted CSV fields,
collapses whitespace (including embedded line breaks), drops incomplete pairs together,
and removes duplicate normalized pairs. It writes new `all.src` and `all.tgt` files;
rerunning the command replaces those generated files.

```bash
python3 - <<'PY'
import csv
from pathlib import Path

folder = Path("data/raw/english-swahili")
seen = set()
with (folder / "english_swahili_sentence_pairs_train.csv").open(
    encoding="utf-8-sig", newline=""
) as csv_file, (folder / "all.src").open(
    "w", encoding="utf-8"
) as src_file, (folder / "all.tgt").open("w", encoding="utf-8") as tgt_file:
    for row in csv.DictReader(csv_file):
        source = " ".join(row["English sentence"].split())
        target = " ".join(row["Swahili Translation"].split())
        pair = (source, target)
        if not source or not target or pair in seen:
            continue
        seen.add(pair)
        src_file.write(source + "\n")
        tgt_file.write(target + "\n")
print(f"Exported {len(seen):,} aligned pairs")
PY
```

With the backend dependencies installed, create local training, validation, and test
splits from the exported corpus:

```bash
PYTHONPATH=backend python3 backend/scripts/preprocess.py \
  --src data/raw/english-swahili/all.src \
  --tgt data/raw/english-swahili/all.tgt \
  --output data/processed/english-swahili/ \
  --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
```

The preprocessor defaults to a maximum sentence length of 200 characters, so it will
remove longer pairs. Adjust `--max-length` if needed. Output counts will differ from
the raw CSV count after normalization, deduplication, and cleaning. These derived
splits are local evaluation splits, not official held-out benchmark data.

See the [data format guide](../../docs/data_format.md) for the expected `.src` / `.tgt`
layout and the [backend README](../../backend/README.md) for setup and training.

## Repository contribution

Commit this folder's `README.md` and `config.json` with the language registry update.
The repository excludes `data/raw/` and `data/processed/` from Git, so the CSV and
generated text files are not included in a normal commit. Use the Hugging Face source
linked above to obtain the dataset.

For corrections or bilingual review, open a repository issue mentioning
`english-swahili` and the affected CSV record.
