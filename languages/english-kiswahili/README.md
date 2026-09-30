# English → Kiswahili

## About this language

**Language name:** Kiswahili (Standard/Sanifu)
**Also known as:** Swahili
**Region(s) spoken:** Kenya, Tanzania, Uganda, Rwanda, DRC, Burundi, Mozambique and across East and Central Africa
**Approximate number of speakers:** 200+ million (native and second language speakers)
**Language family:** Bantu (Niger-Congo)
**Writing system / script:** Latin alphabet, left to right

---

## Why this language needs machine translation

Kiswahili is one of Africa's most widely spoken languages and serves as a
lingua franca across East and Central Africa. Despite its reach, it remains
significantly underserved by mainstream AI translation tools. Communities
that rely on Kiswahili for education, healthcare, commerce and government
communication have little access to quality automated translation. Adding
Kiswahili to La Lango AI directly serves over 200 million people.

---

## Dataset

**Source of the parallel corpus:** QED English-Kiswahili parallel corpus subset.
**Corpus size:** 3,000 sentence pairs
**Domains covered:** Educational and conversational text
**License:** See corpus LICENSE file from the QED dataset distribution

This language pair uses a curated 3,000-pair subset, randomly sampled from the
full QED English-Kiswahili corpus (18,192 raw pairs), after filtering and
deduplication (see below). Using the full raw corpus directly will NOT match
the documented 3,000-pair size and will include broken/misaligned lines.

**How to obtain the data:**
Data is not committed to the repository (see data/README.md).

To prepare locally:

1. Download the QED English-Kiswahili corpus from OPUS
   (https://opus.nlpl.eu, QED corpus, en-sw language pair). This gives you
   `QED.en-swa.en` (18,192 lines) and `QED.en-swa.swa` (18,192 lines).
2. Filter and sample the corpus — do NOT use the raw files directly:
   - Drop any pair where either side is empty or under 5 characters.
   - Drop any pair where either side exceeds 180 characters (the raw corpus
     contains some badly-aligned lines with 100s–1000+ words from merged
     subtitle segments — these skew training if included).
   - Deduplicate exact-match pairs (the raw corpus has ~900+ duplicates).
   - Randomly sample 3,000 pairs from what remains (a fixed random seed is
     recommended for reproducibility).
3. Save the sampled pairs as:
   - data/raw/english-kiswahili/all.src (English sentences, one per line)
   - data/raw/english-kiswahili/all.tgt (Kiswahili translations, one per line)
4. Run the preprocessing script, which cleans and splits into train/val/test (80/10/10):
   - Windows: $env:PYTHONPATH="backend"
     python backend/scripts/preprocess.py --src data/raw/english-kiswahili/all.src --tgt data/raw/english-kiswahili/all.tgt --output data/processed/english-kiswahili/
   - Linux/Mac: PYTHONPATH=backend python backend/scripts/preprocess.py --src data/raw/english-kiswahili/all.src --tgt data/raw/english-kiswahili/all.tgt --output data/processed/english-kiswahili/

Expected result: "Cleaned corpus: kept 3000 pairs, skipped 0", split into
2,400 train / 300 val / 300 test pairs.

---

## Dataset (CSV, committed)

**File:** [`dataset.csv`](dataset.csv)
**Pairs:** 4,381 English → Kiswahili sentence pairs
**Source:** Tatoeba English–Kiswahili sentence links
**License:** CC BY 2.0 FR — see [ATTRIBUTION.md](ATTRIBUTION.md)
**Domains covered:** everyday conversation, short stories, education, proverbs

Tatoeba sentences are written and translated by native speakers, so every pair
here is human-made. That makes this file smaller than the QED corpus
(4,381 vs 18,192 raw pairs), but it is the one we are allowed to commit: QED is
distributed for research use only, so it stays a local-only option (documented
in the section above).

### Columns

Two columns, in the order the project uses for every language dataset:
English first, target language second.

| Column      | Meaning                        |
|-------------|--------------------------------|
| `english`   | English sentence               |
| `kiswahili` | Kiswahili (Sanifu) sentence    |

### How to load it

The preprocessing script reads the CSV directly — no need to split it by hand:

```bash
# Windows
$env:PYTHONPATH="backend"
python backend/scripts/preprocess.py --csv languages/english-kiswahili/dataset.csv --output data/processed/english-kiswahili/

# Linux/Mac
PYTHONPATH=backend python backend/scripts/preprocess.py --csv languages/english-kiswahili/dataset.csv --output data/processed/english-kiswahili/
```

Expected result: "Loaded 4381 sentence pairs" then "Cleaned corpus: kept 4373
pairs, skipped 8" — the 8 skipped pairs are longer than the default
`--max-length 200` — split into 3,498 train / 437 val / 438 test pairs.

No column flags are needed: the script reads `english` and then the next
column. If a CSV uses other names, point the script at them:

```bash
PYTHONPATH=backend python backend/scripts/preprocess.py \
  --csv my_dataset.csv --source-column english --target-column kiswahili \
  --output data/processed/my-pair/
```

### How this file was built (reproducible)

1. Download the Tatoeba per-language exports (all under
   `https://downloads.tatoeba.org/exports/per_language/`):
   - `eng/eng_sentences.tsv.bz2` — English sentences (`id`, `lang`, `text`)
   - `swh/swh_sentences.tsv.bz2` — Kiswahili sentences
   - `swh/swh-eng_links.tsv.bz2` — the links between them (4,403 links)
2. Join each link to both sentence tables.
3. Drop, in this order: links whose sentence id is missing (21), empty cells,
   sentences shorter than 2 or longer than 300 characters, sentences with no
   Latin letters, sentences containing Arabic/Cyrillic/Devanagari characters,
   sentences containing URLs, sentences containing bracketed annotations (1),
   pairs where both sides are identical, exact duplicate pairs, and pairs whose
   Kiswahili side is actually English.
   **Result: 4,381 pairs kept** out of 4,403 links, covering 4,368 unique
   English sentences (26 English sentences have two Kiswahili translations).
4. Text is stored Unicode-normalised (NFC) with collapsed whitespace and tidy
   punctuation spacing (`rafiki ; lakini` → `rafiki; lakini`). Casing and
   punctuation marks are preserved, so do not lowercase this corpus blindly.

---

## Linguistic notes for contributors

- Kiswahili uses spaces between words - tokenization is straightforward
- No tone marks or diacritics that affect meaning
- Language is agglutinative - verb roots take many prefixes and suffixes
  Example: "Nitakupenda" = "I will love you" (ni-ta-ku-penda)
- Standard Kiswahili (Sanifu) has consistent spelling - minimal variation
- This dataset covers Standard Kiswahili only, not Sheng or coastal dialects
- Noun class system - nouns belong to classes that affect agreement

---

## Known issues / limitations

- Corpus is currently a 3,000-pair QED subset (local only) plus the committed
  4,381-pair Tatoeba CSV - contributions welcome
- The committed CSV is conversational and literary, not technical, medical or
  legal text, so domain coverage is narrow
- 26 English sentences have more than one Kiswahili translation, which is fine
  for training but means the source side is not strictly unique
- Model not yet trained - Phase 1 implementation in progress

---

## Contact

**Contributors:**

- [@reuben-vitalis](https://github.com/reuben-vitalis) - QED recipe and language notes
- [@Johnnierad24](https://github.com/Johnnierad24) - Tatoeba `dataset.csv` and CSV loading support

If you are a native Kiswahili speaker and want to help evaluate translations
or contribute sentence pairs, please open a GitHub issue or reach out via
Discussions.