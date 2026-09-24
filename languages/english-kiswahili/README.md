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

- Corpus is currently a 3,000-pair QED subset - contributions welcome
- Model not yet trained - Phase 1 implementation in progress

---

## Contact

**Contributor:** [@reuben-vitalis](https://github.com/reuben-vitalis)

If you are a native Kiswahili speaker and want to help evaluate translations
or contribute sentence pairs, please open a GitHub issue or reach out via
Discussions.