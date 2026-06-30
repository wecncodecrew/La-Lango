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

**Source of the parallel corpus:** Community collected - everyday conversational Kiswahili (Sanifu). Initial corpus of 20 sentence pairs covering common greetings and phrases, manually curated by contributor.
**Corpus size:** 20 sentence pairs (initial — contributions welcome)
**Domains covered:** Everyday conversation, greetings, common phrases, basic needs
**License:** CC0 (Public Domain) — freely usable
**How to obtain the data:**
Data is not committed to the repository (see data/README.md).
To prepare locally:
1. Create the following files:
   - data/raw/english-kiswahili/train.src (English sentences, one per line)
   - data/raw/english-kiswahili/train.tgt (Kiswahili translations, one per line)
2. Run the preprocessing script:
   - Windows: $env:PYTHONPATH="backend"
     python backend/scripts/preprocess.py --src data/raw/english-kiswahili/train.src --tgt data/raw/english-kiswahili/train.tgt --output data/processed/english-kiswahili/
   - Linux/Mac: PYTHONPATH=backend python backend/scripts/preprocess.py --src data/raw/english-kiswahili/train.src --tgt data/raw/english-kiswahili/train.tgt --output data/processed/english-kiswahili/

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

- Parallel corpus is currently being collected - contributions welcome
- Model not yet trained - Phase 1 implementation in progress

---

## Contact

**Contributor:** [@reuben-vitalis](https://github.com/reuben-vitalis)

If you are a native Kiswahili speaker and want to help evaluate translations
or contribute sentence pairs, please open a GitHub issue or reach out via
Discussions.