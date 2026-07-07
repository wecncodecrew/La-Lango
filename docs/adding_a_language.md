# How to Add a New Language

Adding a new language is one of the most impactful contributions you can make.
This guide walks you through the exact steps.

---

## Step 1: Open an issue first

Before you start, open a GitHub issue using the "Add a new language" template.
This lets the community know you are working on it and avoids duplicate work.

---

## Step 2: Create your language folder

Create a new folder under `languages/` using this naming convention:

```
languages/<source-language>-<target-language>/
```

Examples:
```
languages/konkani-english/
languages/yoruba-english/
languages/tulu-kannada/
```

Use lowercase, with a hyphen between the two language names.

---

## Step 3: Add a config.json

Create `languages/<your-pair>/config.json` with the following fields:

```json
{
  "source_language": "Konkani",
  "target_language": "English",
  "source_iso_code": "kok",
  "target_iso_code": "eng",
  "source_script": "Latin",
  "target_script": "Latin",
  "direction": "ltr",
  "contributor": "your-github-username",
  "data_source": "Describe where the data comes from",
  "notes": "Any special notes about this language or dialect"
}
```

**direction** should be `"ltr"` (left-to-right) or `"rtl"` (right-to-left, e.g. Arabic, Urdu).

---

## Step 4: Add a README.md

Create `languages/<your-pair>/README.md` with some background on the language.
This helps future contributors understand the language better.

A good README includes:
- Where the language is spoken
- How many speakers it has
- Any interesting linguistic features (tones, complex morphology, special characters)
- Where your data came from
- How to reach you if someone has questions

See `languages/example_dialect/README.md` for a template.

---

## Step 5: Add your data

Follow the [data format guide](data_format.md) to prepare your parallel corpus,
then run the preprocessing script:

```bash
PYTHONPATH=backend python backend/scripts/preprocess.py \
  --src data/raw/<your-pair>/train.src \
  --tgt data/raw/<your-pair>/train.tgt \
  --output data/processed/<your-pair>/
```

---

## Step 6: Open a Pull Request

Once your folder is set up, open a PR with:
- The `languages/<your-pair>/` folder (config.json + README.md)
- The label `data` on your PR

You do not need to include the raw data files in the PR -
just the config and documentation. Point contributors to where
they can download the data in your README.

---

## Questions?

Comment on your issue or open a [GitHub Discussion](https://github.com/Wecncode/la-lango-ai/discussions).
