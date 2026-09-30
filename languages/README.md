# Language Registry

This folder contains one subfolder for each language pair supported by La Lango AI.
Every language is contributed and maintained by community members.

---

## Currently supported languages

| Language pair | Status | Contributor | Corpus size |
|---------------|--------|-------------|-------------|
| English → Kiswahili (Sanifu) | In progress | reuben-vitalis, Johnnierad24 | 4,381 sentence pairs (`dataset.csv`, Tatoeba) + 3,000-pair QED recipe |

---

## Want to add a language?

Read the full guide: [docs/adding_a_language.md](../docs/adding_a_language.md)

Short version:
1. Open a GitHub issue using the "Add a new language" template
2. Fork the repo and create a folder: `languages/<source>-<target>/`
3. Add a `config.json` (copy from `example_dialect/`) and a `README.md`
4. Open a Pull Request with the label `data`

---

## Folder structure for each language

```
languages/
└── konkani-english/
    ├── config.json     ← language metadata (name, script, contributor, data source)
    └── README.md       ← background on the language and dataset
```

Data files (`.src`, `.tgt`) are **not** committed to this repo.
See `docs/data_format.md` for how to prepare your data locally.
