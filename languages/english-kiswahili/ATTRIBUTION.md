# Attribution

## dataset.csv

**Source:** Tatoeba — <https://tatoeba.org>

Every English sentence in `dataset.csv` is linked to its Kiswahili translation
through the Tatoeba sentence-link graph. The sentences were written and
translated by Tatoeba contributors, many of them native Kiswahili speakers.

**License:** [CC BY 2.0 FR](https://creativecommons.org/licenses/by/2.0/fr/)

**Attribution required by that license:**

> Tatoeba sentence data by Tatoeba contributors, licensed under
> CC BY 2.0 FR. <https://tatoeba.org>

`dataset.csv` holds two columns, `english` and `kiswahili`, in the project's
standard format, so it does not carry a per-row link. To trace any pair back to
tatoeba.org, paste the English sentence into the search box at
<https://tatoeba.org/en/sentences/search?from=eng&to=swh&query=...> and look for
the Kiswahili translation, or re-run the export steps below. Please keep this
credit if you redistribute this data or a subset of it.

**How the file was produced** (exports snapshot: 19 September 2026):

- <https://downloads.tatoeba.org/exports/per_language/eng/eng_sentences.tsv.bz2>
- <https://downloads.tatoeba.org/exports/per_language/swh/swh_sentences.tsv.bz2>
- <https://downloads.tatoeba.org/exports/per_language/swh/swh-eng_links.tsv.bz2>

The filters applied to those exports are listed in
[README.md](README.md#how-this-file-was-built-reproducible).

## QED (not committed)

The QED English–Kiswahili corpus recipe documented in [README.md](README.md)
is **not** included in this repository. QED is distributed by the Qatar
Computing Research Institute for research purposes only, so it stays a
local-only download. Do not commit QED text to this repository.
