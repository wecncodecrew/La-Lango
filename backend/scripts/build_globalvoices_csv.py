"""
Build an English-Kiswahili parallel CSV from the OPUS GlobalVoices corpus.

Source : GlobalVoices (via OPUS) — https://opus.nlpl.eu/GlobalVoices/
Licence: CC BY 3.0 (Global Voices) — attribution required, commercial use allowed.
         https://globalvoices.org/about/global-voices-attribution-policy/

Two ways to run it
------------------
1. Let the script download the corpus for you:

       python build_globalvoices_csv.py --download --target 5000 --lowercase \
           --out languages/english-kiswahili/english-kiswahili-data.csv

2. Download the Moses zip yourself from opus.nlpl.eu (GlobalVoices → en-sw →
   "moses"), unzip it, then point the script at the two text files:

       python build_globalvoices_csv.py \
           --en GlobalVoices.en-sw.en --sw GlobalVoices.en-sw.sw \
           --target 5000 --lowercase \
           --out languages/english-kiswahili/english-kiswahili-data.csv
"""

import argparse
import csv
import io
import random
import re
import sys
import unicodedata
import urllib.request
import zipfile
from pathlib import Path

# Team format (see sample_data.csv): two columns, english,<target language>
ENG_COL = "english"
SWA_COL = "kiswahili"

OPUS_ZIP = "https://object.pouta.csc.fi/OPUS-GlobalVoices/v2018q4/moses/en-sw.txt.zip"

# Collapse runs of whitespace; strip stray control characters.
WS = re.compile(r"\s+")
CTRL = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f]")
LATIN_WORD = re.compile(r"[A-Za-z']+")


def clean(text):
    """Normalise unicode, strip control chars, collapse whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = CTRL.sub(" ", text)
    return WS.sub(" ", text).strip()


def download_corpus():
    """Fetch the Moses-format zip from OPUS and return (en_lines, sw_lines)."""
    print(f"Downloading {OPUS_ZIP}")
    print("(this is ~10 MB; it may take a minute)")
    try:
        with urllib.request.urlopen(OPUS_ZIP, timeout=120) as resp:
            blob = resp.read()
    except Exception as exc:
        sys.exit(
            f"ERROR: download failed: {exc}\n\n"
            "Download it manually instead:\n"
            "  1. Go to https://opus.nlpl.eu/GlobalVoices/\n"
            "  2. Pick en-sw, download the 'moses' package\n"
            "  3. Unzip it, then rerun this script with --en and --sw\n"
        )

    en_lines = sw_lines = None
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        for name in zf.namelist():
            if name.endswith(".en"):
                en_lines = zf.read(name).decode("utf-8").splitlines()
            elif name.endswith(".sw"):
                sw_lines = zf.read(name).decode("utf-8").splitlines()

    if en_lines is None or sw_lines is None:
        sys.exit("ERROR: could not find .en / .sw files inside the downloaded zip.")

    return en_lines, sw_lines


def read_file(path):
    p = Path(path)
    if not p.exists():
        sys.exit(f"ERROR: file not found: {path}")
    return p.read_text(encoding="utf-8").splitlines()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true",
                    help="Fetch the corpus from OPUS instead of reading local files.")
    ap.add_argument("--en", help="Path to the English side (.en) if not downloading.")
    ap.add_argument("--sw", help="Path to the Swahili side (.sw) if not downloading.")
    ap.add_argument("--out", required=True, help="CSV file to write.")
    ap.add_argument("--target", type=int, default=5000,
                    help="How many pairs to keep (default 5000).")
    ap.add_argument("--min-words", type=int, default=3)
    ap.add_argument("--max-words", type=int, default=50)
    ap.add_argument("--max-ratio", type=float, default=2.0,
                    help="Reject pairs where one side is this many times longer.")
    ap.add_argument("--lowercase", action="store_true",
                    help="Lowercase both columns before writing (team format).")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed, so the sample is reproducible.")
    args = ap.parse_args()

    if args.download:
        en_lines, sw_lines = download_corpus()
    elif args.en and args.sw:
        en_lines, sw_lines = read_file(args.en), read_file(args.sw)
    else:
        sys.exit("ERROR: pass --download, or both --en and --sw.")

    if len(en_lines) != len(sw_lines):
        sys.exit(
            f"ERROR: the two sides are not aligned "
            f"({len(en_lines)} English vs {len(sw_lines)} Swahili lines)."
        )

    print(f"\nLoaded {len(en_lines)} raw pairs\n")

    stats = {
        "blank": 0,
        "too_short": 0,
        "too_long": 0,
        "ratio": 0,
        "identical": 0,
        "duplicate": 0,
    }

    seen = set()
    kept = []

    for eng_raw, swa_raw in zip(en_lines, sw_lines):
        eng, swa = clean(eng_raw), clean(swa_raw)

        if not eng or not swa:
            stats["blank"] += 1
            continue

        e_words = LATIN_WORD.findall(eng)
        s_words = LATIN_WORD.findall(swa)

        if len(e_words) < args.min_words or len(s_words) < args.min_words:
            stats["too_short"] += 1
            continue

        if len(e_words) > args.max_words or len(s_words) > args.max_words:
            stats["too_long"] += 1
            continue

        ratio = max(len(e_words), len(s_words)) / min(len(e_words), len(s_words))
        if ratio > args.max_ratio:
            stats["ratio"] += 1
            continue

        # Identical both sides means the "translation" is just a copy.
        if eng.casefold() == swa.casefold():
            stats["identical"] += 1
            continue

        key = eng.casefold()
        if key in seen:
            stats["duplicate"] += 1
            continue
        seen.add(key)

        kept.append((eng, swa))

    print("Filtered out:")
    for name, count in stats.items():
        print(f"  {name:<12} {count:>7}")
    print(f"\nSurviving pairs: {len(kept)}")

    if len(kept) < args.target:
        print(
            f"\nWARNING: only {len(kept)} pairs survived, fewer than the "
            f"{args.target} requested. Writing all of them.\n"
            "Loosen --max-ratio or --max-words if you need more."
        )
        sample = kept
    else:
        random.seed(args.seed)
        sample = random.sample(kept, args.target)
        print(f"Randomly sampled {args.target} (seed {args.seed})")

    # Keep the sample in a stable order so the file diffs predictably.
    if args.lowercase:
        sample = [(eng.lower(), swa.lower()) for eng, swa in sample]

    sample.sort()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Minimal quoting and LF line endings, matching sample_data.csv. Fields
    # containing commas or quotes are still quoted, as the CSV format requires.
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        w.writerow([ENG_COL, SWA_COL])
        w.writerows(sample)

    print(f"\nWrote {len(sample)} pairs to {out}")
    print(f"Columns: {ENG_COL},{SWA_COL}" + (" (lowercased)" if args.lowercase else ""))
    print("\nRemember: GlobalVoices is CC BY 3.0 — attribution is required.")


if __name__ == "__main__":
    main()
