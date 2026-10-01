# backend/scripts/csv_to_parallel.py
#
# Convert a parallel corpus CSV into the .src / .tgt text files that
# preprocess.py expects.
#
# preprocess.py reads two plain-text files, one sentence per line, where line N
# of the source file is the translation of line N of the target file. Datasets
# in this repository are distributed as CSV, so this script bridges the two.
#
# Usage:
#   python backend/scripts/csv_to_parallel.py \
#     --csv languages/english-kiswahili/english-kiswahili-data.csv \
#     --output data/raw/english-kiswahili/
#
# That writes all.src and all.tgt into the output directory. Then run:
#   PYTHONPATH=backend python backend/scripts/preprocess.py \
#     --src data/raw/english-kiswahili/all.src \
#     --tgt data/raw/english-kiswahili/all.tgt \
#     --output data/processed/english-kiswahili/

import argparse
import csv
import os
import sys

# Team format (see sample_data.csv): english,<target language>
DEFAULT_SOURCE_COLUMN = "english"


def main():
    parser = argparse.ArgumentParser(
        description="Convert a parallel corpus CSV into .src / .tgt files."
    )
    parser.add_argument("--csv", required=True, help="Path to the input CSV file.")
    parser.add_argument(
        "--output", required=True,
        help="Directory to write all.src and all.tgt into."
    )
    parser.add_argument(
        "--source-column", default=DEFAULT_SOURCE_COLUMN,
        help=f"CSV column holding the source text. Default: {DEFAULT_SOURCE_COLUMN!r}"
    )
    parser.add_argument(
        "--target-column", default=None,
        help="CSV column holding the translation. Default: the first column "
             "that is not the source column."
    )
    parser.add_argument(
        "--prefix", default="all",
        help="Base name for the output files. Default: 'all' (all.src, all.tgt)."
    )
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"Error: File not found: {args.csv}")
        sys.exit(1)

    with open(args.csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            print(f"Error: {args.csv} appears to be empty.")
            sys.exit(1)

        if args.target_column is None:
            others = [c for c in reader.fieldnames if c != args.source_column]
            if not others:
                print(f"Error: no target column found in {reader.fieldnames}")
                sys.exit(1)
            args.target_column = others[0]
            print(f"  Using columns: source={args.source_column!r}, "
                  f"target={args.target_column!r}")

        missing = [
            column for column in (args.source_column, args.target_column)
            if column not in reader.fieldnames
        ]
        if missing:
            print(f"Error: column(s) not found in {args.csv}: {missing}")
            print(f"Columns present: {reader.fieldnames}")
            print("Use --source-column / --target-column if your CSV differs.")
            sys.exit(1)

        source_sentences = []
        target_sentences = []
        skipped = 0

        for row in reader:
            source = (row.get(args.source_column) or "").strip()
            target = (row.get(args.target_column) or "").strip()

            # A pair is only useful if both sides are present, and an embedded
            # newline would break the one-sentence-per-line contract.
            if not source or not target:
                skipped += 1
                continue

            source_sentences.append(" ".join(source.split()))
            target_sentences.append(" ".join(target.split()))

    os.makedirs(args.output, exist_ok=True)
    source_path = os.path.join(args.output, f"{args.prefix}.src")
    target_path = os.path.join(args.output, f"{args.prefix}.tgt")

    for path, sentences in ((source_path, source_sentences),
                            (target_path, target_sentences)):
        with open(path, "w", encoding="utf-8") as f:
            for sentence in sentences:
                f.write(sentence + "\n")
        print(f"  Saved {len(sentences)} sentences -> {path}")

    if skipped:
        print(f"  Skipped {skipped} row(s) where one side was empty.")

    print("\nNext step:")
    print("  PYTHONPATH=backend python backend/scripts/preprocess.py \\")
    print(f"    --src {source_path} --tgt {target_path} \\")
    print("    --output data/processed/<your-pair>/")


if __name__ == "__main__":
    main()
