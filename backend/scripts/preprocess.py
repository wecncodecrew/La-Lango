# scripts/preprocess.py
#
# Command-line script to preprocess a raw parallel corpus.
#
# This script takes two raw text files (source and target),
# cleans them, splits them into train/val/test sets,
# and saves the results in the standard format.
#
# Usage:
#   python scripts/preprocess.py \
#     --src data/raw/konkani-english/all.src \
#     --tgt data/raw/konkani-english/all.tgt \
#     --output data/processed/konkani-english/
#
# If you already have pre-split files:
#   python scripts/preprocess.py \
#     --src-train data/raw/train.src --tgt-train data/raw/train.tgt \
#     --src-val   data/raw/val.src   --tgt-val   data/raw/val.tgt \
#     --src-test  data/raw/test.src  --tgt-test  data/raw/test.tgt \
#     --output    data/processed/konkani-english/

import argparse
import os
import sys

# Make sure the project root is on the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lalango.data.cleaner import clean_parallel_corpus  # noqa: E402
from lalango.data.splitter import split_corpus  # noqa: E402


def save_split(sentences, filepath):
    """Write a list of sentences to a file, one per line."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for sentence in sentences:
            f.write(sentence + "\n")
    print(f"  Saved {len(sentences)} sentences → {filepath}")


def load_file(filepath):
    """Read a text file into a list of sentences."""
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        sys.exit(1)
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess a parallel corpus for La Lango AI training."
    )

    # Option 1: provide a single combined file and let the script split it
    parser.add_argument("--src", help="Path to the combined source language file.")
    parser.add_argument("--tgt", help="Path to the combined target language file.")

    # Option 2: provide pre-split files
    parser.add_argument("--src-train", help="Pre-split source train file.")
    parser.add_argument("--tgt-train", help="Pre-split target train file.")
    parser.add_argument("--src-val",   help="Pre-split source val file.")
    parser.add_argument("--tgt-val",   help="Pre-split target val file.")
    parser.add_argument("--src-test",  help="Pre-split source test file.")
    parser.add_argument("--tgt-test",  help="Pre-split target test file.")

    # Output
    parser.add_argument(
        "--output", required=True,
        help="Directory where cleaned and split files will be saved."
    )

    # Optional cleaning settings
    parser.add_argument(
        "--lowercase", action="store_true",
        help="Lowercase all text. Only use for case-insensitive languages."
    )
    parser.add_argument(
        "--max-length", type=int, default=200,
        help="Remove sentences longer than this many characters. Default: 200."
    )
    parser.add_argument(
        "--min-length", type=int, default=1,
        help="Remove sentences shorter than this many characters. Default: 1."
    )

    # Split ratios (only used when --src and --tgt are provided)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio",   type=float, default=0.1)
    parser.add_argument("--test-ratio",  type=float, default=0.1)

    args = parser.parse_args()

    # ---------------------------------------------------------------------------
    # Load the data
    # ---------------------------------------------------------------------------
    if args.src and args.tgt:
        # Single combined file → load, clean, then split
        print("\nLoading combined corpus...")
        print(f"  Source: {args.src}")
        print(f"  Target: {args.tgt}")

        source_sentences = load_file(args.src)
        target_sentences = load_file(args.tgt)

        print(f"\nCleaning {len(source_sentences)} sentence pairs...")
        source_sentences, target_sentences = clean_parallel_corpus(
            source_sentences,
            target_sentences,
            min_length=args.min_length,
            max_length=args.max_length,
        )

        print("\nSplitting into train/val/test...")
        splits = split_corpus(
            source_sentences,
            target_sentences,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )

        splits_to_save = {
            "train": (splits["train"]["source"], splits["train"]["target"]),
            "val":   (splits["val"]["source"],   splits["val"]["target"]),
            "test":  (splits["test"]["source"],  splits["test"]["target"]),
        }

    elif args.src_train and args.tgt_train:
        # Pre-split files → load each split separately and clean
        print("\nLoading pre-split corpus...")
        splits_to_save = {}
        for split_name, src_path, tgt_path in [
            ("train", args.src_train, args.tgt_train),
            ("val",   args.src_val,   args.tgt_val),
            ("test",  args.src_test,  args.tgt_test),
        ]:
            if src_path and tgt_path:
                print(f"  Loading {split_name}: {src_path}, {tgt_path}")
                src = load_file(src_path)
                tgt = load_file(tgt_path)
                src, tgt = clean_parallel_corpus(src, tgt, args.min_length, args.max_length)
                splits_to_save[split_name] = (src, tgt)
    else:
        print("Error: Provide either --src and --tgt, or --src-train/--tgt-train etc.")
        parser.print_help()
        sys.exit(1)

    # ---------------------------------------------------------------------------
    # Save the processed splits
    # ---------------------------------------------------------------------------
    print(f"\nSaving processed files to: {args.output}")
    for split_name, (src_sentences, tgt_sentences) in splits_to_save.items():
        save_split(src_sentences, os.path.join(args.output, f"{split_name}.src"))
        save_split(tgt_sentences, os.path.join(args.output, f"{split_name}.tgt"))

    print(f"\nPreprocessing complete! Your data is ready in: {args.output}")
    print("Next step: run scripts/train.py to train a model.")


if __name__ == "__main__":
    main()
