# lalango/data/dataset.py
#
# Dataset loading and batching utilities.
#
# This file handles:
#   1. Loading a parallel corpus from text files or from a CSV file
#   2. Encoding sentences using a tokenizer
#   3. Creating mini-batches for training
#
# A "batch" is a small group of sentence pairs that the model trains on at once.
# Instead of updating the model after every single sentence (slow),
# we process e.g. 32 sentences together (faster, more stable training).

import csv
import os


def load_corpus_from_csv(
    csv_file,
    source_column="english",
    target_column=None,
    delimiter=",",
):
    """
    Load a parallel corpus from a single CSV file.

    Many community datasets are published as a spreadsheet-like table with one
    sentence pair per row, so we read both languages from the same file instead
    of from two separate .src/.tgt files.

    The project-wide format is a header row plus two columns: the source
    language and the target language, named in the header, e.g.
    "english,kiswahili" or "english,spanish". Because every language names its
    target column differently, target_column defaults to the first column after
    the source column, so a two-column file loads with no extra arguments.

    The file must have a header row. Rows where either language is empty are
    skipped, and the remaining pairs keep their original order.

    Args:
        csv_file (str): Path to the CSV file.
        source_column (str): Name of the column holding the source sentences.
        target_column (str, optional): Name of the column holding the target
            sentences. Defaults to the first column that is not source_column.
        delimiter (str): Column separator. Use "\\t" for tab-separated files.

    Returns:
        tuple: (source_sentences, target_sentences) — two lists of strings.

    Raises:
        FileNotFoundError: If the file does not exist.
        KeyError: If the requested columns are not in the header, or if the
            file has no second column to use as the target language.

    Example:
        >>> src, tgt = load_corpus_from_csv(
        ...     "languages/english-kiswahili/dataset.csv"
        ... )
        >>> src[0]
        "Are you sure?"
        >>> tgt[0]
        "Je, una uhakika?"
    """
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Could not find {csv_file}.")

    source_sentences = []
    target_sentences = []
    skipped = 0

    with open(csv_file, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)

        # DictReader puts the header in fieldnames; fail early with a clear
        # message rather than silently producing an empty corpus.
        available_columns = reader.fieldnames or []

        if source_column not in available_columns:
            raise KeyError(
                f"Column '{source_column}' not found in {csv_file}. "
                f"Available columns: {', '.join(available_columns)}"
            )

        if target_column is None:
            # Two-column convention: the target language is simply the other
            # column, whatever it is named in the header.
            other_columns = [
                column for column in available_columns if column != source_column
            ]
            if not other_columns:
                raise KeyError(
                    f"{csv_file} only has a '{source_column}' column, so there is "
                    f"no target language column. Expected a header like "
                    f"'english,<target language>'."
                )
            target_column = other_columns[0]
        elif target_column not in available_columns:
            raise KeyError(
                f"Column '{target_column}' not found in {csv_file}. "
                f"Available columns: {', '.join(available_columns)}"
            )

        for row in reader:
            source = (row.get(source_column) or "").strip()
            target = (row.get(target_column) or "").strip()

            if not source or not target:
                skipped += 1
                continue

            source_sentences.append(source)
            target_sentences.append(target)

    print(f"Loaded {len(source_sentences)} sentence pairs from {csv_file}")
    if skipped:
        print(f"  Skipped {skipped} rows with an empty {source_column} or {target_column}")

    return source_sentences, target_sentences


def load_corpus_from_files(src_file, tgt_file):
    """
    Load a parallel corpus from two plain text files.

    Each file should have one sentence per line.
    Line N in the source file corresponds to line N in the target file.

    Args:
        src_file (str): Path to the source language file.
        tgt_file (str): Path to the target language file.

    Returns:
        tuple: (source_sentences, target_sentences) — two lists of strings.

    Example:
        >>> src, tgt = load_corpus_from_files("data/raw/train.src", "data/raw/train.tgt")
        >>> src[0]
        "How are you?"
        >>> tgt[0]
        "Koso asa?"
    """
    with open(src_file, "r", encoding="utf-8") as f:
        source_sentences = [line.strip() for line in f if line.strip()]

    with open(tgt_file, "r", encoding="utf-8") as f:
        target_sentences = [line.strip() for line in f if line.strip()]

    assert len(source_sentences) == len(target_sentences), (
        f"Source file has {len(source_sentences)} lines but "
        f"target file has {len(target_sentences)} lines. They must match."
    )

    print(f"Loaded {len(source_sentences)} sentence pairs")
    return source_sentences, target_sentences


def load_processed_dataset(data_dir, split="train"):
    """
    Load a processed dataset from the standard directory structure.

    Expects files named:
        <data_dir>/train.src  and  <data_dir>/train.tgt
        <data_dir>/val.src    and  <data_dir>/val.tgt
        <data_dir>/test.src   and  <data_dir>/test.tgt

    Args:
        data_dir (str): Path to the processed data folder.
        split (str): Which split to load — "train", "val", or "test".

    Returns:
        tuple: (source_sentences, target_sentences)
    """
    src_file = os.path.join(data_dir, f"{split}.src")
    tgt_file = os.path.join(data_dir, f"{split}.tgt")

    if not os.path.exists(src_file):
        raise FileNotFoundError(
            f"Could not find {src_file}. "
            f"Run scripts/preprocess.py first to prepare the data."
        )

    return load_corpus_from_files(src_file, tgt_file)


def encode_corpus(source_sentences, target_sentences, tokenizer):
    """
    Encode all sentences in a corpus using the given tokenizer.

    Source sentences get EOS added (the encoder needs to know the sentence ended).
    Target sentences get both SOS and EOS (the decoder needs to know where to start
    and stop).

    Args:
        source_sentences (list of str): The source language sentences.
        target_sentences (list of str): The target language sentences.
        tokenizer: Any tokenizer with an .encode() method (CharTokenizer or BPETokenizer).

    Returns:
        list of dict: Each item has 'source' and 'target' keys with encoded indices.
    """
    encoded_pairs = []

    for src, tgt in zip(source_sentences, target_sentences):
        encoded_pairs.append({
            # Source: just add EOS so the encoder knows the sentence ended
            "source": tokenizer.encode(src, add_sos=False, add_eos=True),
            # Target: add SOS at the start (decoder input) and EOS at the end (label)
            "target": tokenizer.encode(tgt, add_sos=True, add_eos=True),
        })

    return encoded_pairs


def create_batches(encoded_pairs, batch_size, pad_idx=0):
    """
    Group encoded sentence pairs into mini-batches for training.

    Sentences in each batch are padded to the same length.

    Args:
        encoded_pairs (list of dict): Output from encode_corpus().
        batch_size (int): How many sentence pairs per batch.
        pad_idx (int): The padding index (should match your tokenizer's PAD_IDX).

    Returns:
        list of dict: Each batch has 'source' and 'target' keys,
                      each containing a 2D list [batch_size x max_seq_len].

    Example:
        >>> batches = create_batches(encoded_pairs, batch_size=32)
        >>> len(batches[0]['source'])
        32
    """
    batches = []

    # Walk through the data in steps of batch_size
    for i in range(0, len(encoded_pairs), batch_size):
        batch_pairs = encoded_pairs[i: i + batch_size]

        # Collect all source sequences in this batch
        source_seqs = [pair["source"] for pair in batch_pairs]
        target_seqs = [pair["target"] for pair in batch_pairs]

        # Pad all source sequences to the same length
        max_src_len = max(len(s) for s in source_seqs)
        padded_source = [
            s + [pad_idx] * (max_src_len - len(s)) for s in source_seqs
        ]

        # Pad all target sequences to the same length
        max_tgt_len = max(len(t) for t in target_seqs)
        padded_target = [
            t + [pad_idx] * (max_tgt_len - len(t)) for t in target_seqs
        ]

        batches.append({
            "source": padded_source,   # shape: [batch_size, max_src_len]
            "target": padded_target,   # shape: [batch_size, max_tgt_len]
        })

    return batches
