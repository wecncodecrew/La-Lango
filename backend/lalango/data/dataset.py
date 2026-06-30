# lalango/data/dataset.py
#
# Dataset loading and batching utilities.
#
# This file handles:
#   1. Loading a processed parallel corpus from disk
#   2. Encoding sentences using a tokenizer
#   3. Creating mini-batches for training
#
# A "batch" is a small group of sentence pairs that the model trains on at once.
# Instead of updating the model after every single sentence (slow),
# we process e.g. 32 sentences together (faster, more stable training).

import os


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
