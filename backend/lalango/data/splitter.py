# lalango/data/splitter.py
#
# Splits a parallel corpus into train, validation, and test sets.
#
# Why do we need three splits?
#
#   - train:  The model learns from this. It sees these sentences many times.
#   - val:    We check the model's progress on this during training.
#             The model never trains on it, but we peek at it to catch overfitting.
#   - test:   The final exam. We only evaluate on this ONCE, at the very end.
#             If you use the test set to tune your model, your results are misleading.

import random


def split_corpus(
    source_sentences,
    target_sentences,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    shuffle=True,
    seed=42,
):
    """
    Split a parallel corpus into train, validation, and test sets.

    Args:
        source_sentences (list of str): Source language sentences.
        target_sentences (list of str): Matching target language sentences.
        train_ratio (float): Fraction of data to use for training. Default: 0.8 (80%)
        val_ratio (float): Fraction for validation. Default: 0.1 (10%)
        test_ratio (float): Fraction for testing. Default: 0.1 (10%)
        shuffle (bool): Whether to shuffle before splitting. Default: True.
                        Always shuffle unless the order of sentences matters.
        seed (int): Random seed for reproducibility. Using the same seed
                    means you get the same split every time.

    Returns:
        dict with keys 'train', 'val', 'test'. Each value is a dict with
        keys 'source' and 'target'.

    Example:
        >>> splits = split_corpus(src_sentences, tgt_sentences)
        >>> len(splits['train']['source'])   # 80% of data
        >>> len(splits['val']['source'])     # 10% of data
        >>> len(splits['test']['source'])    # 10% of data
    """
    # Sanity check: ratios must add up to 1.0
    total = train_ratio + val_ratio + test_ratio
    assert abs(total - 1.0) < 1e-6, (
        f"train_ratio + val_ratio + test_ratio must equal 1.0, got {total:.2f}"
    )

    # Sanity check: same number of source and target sentences
    assert len(source_sentences) == len(target_sentences), (
        "source_sentences and target_sentences must have the same length"
    )

    total_count = len(source_sentences)

    # Pair up source and target so they stay aligned when we shuffle
    pairs = list(zip(source_sentences, target_sentences))

    # Shuffle the pairs (using a fixed seed so the split is reproducible)
    if shuffle:
        random.seed(seed)
        random.shuffle(pairs)

    # Calculate where to cut the list
    train_end = int(total_count * train_ratio)
    val_end = train_end + int(total_count * val_ratio)

    # Slice the list into three parts
    train_pairs = pairs[:train_end]
    val_pairs = pairs[train_end:val_end]
    test_pairs = pairs[val_end:]

    # Unzip the pairs back into separate source and target lists
    def unzip(pairs):
        if not pairs:
            return [], []
        sources, targets = zip(*pairs)
        return list(sources), list(targets)

    train_src, train_tgt = unzip(train_pairs)
    val_src, val_tgt = unzip(val_pairs)
    test_src, test_tgt = unzip(test_pairs)

    print("Split complete:")
    print(f"  Train: {len(train_src)} pairs ({train_ratio*100:.0f}%)")
    print(f"  Val:   {len(val_src)} pairs ({val_ratio*100:.0f}%)")
    print(f"  Test:  {len(test_src)} pairs ({test_ratio*100:.0f}%)")

    return {
        "train": {"source": train_src, "target": train_tgt},
        "val":   {"source": val_src,   "target": val_tgt},
        "test":  {"source": test_src,  "target": test_tgt},
    }
