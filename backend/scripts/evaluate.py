# scripts/evaluate.py
#
# Command-line script to evaluate a trained model on the test set.
#
# Usage:
#   python scripts/evaluate.py \
#     --checkpoint checkpoints/konkani-english/ \
#     --data data/processed/konkani-english/
#
# This will print a report with BLEU and chrF scores
# and show a sample of translations side-by-side with the references.

import argparse
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lalango.data.dataset import load_processed_dataset  # noqa: E402
from lalango.tokenizers.char_tokenizer import CharTokenizer  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained La Lango AI model on the test set."
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to the checkpoint directory (produced by scripts/train.py)."
    )
    parser.add_argument(
        "--data", required=True,
        help="Path to the processed data directory."
    )
    parser.add_argument(
        "--split", default="test", choices=["train", "val", "test"],
        help="Which split to evaluate on. Always use 'test' for final results."
    )
    parser.add_argument(
        "--num-examples", type=int, default=5,
        help="How many example translations to show in the report."
    )
    parser.add_argument(
        "--lang-pair", default=None,
        help="Language pair name for the report (e.g. 'konkani-english'). "
             "Inferred from the data path if not provided."
    )

    args = parser.parse_args()

    # Infer lang-pair from data path if not provided
    if args.lang_pair is None:
        args.lang_pair = os.path.basename(args.data.rstrip("/"))

    # ---------------------------------------------------------------------------
    # Load test data
    # ---------------------------------------------------------------------------
    print(f"Loading {args.split} set from {args.data}...")
    test_src, test_tgt = load_processed_dataset(args.data, split=args.split)
    print(f"  Loaded {len(test_src)} sentence pairs.\n")

    # ---------------------------------------------------------------------------
    # Load tokenizers
    # ---------------------------------------------------------------------------
    src_vocab_path = os.path.join(args.checkpoint, "src_vocab.json")
    tgt_vocab_path = os.path.join(args.checkpoint, "tgt_vocab.json")

    if not os.path.exists(src_vocab_path):
        print(f"Error: Could not find {src_vocab_path}.")
        print("Make sure you have run scripts/train.py first.")
        sys.exit(1)

    print("Loading tokenizers...")
    with open(src_vocab_path, "r") as f:
        src_char_to_idx = json.load(f)
    with open(tgt_vocab_path, "r") as f:
        tgt_char_to_idx = json.load(f)

    src_tokenizer = CharTokenizer()
    src_tokenizer.char_to_idx = src_char_to_idx
    src_tokenizer.idx_to_char = {int(v): k for k, v in src_char_to_idx.items()}
    src_tokenizer.vocab_size = len(src_char_to_idx)

    tgt_tokenizer = CharTokenizer()
    tgt_tokenizer.char_to_idx = tgt_char_to_idx
    tgt_tokenizer.idx_to_char = {int(v): k for k, v in tgt_char_to_idx.items()}
    tgt_tokenizer.vocab_size = len(tgt_char_to_idx)

    # ---------------------------------------------------------------------------
    # Load model and generate translations
    # ---------------------------------------------------------------------------
    # TODO (Phase 1):
    #   Uncomment the code below once Seq2SeqLSTM and its translate() method
    #   are implemented.
    #
    # import torch
    # from lalango.models.seq2seq_lstm import Seq2SeqLSTM
    #
    # model_path = os.path.join(args.checkpoint, "best.pt")
    # checkpoint = torch.load(model_path, map_location="cpu")
    #
    # model = Seq2SeqLSTM(
    #     src_vocab_size=src_tokenizer.vocab_size,
    #     tgt_vocab_size=tgt_tokenizer.vocab_size,
    # )
    # model.load_state_dict(checkpoint["model_state_dict"])
    # model.eval()
    #
    # print("Generating translations...")
    # hypotheses = []
    # with torch.no_grad():
    #     for src_sentence in test_src:
    #         encoded = src_tokenizer.encode(src_sentence, add_sos=False, add_eos=True)
    #         source_tensor = torch.tensor([encoded])
    #         output_indices = model.translate(source_tensor)
    #         translation = tgt_tokenizer.decode(output_indices)
    #         hypotheses.append(translation)
    #
    # generate_report(hypotheses, test_tgt, args.lang_pair, "seq2seq_lstm", args.num_examples)

    print("Model not yet implemented.")
    print("Complete Phase 1 in lalango/models/seq2seq_lstm.py,")
    print("then train with scripts/train.py before evaluating.")


if __name__ == "__main__":
    main()
