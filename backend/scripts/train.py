# scripts/train.py
#
# Command-line script to train a translation model.
#
# Usage:
#   python scripts/train.py \
#     --lang-pair konkani-english \
#     --data data/processed/konkani-english/ \
#     --epochs 30
#
# The script will:
#   1. Load training and validation data
#   2. Build character vocabularies
#   3. Create the model
#   4. Run the training loop
#   5. Save the best checkpoint to checkpoints/<lang-pair>/

import argparse
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lalango.data.dataset import load_processed_dataset, encode_corpus, create_batches  # noqa: E402
from lalango.tokenizers.char_tokenizer import CharTokenizer  # noqa: E402

# We will import the model once Phase 1 is implemented.
# For now this import is commented out so the script runs cleanly.
# from lalango.models.seq2seq_lstm import Seq2SeqLSTM


def save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_dir, filename="best.pt"):
    """
    Save a model checkpoint to disk.

    We save both the model weights and the optimizer state.
    The optimizer state lets us resume training from this point if needed.

    Args:
        model: The PyTorch model.
        optimizer: The optimizer used during training.
        epoch (int): The current epoch number.
        val_loss (float): The validation loss at this checkpoint.
        checkpoint_dir (str): Directory to save the checkpoint in.
        filename (str): Filename for the checkpoint file.
    """
    # We import torch here (inside the function) so that the rest of
    # the script can run even if PyTorch is not installed.
    import torch
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
    }, filepath)
    print(f"  Checkpoint saved → {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Train a La Lango AI translation model.")

    parser.add_argument(
        "--lang-pair", required=True,
        help="Language pair identifier, e.g. 'konkani-english'. "
             "Must match a folder under data/processed/."
    )
    parser.add_argument(
        "--data", required=True,
        help="Path to the processed data directory (output of scripts/preprocess.py)."
    )
    parser.add_argument("--epochs",     type=int,   default=30,   help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int,   default=32,   help="Mini-batch size.")
    parser.add_argument("--embed-dim", type=int, default=128, help="Embedding dimension.")
    parser.add_argument("--hidden-dim", type=int, default=256, help="LSTM hidden dimension.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument(
        "--checkpoint-dir", default=None,
        help="Where to save model checkpoints. Defaults to checkpoints/<lang-pair>/."
    )

    args = parser.parse_args()

    if args.checkpoint_dir is None:
        args.checkpoint_dir = os.path.join("checkpoints", args.lang_pair)

    print("=" * 60)
    print("  La Lango AI — Training")
    print("=" * 60)
    print(f"  Language pair : {args.lang_pair}")
    print(f"  Data directory: {args.data}")
    print(f"  Epochs        : {args.epochs}")
    print(f"  Batch size    : {args.batch_size}")
    print(f"  Embedding dim : {args.embed_dim}")
    print(f"  Hidden dim    : {args.hidden_dim}")
    print(f"  Learning rate : {args.lr}")
    print(f"  Checkpoints   : {args.checkpoint_dir}")
    print("=" * 60 + "\n")

    # ---------------------------------------------------------------------------
    # Step 1: Load the data
    # ---------------------------------------------------------------------------
    print("Step 1: Loading data...")
    train_src, train_tgt = load_processed_dataset(args.data, split="train")
    val_src, val_tgt = load_processed_dataset(args.data, split="val")
    print(f"  Train: {len(train_src)} pairs")
    print(f"  Val:   {len(val_src)} pairs\n")

    # ---------------------------------------------------------------------------
    # Step 2: Build vocabularies
    # ---------------------------------------------------------------------------
    print("Step 2: Building vocabularies...")

    # Build source vocabulary from training source sentences only
    # (never use val or test data to build the vocabulary)
    src_tokenizer = CharTokenizer()
    src_tokenizer.build(train_src)

    # Build target vocabulary from training target sentences
    tgt_tokenizer = CharTokenizer()
    tgt_tokenizer.build(train_tgt)

    # Save the tokenizers so we can use them at inference time
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    with open(os.path.join(args.checkpoint_dir, "src_vocab.json"), "w") as f:
        json.dump(src_tokenizer.char_to_idx, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.checkpoint_dir, "tgt_vocab.json"), "w") as f:
        json.dump(tgt_tokenizer.char_to_idx, f, ensure_ascii=False, indent=2)

    print(f"  Source vocab size: {src_tokenizer.vocab_size}")
    print(f"  Target vocab size: {tgt_tokenizer.vocab_size}\n")

    # ---------------------------------------------------------------------------
    # Step 3: Encode and batch the data
    # ---------------------------------------------------------------------------
    print("Step 3: Encoding and batching data...")
    train_encoded = encode_corpus(train_src, train_tgt, src_tokenizer)
    val_encoded = encode_corpus(val_src, val_tgt, src_tokenizer)

    train_batches = create_batches(train_encoded, batch_size=args.batch_size)
    val_batches = create_batches(val_encoded, batch_size=args.batch_size)
    print(f"  Train batches: {len(train_batches)}")
    print(f"  Val batches:   {len(val_batches)}\n")

    # ---------------------------------------------------------------------------
    # Step 4: Create the model
    # ---------------------------------------------------------------------------
    # TODO (Phase 1):
    #   Uncomment the code below once Seq2SeqLSTM is implemented.
    #
    # import torch
    # import torch.nn as nn
    # from lalango.models.seq2seq_lstm import Seq2SeqLSTM
    #
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Step 4: Creating model (device: {device})...")
    #
    # model = Seq2SeqLSTM(
    #     src_vocab_size=src_tokenizer.vocab_size,
    #     tgt_vocab_size=tgt_tokenizer.vocab_size,
    #     embed_dim=args.embed_dim,
    #     hidden_dim=args.hidden_dim,
    # ).to(device)
    #
    # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"  Trainable parameters: {total_params:,}\n")

    print("Step 4: Skipping model creation (Seq2SeqLSTM not yet implemented).")
    print("  → Complete Phase 1 in lalango/models/seq2seq_lstm.py first.\n")

    # ---------------------------------------------------------------------------
    # Step 5: Training loop
    # ---------------------------------------------------------------------------
    # TODO (Phase 1):
    #   Uncomment the training loop below once the model is implemented.
    #
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    #
    # # CrossEntropyLoss measures how wrong the model's predictions are.
    # # ignore_index=0 means we do not compute loss for PAD tokens.
    # criterion = nn.CrossEntropyLoss(ignore_index=0)
    #
    # best_val_loss = float("inf")
    #
    # for epoch in range(1, args.epochs + 1):
    #     # --- Training ---
    #     model.train()
    #     train_loss = 0.0
    #
    #     for batch in train_batches:
    #         # Convert to tensors and move to device
    #         source = torch.tensor(batch["source"]).to(device)
    #         target = torch.tensor(batch["target"]).to(device)
    #
    #         # Zero the gradients from the previous step
    #         optimizer.zero_grad()
    #
    #         # Forward pass: get predictions for each step
    #         # predictions shape: [batch_size, tgt_seq_len, tgt_vocab_size]
    #         predictions = model(source, target)
    #
    #         # Reshape for loss calculation:
    #         #   predictions: [batch * tgt_seq_len, vocab_size]
    #         #   targets:     [batch * tgt_seq_len]
    #         # We skip the first token (SOS) in the target because the model
    #         # predicts what comes *after* each token.
    #         predictions = predictions[:, 1:, :].reshape(-1, tgt_tokenizer.vocab_size)
    #         targets = target[:, 1:].reshape(-1)
    #
    #         loss = criterion(predictions, targets)
    #
    #         # Backward pass: compute gradients
    #         loss.backward()
    #
    #         # Gradient clipping prevents exploding gradients (a common LSTM issue)
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    #
    #         # Update model weights
    #         optimizer.step()
    #
    #         train_loss += loss.item()
    #
    #     avg_train_loss = train_loss / len(train_batches)
    #
    #     # --- Validation ---
    #     model.eval()
    #     val_loss = 0.0
    #     with torch.no_grad():
    #         for batch in val_batches:
    #             source = torch.tensor(batch["source"]).to(device)
    #             target = torch.tensor(batch["target"]).to(device)
    #             predictions = model(source, target, teacher_forcing_ratio=0.0)
    #             predictions = predictions[:, 1:, :].reshape(-1, tgt_tokenizer.vocab_size)
    #             targets = target[:, 1:].reshape(-1)
    #             val_loss   += criterion(predictions, targets).item()
    #
    #     avg_val_loss = val_loss / len(val_batches)
    #
    #     print(f"Epoch {epoch:3d}/{args.epochs} | "
    #           f"Train loss: {avg_train_loss:.4f} | "
    #           f"Val loss: {avg_val_loss:.4f}")
    #
    #     # Save the model if it is the best so far
    #     if avg_val_loss < best_val_loss:
    #         best_val_loss = avg_val_loss
    #         save_checkpoint(model, optimizer, epoch, avg_val_loss, args.checkpoint_dir)
    #         print(f"  ↑ New best model saved.")
    #
    # print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    # print(f"Run scripts/evaluate.py to measure BLEU and chrF on the test set.")

    print("Training loop is ready and waiting for the model implementation.")
    print("Once Seq2SeqLSTM is implemented, uncomment the training loop above.")


if __name__ == "__main__":
    main()
