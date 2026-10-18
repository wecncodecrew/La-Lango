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
import random
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lalango.data.dataset import load_processed_dataset, create_batches  # noqa: E402
from lalango.tokenizers.char_tokenizer import CharTokenizer, PAD_IDX  # noqa: E402


def encode_pairs(source_sentences, target_sentences, src_tokenizer, tgt_tokenizer):
    """
    Encode a parallel corpus, using the right vocabulary for each side.

    The source and target languages have different vocabularies, so each side
    must be encoded with its own tokenizer. Encoding the target with the source
    tokenizer would turn every target-only character into <UNK> and produce
    indices that do not line up with the decoder's output layer.

    Source sentences get EOS (the encoder needs to know where the sentence ends).
    Target sentences get SOS and EOS (the decoder needs to know where to start
    and stop).

    Returns:
        list of dict: Each item has 'source' and 'target' keys with encoded indices,
        in the same shape create_batches() expects.
    """
    return [
        {
            "source": src_tokenizer.encode(src, add_sos=False, add_eos=True),
            "target": tgt_tokenizer.encode(tgt, add_sos=True, add_eos=True),
        }
        for src, tgt in zip(source_sentences, target_sentences)
    ]


def compute_loss(model, batch, criterion, device, teacher_forcing_ratio):
    """
    Run the forward pass on one batch and return the loss.

    The model predicts the character that comes after each target position, so
    the first target token (SOS) is skipped when comparing predictions to labels.
    """
    import torch

    source = torch.tensor(batch["source"], dtype=torch.long, device=device)
    target = torch.tensor(batch["target"], dtype=torch.long, device=device)

    # predictions shape: [batch_size, tgt_seq_len, tgt_vocab_size]
    predictions = model(source, target, teacher_forcing_ratio=teacher_forcing_ratio)
    vocab_size = predictions.shape[-1]

    # Flatten for the loss:
    #   predictions: [batch * (tgt_seq_len - 1), vocab_size]
    #   targets:     [batch * (tgt_seq_len - 1)]
    predictions = predictions[:, 1:, :].reshape(-1, vocab_size)
    targets = target[:, 1:].reshape(-1)

    return criterion(predictions, targets)


def train_one_epoch(model, batches, optimizer, criterion, device,
                    teacher_forcing_ratio=0.5, clip=1.0):
    """
    Train the model for one pass over the training batches.

    For each batch: forward pass, loss calculation, backward pass, gradient
    clipping, and an optimizer step.

    Returns:
        float: The average training loss over all batches.
    """
    import torch

    model.train()
    total_loss = 0.0

    for batch in batches:
        # Clear the gradients left over from the previous step
        optimizer.zero_grad()

        # Forward pass + loss
        loss = compute_loss(model, batch, criterion, device, teacher_forcing_ratio)

        # Backward pass: compute gradients
        loss.backward()

        # Clipping stops gradients from exploding, a common problem with LSTMs
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)

        # Update the model's weights
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(batches), 1)


def evaluate(model, batches, criterion, device):
    """
    Measure the loss on a set of batches without updating the model.

    Teacher forcing is turned off, so the decoder relies on its own predictions,
    which is how it will be used at inference time.

    Returns:
        float: The average loss over all batches.
    """
    import torch

    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in batches:
            loss = compute_loss(model, batch, criterion, device, teacher_forcing_ratio=0.0)
            total_loss += loss.item()

    return total_loss / max(len(batches), 1)


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
        "--teacher-forcing", type=float, default=0.5,
        help="Probability of feeding the decoder the correct previous character "
             "during training (0.0 to 1.0)."
    )
    parser.add_argument("--clip", type=float, default=1.0, help="Gradient clipping max norm.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
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

    # Save the tokenizers so we can use them at inference time.
    # encoding="utf-8" matters on Windows, where the default encoding cannot
    # represent every character that appears in non-English text.
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    with open(os.path.join(args.checkpoint_dir, "src_vocab.json"), "w", encoding="utf-8") as f:
        json.dump(src_tokenizer.char_to_idx, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.checkpoint_dir, "tgt_vocab.json"), "w", encoding="utf-8") as f:
        json.dump(tgt_tokenizer.char_to_idx, f, ensure_ascii=False, indent=2)

    print(f"  Source vocab size: {src_tokenizer.vocab_size}")
    print(f"  Target vocab size: {tgt_tokenizer.vocab_size}\n")

    # ---------------------------------------------------------------------------
    # Step 3: Encode and batch the data
    # ---------------------------------------------------------------------------
    print("Step 3: Encoding and batching data...")
    # Each side is encoded with its own tokenizer (source with the source
    # vocabulary, target with the target vocabulary).
    train_encoded = encode_pairs(train_src, train_tgt, src_tokenizer, tgt_tokenizer)
    val_encoded = encode_pairs(val_src, val_tgt, src_tokenizer, tgt_tokenizer)

    train_batches = create_batches(train_encoded, batch_size=args.batch_size)
    val_batches = create_batches(val_encoded, batch_size=args.batch_size)
    print(f"  Train batches: {len(train_batches)}")
    print(f"  Val batches:   {len(val_batches)}\n")

    # ---------------------------------------------------------------------------
    # Step 4: Create the model
    # ---------------------------------------------------------------------------
    import torch
    import torch.nn as nn
    from lalango.models.seq2seq_lstm import Seq2SeqLSTM

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Step 4: Creating model (device: {device})...")

    model = Seq2SeqLSTM(
        src_vocab_size=src_tokenizer.vocab_size,
        tgt_vocab_size=tgt_tokenizer.vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {total_params:,}\n")

    # ---------------------------------------------------------------------------
    # Step 5: Training loop
    # ---------------------------------------------------------------------------
    print("Step 5: Training...")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # CrossEntropyLoss measures how wrong the model's predictions are.
    # ignore_index=PAD_IDX means we do not compute loss for PAD tokens.
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # Shuffle the order of the batches each epoch so the model does not
        # learn anything from the order the data happens to be in.
        random.shuffle(train_batches)

        avg_train_loss = train_one_epoch(
            model, train_batches, optimizer, criterion, device,
            teacher_forcing_ratio=args.teacher_forcing, clip=args.clip,
        )
        avg_val_loss = evaluate(model, val_batches, criterion, device)

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train loss: {avg_train_loss:.4f} | "
              f"Val loss: {avg_val_loss:.4f}")

        # Save the model if it is the best so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(model, optimizer, epoch, avg_val_loss, args.checkpoint_dir)
            print("  ↑ New best model saved.")

    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    print("Run scripts/evaluate.py to measure BLEU and chrF on the test set.")


if __name__ == "__main__":
    main()
