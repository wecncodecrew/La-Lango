# tests/test_train.py
#
# Tests for the training loop in scripts/train.py.

import importlib.util
import os

import pytest
import torch
import torch.nn as nn

from lalango.data.dataset import create_batches
from lalango.models.seq2seq_lstm import Seq2SeqLSTM
from lalango.tokenizers.char_tokenizer import CharTokenizer, PAD_IDX, SOS_IDX, EOS_IDX

# scripts/ is not a package, so load train.py directly from its file path.
_TRAIN_PATH = os.path.join(os.path.dirname(__file__), "..", "scripts", "train.py")
_spec = importlib.util.spec_from_file_location("train", _TRAIN_PATH)
train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train)


@pytest.fixture
def setup(small_corpus):
    """A small model, optimizer, loss and batches built from the shared corpus."""
    source, target = small_corpus
    src_tok, tgt_tok = CharTokenizer(), CharTokenizer()
    src_tok.build(source)
    tgt_tok.build(target)

    pairs = train.encode_pairs(source, target, src_tok, tgt_tok)
    batches = create_batches(pairs, batch_size=2)

    torch.manual_seed(0)
    model = Seq2SeqLSTM(src_tok.vocab_size, tgt_tok.vocab_size, embed_dim=16, hidden_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    return model, optimizer, criterion, batches, src_tok, tgt_tok


def test_encode_pairs_uses_target_vocabulary(small_corpus):
    """Target sentences must be encoded with the target tokenizer, not the source one."""
    source, target = small_corpus
    src_tok, tgt_tok = CharTokenizer(), CharTokenizer()
    src_tok.build(source)
    tgt_tok.build(target)

    pairs = train.encode_pairs(source, target, src_tok, tgt_tok)

    assert pairs[0]["source"] == src_tok.encode(source[0], add_sos=False, add_eos=True)
    assert pairs[0]["target"] == tgt_tok.encode(target[0], add_sos=True, add_eos=True)
    # Every target index must fit inside the decoder's output layer.
    assert all(i < tgt_tok.vocab_size for p in pairs for i in p["target"])


def test_encode_pairs_adds_special_tokens(small_corpus):
    source, target = small_corpus
    src_tok, tgt_tok = CharTokenizer(), CharTokenizer()
    src_tok.build(source)
    tgt_tok.build(target)

    pair = train.encode_pairs(source, target, src_tok, tgt_tok)[0]
    assert pair["source"][-1] == EOS_IDX
    assert pair["target"][0] == SOS_IDX
    assert pair["target"][-1] == EOS_IDX


def test_train_one_epoch_returns_finite_loss(setup):
    model, optimizer, criterion, batches, _, _ = setup
    loss = train.train_one_epoch(model, batches, optimizer, criterion, torch.device("cpu"))
    assert isinstance(loss, float)
    assert loss > 0
    assert loss == loss  # not NaN


def test_train_one_epoch_updates_weights(setup):
    """The backward pass and optimizer step must actually change the weights."""
    model, optimizer, criterion, batches, _, _ = setup
    before = [p.detach().clone() for p in model.parameters()]
    train.train_one_epoch(model, batches, optimizer, criterion, torch.device("cpu"))
    after = list(model.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))


def test_training_reduces_loss(setup):
    """Training repeatedly on a tiny corpus should drive the loss down."""
    model, optimizer, criterion, batches, _, _ = setup
    device = torch.device("cpu")
    first = train.train_one_epoch(model, batches, optimizer, criterion, device,
                                  teacher_forcing_ratio=1.0)
    for _ in range(40):
        last = train.train_one_epoch(model, batches, optimizer, criterion, device,
                                     teacher_forcing_ratio=1.0)
    assert last < first * 0.5


def test_evaluate_does_not_change_weights(setup):
    model, _, criterion, batches, _, _ = setup
    before = [p.detach().clone() for p in model.parameters()]
    loss = train.evaluate(model, batches, criterion, torch.device("cpu"))
    assert loss > 0
    assert all(torch.equal(b, a) for b, a in zip(before, model.parameters()))


def test_save_checkpoint_writes_loadable_file(setup, tmp_path):
    model, optimizer, _, _, _, _ = setup
    train.save_checkpoint(model, optimizer, epoch=3, val_loss=1.5, checkpoint_dir=str(tmp_path))
    checkpoint = torch.load(tmp_path / "best.pt", weights_only=False)
    assert checkpoint["epoch"] == 3
    assert checkpoint["val_loss"] == 1.5
    model.load_state_dict(checkpoint["model_state_dict"])
