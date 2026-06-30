# lalango/models/transformer.py
#
# Transformer Translation Model  —  Phase 4
#
# This is a Phase 4 contribution. Before implementing this:
#   - Make sure Phase 1 (LSTM) and Phase 2 (Attention) are complete
#   - Read docs/architecture.md (the Phase 4 section)
#   - Read the original paper: "Attention Is All You Need" https://arxiv.org/abs/1706.03762
#   - Check experiments/04_transformer_walkthrough.ipynb when it is available
#
# The Transformer replaces the sequential LSTM with self-attention,
# which can look at all positions in a sequence simultaneously.
# This makes it faster to train and better at capturing long-range dependencies.
#
# Key components you need to implement:
#   1. Positional Encoding    — since Transformers have no recurrence, we need to
#                               inject information about word order explicitly
#   2. Multi-Head Attention   — attend to multiple aspects of the sequence at once
#   3. Feed-Forward Layer     — a simple MLP applied to each position
#   4. Encoder Layer          — self-attention + feed-forward
#   5. Decoder Layer          — self-attention + cross-attention + feed-forward
#   6. Full Transformer       — stack of encoder and decoder layers
#
# This file contains the class skeletons. Fill them in as part of Phase 4.

import torch  # noqa: F401
import torch.nn as nn
import math  # noqa: F401


class PositionalEncoding(nn.Module):
    """
    Adds positional information to the token embeddings.

    Because the Transformer processes all tokens in parallel (no recurrence),
    it has no built-in sense of order. Positional encoding injects a pattern
    based on the position of each token in the sequence.

    The classic formulation uses sine and cosine functions of different frequencies.
    See the original paper (Section 3.5) for details.

    TODO (Phase 4):
        Implement the __init__ and forward methods.
        The forward method should add the positional encoding to the input embeddings.
    """

    def __init__(self, embed_dim, max_seq_len=512, dropout=0.1):
        super().__init__()
        # TODO (Phase 4): Create the positional encoding matrix using sin/cos
        # Hint: pe[pos][2i] = sin(pos / 10000^(2i/embed_dim))
        #        pe[pos][2i+1] = cos(pos / 10000^(2i/embed_dim))
        raise NotImplementedError(
            "PositionalEncoding.__init__ is a Phase 4 task."
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Shape [batch_size, seq_len, embed_dim]
        Returns:
            Tensor: Same shape — embeddings with positional encoding added.
        """
        # TODO (Phase 4): Add the positional encoding to x and return
        raise NotImplementedError("PositionalEncoding.forward is a Phase 4 task.")


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention.

    Instead of computing one attention function, we compute h attention
    functions in parallel (h = num_heads). Each head can attend to different
    aspects of the sequence. Their outputs are concatenated and projected.

    TODO (Phase 4):
        Implement __init__ and forward.
        Key parameters: embed_dim, num_heads
        Key operations: Q, K, V projections → scaled dot-product attention
                        → concat → output projection
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        # TODO (Phase 4): set up the Q, K, V projection layers and output projection
        raise NotImplementedError("MultiHeadAttention.__init__ is a Phase 4 task.")

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query (Tensor): Shape [batch, seq_len, embed_dim]
            key   (Tensor): Shape [batch, seq_len, embed_dim]
            value (Tensor): Shape [batch, seq_len, embed_dim]
            mask  (Tensor): Optional mask to prevent attending to certain positions
                            (used for padding and for the decoder's future mask).
        Returns:
            output (Tensor): Shape [batch, seq_len, embed_dim]
        """
        # TODO (Phase 4): implement scaled dot-product attention across multiple heads
        raise NotImplementedError("MultiHeadAttention.forward is a Phase 4 task.")


class TransformerEncoderLayer(nn.Module):
    """
    One layer of the Transformer Encoder.

    Structure:
        Self-Attention → Add & Norm → Feed-Forward → Add & Norm

    "Add & Norm" means: add the input (residual connection) then apply LayerNorm.
    Residual connections help gradients flow during training.

    TODO (Phase 4): Implement __init__ and forward using MultiHeadAttention above.
    """

    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        # TODO (Phase 4): set up self-attention, feed-forward, layer norms, dropout
        raise NotImplementedError("TransformerEncoderLayer.__init__ is a Phase 4 task.")

    def forward(self, x, src_mask=None):
        raise NotImplementedError("TransformerEncoderLayer.forward is a Phase 4 task.")


class TransformerDecoderLayer(nn.Module):
    """
    One layer of the Transformer Decoder.

    Structure:
        Masked Self-Attention → Add & Norm
        → Cross-Attention (attends to encoder output) → Add & Norm
        → Feed-Forward → Add & Norm

    The "masked" self-attention prevents the decoder from seeing future tokens.

    TODO (Phase 4): Implement __init__ and forward.
    """

    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        # TODO (Phase 4): set up masked self-attention, cross-attention, feed-forward, layer norms
        raise NotImplementedError("TransformerDecoderLayer.__init__ is a Phase 4 task.")

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        raise NotImplementedError("TransformerDecoderLayer.forward is a Phase 4 task.")


class Transformer(nn.Module):
    """
    The full Transformer model for sequence-to-sequence translation.

    Stacks multiple encoder and decoder layers.

    TODO (Phase 4): Implement __init__, forward, and translate.
    """

    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        embed_dim=256,
        num_heads=8,
        num_layers=3,
        ff_dim=512,
        dropout=0.1,
        max_seq_len=512,
    ):
        super().__init__()
        # TODO (Phase 4): set up source/target embeddings, positional encoding,
        # encoder layers, decoder layers, and output projection
        raise NotImplementedError("Transformer.__init__ is a Phase 4 task.")

    def forward(self, source, target, src_mask=None, tgt_mask=None):
        raise NotImplementedError("Transformer.forward is a Phase 4 task.")

    def translate(self, source_indices, sos_idx=2, eos_idx=3, max_length=100):
        raise NotImplementedError("Transformer.translate is a Phase 4 task.")
