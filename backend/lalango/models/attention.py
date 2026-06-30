# lalango/models/attention.py
#
# Bahdanau Attention  —  Phase 2
#
# Read docs/architecture.md (the "Phase 2" section) before diving in here.
#
# The problem attention solves:
#   In Phase 1, the encoder compresses the entire source sentence into
#   a single hidden state vector. For long sentences this loses information.
#
#   Attention gives the decoder a way to "look back" at all the encoder's
#   hidden states and decide which source words are most relevant
#   when generating each target word.
#
# How Bahdanau attention works (step by step):
#
#   For each decoding step t:
#     1. Score: compare the decoder's current hidden state with each encoder
#        hidden state to get a relevance score for each source position.
#        score(decoder_hidden, encoder_hidden_i) = tanh(W1 * decoder + W2 * encoder_i)
#
#     2. Weights: apply softmax to the scores to get a probability distribution.
#        These are the "attention weights" — they sum to 1.0.
#        A high weight on position i means "pay attention to source word i".
#
#     3. Context: compute a weighted sum of the encoder hidden states.
#        context = sum(weight_i * encoder_hidden_i)
#        This gives a single vector that blends all source information,
#        weighted by relevance.
#
#     4. Feed context + decoder hidden state into the output layer.
#
# Reference: "Neural Machine Translation by Jointly Learning to Align and Translate"
#             Bahdanau et al., 2015  https://arxiv.org/abs/1409.0473

import torch.nn as nn


class BahdanauAttention(nn.Module):
    """
    Bahdanau (additive) attention mechanism.

    This module computes attention weights and a context vector
    given the decoder's current hidden state and all encoder outputs.
    """

    def __init__(self, hidden_dim):
        """
        Args:
            hidden_dim (int): The hidden dimension of the LSTM.
                              Must match the hidden_dim used in the encoder and decoder.
        """
        super().__init__()

        # W1 projects the decoder hidden state into the attention space
        self.W1 = nn.Linear(hidden_dim, hidden_dim)

        # W2 projects each encoder hidden state into the attention space
        self.W2 = nn.Linear(hidden_dim, hidden_dim)

        # V collapses the combined projection to a single score per position
        self.V = nn.Linear(hidden_dim, 1)

    def forward(self, decoder_hidden, encoder_outputs):
        """
        Compute attention weights and context vector.

        Args:
            decoder_hidden (Tensor): Shape [batch_size, hidden_dim]
                                     The decoder's current hidden state.

            encoder_outputs (Tensor): Shape [batch_size, src_seq_len, hidden_dim]
                                      All hidden states from the encoder
                                      (one per source character).

        Returns:
            context (Tensor):         Shape [batch_size, hidden_dim]
                                      The weighted sum of encoder outputs.
            attention_weights (Tensor): Shape [batch_size, src_seq_len]
                                        The weight given to each source position.
                                        Useful for visualization.

        TODO (Phase 2):
            Step 1 — Score each encoder position:
                # decoder_hidden is [batch, hidden_dim]
                # We need it to be [batch, 1, hidden_dim] to broadcast across src_seq_len
                decoder_hidden = decoder_hidden.unsqueeze(1)

                # Compute the score for each encoder position
                # tanh( W1(decoder) + W2(encoder) )
                # Both terms broadcast to shape [batch, src_seq_len, hidden_dim]
                scores = self.V(torch.tanh(self.W1(decoder_hidden) + self.W2(encoder_outputs)))
                # scores shape: [batch, src_seq_len, 1]

            Step 2 — Convert scores to weights using softmax:
                # Squeeze the last dimension: [batch, src_seq_len]
                scores = scores.squeeze(2)
                attention_weights = torch.softmax(scores, dim=1)

            Step 3 — Compute the context vector:
                # Weighted sum: [batch, 1, src_seq_len] x [batch, src_seq_len, hidden_dim]
                # → [batch, 1, hidden_dim] → squeeze to [batch, hidden_dim]
                context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
                context = context.squeeze(1)

            Step 4 — Return context and attention_weights
        """
        # --- Your code here ---
        raise NotImplementedError(
            "BahdanauAttention.forward() is not implemented yet. "
            "This is a Phase 2 task. See the TODO above for step-by-step guidance."
        )
