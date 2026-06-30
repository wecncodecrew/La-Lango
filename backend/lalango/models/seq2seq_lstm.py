# lalango/models/seq2seq_lstm.py
#
# Sequence-to-Sequence LSTM Translation Model  —  Phase 1
#
# This file contains the core translation model.
# Read docs/architecture.md first for a plain-English explanation of how it works.
#
# There are two parts to this model:
#   1. Encoder — reads the source sentence and builds a summary (hidden state)
#   2. Decoder — takes that summary and generates the translation word by word
#
# This file uses PyTorch. If you want to implement in TensorFlow or JAX,
# create a new file (e.g. seq2seq_lstm_tf.py) following the same structure.
#
# To run this model end-to-end, see scripts/train.py.

import torch  # noqa: F401
import torch.nn as nn


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """
    The Encoder reads the source sentence character by character and
    compresses it into a fixed-size vector called the "hidden state".

    Think of it as the model "understanding" the source sentence before
    attempting to translate it.

    Architecture:
        Embedding layer  →  maps character indices to dense vectors
        LSTM layer       →  reads the vectors one by one, updating its memory
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim):
        """
        Args:
            vocab_size (int): Number of unique characters in the source vocabulary.
            embed_dim (int): Size of the character embedding vectors.
                             A good starting value: 64 or 128.
            hidden_dim (int): Size of the LSTM's hidden state (its "memory").
                              A good starting value: 256.
        """
        super().__init__()

        # The embedding layer turns character indices (integers) into dense vectors.
        # e.g. index 5 → [0.2, -0.1, 0.8, ...]  (a vector of length embed_dim)
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0  # PAD tokens will have zero gradient
        )

        # The LSTM reads the sequence of embedding vectors.
        # At each step it updates its hidden state based on:
        #   - the current input vector
        #   - its previous hidden state
        # batch_first=True means our tensors are shaped [batch, seq_len, features]
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

    def forward(self, source_indices):
        """
        Run the encoder on a batch of source sentences.

        Args:
            source_indices (Tensor): Shape [batch_size, seq_len]
                                     Each row is a source sentence encoded as integers.

        Returns:
            hidden (Tensor): Shape [1, batch_size, hidden_dim]
                             The final hidden state — the "summary" of the sentence.
            cell (Tensor):   Shape [1, batch_size, hidden_dim]
                             The LSTM cell state (internal memory). Passed to the decoder.

        TODO (Phase 1):
            The structure is already set up. Your task:
            1. Pass source_indices through self.embedding to get embedded vectors
               Shape goes from [batch, seq_len] → [batch, seq_len, embed_dim]
            2. Pass the embedded vectors through self.lstm
               The LSTM returns: (all_outputs, (hidden, cell))
               We only need hidden and cell — we do not use all_outputs in Phase 1
            3. Return hidden, cell

            Hint: embedded = self.embedding(source_indices)
                  outputs, (hidden, cell) = self.lstm(embedded)
        """
        # --- Your code here ---
        raise NotImplementedError(
            "Encoder.forward() is not implemented yet. "
            "See the TODO above for step-by-step guidance."
        )


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class Decoder(nn.Module):
    """
    The Decoder takes the encoder's hidden state and generates the translation
    one character at a time.

    At each step, it:
        1. Takes the previously generated character as input
        2. Updates its hidden state
        3. Predicts the probability of each character in the target vocabulary

    The first input is always the SOS (Start Of Sentence) token.
    It stops when it predicts the EOS (End Of Sentence) token.
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim):
        """
        Args:
            vocab_size (int): Number of unique characters in the TARGET vocabulary.
            embed_dim (int): Size of character embeddings.
            hidden_dim (int): Must match the encoder's hidden_dim.
        """
        super().__init__()

        # Embedding for target characters
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0
        )

        # The LSTM that generates the translation step by step
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        # A linear layer that maps the hidden state to a probability
        # distribution over all characters in the target vocabulary.
        # e.g. hidden state (size 256) → scores for each of 80 characters
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def forward_step(self, input_token, hidden, cell):
        """
        Run the decoder for ONE step (one character at a time).

        Args:
            input_token (Tensor): Shape [batch_size]
                                  The character predicted in the previous step.
                                  For the first step, this is the SOS token.
            hidden (Tensor): Shape [1, batch_size, hidden_dim]
            cell (Tensor):   Shape [1, batch_size, hidden_dim]

        Returns:
            prediction (Tensor): Shape [batch_size, vocab_size]
                                  Raw scores for each character in the vocabulary.
                                  Use argmax to get the predicted character index.
            hidden (Tensor): Updated hidden state.
            cell (Tensor):   Updated cell state.

        TODO (Phase 1):
            1. input_token is shape [batch_size]. Add a dimension to make it
               [batch_size, 1] so the LSTM sees it as a sequence of length 1.
               Hint: input_token = input_token.unsqueeze(1)
            2. Pass through self.embedding → shape [batch_size, 1, embed_dim]
            3. Pass through self.lstm with (hidden, cell)
               Returns: (output, (new_hidden, new_cell))
               output shape: [batch_size, 1, hidden_dim]
            4. Squeeze output to [batch_size, hidden_dim]
               Hint: output = output.squeeze(1)
            5. Pass through self.output_layer → shape [batch_size, vocab_size]
            6. Return prediction, new_hidden, new_cell
        """
        # --- Your code here ---
        raise NotImplementedError(
            "Decoder.forward_step() is not implemented yet. "
            "See the TODO above for step-by-step guidance."
        )


# ---------------------------------------------------------------------------
# Full Seq2Seq model
# ---------------------------------------------------------------------------

class Seq2SeqLSTM(nn.Module):
    """
    The complete Seq2Seq model: Encoder + Decoder working together.

    This is what you train and what the API calls to generate translations.
    """

    def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim=128, hidden_dim=256):
        """
        Args:
            src_vocab_size (int): Source vocabulary size (from your source tokenizer).
            tgt_vocab_size (int): Target vocabulary size (from your target tokenizer).
            embed_dim (int): Embedding dimension. Default: 128.
            hidden_dim (int): LSTM hidden dimension. Default: 256.
        """
        super().__init__()

        self.encoder = Encoder(src_vocab_size, embed_dim, hidden_dim)
        self.decoder = Decoder(tgt_vocab_size, embed_dim, hidden_dim)

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        """
        Run a full forward pass through the model during training.

        Teacher forcing: During training, we sometimes feed the decoder the
        correct previous character (from the reference translation) instead of
        its own previous prediction. This helps training converge faster.
        teacher_forcing_ratio=0.5 means we do this 50% of the time.

        Args:
            source (Tensor): Shape [batch_size, src_seq_len] — encoded source sentences.
            target (Tensor): Shape [batch_size, tgt_seq_len] — encoded target sentences.
            teacher_forcing_ratio (float): How often to use teacher forcing (0.0 to 1.0).

        Returns:
            all_predictions (Tensor): Shape [batch_size, tgt_seq_len, tgt_vocab_size]
                                      The raw scores at each decoding step.

        TODO (Phase 1):
            1. Run the encoder:
               hidden, cell = self.encoder(source)

            2. Prepare to collect outputs:
               tgt_seq_len = target.shape[1]
               batch_size = target.shape[0]
               tgt_vocab_size = self.decoder.output_layer.out_features
               all_predictions = torch.zeros(batch_size, tgt_seq_len, tgt_vocab_size)

            3. The first input to the decoder is the SOS token (index 2):
               decoder_input = target[:, 0]   ← this is the SOS token for the whole batch

            4. Loop from step 1 to tgt_seq_len - 1:
               a. Call self.decoder.forward_step(decoder_input, hidden, cell)
               b. Store the prediction in all_predictions[:, step, :]
               c. Decide the next input using teacher forcing:
                  - With probability teacher_forcing_ratio: use target[:, step] (the correct char)
                  - Otherwise: use the model's own prediction → prediction.argmax(dim=1)

            5. Return all_predictions
        """
        # --- Your code here ---
        raise NotImplementedError(
            "Seq2SeqLSTM.forward() is not implemented yet. "
            "See the TODO above for step-by-step guidance."
        )

    def translate(self, source_indices, sos_idx=2, eos_idx=3, max_length=100):
        """
        Translate a single encoded source sentence at inference time.

        At inference time we do NOT have access to the target sentence,
        so teacher forcing is off. The decoder uses its own output as the
        next input (this is called "greedy decoding").

        Args:
            source_indices (Tensor): Shape [1, src_seq_len] — one encoded sentence.
            sos_idx (int): Index of the SOS token. Default: 2.
            eos_idx (int): Index of the EOS token. Default: 3.
            max_length (int): Stop after this many characters even if EOS not seen.

        Returns:
            list of int: The predicted character indices (the translation).

        TODO (Phase 1 — stretch goal):
            This is very similar to forward(), but simpler because:
            - batch_size is always 1
            - There is no teacher forcing
            - We stop when we predict EOS or reach max_length

            1. Run the encoder to get hidden, cell
            2. Start with decoder_input = tensor([[sos_idx]])
            3. Loop up to max_length times:
               a. Run decoder.forward_step()
               b. Take the argmax of the prediction → the predicted character
               c. If the predicted character is EOS, stop
               d. Append the predicted character to the output list
               e. Use the predicted character as the next input
            4. Return the list of predicted indices
        """
        # --- Your code here ---
        raise NotImplementedError(
            "Seq2SeqLSTM.translate() is not implemented yet. "
            "This is a stretch goal for Phase 1."
        )
