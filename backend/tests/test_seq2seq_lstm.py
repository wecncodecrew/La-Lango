import torch
import pytest
from lalango.models.seq2seq_lstm import Encoder, Decoder, Seq2SeqLSTM

SRC_VOCAB  = 20
TGT_VOCAB  = 25
EMBED_DIM  = 16
HIDDEN_DIM = 32
BATCH      = 3
SRC_LEN    = 7
TGT_LEN    = 9

@pytest.fixture
def encoder():
    return Encoder(SRC_VOCAB, EMBED_DIM, HIDDEN_DIM)

@pytest.fixture
def decoder():
    return Decoder(TGT_VOCAB, EMBED_DIM, HIDDEN_DIM)

@pytest.fixture
def model():
    return Seq2SeqLSTM(SRC_VOCAB, TGT_VOCAB, EMBED_DIM, HIDDEN_DIM)

@pytest.fixture
def source():
    return torch.randint(1, SRC_VOCAB, (BATCH, SRC_LEN))

@pytest.fixture
def target():
    t = torch.randint(1, TGT_VOCAB, (BATCH, TGT_LEN))
    t[:, 0] = 2
    return t

class TestEncoder:

    def test_hidden_shape(self, encoder, source):
        hidden, cell = encoder(source)
        assert hidden.shape == (1, BATCH, HIDDEN_DIM)

    def test_cell_shape(self, encoder, source):
        hidden, cell = encoder(source)
        assert cell.shape == (1, BATCH, HIDDEN_DIM)

    def test_returns_two_tensors(self, encoder, source):
        result = encoder(source)
        assert len(result) == 2

class TestDecoder:

    def test_prediction_shape(self, decoder, encoder, source):
        hidden, cell = encoder(source)
        input_token = torch.randint(1, TGT_VOCAB, (BATCH,))
        prediction, new_hidden, new_cell = decoder.forward_step(input_token, hidden, cell)
        assert prediction.shape == (BATCH, TGT_VOCAB)

    def test_hidden_shape_unchanged(self, decoder, encoder, source):
        hidden, cell = encoder(source)
        input_token = torch.randint(1, TGT_VOCAB, (BATCH,))
        _, new_hidden, new_cell = decoder.forward_step(input_token, hidden, cell)
        assert new_hidden.shape == (1, BATCH, HIDDEN_DIM)

    def test_cell_shape_unchanged(self, decoder, encoder, source):
        hidden, cell = encoder(source)
        input_token = torch.randint(1, TGT_VOCAB, (BATCH,))
        _, new_hidden, new_cell = decoder.forward_step(input_token, hidden, cell)
        assert new_cell.shape == (1, BATCH, HIDDEN_DIM)

class TestSeq2SeqLSTMForward:

    def test_output_shape(self, model, source, target):
        predictions = model(source, target)
        assert predictions.shape == (BATCH, TGT_LEN, TGT_VOCAB)

    def test_output_is_tensor(self, model, source, target):
        predictions = model(source, target)
        assert isinstance(predictions, torch.Tensor)

    def test_no_teacher_forcing(self, model, source, target):
        predictions = model(source, target, teacher_forcing_ratio=0.0)
        assert predictions.shape == (BATCH, TGT_LEN, TGT_VOCAB)

    def test_full_teacher_forcing(self, model, source, target):
        predictions = model(source, target, teacher_forcing_ratio=1.0)
        assert predictions.shape == (BATCH, TGT_LEN, TGT_VOCAB)

class TestSeq2SeqLSTMTranslate:

    def test_returns_list(self, model, source):
        single_source = source[0].unsqueeze(0)
        result = model.translate(single_source)
        assert isinstance(result, list)

    def test_returns_list_of_ints(self, model, source):
        single_source = source[0].unsqueeze(0)
        result = model.translate(single_source)
        assert all(isinstance(idx, int) for idx in result)

    def test_respects_max_length(self, model, source):
        single_source = source[0].unsqueeze(0)
        max_length = 10
        result = model.translate(single_source, max_length=max_length)
        assert len(result) <= max_length

    def test_eos_stops_decoding(self):
        tiny_model = Seq2SeqLSTM(10, 10, 8, 16)
        with torch.no_grad():
            tiny_model.decoder.output_layer.weight.zero_()
            tiny_model.decoder.output_layer.bias.zero_()
            tiny_model.decoder.output_layer.bias[3] = 100.0
        source = torch.randint(1, 10, (1, 5))
        result = tiny_model.translate(source, max_length=50)
        assert len(result) == 0
