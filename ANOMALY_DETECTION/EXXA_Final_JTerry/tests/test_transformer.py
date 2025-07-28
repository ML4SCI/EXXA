import pytest
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.new_transformer import (
    IntegratedTransformerSeq2Seq,
    TransformerBlock,
    TransformerDecoder,
    TransformerEncoder,
)


# Initialization
def test_transformer_block_initialization():
    model = TransformerBlock(
        input_dim=1,
        hid_dim=512,
    )
    assert model is not None, "Failed to initialize TransformerBlock"


# Forward Pass
@pytest.mark.parametrize("input_shape", [(1, 101, 1), (5, 101, 1)])
def test_transformer_block_forward_pass(input_shape):
    model = TransformerBlock(  # input_dim=input_shape[2],
        input_dim=1,
        seq_length=input_shape[1],
    )
    input_tensor = torch.randn(input_shape)
    output = model(input_tensor)
    output_shape = torch.Size([input_shape[0], input_shape[1], model.hid_dim])
    assert (
        output.shape == output_shape
    ), f"Expected output shape {output_shape}, but got {output.shape}"


@pytest.mark.parametrize("input_shape", [(10, 100, 1), (5, 50, 1), (1, 10, 1)])
def test_transformer_encoder_forward_pass(input_shape):
    model = TransformerEncoder(  # input_dim=input_shape[2],
        input_dim=1, seq_length=input_shape[1], device="cpu"
    )
    input_tensor = torch.randn(input_shape)
    output = model(input_tensor)
    output_shape = torch.Size([input_shape[0], input_shape[1], model.hid_dim])
    assert (
        output.shape == output_shape
    ), f"Expected output shape {output_shape}, but got {output.shape}"


# Training Loop
def test_integrated_transformer_seq2seq_training():
    class TestModel(IntegratedTransformerSeq2Seq):
        def train_dataloader(self):
            X = torch.randn(100, 50)
            y = torch.randn(100, 50)
            dataset = TensorDataset(X, y)
            return DataLoader(dataset, batch_size=10, shuffle=True)

    model = TestModel(
        input_dim=1, device="cpu", seq_length=50, hid_dim=512, n_layers=6, n_heads=8
    )
    trainer = pl.Trainer(max_epochs=1, fast_dev_run=True)
    trainer.fit(model)


# Edge Cases
def test_transformer_encoder_empty_input():
    model = TransformerEncoder(input_dim=1, device="cpu")
    input_tensor = torch.tensor([])
    with pytest.raises(Exception, match="Expected input tensor to be non-empty"):
        model(input_tensor)


def test_transformer_encoder_mismatched_dimensions():
    seq_len = 101
    model = TransformerEncoder(input_dim=1, seq_length=seq_len, device="cpu")
    input_tensor = torch.randn(10, seq_len, 5)  # Dimension mismatch
    with pytest.raises(Exception, match="Mismatched input dimension"):
        model(input_tensor)


# Non-default Parameters
@pytest.mark.parametrize("param_set", [{"n_layers": 4}, {"n_heads": 4}, {"dropout": 0.5}])
def test_transformer_encoder_non_default_parameters(param_set):
    input_tensor = torch.randn(10, 100, 1)
    hid_dim = 4
    model = TransformerEncoder(
        seq_length=input_tensor.shape[1], hid_dim=hid_dim, device="cpu", **param_set
    )
    output = model(input_tensor)
    expected_hid_dim = param_set.get("hid_dim", hid_dim)
    assert output.shape == (
        10,
        100,
        expected_hid_dim,
    ), f"Expected output shape (10, 100, {expected_hid_dim}), but got {output.shape}"


def test_model_initialization():
    model = IntegratedTransformerSeq2Seq(
        input_dim=1, output_dim=1, device="cpu", seq_length=50, hid_dim=512, n_layers=6, n_heads=8
    )
    assert isinstance(model.encoder, TransformerEncoder), "Encoder initialization failed"
    assert isinstance(model.decoder, TransformerDecoder), "Decoder initialization failed"


# Forward Pass Tests
@pytest.mark.parametrize("input_shape", [(10, 100), (5, 50), (1, 10)])
def test_forward_pass_shape(input_shape):
    model = IntegratedTransformerSeq2Seq(
        input_dim=1, output_dim=1, device="cpu", seq_length=50, hid_dim=512, n_layers=6, n_heads=8
    )
    input_tensor = torch.randn(input_shape)
    output = model(input_tensor)
    assert (
        output.shape == input_tensor.shape
    ), f"Expected output shape {input_shape}, but got {output.shape}"


def test_forward_pass_zero_tensor():
    seq_len = 100
    model = IntegratedTransformerSeq2Seq(
        input_dim=1,
        output_dim=1,
        device="cpu",
        seq_length=seq_len,
        hid_dim=512,
        n_layers=6,
        n_heads=8,
    )
    input_tensor = torch.zeros((10, seq_len))
    output = model(input_tensor)
    assert not torch.allclose(
        output, torch.zeros_like(output)
    ), "Output tensor should deviate from zero tensor"


def test_forward_pass_with_trg_eq_zero():
    seq_len = 100
    model = IntegratedTransformerSeq2Seq(
        input_dim=1,
        output_dim=1,
        device="cpu",
        seq_length=seq_len,
        hid_dim=512,
        n_layers=6,
        n_heads=8,
    )
    input_tensor = torch.randn((10, seq_len))
    model(input_tensor)
