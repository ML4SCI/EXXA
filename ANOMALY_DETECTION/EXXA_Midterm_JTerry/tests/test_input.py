import sys

import pytest
import torch

sys.path.insert(0, "./")
from models.autoencoder import Autoencoder
from models.gan import Discriminator, Generator
from models.transformer import TransformerSeq2Seq


def test_input(test_model_architecture):
    """Passes example data through models"""

    gen = test_model_architecture[0]
    disc = test_model_architecture[1]
    ae = test_model_architecture[2]
    transformer = test_model_architecture[3]
    spec_length = test_model_architecture[4]
    batch_size = test_model_architecture[5]

    example_gen_output = torch.rand((batch_size, spec_length), dtype=torch.float)
    example_disc_output = torch.rand((batch_size, 1), dtype=torch.float)
    example_ae_output = torch.rand((batch_size, spec_length), dtype=torch.float)
    example_transformer_output = torch.rand((batch_size, spec_length), dtype=torch.float)

    assert example_gen_output.shape == gen.shape
    assert example_disc_output.shape == disc.shape
    assert example_ae_output.shape == ae.shape
    assert example_transformer_output.shape == transformer.shape


@pytest.fixture()
def test_model_architecture():
    spec_length = 101
    latent_dim = 32
    batch_size = 2

    gen = Generator(latent_dim=latent_dim, max_seq_length=spec_length)
    disc = Discriminator(max_seq_length=spec_length)
    ae = Autoencoder(latent_dim=latent_dim, max_seq_length=spec_length)
    transformer = TransformerSeq2Seq(seq_length=spec_length)

    example_gen_input_tensor = torch.rand((batch_size, latent_dim), dtype=torch.float)
    example_vel_input_tensor = torch.rand((batch_size, spec_length), dtype=torch.float)
    example_disc_input_tensor = torch.rand((batch_size, spec_length), dtype=torch.float)
    example_ae_input_tensor = torch.rand((batch_size, spec_length), dtype=torch.float)
    example_transformer_input_tensor = torch.rand((batch_size, spec_length), dtype=torch.float)

    return (
        gen(example_gen_input_tensor, example_vel_input_tensor),
        disc(example_disc_input_tensor, example_vel_input_tensor),
        ae(example_ae_input_tensor),
        transformer(example_transformer_input_tensor),
        spec_length,
        batch_size,
    )
