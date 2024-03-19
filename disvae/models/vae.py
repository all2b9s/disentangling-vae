"""
Module containing the main VAE class.
"""
import torch
from torch import nn, optim
from torch.nn import functional as F

from disvae.utils.initialization import weights_init
from .encoders import EncoderSpec
from .decoders import DecoderSpec

MODELS = ["Burgess"]
cos_dim = 3

def init_specific_model(model_type, spec_length, latent_dim, hyperparameters):
    """Return an instance of a VAE with encoder and decoder from `model_type`."""
    model_type = model_type.lower().capitalize()
    if model_type not in MODELS:
        err = "Unkown model_type={}. Possible values: {}"
        raise ValueError(err.format(model_type, MODELS))

    encoder = get_encoder(model_type)
    decoder = get_decoder(model_type)
    model = VAE(spec_length, encoder, decoder, latent_dim, hyperparameters)
    model.model_type = model_type  # store to help reloading
    return model


class VAE(nn.Module):
    def __init__(self, spec_dim, latent_dim, hyperparameters = [3,3], cos_dim=4):
        """
        Class which defines model and forward pass.

        input size: [1, 1, 4, spec_dim+cos_dim]
        """
        super(VAE, self).__init__()
        e_layers, d_layers = hyperparameters
        self.spec_dim = spec_dim
        self.latent_dim = latent_dim
        self.encoder = EncoderSpec(spec_dim, e_layers, latent_dim = latent_dim, cos_dim = cos_dim)
        self.decoder = DecoderSpec(spec_dim, d_layers, latent_dim = latent_dim, cos_dim = cos_dim)

        self.reset_parameters()

    def reparameterize(self, mean, logvar):
        """
        Samples from a normal distribution using the reparameterization trick.

        Parameters
        ----------
        mean : torch.Tensor
            Mean of the normal distribution. Shape (batch_size, latent_dim)

        logvar : torch.Tensor
            Diagonal log variance of the normal distribution. Shape (batch_size,
            latent_dim)
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + std * eps
        else:
            # Reconstruction mode
            return mean

    def forward(self, x):
        """
        Forward pass of model.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, n_spec+n_cos)
        """
        batch_size = x.size(0)
        input_data = x[:,:2,:]
        target_data = x[:,2:3,:].clone().view((batch_size,-1))
        latent_dist = self.encoder(input_data)
        latent_sample = self.reparameterize(*latent_dist)
        dc_input = torch.cat([latent_sample,target_data],dim=1)
        reconstruct = self.decoder(dc_input)
        return reconstruct, latent_dist, latent_sample

    def reset_parameters(self):
        self.apply(weights_init)

    def sample_latent(self, x):
        """
        Returns a sample from the latent distribution.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        latent_dist = self.encoder(x)
        latent_sample = self.reparameterize(*latent_dist)
        return latent_sample
