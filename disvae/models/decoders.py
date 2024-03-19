"""
Module containing the decoders.
"""


import torch
from torch import nn


# ALL decoders should be called Decoder<Model>
def get_decoder(model_type):
    model_type = model_type.lower().capitalize()
    return eval("Decoder{}".format(model_type))


class DecoderSpec(nn.Module):
    def __init__(self, spec_dim, d_layers, cos_dim=4,
                 latent_dim=3, hidden_dim=64):
        r"""Decoder of the model proposed in [1].

        Parameters
        ----------
        input_size = [:, 1, latent_dim+spec_dim+cos_dim]

        latent_dim : int
            Dimensionality of latent output.

        Model Architecture (transposed for decoder)
        ------------
        - 4 convolutional layers (each with 32 channels), (4 x 4 kernel), (stride of 2)
        - 2 fully connected layers (each of 256 units)
        - Latent distribution:
            - 1 fully connected layer of 20 units (log variance and mean for 10 Gaussians)

        References:
            [1] Burgess, Christopher P., et al. "Understanding disentangling in
            $\beta$-VAE." arXiv preprint arXiv:1804.03599 (2018).
        """
        super(DecoderSpec, self).__init__()

        # Layer parameters
        self.hidden_dim = hidden_dim

        # Fully connected layers
        self.lin_1 = nn.Linear(latent_dim+spec_dim+cos_dim, self.hidden_dim)
        lin_mid = []
        for i in range(0,d_layers):
            lin_mid.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            lin_mid.append(nn.ReLU())
        self.lin_mid = nn.Sequential(*lin_mid)
        self.lin_f = nn.Linear(self.hidden_dim, spec_dim)


    def forward(self, z):
        batch_size = z.size(0)
        z = z.view((batch_size,-1))

        # Fully connected layers with ReLu activations
        x = torch.relu(self.lin_1(z))
        x = self.lin_mid(x)
        x = self.lin_f(x)
        x = x.view((batch_size,-1))

        return x
