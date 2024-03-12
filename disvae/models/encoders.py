"""
Module containing the encoders.
"""
import numpy as np

import torch
from torch import nn


# ALL encoders should be called Enccoder<Model>
def get_encoder(model_type):
    model_type = model_type.lower().capitalize()
    return eval("Encoder{}".format(model_type))

class EncoderSpec(nn.Module):
    def __init__(self, spec_dim, e_layers, cos_dim=4,
                 latent_dim=10):
        r"""Encoder of the model proposed in [1].

        Parameters
        ----------
        spec_dim : int
            The number of point for a given spectrum, 18 in our case.
            
        e_layers: int
            The number of layers of encoder
        
        cos_dim: int
            The number of cosmology parameters, 4 in our case

        latent_dim : int
            Dimensionality of latent output.

        """
        super(EncoderSpec, self).__init__()

        # Layer parameters
        self.spec_dim = spec_dim
        self.hidden_dim = 256
        self.latent_dim = latent_dim
        self.e_layers = e_layers
        self.extend = 5
        self.cos_dim = cos_dim
        
        self.cos_extend = nn.Linear(self.cos_dim*2, self.cos_dim*self.extend)
        
        self.extend = nn.Linear(2*spec_dim, self.extend*self.spec_dim)
        MLP = [nn.Linear(5*(spec_dim+cos_dim), self.hidden_dim),nn.ReLU()]
        
        for i in range(0,e_layers):
            MLP.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            MLP.append(nn.ReLU())
        self.MLP = nn.Sequential(*MLP)
        
        self.mu_logvar_gen = nn.Linear(self.hidden_dim, self.latent_dim * 2)


    def forward(self, x):
        batch_size = x.size(0)
        spec = x[:,:,:self.spec_dim].clone().reshape([batch_size, -1])
        cos_para = x[:,:,self.spec_dim:].clone().view((batch_size, -1))
        # Fully connected layers with ReLu activations
        cos_para = torch.relu(self.cos_extend(cos_para))
        spec = torch.relu(self.extend(spec))
        
        x = torch.cat([spec,cos_para],dim=1)
        x = self.MLP(x)
        # Fully connected layer for log variance and mean
        # Log std-dev in paper (bear in mind)
        mu_logvar = self.mu_logvar_gen(x)
        mu, logvar = mu_logvar.view(-1, self.latent_dim, 2).unbind(-1)
        logvar = torch.abs(logvar)

        return mu, logvar