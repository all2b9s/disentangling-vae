from collections import OrderedDict
import math

import torch
from torch import nn


def module_with_param_init(Module):
    class ModuleWithParamInit(Module):
        def __init__(self, *args, linear=False, leaky_slope=0, **kwargs):
            self.nonlinearity = 'linear' if linear else 'leaky_relu'
            self.leaky_slope = leaky_slope
            super().__init__(*args, **kwargs)

        def reset_parameters(self):
            nn.init.kaiming_uniform_(self.weight, a=self.leaky_slope,
                                    nonlinearity=self.nonlinearity)
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)

    return ModuleWithParamInit

Linear = module_with_param_init(nn.Linear)
Conv1d = module_with_param_init(nn.Conv1d)


def get_block(header, l, in_width, out_width, leaky_slope, bn_mode):
    if header[1] == 'c':
        lin = f'{header}c{l}', Conv1d(in_width, out_width, 1, leaky_slope=leaky_slope)
    elif header[1] == 'm':
        lin = f'{header}l{l}', Linear(in_width, out_width, leaky_slope=leaky_slope)
    else:
        raise ValueError(f'{header=} not supported')

    act = f'{header}a{l}', nn.LeakyReLU(negative_slope=leaky_slope)
    if bn_mode == 'A':
        return [lin, act]

    bn = f'{header}b{l}', nn.BatchNorm1d(out_width)
    if bn_mode == 'BA':
        return [lin, bn, act]
    elif bn_mode == 'AB':
        return [lin, act, bn]
    else:
        raise ValueError(f'{bn_mode=} not supported')


class VAE(nn.Module):
    def __init__(self, num_z=2, P_shape=(2, 4, 18), num_p=3,
                 conv_depth=2, conv_width=4, mlp_depth=3, mlp_width=64,
                 bn_mode='A'):  #FIXME use optuna best hyperparams
        super().__init__()

        self.num_z = num_z
        two, num_a, num_k = P_shape
        self.num_a = num_a
        self.num_k = num_k
        self.num_p = num_p
        self.conv_depth = conv_depth
        self.conv_width = conv_width
        self.mlp_depth = mlp_depth
        self.mlp_width = mlp_width
        assert conv_depth >= 2 and conv_width >= num_a and mlp_depth >= 2

        leaky_slope = 1 / torch.e

        self.econv = OrderedDict()
        for l in range(conv_depth):
            in_chan = two * num_a + num_p if l == 0 else conv_width
            self.econv.update(get_block('ec', l, in_chan, conv_width, leaky_slope,
                                        bn_mode))
        self.econv = nn.Sequential(self.econv)

        self.emlp = OrderedDict()
        for l in range(mlp_depth - 1):
            in_feat = conv_width * num_k + num_p if l == 0 else mlp_width
            self.emlp.update(get_block('em', l, in_feat, mlp_width, leaky_slope,
                                       bn_mode))
        out_feat = 2 * num_z
        self.emlp[f'eml{mlp_depth-1}'] = Linear(mlp_width, out_feat, linear=True)
        self.emlp = nn.Sequential(self.emlp)

        self.dmlp = OrderedDict()
        for l in range(mlp_depth):
            in_feat = num_z + num_p if l == 0 else mlp_width
            out_feat = conv_width * num_k if l == mlp_depth - 1 else mlp_width
            self.dmlp.update(get_block('dm', l, in_feat, out_feat, leaky_slope,
                                       bn_mode))
        self.dmlp = nn.Sequential(self.dmlp)

        self.dconv = OrderedDict()
        for l in range(conv_depth - 1):
            in_chan = conv_width + num_a + num_p if l == 0 else conv_width
            leaky_slope_ = 0 if l == conv_depth - 2 else leaky_slope
            self.dconv.update(get_block('dc', l, in_chan, conv_width, leaky_slope_,
                                        bn_mode))
        out_chan = num_a
        self.dconv[f'dcc{conv_depth-1}'] = Conv1d(conv_width, out_chan, 1, bias=False,
                                                  linear=True)
        self.dconv = nn.Sequential(self.dconv)

    def encode(self, P, p):
        P = P.flatten(start_dim=1, end_dim=2)
        x = torch.cat([P, p.unsqueeze(2).repeat(1, 1, self.num_k)], dim=1)
        x = self.econv(x)

        x = x.flatten(start_dim=1, end_dim=2)
        x = torch.cat([x, p], dim=1)
        x = self.emlp(x)

        x = x.unflatten(1, (2, self.num_z))
        mean, logvar = x[:, 0], x[:, 1]
        return mean, logvar

    def reparameterize(self, mean, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + std * eps
        return mean

    def decode(self, z, Pdmo, p):
        x = torch.cat([z, p], dim=1)
        x = self.dmlp(x)

        x = x.unflatten(1, (self.conv_width, self.num_k))
        x = torch.cat([x, Pdmo, p.unsqueeze(2).repeat(1, 1, self.num_k)], dim=1)
        Prec = self.dconv(x)

        return Prec

    def forward(self, P, p):
        mean, logvar = self.encode(P, p)

        z = self.reparameterize(mean, logvar)

        Pdmo = P[:, 0]
        Prec = self.decode(z, Pdmo, p)

        return mean, logvar, z, Prec
