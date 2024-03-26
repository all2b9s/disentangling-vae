import math

import torch
from torch.nn import functional as F


class BtcvaeLoss:
    """VAE loss with minibatch stratified sampling following [1]_.

    Parameters
    ----------
    data_size : int

    lamda : float, optional
        Weight of the mutual information, total correlation, and dimension-wise KL
        terms relative to the reconstruction term.

    alpha : float, optional
        Weight of the mutual information term.

    beta : float, optional
        Weight of the total correlation term.

    gamma : float, optional
        Weight of the dimension-wise KL term.

    References
    ----------
    .. [1] Chen, Tian Qi, et al. "Isolating sources of disentanglement in variational
            autoencoders." Advances in Neural Information Processing Systems. 2018.

    """

    def __init__(self, data_size, lamda=1, alpha=1, beta=1, gamma=1):  #FIXME replace with optuna best hyperparams
        super().__init__()
        self.data_size = data_size
        self.lamda = lamda
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, mean, logvar, z, Prec, Prat, storer):
        rec = _reconstruction_loss(Prec, Prat, storer=storer)

        if mean is None:
            return rec

        log_pz, log_qz, log_prod_qzi, log_q_zCx = _get_log_pz_qz_prodzi_qzCx(
            mean, logvar, z, self.data_size)
        # I[z;x] = KL[q(z,x)||q(x)q(z)] = E_x[KL[q(z|x)||q(z)]]
        mi = (log_q_zCx - log_qz).mean()
        # TC[z] = KL[q(z)||\prod_i z_i]
        tc = (log_qz - log_prod_qzi).mean()
        # dwkl is KL[q(z)||p(z)] instead of usual KL[q(z|x)||p(z))]
        dwkl = (log_prod_qzi - log_pz).mean()

        # total loss
        loss = rec + self.lamda * (self.alpha * mi + self.beta * tc + self.gamma * dwkl)
        if storer is not None:
            storer['loss'] += loss.detach()
            storer['mi'] += mi.detach()
            storer['tc'] += tc.detach()
            storer['dwkl'] += dwkl.detach()
            _ = _kl_normal_loss(mean, logvar, storer)  # for comparaison purposes

        return loss


def _reconstruction_loss(Prec, Prat, storer=None):
    loss = F.mse_loss(Prec, Prat)

    if storer is not None:
        storer['rec'] += loss.detach()

    return loss


def _kl_normal_loss(mean, logvar, storer=None):
    num_z = mean.shape[1]

    # batch mean of kl for each latent dimension
    kl = 0.5 * (-1 - logvar + torch.exp(logvar) + mean ** 2).mean(dim=0)
    total_kl = kl.sum()

    if storer is not None:
        storer['kl'] += total_kl.detach()
        for d in range(num_z):
            storer[f'kl_{d}'] += kl[d].detach()

    return total_kl


def _get_log_pz_qz_prodzi_qzCx(mean, logvar, z, data_size):
    batch_size = mean.shape[0]
    device = mean.device

    # log q(z|x)
    log_q_zCx = log_gaussian_density(mean, logvar, z).sum(dim=1)

    # log p(z) with zero mean and logvar
    zeros = torch.zeros_like(z)
    log_pz = log_gaussian_density(zeros, zeros, z).sum(dim=1)

    mat_log_qz = log_gaussian_density(mean, logvar, z.unsqueeze(1))
    mat_log_w = log_weight_matrix(batch_size, data_size, device=device)

    # log \prod_i q(z_i)
    mat_log_qzi = mat_log_qz + mat_log_w.unsqueeze(2)
    log_prod_qzi = mat_log_qzi.logsumexp(1).sum(dim=1)

    # log q(z)
    mat_log_qz = mat_log_qz.sum(dim=2) + mat_log_w
    log_qz = mat_log_qz.logsumexp(1)

    return log_pz, log_qz, log_prod_qzi, log_q_zCx


def log_gaussian_density(mean, logvar, z):
    norm = - 0.5 * (math.log(2 * math.pi) + logvar)
    inv_var = torch.exp(-logvar)
    log_density = norm - 0.5 * ((z - mean)**2 * inv_var)
    return log_density


def log_weight_matrix(batch_size, data_size, device):
    """Weight matrix, improved upon Chen et al. 2018"""
    M, N = batch_size - 1, data_size
    strat_weight = (N - 1) / (N * M)
    W = torch.full((batch_size, batch_size), strat_weight, device=device)
    W.fill_diagonal_(1 / N)
    return torch.log(W)
