import math

import torch
from torch.nn import functional as F


class BtcvaeLoss:
    """VAE loss with minibatch weighted or stratified sampling following [1]_.

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

    sampling : {'mws', 'mss'}, optional
        Whether to use minibatch stratified sampling instead of minibatch weighted
        sampling.

    References
    ----------
    .. [1] Chen, Tian Qi, et al. "Isolating sources of disentanglement in variational
            autoencoders." Advances in Neural Information Processing Systems. 2018.

    """

    def __init__(self, data_size, lamda=1, alpha=1, beta=1, gamma=1, sampling='mss'):  #FIXME replace with optuna best hyperparams
        super().__init__()
        self.data_size = data_size
        self.lamda = lamda
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.sampling = sampling
        assert sampling in ['mws', 'mss']

    def __call__(self, mean, logvar, z, Prec, Prat, storer):
        rec = _reconstruction_loss(Prec, Prat, storer=storer)
        log_pz, log_qz, log_prod_qzi, log_q_zCx = _get_log_pz_qz_prodzi_qzCx(
            mean, logvar, z, self.data_size, self.sampling)
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


#TODO compare mss and mws
def _get_log_pz_qz_prodzi_qzCx(mean, logvar, z, data_size, sampling):
    batch_size = mean.shape[0]

    # log q(z|x)
    log_q_zCx = log_density_gaussian(mean, logvar, z).sum(dim=1)

    # log p(z) with zero mean and logvar
    zeros = torch.zeros_like(z)
    log_pz = log_density_gaussian(zeros, zeros, z).sum(dim=1)

    mat_log_qz = log_density_gaussian(mean, logvar, z.unsqueeze(1))
    mat_log_qz_ = mat_log_qz.sum(dim=2)

    if sampling == 'mss':
        log_iw_mat = log_importance_weight_matrix(batch_size, data_size, device=z.device)
        mat_log_qz_ = mat_log_qz_ + log_iw_mat
        mat_log_qz = mat_log_qz + log_iw_mat.unsqueeze(2)

    log_qz = torch.logsumexp(mat_log_qz_, dim=1, keepdim=False)
    log_prod_qzi = torch.logsumexp(mat_log_qz, dim=1, keepdim=False).sum(dim=1)

    return log_pz, log_qz, log_prod_qzi, log_q_zCx


def log_density_gaussian(mean, logvar, z):
    norm = - 0.5 * (math.log(2 * math.pi) + logvar)
    inv_var = torch.exp(-logvar)
    log_density = norm - 0.5 * ((z - mean)**2 * inv_var)
    return log_density


def log_importance_weight_matrix(batch_size, data_size, device):
    M, N = batch_size - 1, data_size
    strat_weight = (N - M) / (N * M)
    W = torch.full((batch_size, batch_size), 1 / M, device=device)
    W.view(-1)[::M + 1] = 1 / N
    W.view(-1)[1::M + 1] = strat_weight
    W[M - 1, 0] = strat_weight
    return torch.log(W)
