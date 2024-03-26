import math
import sys
from datetime import date

import numpy as np
import torch
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import TensorDataset, DataLoader
import optuna
from optuna.storages import JournalStorage, JournalFileStorage
from optuna.trial import TrialState

from disvae.models.vae import VAE
from disvae.models.losses import BtcvaeLoss


class Objective:
    def __init__(self, num_z, P_shape, num_p, t_dataset, v_dataset, num_epoch, device):
        self.num_z = num_z
        self.P_shape = P_shape
        self.num_p = num_p
        self.t_dataset = t_dataset
        self.v_dataset = v_dataset
        self.num_epoch = num_epoch
        self.device = device

    def __call__(self, trial):
        #model hyperparams
        conv_depth = trial.suggest_int('Dc', 2, 4)
        conv_width = trial.suggest_int('Wc', 4, 9)  # number of channels
        mlp_depth = trial.suggest_int('Dm', 2, 5)
        mlp_width = 2 ** trial.suggest_int('log2_Wm', 4, 9)
        bn_mode = trial.suggest_categorical('BA', ['BA', 'A', 'AB'])

        #loss hyperparams
        lamda = alpha = beta = gamma = 1
        if self.num_z >= 1:
            lamda = trial.suggest_float('lambda', 0.01, 10, log=True)
            if self.num_z >= 2:
                alpha = trial.suggest_float('alpha', 0.1, 10, log=True)
                beta = trial.suggest_float('beta', 1, 100, log=True)
                gamma = trial.suggest_float('gamma', 0.1, 10, log=True)

        #training hyperparams
        lr = trial.suggest_float('lr', 1e-6, 1e-1, log=True)
        b1 = 1 - trial.suggest_float('1-b1', 0.1, 0.5, log=True)
        b2 = 1 - trial.suggest_float('1-b2', 1e-3, 0.1, log=True)
        wd = trial.suggest_float('wd', 1e-6, 1e-3, log=True)
        batch_size = 2 ** trial.suggest_int('log2_B', 6, 9)

        t_loader = DataLoader(self.t_dataset, batch_size=batch_size, shuffle=True)
        v_loader = DataLoader(self.v_dataset, batch_size=len(self.v_dataset),
                              shuffle=False)
        model = VAE(self.num_z, self.P_shape, self.num_p,
                    conv_depth, conv_width, mlp_depth, mlp_width,
                    bn_mode).to(self.device, non_blocking=True)
        criterion = BtcvaeLoss(len(self.t_dataset), lamda=lamda, alpha=alpha, beta=beta,
                               gamma=gamma)
        assert len(self.t_dataset) == len(self.v_dataset), 'otherwise need 2 criteria'
        optimizer = AdamW(model.parameters(), lr=lr, betas=(b1, b2), weight_decay=wd)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=10,
                                                   threshold=0.01)

        t_loss, v_loss = math.inf, math.inf
        t_rec, v_rec = math.inf, math.inf
        t_kl, v_kl = math.inf, math.inf
        t_mi, v_mi = math.inf, math.inf
        t_tc, v_tc = math.inf, math.inf
        t_dwkl, v_dwkl = math.inf, math.inf
        for epoch in range(self.num_epoch):
            if optimizer.param_groups[0]['lr'] <= 1e-8:
                break

            model.train()
            storer = {k: torch.zeros(1, device=self.device) for k in
                    ['loss', 'rec', 'mi', 'tc', 'dwkl', 'kl']
                    + [f'kl_{d}' for d in range(num_z)]}
            for P, p in t_loader:
                Prat = P[:, 1]
                loss = criterion(*model(P, p), Prat, storer=storer)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            t_loss = min(storer['loss'].item() / len(t_loader), t_loss)
            t_rec = min(storer['rec'].item() / len(t_loader), t_rec)
            t_kl = min(storer['kl'].item() / len(t_loader), t_kl)
            t_mi = min(storer['mi'].item() / len(t_loader), t_mi)
            t_tc = min(storer['tc'].item() / len(t_loader), t_tc)
            t_dwkl = min(storer['dwkl'].item() / len(t_loader), t_dwkl)

            model.eval()
            storer = {k: torch.zeros(1, device=self.device) for k in
                    ['loss', 'rec', 'mi', 'tc', 'dwkl', 'kl']
                    + [f'kl_{d}' for d in range(num_z)]}
            with torch.no_grad():
                for P, p in v_loader:
                    Prat = P[:, 1]
                    _ = criterion(*model(P, p), Prat, storer=storer)
                v_loss = min(storer['loss'].item() / len(v_loader), v_loss)
                v_rec = min(storer['rec'].item() / len(v_loader), v_rec)
                v_kl = min(storer['kl'].item() / len(v_loader), v_kl)
                v_mi = min(storer['mi'].item() / len(v_loader), v_mi)
                v_tc = min(storer['tc'].item() / len(v_loader), v_tc)
                v_dwkl = min(storer['dwkl'].item() / len(v_loader), v_dwkl)

            scheduler.step(v_loss)

        print(f'{trial.number=}, {epoch=}, {t_loss=:.1e}, {v_loss=:.1e}')
        print(f'  {t_rec=:.1e}, {t_kl=:.1e}, {t_mi=:.1e}, {t_tc=:.1e}, {t_dwkl=:.1e}')
        print(f'  {v_rec=:.1e}, {v_kl=:.1e}, {v_mi=:.1e}, {v_tc=:.1e}, {v_dwkl=:.1e}',
              flush=True)

        trial.set_user_attr('epoch', epoch)
        trial.set_user_attr('t_loss', t_loss)
        trial.set_user_attr('v_loss', v_loss)
        trial.set_user_attr('t_rec', t_rec)
        trial.set_user_attr('v_rec', v_rec)
        trial.set_user_attr('t_kl', t_kl)
        trial.set_user_attr('v_kl', v_kl)
        trial.set_user_attr('t_mi', t_mi)
        trial.set_user_attr('v_mi', v_mi)
        trial.set_user_attr('t_tc', t_tc)
        trial.set_user_attr('v_tc', v_tc)
        trial.set_user_attr('t_dwkl', t_dwkl)
        trial.set_user_attr('v_dwkl', v_dwkl)

        torch.cuda.empty_cache()

        if self.num_z == 0:
            return v_rec
        if self.num_z == 1:
            return v_rec, v_kl
        return v_rec, v_tc


def optune(num_z, P_shape, num_p, t_dataset, v_dataset, device):
    n_trials = 2**14
    timeout = 6.9 * 24 * 3600
    n_startup_trials = 2**6

    sampler = optuna.samplers.TPESampler(
        n_startup_trials=n_startup_trials,
        seed=42,
        multivariate=True,
        warn_independent_sampling=True,
        constant_liar=True,
    )

    # both optuna<3.6.0 and file system can crash journal storage
    storage = JournalStorage(JournalFileStorage(f'/dev/shm/optuna_d{num_z}.journal'))
    study = optuna.create_study(
        storage=storage,
        sampler=sampler,
        pruner=None,  # currently not for multi-objective optimization
        study_name=f"{date.today().strftime('%m%d')}",
        load_if_exists=True,
        directions=['minimize'] * (1 if num_z == 0 else 2),
    )

    num_epoch = 1000
    objective = Objective(num_z, P_shape, num_p, t_dataset, v_dataset, num_epoch,
                          device)

    study.optimize(objective, n_trials=n_trials, timeout=timeout, gc_after_trial=True)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print('Study statistics: ')
    print('  Number of finished trials: ', len(study.trials))
    print('  Number of pruned trials: ', len(pruned_trials))
    print('  Number of complete trials: ', len(complete_trials))

    if num_z == 0:
        # single-objective
        print('Best trial:')
        trial = study.best_trial
        print('  Value: ', trial.value)
        print('  Params: ')
        for key, value in trial.params.items():
            print('    {}: {}'.format(key, value))
    elif num_z == 1:
        # multi-objective
        print('Pareto front:')
        trials = sorted(study.best_trials, key=lambda trial: trial.values)
        for trial in trials:
            rec, kl = trial.values
            print(f'  Trial#{trial.number}')
            print(f'    Values: {rec=}, {kl=}')
            print(f'    Params: {trial.params}')
    else:
        # multi-objective
        print('Pareto front:')
        trials = sorted(study.best_trials, key=lambda trial: trial.values)
        for trial in trials:
            rec, tc = trial.values
            print(f'  Trial#{trial.number}')
            print(f'    Values: {rec=}, {tc=}')
            print(f'    Params: {trial.params}')

    #print('Trials:')
    #df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'),
    #                            multi_index=True)
    #print(df.to_markdown())


if __name__ == '__main__':
    num_z = int(sys.argv[1])

    device = torch.device('cuda', 0)

    P_set = np.load('IllustrisTNG_powers.npy')[:, :2]  # shape = (1024, 2, 4, 18)
    P_set = torch.from_numpy(P_set).to(device)
    P_set = torch.log(P_set)
    P_set[:, 1] -= P_set[:, 0]  # log(P_hydro / P_dmo)
    Pt_set, Pv_set = P_set[:512], P_set[512:]

    p_set = np.load('IllustrisTNG_params.npy')[:, 0, -1]  # shape = (1024, 3)
    p_set = torch.from_numpy(p_set).to(device)
    p_set = torch.log(p_set)
    pt_set, pv_set = p_set[:512], p_set[512:]

    P_shape, num_p = P_set.shape[1:], p_set.shape[1]

    t_dataset = TensorDataset(Pt_set, pt_set)
    v_dataset = TensorDataset(Pv_set, pv_set)

    optune(num_z, P_shape, num_p, t_dataset, v_dataset, device)
