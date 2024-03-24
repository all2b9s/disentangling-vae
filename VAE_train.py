import math
import os
import sys
from datetime import date

import numpy as np
import torch
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import TensorDataset, DataLoader
import optuna
from optuna.trial import TrialState

from disvae.models.vae import VAE
from disvae.models.losses import BtcvaeLoss


def objective(trial):
    #model hyperparams
    conv_depth = trial.suggest_int('Dꟲ', 2, 4)
    conv_width = trial.suggest_int('Wꟲ', 4, 9)  # number of channels
    mlp_depth = trial.suggest_int('Dᴹ', 2, 5)
    mlp_width = 2 ** trial.suggest_int('㏒₂Wᴹ', 4, 9)
    bn_mode = trial.suggest_categorical('BA', ['BA', 'A', 'AB'])

    #loss hyperparams
    lamda = trial.suggest_float('λ', 0.01, 10, log=True)
    alpha = - trial.suggest_int('-α', 1, 100, log=True)
    beta = trial.suggest_int('β', 1, 100, log=True)
    gamma = trial.suggest_float('γ', 0.1, 10, log=True)

    #training hyperparams
    lr = trial.suggest_float('lr', 1e-6, 1e-2, log=True)
    b1 = 1 - trial.suggest_float('1-β₁', 0.1, 0.5, log=True)
    b2 = 1 - trial.suggest_float('1-β₂', 1e-3, 0.1, log=True)
    wd = trial.suggest_float('wd', 1e-6, 1e-3, log=True)
    batch_size = 2 ** trial.suggest_int('㏒₂B', 6, 9)

    t_loader = DataLoader(t_dataset, batch_size=batch_size, shuffle=True)
    v_loader = DataLoader(v_dataset, batch_size=len(v_dataset), shuffle=False)
    model = VAE(num_z, P_shape, num_p,
                conv_depth, conv_width, mlp_depth, mlp_width,
                bn_mode).to(device, non_blocking=True)
    criterion = BtcvaeLoss(len(t_dataset), lamda=lamda, alpha=alpha, beta=beta,
                           gamma=gamma, sampling=sampling)
    assert len(t_dataset) == len(v_dataset), 'otherwise we need two criteria'
    optimizer = AdamW(model.parameters(), lr=lr, betas=(b1, b2), weight_decay=wd)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, threshold=0.01)

    min_train, min_valid = math.inf, math.inf
    min_rec, max_mi, min_tc, min_dwkl = math.inf, -math.inf, math.inf, math.inf
    min_kl = math.inf
    for epoch in range(1000):
        if optimizer.param_groups[0]['lr'] <= 1e-8:
            break

        model.train()
        storer = {k: torch.zeros(1, device=device) for k in
                  ['loss', 'rec', 'mi', 'tc', 'dwkl', 'kl']
                  + [f'kl_{d}' for d in range(num_z)]}
        for P, p in t_loader:
            Prat = P[:, 1]
            loss = criterion(*model(P, p), Prat, storer=storer)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_loss = storer['loss'].item() / len(t_loader)
        min_train = min(epoch_loss, min_train)

        model.eval()
        storer = {k: torch.zeros(1, device=device) for k in
                  ['loss', 'rec', 'mi', 'tc', 'dwkl', 'kl']
                  + [f'kl_{d}' for d in range(num_z)]}
        with torch.no_grad():
            for P, p in v_loader:
                Prat = P[:, 1]
                loss = criterion(*model(P, p), Prat, storer=storer)
            epoch_loss = storer['loss'].item() / len(v_loader)
            rec = storer['rec'].item() / len(v_loader)
            mi = storer['mi'].item() / len(v_loader)
            tc = storer['tc'].item() / len(v_loader)
            dwkl = storer['dwkl'].item() / len(v_loader)
            kl = storer['kl'].item() / len(v_loader)
            min_valid = min(epoch_loss, min_valid)
            min_rec = min(rec, min_rec)
            max_mi = max(mi, max_mi)
            min_tc = min(tc, min_tc)
            min_dwkl = min(dwkl, min_dwkl)
            min_kl = min(kl, min_kl)

        scheduler.step(epoch_loss)

    print(f'{trial.number=}, {epoch=}, {min_train=}, {min_valid=}')
    print(f'  {min_rec=}, {max_mi=}, {min_tc=}, {min_dwkl=} {min_kl=}', flush=True)

    trial.set_user_attr('epoch', epoch)
    trial.set_user_attr('min_train', min_train)
    trial.set_user_attr('min_valid', min_valid)
    trial.set_user_attr('min_rec', min_rec)
    trial.set_user_attr('max_mi', max_mi)
    trial.set_user_attr('min_tc', min_tc)
    trial.set_user_attr('min_dwkl', min_dwkl)
    trial.set_user_attr('min_kl', min_kl)

    #states = {
    #    'epoch': epoch,
    #    'model': model.state_dict(),
    #    'optimizer': optimizer.state_dict(),
    #    'scheduler': scheduler.state_dict(),
    #    'rng': torch.get_rng_state(),
    #    'min_train': min_train,
    #    'min_valid': min_valid,
    #    'min_rec': min_rec,
    #    'max_mi': max_mi,
    #    'min_tc': min_tc,
    #    'min_dwkl': min_dwkl,
    #    'min_kl': min_kl,
    #}
    #torch.save(states, f'{study_path}/{trial.number}.pt')

    torch.cuda.empty_cache()

    return min_rec, max_mi, min_tc


def main():
    n_trials = None  #2**11
    timeout = 6.9 * 24 * 3600
    n_startup_trials = 2**6

    #n_trials = 1
    #timeout = 7200
    #n_startup_trials = 0

    study = optuna.create_study(
        storage=f'sqlite:///{study_path}/optuna.db',
        sampler=optuna.samplers.TPESampler(n_startup_trials=n_startup_trials, seed=42),
        pruner=None,  # currently not for multi-objective optimization
        load_if_exists=True,
        directions=['minimize', 'maximize', 'minimize'],
        #directions=['minimize', 'maximize', 'minimize', 'minimize', 'minimize'],
    )
    study.set_metric_names(['rec', 'mi', 'tc'])
    #study.set_metric_names(['rec', 'mi', 'tc', 'dwkl', 'kl'])
    study.optimize(objective, n_trials=n_trials, timeout=timeout, gc_after_trial=True)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print('Study statistics: ')
    print('  Number of finished trials: ', len(study.trials))
    print('  Number of pruned trials: ', len(pruned_trials))
    print('  Number of complete trials: ', len(complete_trials))

    # single-objective
    #print('Best trial:')
    #trial = study.best_trial
    #print('  Value: ', trial.value)
    #print('  Params: ')
    #for key, value in trial.params.items():
    #    print('    {}: {}'.format(key, value))

    # multi-objective
    print('Pareto front:')
    trials = sorted(study.best_trials, key=lambda trial: trial.values)
    for trial in trials:
        rec, mi, tc = trial.values
        #rec, mi, tc, dwkl, kl = trial.values
        print(f'  Trial#{trial.number}')
        print(f'    Values: {rec=}, {mi=}, {tc=}')
        #print(f'    Values: {rec=}, {mi=}, {tc=}, {dwkl=} {kl=}')
        print(f'    Params: {trial.params}')

    print('Trials:')
    df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'),
                                multi_index=True)
    print(df.to_markdown())


if __name__ == '__main__':
    num_z = int(sys.argv[1])
    sampling = sys.argv[2]

    study_path = f"models/d{num_z}_{sampling}_{date.today().strftime('%m%d')}"
    os.makedirs(study_path, exist_ok=True)
    device = torch.device('cuda', 0)

    P_set = np.load('IllustrisTNG_powers.npy')[:, :2]  # shape = (1024, 2, 4, 18)
    P_set = torch.from_numpy(P_set).to(device)
    P_set = torch.log(P_set)
    P_set[:, 1] -= P_set[:, 0]
    Pt_set, Pv_set = P_set[:512], P_set[512:]

    p_set = np.load('IllustrisTNG_params.npy')[:, 0, -1]  # shape = (1024, 3)
    p_set = torch.from_numpy(p_set).to(device)
    p_set = torch.log(p_set)
    pt_set, pv_set = p_set[:512], p_set[512:]

    P_shape, num_p = P_set.shape[1:], p_set.shape[1]

    t_dataset = TensorDataset(Pt_set, pt_set)
    v_dataset = TensorDataset(Pv_set, pv_set)

    main()
