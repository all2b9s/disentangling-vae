import math
import os
import sys
from datetime import date

import numpy as np
import torch
from torch.optim import lr_scheduler
import optuna
from optuna.trial import TrialState

from disvae.models.vae import VAE
from disvae.models.losses import BtcvaeLoss


def objective(trial):
    #training hyperparams
    lr = trial.suggest_float('lr', 1e-6, 1e-3, log=True)
    b1 = 1 - trial.suggest_float('1-β₁', 0.1, 0.5, log=True)
    b2 = 1 - trial.suggest_float('1-β₂', 1e-3, 0.1, log=True)
    wd = trial.suggest_float('wd', 1e-6, 1e-3, log=True)
    batch_size = 2 ** trial.suggest_int('㏒₂B', 7, 13)

    #model hyperparams
    depth = trial.suggest_int('D', 1, 7)
    width = 2 ** trial.suggest_int('㏒₂W', 4, 9)
    cos_dim = 4
    spec_dim = 18

    #loss hyperparams
    tau = trial.suggest_float('τ', 0.1, 100, log=True)
    alpha = - trial.suggest_int('-α', 1, 100, log=True)
    beta = trial.suggest_int('β', 1, 100, log=True)
    gamma = trial.suggest_float('γ', 0.1, 10, log=True)

    t_loader = torch.utils.data.DataLoader(t_dataset, batch_size=batch_size, shuffle=True)
    v_loader = torch.utils.data.DataLoader(v_dataset, batch_size=len(v_dataset), shuffle=False)
    model = VAE(spec_dim=spec_dim,
                latent_dim=latent_dim,
                hyperparameters=[depth, depth, width],
                cos_dim=cos_dim,
               ).to(device, non_blocking=True)
    #TODO confirm whether to use 512*16, 512*4, or 512 as n_data here
    criterion = BtcvaeLoss(len(t_dataset), alpha=alpha, beta=beta, gamma=gamma, tau=tau)
    assert len(t_dataset) == len(v_dataset), 'otherwise we need two criteria'
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(b1, b2), weight_decay=wd)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, threshold=0.01)

    epoch = 0
    min_train_loss, min_valid_loss = math.inf, math.inf
    min_recon_loss, max_mi_loss = math.inf, -math.inf
    min_tc_loss, min_dw_kl_loss = math.inf, math.inf
    min_kl_loss = math.inf
    while optimizer.param_groups[0]['lr'] > 1e-8:
        model.train()
        storer = {k: torch.zeros(1, device=device) for k in
                  ['loss', 'recon_loss', 'mi_loss', 'tc_loss', 'dw_kl_loss', 'kl_loss']
                  + [f'kl_loss_{d}' for d in range(latent_dim)]}
        for x_batch, y_batch in t_loader:
            z, latent_dis, latent_sample = model(x_batch)
            loss = criterion(y_batch, z, latent_dist=latent_dis, is_train=True,
                             storer=storer, latent_sample=latent_sample)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_loss = storer['loss'].item() / len(t_loader) / tau
        min_train_loss = min(epoch_loss, min_train_loss)

        model.eval()
        storer = {k: torch.zeros(1, device=device) for k in
                  ['loss', 'recon_loss', 'mi_loss', 'tc_loss', 'dw_kl_loss', 'kl_loss']
                  + [f'kl_loss_{d}' for d in range(latent_dim)]}
        with torch.no_grad():
            for x_batch, y_batch in v_loader:
                z, latent_dis, latent_sample = model(x_batch)
                loss = criterion(y_batch, z, latent_dist=latent_dis, is_train=False,
                                 storer=storer,latent_sample=latent_sample)
            epoch_loss = storer['loss'].item() / len(v_loader) / tau
            recon_loss = storer['recon_loss'].item() / len(v_loader)
            mi_loss = storer['mi_loss'].item() / len(v_loader)
            tc_loss = storer['tc_loss'].item() / len(v_loader)
            dw_kl_loss = storer['dw_kl_loss'].item() / len(v_loader)
            kl_loss = storer['kl_loss'].item() / len(v_loader)
            min_valid_loss = min(epoch_loss, min_valid_loss)
            min_recon_loss = min(recon_loss, min_recon_loss)
            max_mi_loss = max(mi_loss, max_mi_loss)
            min_tc_loss = min(tc_loss, min_tc_loss)
            min_dw_kl_loss = min(dw_kl_loss, min_dw_kl_loss)
            min_kl_loss = min(kl_loss, min_kl_loss)

        scheduler.step(epoch_loss)

        epoch += 1

    print(f'{trial.number=}, {epoch=}, {min_train_loss=}, {min_valid_loss=}')
    print(f'  {min_recon_loss=}, {max_mi_loss=}, {min_tc_loss=}, {min_dw_kl_loss=} '
          + f'{min_kl_loss=}', flush=True)

    trial.set_user_attr('epoch', epoch)
    trial.set_user_attr('min_train_loss', min_train_loss)
    trial.set_user_attr('min_valid_loss', min_valid_loss)
    trial.set_user_attr('min_recon_loss', min_recon_loss)
    trial.set_user_attr('max_mi_loss', max_mi_loss)
    trial.set_user_attr('min_tc_loss', min_tc_loss)
    trial.set_user_attr('min_dw_kl_loss', min_dw_kl_loss)
    trial.set_user_attr('min_kl_loss', min_kl_loss)

    states = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'rng': torch.get_rng_state(),
        'min_train_loss': min_train_loss,
        'min_valid_loss': min_valid_loss,
        'min_recon_loss': min_recon_loss,
        'max_mi_loss': max_mi_loss,
        'min_tc_loss': min_tc_loss,
        'min_dw_kl_loss': min_dw_kl_loss,
        'min_kl_loss': min_kl_loss,
    }
    torch.save(states, f'{study_path}/{trial.number}.pt')

    torch.cuda.empty_cache()

    return min_recon_loss, max_mi_loss, min_tc_loss


def main():
    n_trials = None  #2**11
    timeout = 6.9 * 24 * 3600
    n_startup_trials = 2**6

    #n_trials = 1
    #timeout = 7200
    #n_startup_trials = 0

    study = optuna.create_study(
        storage=f'sqlite:///{study_path}/optuna.db',
        sampler=optuna.samplers.TPESampler(n_startup_trials=n_startup_trials),
        pruner=None,  # currently not for multi-objective optimization
        load_if_exists=True,
        directions=['minimize', 'maximize', 'minimize'],
        #directions=['minimize', 'maximize', 'minimize', 'minimize', 'minimize'],
    )
    study.set_metric_names(['recon_loss', 'mi_loss', 'tc_loss'])
    #study.set_metric_names(['recon_loss', 'mi_loss', 'tc_loss', 'dw_kl_loss', 'kl_loss'])
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
        recon_loss, mi_loss, tc_loss = trial.values
        #recon_loss, mi_loss, tc_loss, dw_kl_loss, kl_loss = trial.values
        print(f'  Trial#{trial.number}')
        print(f'    Values: {recon_loss=}, {mi_loss=}, {tc_loss=}')
        #print(f'    Values: {recon_loss=}, {mi_loss=}, {tc_loss=}, {dw_kl_loss=} {kl_loss=}')
        print(f'    Params: {trial.params}')

    print('Trials:')
    df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'),
                                multi_index=True)
    print(df.to_markdown())


if __name__ == '__main__':
    latent_dim = int(sys.argv[1])

    study_path = f"models/d{latent_dim}_{date.today().strftime('%m%d')}"
    os.makedirs(study_path, exist_ok=True)
    device = torch.device('cuda', 0)

    data_set = np.load('repeat_data.npy')
    data_set = data_set.reshape([1024*16, 4, 22]).astype(np.float32)
    t_set = torch.tensor(data_set[:512*16]).to(device)
    t_dataset = torch.utils.data.TensorDataset(t_set[:, :3], t_set[:, 3, :18])
    v_set = torch.tensor(data_set[512*16:]).to(device)
    v_dataset = torch.utils.data.TensorDataset(v_set[:, :3], v_set[:, 3, :18])

    main()
