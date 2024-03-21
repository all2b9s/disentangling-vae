import os
from collections import defaultdict

import numpy as np
import torch
from torch.optim import lr_scheduler
import optuna
from optuna.trial import TrialState

from disvae.models.vae import VAE
from disvae.models.losses import BtcvaeLoss
#nohup python -u VAE_train.py >> ./logs/ti_d2_3.out 2>&1 &


device=torch.device(0 if torch.cuda.is_available() else 'cpu')
print(device)

study_name = 'ti_nan'
storage = 'sqlite:///./models/'+study_name+'.db'

data_set = np.load('./repeat_data.npy')
data_set = data_set.reshape([1024*16,4,22]).astype(np.float32)

t_set = torch.tensor(data_set[:512*16]).to(device)
t_dataset = torch.utils.data.TensorDataset(t_set, t_set[:,3,:18])
#t_loader = torch.utils.data.DataLoader(t_dataset, batch_size=None, shuffle=True)

v_set = torch.tensor(data_set[512*16:]).to(device)
v_dataset = torch.utils.data.TensorDataset(v_set, v_set[:,3,:18])
#v_loader = torch.utils.data.DataLoader(v_dataset, batch_size=None, shuffle=False)


def objective(trial):
    #training hyperparams
    lr = 2e-4 #trial.suggest_float('lr', 1e-6, 1e-3, log=True)
    b1 = 1 - trial.suggest_float('1-β₁', 0.1, 0.5, log=True)
    b2 = trial.suggest_float('β₂', 1e-3, 0.1, log=True)
    wd = trial.suggest_float('wd', 1e-6, 1e-3, log=True)
    batch_size = 2**10 #2 ** trial.suggest_int('log2(batch_size)', 7, 11)
    epoches = 2**11

    #model hyperparams
    depth = 4 #trial.suggest_int('depth', 1, 7)
    width = 2**9 #2 ** trial.suggest_int('log2(width)', 4, 9)
    latent_dim = 2
    cos_dim = 4
    spec_dim = 18
    n_data = 512*16

    #loss hyperparams
    tau = 100 #trial.suggest_float('τ', 0.1, 100, log=True)
    alpha = -21 #- trial.suggest_int('-α', 1, 100, log=True)
    beta = 7 #trial.suggest_int('β', 1, 100, log=True)
    gamma = 0.25 #trial.suggest_float('γ', 0.1, 10, log=True)

    min_valid = 1e40
    fout = './models/'+study_name+'.out'
    fmodel = './models/'+study_name+'/'+study_name+'_%d.pt'%(trial.number)
    fmodel_ = './models/'+study_name+'/'+study_name+'_'+str(trial.number)+'_final.pt'
    path = './models/'+study_name+'/'
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    f = open(fout, 'a')
    f.write('Trial %d starts! \n' % (trial.number))

    t_loader = torch.utils.data.DataLoader(t_dataset, batch_size=batch_size, shuffle=True)
    v_loader = torch.utils.data.DataLoader(v_dataset, batch_size=batch_size, shuffle=False)
    model = VAE(spec_dim=spec_dim,
                latent_dim = latent_dim,
                hyperparameters=[depth, depth, width],
                cos_dim = cos_dim
               ).to(device, non_blocking=True)
    criterion = BtcvaeLoss(n_data, alpha=alpha, beta=beta, gamma=gamma, tau=tau, record_loss_every=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) #, betas=(b1, b2), weight_decay=wd
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=50, threshold=1e-2,)

    n_count = 0
    f_count = 0
    for epoch in range (0, epoches):
        if n_count>=100 and epoch>700:
            return min_valid
        storer = defaultdict(list)
        for i, (x_batch, y_batch) in enumerate(t_loader):
                model = model
                optimizer.zero_grad()
                z, latent_dis, latent_sample = model(x_batch)
                loss = criterion(y_batch, z, latent_dist = latent_dis, is_train=True, storer= storer,latent_sample= latent_sample)
                #print(latent_dis)
                #print(z)
                loss.backward()
                optimizer.step()
        #print(z)
        with torch.no_grad():
            for i, (x_batch, y_batch) in enumerate(v_loader):
                    model = model
                    optimizer.zero_grad()
                    z, latent_dis, latent_sample = model(x_batch)
                    loss = criterion(y_batch, z, latent_dist = latent_dis, is_train=False, storer= storer,latent_sample= latent_sample)
            valid_loss = -np.array(storer['mi_loss']).mean() #Remember to put "-" if using mi_loss
            print(epoch, np.array(storer['recon_loss']).mean(), np.array(storer['mi_loss']).mean())
            
            if valid_loss<min_valid:
                n_count = 0
                min_valid = valid_loss
                torch.save(model, fmodel)
            else:
                n_count+=1
            scheduler.step(min_valid)
        trial.report(min_valid,epoch)
        f.write('%d %.5e %.5e \n'%(epoch,np.array(storer['recon_loss']).mean(),np.array(storer['mi_loss']).mean()))
        torch.save(model, fmodel_)
    f.close()
    torch.cuda.empty_cache()
    return min_valid


def main():
    n_trials = 2**7  # set to None for infinite
    n_startup_trials = 2**5

    optuna.delete_study(study_name=study_name,storage= storage)
    sampler = optuna.samplers.TPESampler(n_startup_trials=n_startup_trials)
    study = optuna.create_study(direction="minimize",
                                study_name=study_name,
                                sampler=sampler,
                                storage=storage,
                                load_if_exists=True)
    study.optimize(objective, n_trials=n_trials,timeout = 3600*48)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    print(df)


if __name__ == "__main__":
    main()