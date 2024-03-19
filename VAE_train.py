import torch
import numpy as np
from torch.nn import functional as F
from torch.optim import lr_scheduler
from disvae.models.vae import VAE
from disvae.models.losses import BtcvaeLoss
from collections import defaultdict
import matplotlib.pyplot as plt
import os

import optuna
from optuna.trial import TrialState
#nohup python -u VAE_train.py >> ./logs/ti_d2_1.out 2>&1 &

device=torch.device(0 if torch.cuda.is_available() else 'cpu')
print(device)

study_name = 'VAE_test_ti_d2'
storage = 'sqlite:///./models/'+study_name+'.db'

data_set = np.load('./repeat_data.npy')
data_set = data_set.reshape([1024*16,4,22]).astype(np.float32)

t_set = torch.tensor(data_set[:512*16]).to(device)
t_dataset = torch.utils.data.TensorDataset(t_set, t_set[:,3,:18])
t_loader = torch.utils.data.DataLoader(t_dataset, batch_size=2048, shuffle=True)
    
v_set = torch.tensor(data_set[512*16:]).to(device)
v_dataset = torch.utils.data.TensorDataset(v_set, v_set[:,3,:18])
v_loader = torch.utils.data.DataLoader(v_dataset, batch_size=2048, shuffle=False)


def objective(trial):
    ############################################################################
    ############Training parameters
    e_layers = trial.suggest_int("encoder_layers", 2, 7)
    #d_layers = trial.suggest_int("decoder_layers", 2, 6)
    num_filters = trial.suggest_int("num_filters", 2**4, 2**9,log=True) 
    lr = trial.suggest_float("lr", 1e-6, 1e-4, log=True)
    #weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch_size = 2048 #trial.suggest_int("batch_size", 1, 32,log=True6
    epoches = 3000
    ############model parameters
    latent_dim = 2
    cos_dim = 4
    spec_dim = 18
    n_data = 2048
    ############loss parameters
    tau = trial.suggest_float("tau", 0.1, 100, log=True)
    alpha = -trial.suggest_int("alpha", 1, 100, log=True)
    beta = trial.suggest_int("beta", 1, 100, log=True)
    gamma = trial.suggest_float("gamma", 0.1, 10, log=True)
    ############################################################################
    min_valid = 1e40
    fout = './models/'+study_name+'.out'   
    fmodel = './models/'+study_name+'/'+study_name+'_%d.pt'%(trial.number)
    path = './models/'+study_name+'/'
    folder = os.path.exists(path)
    if not folder:                  
        os.makedirs(path)
    f = open(fout, 'a')
    f.write('Trial %d starts! \n' % (trial.number))
    ############################################################################
    
    model = VAE(spec_dim=spec_dim, 
                latent_dim = latent_dim, 
                hyperparameters=[e_layers, e_layers], 
                cos_dim = cos_dim
               ).to(device)
    criterion                   = BtcvaeLoss(n_data, alpha=alpha, beta=beta, gamma=gamma, record_loss_every=1)
    optimizer                   = torch.optim.Adam(model.parameters(),lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    n_count = 0
    f_count = 0
    for epoch in range (0,epoches):
        if f_count>=3 and epoch>700:
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
        if n_count>=100:
            scheduler.step()
            n_count = 0
            f_count += 1
            print('Forward !')
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
        
        trial.report(min_valid,epoch)
        f.write('%d %.5e %.5e \n'%(epoch,valid_loss,min_valid))
    f.close()
    torch.cuda.empty_cache()
    return min_valid
    
def main():
########################################################################################################
    n_trials       = 100  # set to None for infinite
    startup_trials = 25
########################################################################################################
    #optuna.delete_study(study_name=study_name,storage= storage)
    sampler = optuna.samplers.TPESampler(n_startup_trials=startup_trials)
    study = optuna.create_study(direction="minimize",
                                study_name=study_name,
                                sampler=sampler,
                                storage=storage,
                                load_if_exists=True)
    #study.enqueue_trial({"num_filters": 256,"lr":5e-5,"batch_size":1})
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
