import torch
import numpy as np
from torch.nn import functional as F
from disvae.models.vae import VAE
from disvae.models.losses import BtcvaeLoss
from collections import defaultdict
import matplotlib.pyplot as plt
import os

import optuna
from optuna.trial import TrialState
#nohup python -u VAE_train.py >> ./logs/rcl_1.out 2>&1 &

device=torch.device(1 if torch.cuda.is_available() else 'cpu')
print(device)

study_name = 'VAE_test_rcl'
storage = 'sqlite:///./models/'+study_name+'.db'

params = np.load('./IllustrisTNG_params.npy')
times  = np.load('./IllustrisTNG_times.npy')
times = times[:, :, :, np.newaxis]
params = np.concatenate([params, times], axis=3)
params = params.transpose([0,2,1,3])
params  = torch.tensor(params.reshape([4096,2,4]))
train_params = params[:2048,:,:]
valid_params = params[2048:,:,:]

spec = np.load('./IllustrisTNG_powers.npy')
spec = spec[:,:2,:,:].transpose([0,2,1,3])
spec = torch.tensor(spec.reshape([4096,2,-1]))
spec[:,1,:] = spec[:,1,:]/spec[:,0,:]
spec[:,0] = np.log10(spec[:,0])
spec[:,1] = 10*np.log10(spec[:,1])
train_spec = spec[:2048,:,:]
valid_spec = spec[2048:,:,:]


def objective(trial):
    ############################################################################
    ############Training parameters
    e_layers = trial.suggest_int("encoder_layers", 2, 6)
    d_layers = trial.suggest_int("decoder_layers", 2, 6)
    num_filters = trial.suggest_int("num_filters", 2**4, 2**9,log=True) 
    lr = trial.suggest_float("lr", 1e-7, 1e-4, log=True)
    #weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch_size = 2048 #trial.suggest_int("batch_size", 1, 32,log=True)
    epoches = 1000
    ############model parameters
    latent_dim = 5
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
    
    t_set = torch.cat([train_spec,train_params],dim=2).to(device)
    t_dataset = torch.utils.data.TensorDataset(t_set, t_set[:,1,:18])
    t_loader = torch.utils.data.DataLoader(t_dataset, batch_size=256, shuffle=True)
    
    v_set = torch.cat([valid_spec,valid_params],dim=2).to(device)
    v_dataset = torch.utils.data.TensorDataset(v_set, v_set[:,1,:18])
    v_loader = torch.utils.data.DataLoader(v_dataset, batch_size=2048, shuffle=False)
    
    
    model = VAE(spec_dim=spec_dim, 
                latent_dim = latent_dim, 
                hyperparameters=[e_layers, e_layers], 
                cos_dim = cos_dim
               ).to(device)
    criterion                   = BtcvaeLoss(n_data, alpha=alpha, beta=beta, gamma=gamma, record_loss_every=1)
    optimizer                   = torch.optim.Adam(model.parameters(),lr=lr)
    n_count = 0
    for epoch in range (0,epoches):
        if n_count>=100 and epoch>700:
            return min_valid
        storer = defaultdict(list)
        for i, (x_batch, y_batch) in enumerate(t_loader):
                model = model
                optimizer.zero_grad()
                z, latent_dis, latent_sample = model(x_batch)
                loss = criterion(y_batch, z, latent_dist= latent_dis, is_train=True, storer= storer,latent_sample= latent_sample)
                loss.backward()
                optimizer.step()
                
        with torch.no_grad():
            for i, (x_batch, y_batch) in enumerate(v_loader):
                    model = model
                    optimizer.zero_grad()
                    z, latent_dis, latent_sample = model(x_batch)
                    loss = criterion(y_batch, z, latent_dist= latent_dis, is_train=False, storer= storer,latent_sample= latent_sample)
            valid_loss = np.array(storer['recon_loss']).mean() #Remember to put "-" if using mi_loss
            print(epoch, storer['recon_loss'], storer['mi_loss'])
                    
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
    optuna.delete_study(study_name=study_name,storage= storage)
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
