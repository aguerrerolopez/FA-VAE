import torch
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pandas as pd
import argparse
import favae as sshiba
import pickle

parser = argparse.ArgumentParser(description='SSHIBA IMG VAE with adapative learning over Z latent space')
parser.add_argument('--lr', type=float, default=1.0,
                   help='Learning rate value for adapative Z learning. Default = 1 (no adaptative learning)')
args = parser.parse_args()

print(args.lr)

print("Loading pretrained vanillaVAE...")
pretrained_vae_path = open('../condition_a_VAE/results/pretrained_celeba_vanillavae_v3.pickle', 'rb')
pretrained_vae = pickle.load(pretrained_vae_path)
pretrained_vae_path.close()
del pretrained_vae_path

# ============ CREATE THE PRETRAINED VIEW ===================
print("Creating mu_Q and stdQ...")
mu, var = pretrained_vae['model'].update_x()
q_dist = torch.distributions.Normal(loc=torch.Tensor(mu), scale=torch.Tensor(np.sqrt(var)))
pretrained_X = q_dist.sample().numpy()
del q_dist, mu, var
print(pretrained_X)
torch.cuda.empty_cache()

# ============ READ CELEBA IMAGES ===================
print("Loading dataset")
celeba = datasets.ImageFolder(root="../datasets/celeba/", transform=transforms.Compose([
                                  transforms.Resize((64, 64)),
                                  transforms.ToTensor()
                              ]))

print("Reading dataset")
loader = DataLoader(celeba, batch_size=len(celeba))

celeba_numpy = next(iter(loader))[0]
celeba_train = celeba_numpy[:30000, :, :, :]
del celeba_numpy, loader, celeba

# ============ READ LABELS ===================
print("Reading labels")
def get_annotation(fnmtxt, verbose=True):
    if verbose:
        print("_"*70)
        print(fnmtxt)
    
    rfile = open(fnmtxt , 'r' ) 
    texts = rfile.read().split("\n") 
    rfile.close()

    columns = np.array(texts[1].split(" "))
    columns = columns[columns != ""]
    df = []
    for txt in texts[2:]:
        txt = np.array(txt.split(" "))
        txt = txt[txt!= ""]
    
        df.append(txt)
        
    df = pd.DataFrame(df)

    if df.shape[1] == len(columns) + 1:
        columns = ["image_id"]+ list(columns)
    df.columns = columns   
    df = df.dropna()
    if verbose:
        print(" Total number of annotations {}\n".format(df.shape))
        print(df.head())
    ## cast to integer
    for nm in df.columns:
        if nm != "image_id":
            df[nm] = pd.to_numeric(df[nm],downcast="integer")
    return(df)
attr = get_annotation('../datasets/celeba/list_attr_celeba.txt', verbose=False).iloc[:30000, :]

attr_list = ["image_id", "Smiling", "Wearing_Lipstick", "Male"]
target = attr[attr_list].replace(-1,0)
target_train = target.iloc[:30000, 1:].to_numpy()

# ============ FAVAE MODEL ===================
print("Creating model")
hyper_parameters = {'sshiba': {"prune": 1, "myKc": 100, "pruning_crit": 1e-3, "max_it": 500, "latentspace_lr": args.lr}}

store = False

myModel_new = sshiba.SSHIBA(hyper_parameters['sshiba']['myKc'], hyper_parameters['sshiba']['prune'], latentspace_lr=hyper_parameters['sshiba']['latentspace_lr'])

#Pretrained vanilla vae latent space
input_pre = myModel_new.struct_data(pretrained_X, 'reg')
#Huge VAE
betaVAE = myModel_new.struct_data(celeba_train.numpy(), 'img', latent_dim=100, lr=1e-3, epochs=15, dataset="celeba")
# Labels
labels = myModel_new.struct_data(target_train, 'mult')
# ============ FAVAE TRAINING ===================
print("Training model")
myModel_new.fit(betaVAE, input_pre, labels,
            max_iter=hyper_parameters['sshiba']['max_it'],
            pruning_crit=hyper_parameters['sshiba']['pruning_crit'],
            verbose=1,
            store=store)

import pickle
store = {"model": myModel_new}
with open('./results/model_celeba_pretrained_conditioned_v3_2_'+str(args.lr)+'.pickle', 'wb') as handle:
    pickle.dump(store, handle, protocol=pickle.HIGHEST_PROTOCOL)