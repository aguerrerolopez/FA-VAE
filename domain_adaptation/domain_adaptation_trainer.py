
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import argparse
import favae as sshiba
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import os

parser = argparse.ArgumentParser(description='FA-VAE with adaptive learning over Z latent space')
parser.add_argument('--lr', type=float, default=1.0,
                    help='Learning rate value for adaptive Z learning. Default = 1 (no adaptive learning)')
args = parser.parse_args()

print(args.lr)

torch.manual_seed(0)
# ============ LOAD CELEBA ===================
print("Loading celeba dataset...")
celeba = ImageFolder(root="../datasets/celeba/", transform=transforms.Compose([
                                  transforms.Resize((64, 64)),
                                  transforms.ToTensor()
                              ]))
# ============ READ ATTRIBUTES ===================
print("Reading CELEBA ATTRIBUTES...")
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
attr = get_annotation('../datasets/celeba/list_attr_celeba.txt').iloc[:20000, :]

target = attr.replace(-1,0)
# Create a balanced dataset of hairs and eyeglasses:
blond_noglasses = target[target['Blond_Hair']==1][target['Eyeglasses']==0].sample(int(3333/2), replace=True).index.tolist()
blond_glasses = target[target['Blond_Hair']==1][target['Eyeglasses']==1].sample(int(3333/2)+1, replace=True).index.tolist()

brown_glasses = target[target['Brown_Hair']==1][target['Eyeglasses']==1].sample(int(3333/4), replace=True).index.tolist()
brown_noglasses = target[target['Brown_Hair']==1][target['Eyeglasses']==0].sample(int(3333/4)+1, replace=True).index.tolist()

black_glasses = target[target['Black_Hair']==1][target['Eyeglasses']==1].sample(int(3333/4), replace=True).index.tolist()
black_noglasses = target[target['Black_Hair']==1][target['Eyeglasses']==0].sample(int(3333/4)+1, replace=True).index.tolist()

gray_glasses = target[target['Gray_Hair']==1][target['Eyeglasses']==1].sample(int(3333/2), replace=True).index.tolist()
gray_noglasses = target[target['Gray_Hair']==1][target['Eyeglasses']==0].sample(int(3333/2)+1, replace=True).index.tolist()

idx = np.concatenate((blond_glasses, blond_noglasses, brown_glasses, brown_noglasses, black_glasses, black_noglasses, gray_glasses,gray_noglasses))


loader = DataLoader(celeba, batch_size=len(celeba))

celeba_numpy = next(iter(loader))[0]
celeba_train = celeba_numpy[idx, :, :, :]

# ============ LOAD CARTOON ===================
print("Loading CARTOON dataset...")
data = ImageFolder(root="../datasets/cartoonset10k/", transform=transforms.Compose([
                                transforms.Resize((64,64)),
                                transforms.ToTensor()]))

loader = DataLoader(data, batch_size=len(data), shuffle=False)

traindata = next(iter(loader))[0]


# ============ SORT CARTOON BY ATTRIBUTES ===================
print("Loading CARTOON attributes...")
csv_files = os.path.join("../datasets/cartoonset10k/")
hair = []
glasses = []
csv_sorted = []
for root, _, files in sorted(os.walk(csv_files)):
    for file in sorted(files):
        path = os.path.join(root, file)
        csv_sorted.append(path)
for f in csv_sorted[:10000]:
    df = pd.read_csv(f)
    hair.append(df.iloc[:,1][11])
    glasses.append(df.iloc[:,1][12])

glasses = np.asarray(glasses)
# Create a balanced dataset of hairs and eyeglasses:
blond = np.concatenate((np.where(np.array(hair)==0)[0], np.where(np.array(hair)==1)[0], np.where(np.array(hair)==4)[0]))[:3333]
brown = np.concatenate((np.where(np.array(hair)==5)[0], np.where(np.array(hair)==6)[0]))[:3333]
gray = np.concatenate((np.where(np.array(hair)==8)[0], np.where(np.array(hair)==9)[0]))[:3333]

blond_glasses = np.concatenate((blond[glasses[blond]!=11], np.random.choice(blond[glasses[blond]!=11], 1666-blond[glasses[blond]!=11].shape[0])))
blond_noglasses= np.concatenate((blond[glasses[blond]==11], np.random.choice(blond[glasses[blond]==11], 1667-blond[glasses[blond]==11].shape[0])))

brown_glasses = np.concatenate((brown[glasses[brown]!=11], np.random.choice(brown[glasses[brown]!=11], 1666-brown[glasses[brown]!=11].shape[0])))
brown_noglasses = np.concatenate((brown[glasses[brown]==11], np.random.choice(brown[glasses[brown]==11], 1667-brown[glasses[brown]==11].shape[0])))

gray_glasses = np.concatenate((gray[glasses[gray]!=11], np.random.choice(gray[glasses[gray]!=11], 1667-gray[glasses[gray]!=11].shape[0])))
gray_noglasses = np.concatenate((gray[glasses[gray]==11], np.random.choice(gray[glasses[gray]==11], 1667-gray[glasses[gray]==11].shape[0])))

idx = np.concatenate((blond_glasses, blond_noglasses, brown_glasses, brown_noglasses, gray_glasses,gray_noglasses))
cartoon_train = traindata[idx, :, :, :]

# ============ CREATE ATTRIBUTES VIEW BY 3 HAIR COLORS: BLOND, BRUNET AND GRAY ===================
labels = np.ones((10000,1))
labels[:3333] = 0
labels[3333:6666] = 1
labels[6666:] = 2

ohe = OneHotEncoder(sparse=False)
labels = ohe.fit_transform(labels)

# ============================ FAVAE MODEL ===========================
print("STARTING MODEL...")
hyper_parameters = {'sshiba': {"prune": 1, "myKc": 100, "pruning_crit": 1e-3, "max_it": 100, "latentspace_lr": args.lr}}
store = False
myModel_new = sshiba.SSHIBA(hyper_parameters['sshiba']['myKc'], hyper_parameters['sshiba']['prune'], latentspace_lr=hyper_parameters['sshiba']['latentspace_lr'])

# Celeba hair
celebin = myModel_new.struct_data(celeba_train.numpy(), 'img',  lr=1e-3, latent_dim=100, dataset="celeba")
# Cartoon hair
cartin = myModel_new.struct_data(cartoon_train.numpy(), 'img',  lr=1e-3, latent_dim=100, dataset="cartoon")
# Hair labels
labels = myModel_new.struct_data(labels, 'mult')

# ============================ FAVAE TRAINING ===========================
print("training FAVAE MODEL...")
myModel_new.fit(celebin, cartin, labels,
            max_iter=hyper_parameters['sshiba']['max_it'],
            pruning_crit=hyper_parameters['sshiba']['pruning_crit'],
            verbose=1,
            store=store)
# ============================ FAVAE STORAGE ===========================
import pickle
store = {"model": myModel_new}
with open('./results/celeb2cart_hairandeyeglasses_100iter_nobeta_BasReg_'+str(args.lr)+'.pickle', 'wb') as handle:
    pickle.dump(store, handle, protocol=pickle.HIGHEST_PROTOCOL)
