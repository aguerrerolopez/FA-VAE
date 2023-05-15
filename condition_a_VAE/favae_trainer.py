import torch
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pandas as pd
import argparse
import favae as FAVAE


# # ============================ Adaptive learning argument ===========================
# parser = argparse.ArgumentParser(description='FAVAE IMG VAE with adapative learning over Z latent space')
# parser.add_argument('--lr', type=float,
#                     help='Learning rate value for adapative Z learning. Default = 1 (no adaptative learning)')
# args = parser.parse_args()
# print(args.lr)
samples_to_use = 500
torch.manual_seed(0)
# # ============================ Load CELEBA dataset ===========================
print("Loading dataset...")
celeba = datasets.ImageFolder(
    root="../datasets/celeba/",
    transform=transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()]),
)
loader = DataLoader(celeba, batch_size=len(celeba))

celeba_numpy = next(iter(loader))[0].numpy()
celeba_train = celeba_numpy[:samples_to_use, :, :, :]
del loader
print("Dataset loaded")
# # ============================ Load CELEBA attributes ===========================
print("Loading attributes...")


def get_annotation(fnmtxt, verbose=False):
    if verbose:
        print("_" * 70)
        print(fnmtxt)

    rfile = open(fnmtxt, "r")
    texts = rfile.read().split("\n")
    rfile.close()

    columns = np.array(texts[1].split(" "))
    columns = columns[columns != ""]
    df = []
    for txt in texts[2:]:
        txt = np.array(txt.split(" "))
        txt = txt[txt != ""]

        df.append(txt)

    df = pd.DataFrame(df)

    if df.shape[1] == len(columns) + 1:
        columns = ["image_id"] + list(columns)
    df.columns = columns
    df = df.dropna()
    if verbose:
        print(" Total number of annotations {}\n".format(df.shape))
        print(df.head())
    ## cast to integer
    for nm in df.columns:
        if nm != "image_id":
            df[nm] = pd.to_numeric(df[nm], downcast="integer")
    return df


attr = get_annotation("../datasets/celeba/list_attr_celeba.txt").iloc[
    :samples_to_use, :
]
attr_list = ["image_id", "Smiling", "Wearing_Lipstick", "Male"]
target = attr[attr_list].replace(-1, 0)
target_train = target.iloc[:samples_to_use, 1:].to_numpy()
print("Attributes loaded")
# # ============================ INIT FAVAE ===========================
print("Creating FAVAE model...")
hyper_parameters = {
    "FAVAE": {
        "prune": 1,
        "myKc": 100,
        "pruning_crit": 1e-3,
        "max_it": 500,
        "latentspace_lr": 1,
    }
}
FAVAE_model = FAVAE.SSHIBA(
    hyper_parameters["FAVAE"]["myKc"],
    hyper_parameters["FAVAE"]["prune"],
    latentspace_lr=hyper_parameters["FAVAE"]["latentspace_lr"],
)
# # ============================ Prepare inputs ===========================
# CELEBA images
celeba_in = FAVAE_model.struct_data(
    celeba_train, "img", latent_dim=20, lr=1e-3, dataset="celeba"
)
# CELEBA attributes
attributes = FAVAE_model.struct_data(target_train, "mult")
print("Training FAVAE model...")
FAVAE_model.fit(
    celeba_in,
    attributes,
    max_iter=hyper_parameters["FAVAE"][
        "max_it"
    ],  # Number maximum of iterations till convergence
    pruning_crit=hyper_parameters["FAVAE"][
        "pruning_crit"
    ],  # Criterium to prune K dimensions of Z shared latent space
    verbose=1,
    pretrained=0,
    store=False,
)
# ============================ RESULTS ===========================
print("Storing FAVAE model...")
import pickle

store = {"model": FAVAE_model}
with open("./results/esc1_favae_celeba_attributes_predictiveG.pickle", "wb") as handle:
    pickle.dump(store, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("FAVAE model stored. See you!")
