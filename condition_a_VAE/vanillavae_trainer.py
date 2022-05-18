import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import celeba_deepvanilla_vae as vae
import wandb
torch.manual_seed(0)

# # ============================ Init WandB ===========================
wandb_obj = wandb.init(project="vae-sshiba", group='Escenario 1', job_type='pretrain_celeba_alone', entity="alexjorguer")
# # ============================ Load CELEBA ===========================
print("Loading dataset")
celeba = datasets.ImageFolder(root="../datasets/celeba/", transform=transforms.Compose([
                                  transforms.Resize((64, 64)),
                                  transforms.ToTensor()
                              ]))
loader = DataLoader(celeba, batch_size=len(celeba))
celeba_numpy = next(iter(loader))[0]
celeba_train = celeba_numpy[:30000, :, :, :]
print("Dataset loaded")
# # ============================ Init VAE ===========================
print("Creating the VAE model...")
img_vae = vae.ImgVAE(dimx=100, channels=3, lr=1e-3, h=celeba_train.shape[2], w=celeba_train.shape[3], dataset='celeba')
# # ============================ Train VAE ===========================
print("Training the model...")
img_vae.trainloop(img=celeba_train, Z=None, W=None, b=None, tau=None, epochs=250, wandb=wandb_obj, favae=False)
print("Model trained.")
# # ============================ Finish WandB ===========================
wandb_obj.finish()
# # ============================ Save model ===========================
import pickle
store = {"model": img_vae}
with open('./results/pretrained_celeba_vanillavae_v3.pickle', 'wb') as handle:
    pickle.dump(store, handle, protocol=pickle.HIGHEST_PROTOCOL)
