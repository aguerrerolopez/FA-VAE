import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import betaVAE as vae
import wandb
torch.manual_seed(0)

# # ============================ Init WandB ===========================
wandb_obj = wandb.init(project="vae-sshiba", group='Escenario 2', job_type='Pretrained BVAE', entity="alexjorguer")
# # ============================ Load CELEBA ===========================
celeba = datasets.ImageFolder(root="../datasets/celeba/", transform=transforms.Compose([
                                  transforms.Resize((64, 64)),
                                  transforms.ToTensor()
                              ]))
loader = DataLoader(celeba, batch_size=len(celeba))
celeba_numpy = next(iter(loader))[0]
celeba_train = celeba_numpy[:30000, :, :, :]
celeba_tst = celeba_numpy[30000:, :, :, :]

# # ============================ Init VAE ===========================
img_vae = vae.ImgVAE(dimx=100, channels=3, lr=1e-3, h=celeba_train.shape[2], w=celeba_train.shape[3], dataset='celeba')
# # ============================ Train VAE ===========================
img_vae.trainloop(img=celeba_train, Z=None, W=None, b=None, tau=None, epochs=1000, wandb=wandb_obj, favae=False)
# # ============================ Finish WandB ===========================
wandb_obj.finish()
# # ============================ Save model ===========================
import pickle
store = {"model": img_vae}
with open('./results/betavae.pickle', 'wb') as handle:
    pickle.dump(store, handle, protocol=pickle.HIGHEST_PROTOCOL)

