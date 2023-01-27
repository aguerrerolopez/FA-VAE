import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import celeba_deepvanilla_vae as vae
import wandb
torch.manual_seed(0)

# # ============================ Init WandB ===========================
# wandb_obj = wandb.init(project="vae-sshiba", group='Escenario 1', job_type='pretrain_celeba_alone', entity="alexjorguer")
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
img_vae.trainloop(img=celeba_train, Z=None, W=None, b=None, tau=None, epochs=250, wandb=None, favae=False)
print("Model trained.")
# # ============================ Finish WandB ===========================
# wandb_obj.finish()
# # ============================ Save model ===========================
import pickle
store = {"model": img_vae}
with open('./results/pretrained_celeba_vanillavae_v3.pickle', 'wb') as handle:
    pickle.dump(store, handle, protocol=pickle.HIGHEST_PROTOCOL)

# # ============================ Load model ===========================

with open('./results/pretrained_celeba_vanillavae_v3.pickle', 'rb') as handle:
    store = pickle.load(handle)
img_vae = store["model"]

# Plot and img example from the model in RGB
import matplotlib.pyplot as plt

image_example = img_vae.img[0, :, :, :]
image_example = image_example.permute(1, 2, 0)
image_example = image_example.detach().numpy()
plt.imshow(image_example)

# Pass this image to the model and plot the result
img_vae.eval()
image_example = img_vae.img[0, :, :, :]
image_example = image_example.unsqueeze(0).to(img_vae.device)
# Encode the image
mu, std = img_vae.encoder(image_example)
# Given same mu and std, we can generate different images
# Lets generate 10 different images and store them in a list
x_hat_list = []
z_list = []
for i in range(10):
    # Reparametrization trick
    eps = torch.normal(0, 5, size=std.size()).to(img_vae.device)
    z = eps.mul(std).add_(mu)
    z_list.append(z)
    # Decode the image
    x_hat = img_vae.decoder(z)
    # Plot the result
    x_hat = x_hat[0, :, :, :]
    x_hat = x_hat.permute(1, 2, 0)
    x_hat = x_hat.cpu().detach().numpy()
    x_hat_list.append(x_hat)
    plt.imshow(x_hat)
    plt.show()