# %%
import pickle
from matplotlib import pyplot as plt
import numpy as np
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"


torch.manual_seed(0)
np.random.seed(0)
#================= LOAD MODELS =================
usps_mnist_path = open('./results/esc1_pretrained_conditioned.pickle', 'rb')
model = pickle.load(usps_mnist_path)
usps_mnist_path.close()
del usps_mnist_path

usps_mnist_path = open('./results/pretrained_celeba_vanillavae_v3.pickle', 'rb')
modelpre = pickle.load(usps_mnist_path)
usps_mnist_path.close()
del usps_mnist_path

#%% # ================================= GENERATE IMAGES CONDITIONALLY =================================
import itertools
labels_name = ["Smiling", "Wearing_Lipstick", "Male"]
labels = np.asarray(list(itertools.product([0, 1], repeat=3)))
labels_view = model['model'].struct_data(labels, 'mult')
img_conditional,_,_ = model['model'].predict([1], [0], labels_view)
fig, ax = plt.subplots(nrows=2, ncols=4, sharex=True, sharey=True)
ploti=0
for i in range(2):
    for j in range(4):
        ax[i,j].set_title(labels[ploti])
        ax[i,j].imshow(np.transpose(img_conditional[ploti], axes=[1,2,0]))
        ploti+=1
plt.tight_layout()
plt.show()

#%% # ================================= Modify attributes from the same images=================================

sel = 3000
plt.imshow(np.transpose(model['model'].t[0]['data'][sel], axes=[1,2,0]))
imgs = model['model'].t[0]['data'][np.repeat(sel,8)]

import itertools
labels_name = ["Smiling", "Wearing_Lipstick", "Male"]
labels = np.asarray(list(itertools.product([0, 1], repeat=3)))

labels_view = model['model'].struct_data(labels, 'mult')
img_view = model['model'].struct_data(imgs, 'img', dataset='celeba')


img_conditional,_,_ = model['model'].predict([0,1], [0], img_view, labels_view)
fig, ax = plt.subplots(nrows=2, ncols=4, sharex=True, sharey=True)
ploti=0
for i in range(2):
    for j in range(4):
        ax[i,j].set_title(labels[ploti])
        ax[i,j].imshow(np.transpose(img_conditional[ploti].data.cpu().numpy(), axes=[1,2,0]))
        ploti+=1
plt.tight_layout()
plt.show()
#%% # ================================= Plot ELBO evolution for FA-VAE vs Vanilla-VAE =================================
plt.figure()
plt.title("Gaussian LogLikelihood")
plt.xlabel("Epochs")
plt.plot(-np.concatenate(model['model'].recon_loss), label="FA-VAE")
plt.plot(-np.array(modelpre['model'].reconstruc_during_training), label="Vanilla VAE")
plt.legend()
plt.show()

plt.figure()
plt.title("KL divergence between q and p")
plt.xlabel("Epochs")
plt.plot(np.concatenate(model['model'].kl_q_p)[1:], label="FA-VAE")
plt.plot(np.array(modelpre['model'].KL_QandP[1:]), label="Vanilla VAE")
plt.legend()
plt.show()