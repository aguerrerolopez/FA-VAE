#%%
import pickle
from matplotlib import pyplot as plt
import numpy as np
import torch
from sklearn.metrics import r2_score
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"

#================= LOAD MODELS =================
pretrained_vae_path = open('./results/pretrained_celeba_bigbeta_v3.pickle', 'rb')
pretrained_vae = pickle.load(pretrained_vae_path)
pretrained_vae_path.close()
del pretrained_vae_path

sshiba_conditioned_path = open('./results/model_celeba_pretrained_conditioned_v3_2_1.0.pickle', 'rb')
sshiba_conditioned = pickle.load(sshiba_conditioned_path)
sshiba_conditioned_path.close()
del sshiba_conditioned_path


#%% 
#================= PLOT ELBO TERMS =================
plt.figure()
plt.title("Reconstruction term")
plt.plot(-np.array(pretrained_vae['model'].reconstruc_during_training)[1:200], 'k', alpha=0.7, label=r'$\beta$'+'-VAE')
plt.plot(-np.hstack(sshiba_conditioned['model'].recon_loss)[1:200], 'r', alpha=0.7, label="FA-VAE")
plt.xlabel("Epochs")
plt.ylabel("Gaussian Log-likelihood")
plt.legend()
plt.show()

plt.figure()
plt.title("KL divergence between posterior and prior")
plt.plot(np.array(pretrained_vae['model'].KL_QandP)[1:200], 'k', alpha=0.7, label=r'$\beta$'+'-VAE')
plt.plot(np.hstack(sshiba_conditioned['model'].kl_q_p)[1:200], 'r', alpha=0.7, label="FA-VAE")
plt.xlabel("Epochs")
plt.ylabel("KL(Q||P)")
plt.legend()
plt.show()

#================= PLOT IMAGE RANDOM GENERATION =================

# Pretrained BetaVAE
mean = torch.zeros((1000, 100)).to(device)
std = torch.sqrt(torch.ones_like(mean)*1/torch.from_numpy(np.asarray(1)).to(device))

p = torch.distributions.Normal(mean, std)

generated_images = pretrained_vae['model'].decoder(p.sample().float())

for i in range(10):
    print("Stored images: "+str(i)+"/"+str(1000))
    plt.figure()
    plt.imshow(np.transpose(generated_images[i].cpu().data.numpy(), axes=[1,2,0]))
    # plt.savefig("./ExpResults/escenario2/images/vanilla/random_"+str(i))
    plt.show()
    plt.close()

# SSHIBA UNCONDITIONED ADAPTATIVE
z = np.mean(sshiba_conditioned['model'].q_dist.Z['mean'], axis=0) 
w = sshiba_conditioned['model'].q_dist.W[0]['mean']
b = sshiba_conditioned['model'].q_dist.b[0]['mean']
mean = torch.from_numpy((z@w.T + b).reshape(1,-1)).to(device)
tau = sshiba_conditioned['model'].q_dist.tau_mean(0)
std = torch.sqrt(torch.ones_like(mean)*1/torch.from_numpy(np.asarray(tau)).to(device))

p = torch.distributions.Normal(mean, std)

generated_images = sshiba_conditioned['model'].img_vae[0].decoder(p.sample_n(1000).float())

for i in range(5):
    print("Stored images: "+str(i)+"/"+str(1000))
    plt.figure()
    plt.imshow(np.transpose(generated_images[i].cpu().data.numpy(), axes=[1,2,0]))
    plt.show()
    plt.close()

# ================================= GENERATE IMAGES CONDITIONALLY =================================

import itertools
labels_name = ["Smiling", "Wearing_Lipstick", "Male"]

labels = np.asarray(list(itertools.product([0, 1], repeat=3)))

labels_view = sshiba_conditioned['model'].struct_data(labels, 'mult')
img_conditional = sshiba_conditioned['model'].predict([2], [0], labels_view)

fig, ax = plt.subplots(nrows=2, ncols=4, sharex=True, sharey=True)
ploti=0
for i in range(2):
    for j in range(4):
        ax[i,j].set_title(labels[ploti])
        ax[i,j].imshow(np.transpose(img_conditional[0][ploti], axes=[1,2,0]))
        ploti+=1
plt.tight_layout()
plt.show()


for i in range(8):
    print("Stored images: "+str(i)+"/"+str(20))
    plt.figure()
    plt.title(labels[i])
    plt.imshow(np.transpose(img_conditional[0][i], axes=[1,2,0]))
    plt.show()
    plt.close()

# ================================= Pairs of input vs recs =================================
from torchvision import datasets
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
print("Loading dataset")
celeba = datasets.ImageFolder(root="../datasets/celeba/", transform=transforms.Compose([
                                  transforms.Resize((64, 64)),
                                  transforms.ToTensor()
                              ]))

print("Reading dataset")
loader = DataLoader(celeba, batch_size=len(celeba))

celeba_numpy = next(iter(loader))[0]
celeba_train = celeba_numpy[20000:, :, :, :]

# Pretrain VAE images
pretrained_vae['model'].eval()
model = sshiba_conditioned['model'].img_vae[0]
model.eval()

img_idx = [433,111,154,839,51]
# img_idx = np.random.choice(np.arange(1000), size=4)
preimages, _, _, _ = pretrained_vae['model'].forward(celeba_train[img_idx].to(pretrained_vae['model'].device))
sshibaimages, _, _, _ = model.forward(celeba_train[img_idx].to(model.device))
fig, ax = plt.subplots(nrows=3, ncols=5, sharex=True, sharey=True)
for i in range(3):
    if i==0: images, title = celeba_train[img_idx], "Input"
    if i==1: images, title = preimages, r'$\beta$-VAE'
    if i==2: images, title = sshibaimages, "FA-VAE"
    for j in range(5):
        ax[i,j].set_title(title)
        ax[i,j].imshow(np.transpose(images[j].data.cpu(), axes=[1,2,0]))
plt.tight_layout()
plt.show()

from mpl_toolkits.axes_grid1 import ImageGrid
fig = plt.figure(figsize=(10,10))
imgs = np.concatenate((celeba_train[img_idx].data.cpu(), preimages.data.cpu(), sshibaimages.data.cpu()))
grid = ImageGrid(fig, 111, nrows_ncols=(3,5), axes_pad=(0, 0.30))
aux,i=0,-1
titles = ['Input', r'$\beta-VAE$', 'FA-VAE']
for ax, im in zip(grid, imgs):
    if aux%5==0: i+=1
    ax.set_title(titles[i])
    ax.imshow(np.transpose(im, axes=[1,2,0]))
    ax.set(xticks=[], yticks=[])
    aux+=1
# plt.savefig("esc3_celeb2cart.png")
plt.show()

#R2 scores
images = celeba_train[:3000].to(model.device)
preimages, _, _, _ = pretrained_vae['model'].forward(images)
print("BVAE r2 score: "+str(r2_score(images.data.cpu().ravel(), preimages.data.cpu().ravel())))
bvae_r2 = []
for i, img in enumerate(images.data.cpu()):
    bvae_r2.append(r2_score(img.ravel(), preimages.data.cpu()[i].ravel()))

sshibaimages, _, _, _ = sshiba_conditioned['model'].img_vae[0].forward(images)
print("SSHIBA r2 score: "+str(r2_score(images.data.cpu().ravel(), sshibaimages.data.cpu().ravel())))

favae_r2 = []
for i, img in enumerate(images.data.cpu()):
    favae_r2.append(r2_score(img.ravel(), sshibaimages.data.cpu()[i].ravel()))

# ================================= Latent space study =================================
import itertools
from mpl_toolkits.axes_grid1 import ImageGrid

labels_name = ["Smiling", "Wearing_Lipstick", "Male"]

labels = np.asarray([1,0,0]).reshape(1,-1)
labels_view = sshiba_conditioned['model'].struct_data(labels, 'mult')

def predict(model, m_in, m_outs, *args):
        import copy
        import math
        q = model.q_dist
        if type(args[0]) == dict:
            n_pred = np.shape(args[0]['data'])[0]
        else:
            n_pred = np.shape(args[0][0]['data'])[0]
        aux = np.eye(q.Kc)
        for m in m_in:
            aux += q.tau_mean(m)*np.dot(q.W[m]['mean'].T, q.W[m]['mean'])
        Z_cov = model.myInverse(aux)
        if not np.any(np.isnan(Z_cov)):
            model.Z_mean = np.zeros((n_pred,q.Kc))
            for m,arg in enumerate(args):
                if not (arg['SV'] is None) and not(arg['data'].shape[1] == arg['SV'].shape[0]):
                    V = copy.deepcopy(arg['SV'])
                    X = copy.deepcopy(arg['data'])
                    k = copy.deepcopy(arg['kernel'])
                    sig = copy.deepcopy(arg['sig'])
                    center = copy.deepcopy(arg['center'])
                    #Feature selection
                    #Lineal Kernel
                    if k == 'linear':
                        # var = np.sqrt(model.sparse_K[0].get_params()[1])
                        var = 1
                        arg['data'] = np.dot(var*X, (var*V).T)
                    #RBF Kernel
                    if center:
                        arg['data'] = model.center_K(arg['data'])

                if type(arg) == dict:
                    # NEW FOR CATEGORIES
                    if arg['method'] == 'cat': #categorical
                        X_mean, _ = model.cat_vae[m_in[m]].update_x(arg['data'])
                        model.Z_mean += np.dot(X_mean - q.b[m_in[m]]['mean'], q.W[m_in[m]]['mean']) * q.tau_mean(m_in[m])
                    # NEW FOR IMAGES
                    if arg['method'] == 'img':
                        X_mean, _ = model.img_vae[m_in[m]].update_x(arg['data'])
                        # encoded_img = X_mean + np.random.randn(1)*np.sqrt(X_cov)
                        model.Z_mean += np.dot(X_mean - q.b[m_in[m]]['mean'], q.W[m_in[m]]['mean']) * q.tau_mean(m_in[m])
                    # NEW FOR MULTILABEL
                    if arg['method'] == 'mult':
                        X_mean = np.log(np.abs((arg['data']-0.05))/(1 - np.abs((arg['data']-0.05))))
                        model.Z_mean += np.dot(X_mean - q.b[m_in[m]]['mean'], q.W[m_in[m]]['mean']) * q.tau_mean(m_in[m])
                else:
                    for (m,x) in enumerate(arg):
                        model.Z_mean += np.dot(x['data'] - q.b[m]['mean'], q.W[m_in[m]]['mean']) * q.tau_mean(m_in[m])

            model.Z_mean = np.dot(model.Z_mean,Z_cov)
        else:
            print ('Cov Z is not invertible')

        predictions = {}
        if isinstance(m_outs, int): m_outs = [m_outs]
        for m_out in m_outs:
            #Regression
            if model.method[m_out] == 'reg':
                #Expectation X
                mean_x = np.dot(model.Z_mean,q.W[m_out]['mean'].T) + q.b[m_out]['mean']
                #Variance X
                var_x = q.tau_mean(m_out)**(-1)*np.eye(model.d[m_out]) + np.linalg.multi_dot([q.W[m_out]['mean'], Z_cov, q.W[m_out]['mean'].T])

                predictions["output_view"+str(m_out)] = {'mean_x': mean_x, 'var_x': var_x}

            if model.method[m_out] == 'img':
                Z_in = {'mean': model.Z_mean, 'cov': Z_cov}
                pred_img = model.img_vae[m_out].predict(Z=Z_in, W=q.W[m_out], b=q.b[m_out], tau=q.tau_mean(m_out))
                predictions = pred_img

            #Categorical
            elif model.method[m_out] == 'cat':
                Z_in = {'mean': model.Z_mean, 'cov': Z_cov}
                pred_img = model.cat_vae[m_out].predict(Z=Z_in, W=q.W[m_out], b=q.b[m_out], tau=q.tau_mean(m_out)).data.cpu().numpy()
                predictions = pred_img
                
            #Multilabel
            elif model.method[m_out] == 'mult':
                print("sale multilabel")
                #Expectation X
                m_x = np.dot(model.Z_mean, q.W[m_out]['mean'].T) + q.b[m_out]['mean']
                #Variance X
                var_x = q.tau_mean(m_out)**(-1)*np.eye(model.d[m_out]) + np.linalg.multi_dot([q.W[m_out]['mean'], Z_cov, q.W[m_out]['mean'].T])
                
                mean_t = np.zeros((n_pred,model.d[m_out]))
                var_t = np.zeros((n_pred,model.d[m_out]))
                #Probability t
                for d in np.arange(model.d[m_out]):
                    mean_t[:,d] = model.sigmoid(m_x[:,d]*(1+math.pi/8*var_x[d,d])**(-0.5))
                    # mean_t[:,d] = model.sigmoid(m_x[:,d])
                    var_t[:, d] = np.exp(m_x[:,d]*(1+math.pi/8*var_x[d,d])**(-0.5))/(1+np.exp(m_x[:,d]*(1+math.pi/8*var_x[d,d])**(-0.5))**2)
                    # var_t[:,d] = np.exp(m_x[:, d])/(1+np.exp(m_x[:, d]))**2
                predictions["output_view"+str(m_out)] = {'mean_x': mean_t, 'var_x': var_t}

        return predictions, Z_in

model = sshiba_conditioned['model']
img_conditional, Z = predict(model, [2], [0], labels_view)
W = model.q_dist.W[0]
b = model.q_dist.b[0]
tau = model.q_dist.tau[0]['a']/ model.q_dist.tau[0]['b']

Z_sampled = Z['mean'] + np.random.normal()*np.sqrt(np.diag(Z['cov']))
W_sampled = W['mean'] + np.random.normal()*np.sqrt(np.diag(W['cov']))
b_sampled = b['mean'] + np.random.normal()*np.sqrt(np.diag(b['cov']))

# Calculate moments of sshiba latent space
mean = torch.from_numpy(Z_sampled@W_sampled.T+b_sampled)
std = torch.sqrt(torch.ones_like(mean)/torch.from_numpy(np.asarray(1/tau)))
p = torch.distributions.Normal(mean, std)
latent_space = p.rsample().float()
v = np.arange(-20,24,4)
relf = np.argsort(-latent_space.abs())[0, :10].tolist()

def plot_images(model, feature, v, latent_space):
    imgs = []
    model.eval()
    with torch.no_grad():
        for i, feat in enumerate(feature):
            for j, val in enumerate(v):
                lat_mod = latent_space.clone()
                if j!=5: lat_mod[0, feat] = lat_mod[0, feat]+val
                img_aprox = model.decoder(lat_mod.to(model.device)).data.cpu()
                imgs.append(img_aprox)

    fig = plt.figure(figsize=(10,10))
    grid = ImageGrid(fig, 111, nrows_ncols=(len(feature),len(v)), axes_pad=(0, 0))
    for ax, im in zip(grid, imgs):
        ax.imshow(np.transpose(im[0], axes=[1,2,0]))
        ax.set(xticks=[], yticks=[])
    plt.show()
    return imgs
    # plt.savefig(title+".png")

plot_images(model.img_vae[0], relf, v, latent_space)


# Pretrained BetaVAE
mean = torch.zeros((10, 100))
std = torch.sqrt(torch.ones_like(mean)*1/torch.from_numpy(np.asarray(1)))

p = torch.distributions.Normal(mean, std)
latent_sample = p.rsample().float()
v = np.arange(-20,24,4)
relf = np.argsort(-latent_sample.abs())[0, :10].tolist()

imgs = plot_images(pretrained_vae['model'], relf, v, latent_sample)
