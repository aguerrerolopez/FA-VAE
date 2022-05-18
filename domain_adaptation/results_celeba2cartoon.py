# %%
import pickle
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

torch.manual_seed(0)
np.random.seed(0)
#%% #================= LOAD FAVAE MODEL =================

usps_mnist_path = open('./results/celeb2cart_hairandeyeglasses_100iter_cartbeta_BasReg_1.0.pickle', 'rb')
celeb2cart = pickle.load(usps_mnist_path)
usps_mnist_path.close()
del usps_mnist_path

#%% #================= GENERATE SAMPLES CONDITIONALLY =================

labels_ohe = np.eye(3)
labels_view = celeb2cart['model'].struct_data(labels_ohe, 'mult')
# Generate CARTOON images given label
img_conditional,_,_ = celeb2cart['model'].predict([2], [1], labels_view)
for i in range(3):
    print("Stored images: "+str(i)+"/"+str(20))
    plt.figure()
    plt.title(i)
    plt.imshow(np.transpose(img_conditional[i], axes=[1,2,0]))
    plt.show()
    plt.close()

#%% ====== Given labels generate Cartoon and Celeba images (conditioning multiple VAEs)======
labels = np.ones((10000,1))
labels[:3333] = 0
labels[3333:6666] = 1
labels[6666:] = 2
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
labels = ohe.fit_transform(labels)

labels = celeb2cart['model'].struct_data(labels, 'mult')

cart_condlabel = celeb2cart['model'].predict([2], [1], labels)
celeb_condlabel = celeb2cart['model'].predict([2], [0], labels)
images = [10,50,3350,3360,6680,7790]
for i in images:
    print("Stored images: "+str(i)+"/"+str(20))
    if i<3333: hair = 'Blond hair'
    if i>3333: hair = 'Brown hair'
    if i>6666: hair = 'Gray hair'
    plt.title("Given the input: "+hair)
    plt.imshow(np.transpose(cart_condlabel[0][i], axes=[1,2,0]))
    plt.tight_layout()
    plt.show()
    plt.close()

images = [10,50,3350,3360,6680,7790]
celeb_condlabel = celeb2cart['model'].predict([2], [0], labels)
for i in images:
    print("Stored images: "+str(i)+"/"+str(20))
    if i<3333: hair = 'Blond hair'
    if i>3333: hair = 'Brown hair'
    if i>6666: hair = 'Gray hair'
    plt.title("Given the input: "+hair)
    plt.imshow(np.transpose(celeb_condlabel[0][i], axes=[1,2,0]))
    plt.tight_layout()
    plt.show()
    plt.close()

for i in images:
    print("Stored images: "+str(i)+"/"+str(20))
    if i<3333: hair = 'Blond hair'
    if i>3333: hair = 'Brown hair'
    if i>6666: hair = 'Gray hair'

    fig, ax= plt.subplots(2,1)
    fig.suptitle('Given the categorical input: '+hair)
    ax[0].set_title("Cartoon output is:")
    ax[0].imshow(np.transpose(cart_condlabel[0][i], axes=[1,2,0]))

    ax[1].set_title("CelebA output:")
    ax[1].imshow(np.transpose(celeb_condlabel[0][i], axes=[1,2,0]))
    plt.tight_layout()
    plt.show()
    plt.close()

imgs_toplot = np.concatenate((cart_condlabel[0][images], celeb_condlabel[0][images]))
from mpl_toolkits.axes_grid1 import ImageGrid
fig = plt.figure(figsize=(10,10))
grid = ImageGrid(fig, 111, nrows_ncols=(2,6), axes_pad=(0, 0))
for ax, im in zip(grid, imgs_toplot):
    ax.imshow(np.transpose(im, axes=[1,2,0]))
    ax.set(xticks=[], yticks=[])
plt.show()
plt.close()



#%% ====== Load celeba and test how to translate it to cartoon======
print("Loading dataset")
celeba = ImageFolder(root="../datasets/celeba/", transform=transforms.Compose([
                                  transforms.Resize((64, 64)),
                                  transforms.ToTensor()
                              ]))
loader = DataLoader(celeba, batch_size=len(celeba))

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
attr = get_annotation('../datasets/celeba/list_attr_celeba.txt').iloc[:20000, :]


target = attr.replace(-1,0)
# Create a balanced dataset of beards:
blond_noglasses = target[target['Blond_Hair']==1][target['Eyeglasses']==0].sample(int(3333/2), replace=True).index.tolist()
blond_glasses = target[target['Blond_Hair']==1][target['Eyeglasses']==1].sample(int(3333/2)+1, replace=True).index.tolist()

brown_glasses = target[target['Brown_Hair']==1][target['Eyeglasses']==1].sample(int(3333/4), replace=True).index.tolist()
brown_noglasses = target[target['Brown_Hair']==1][target['Eyeglasses']==0].sample(int(3333/4)+1, replace=True).index.tolist()

black_glasses = target[target['Black_Hair']==1][target['Eyeglasses']==1].sample(int(3333/4), replace=True).index.tolist()
black_noglasses = target[target['Black_Hair']==1][target['Eyeglasses']==0].sample(int(3333/4)+1, replace=True).index.tolist()

gray_glasses = target[target['Gray_Hair']==1][target['Eyeglasses']==1].sample(int(3333/2), replace=True).index.tolist()
gray_noglasses = target[target['Gray_Hair']==1][target['Eyeglasses']==0].sample(int(3333/2)+1, replace=True).index.tolist()

idx = np.concatenate((blond_glasses, blond_noglasses, brown_glasses, brown_noglasses, black_glasses, black_noglasses, gray_glasses,gray_noglasses))


celeba_numpy = next(iter(loader))[0]
celeba_train = celeba_numpy[idx, :, :, :]
labels = np.ones((10000,1))
labels[:3333] = 0
labels[3333:6666] = 1
labels[6666:] = 2
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
labels = ohe.fit_transform(labels)


celebin = celeb2cart['model'].struct_data(celeba_train, 'img', latent_dim=64, lr=1e-3, epochs=15, dataset="svhn")
labels = celeb2cart['model'].struct_data(labels, 'mult')

img_conditional = celeb2cart['model'].predict([0,2], [1], celebin, labels)
images = [1665,1666,1667,3332,3333,3334,4997,4998,4999,6665,6666,6667,8332,8333,8334] 
for i in images:
    print("Stored images: "+str(i)+"/"+str(20))
    fig, ax= plt.subplots(2,1)
    ax[0].set_title("CELEBA input image:")
    ax[0].imshow(np.transpose(celeba_train[i], axes=[1,2,0]))

    ax[1].set_title("Translated image to CARTOON:")
    ax[1].imshow(np.transpose(img_conditional[0][i], axes=[1,2,0]))

    plt.tight_layout()    
    plt.show()
    plt.close()


# %% Check if everything is working

######## OBSERVATIONS ########
# A sample celeba image observed
c0 = celeb2cart['model'].t[0]['data'][0]
plt.imshow(np.transpose(c0, axes=[1,2,0]))
plt.show()

# A sample cartoon image observed
d0 = celeb2cart['model'].t[1]['data'][0]
plt.imshow(np.transpose(d0, axes=[1,2,0]))
plt.show()

######## Given observations translate to the other domain ########
celebin = celeb2cart['model'].struct_data(celeb2cart['model'].t[0]['data'], 'img', latent_dim=100, lr=1e-3, epochs=15, dataset="celeba")
labels = celeb2cart['model'].struct_data(celeb2cart['model'].t[2]['data'], 'mult')

img_conditional, xin, Zshared, xout = celeb2cart['model'].predict([0,2], [1], celebin, labels)
img_selected = [0,3,3333,1677,1682,4501,4621,5525,3358]
for i in img_selected:
    fig, ax= plt.subplots(2,1)
    ax[0].set_title("CelebA input image "+str(i))
    ax[0].imshow(np.transpose(celeb2cart['model'].t[0]['data'][i], axes=[1,2,0]))

    ax[1].set_title("Cartoon adaptation:")
    ax[1].imshow(np.transpose(img_conditional[i], axes=[1,2,0]))
    plt.tight_layout()
    plt.show()

imgs = np.concatenate((celeb2cart['model'].t[0]['data'][img_selected],img_conditional[img_selected] ))

from mpl_toolkits.axes_grid1 import ImageGrid
fig = plt.figure(figsize=(10,10))
grid = ImageGrid(fig, 111, nrows_ncols=(2,9), axes_pad=(0, 0))
for ax, im in zip(grid, imgs):
    ax.imshow(np.transpose(im, axes=[1,2,0]))
    ax.set(xticks=[], yticks=[])
plt.show()


# Latent spaces analysis for selected images
img_selected = [0,3,3333,1677,1682,4501,4621,5525,3358]
bglass = [0,1,2]
blondes = [3,4]
brunet = [5,6]
brunetglass = [7,8]
cdict = {'0': 'Blond', '1': 'Blond+Glasses', '2': 'Brunet', '3':'Brunet+Glasses'}

import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

u = umap.UMAP(n_components=2)
t = TSNE(n_components=2)
p = PCA(n_components=2)

# PCA
xin2 = p.fit_transform(xin[img_selected])
zs2 = p.fit_transform(Zshared['mean'][img_selected])
xout2 = p.fit_transform(xout[img_selected])

# TSNE
xin2 = t.fit_transform(xin[img_selected])
zs2 = t.fit_transform(Zshared['mean'][img_selected])
xout2 = t.fit_transform(xout[img_selected])

# UMAP
xin2 = u.fit_transform(xin[img_selected])
zs2 = u.fit_transform(Zshared['mean'][img_selected])
xout2 = u.fit_transform(xout[img_selected])

fig, ax= plt.subplots(1,3, figsize=(14,3))
for i, lv in enumerate([xin2, zs2, xout2]):
    if i==0: title='UMAP over $x_{n,:}^{(C)}$'
    if i==1: title='UMAP over $z_{n,:}$'
    if i==2: title='UMAP over $x_{n,:}^{(D)}$'
    ax[i].set_title(title)
    b = ax[i].scatter(lv[blondes,0], lv[blondes,1], marker='o', color='b')
    bg = ax[i].scatter(lv[bglass,0], lv[bglass,1], marker='x', color='b')
    m = ax[i].scatter(lv[brunet,0], lv[brunet,1], marker='o', color='m')
    mg = ax[i].scatter(lv[brunetglass,0], lv[brunetglass,1], marker='x', color='m')
fig.legend((b, bg, m, mg),
           ('Blond', 'Blond+Glasses', 'Brunet', 'Brunet+Glasses'),
           scatterpoints=1,
           ncol=1,
           loc=[0.89,.4],
           fontsize=8)
plt.show()
plt.close()


######## Latent space experiment over Zs ########
img1 = img_selected[-1]
img2 = img_selected[3]

lamb = np.arange(0,1,0.1)
zt_mean = [l*Zshared['mean'][img1] + (1-l)*Zshared['mean'][img2] for l in lamb]

imgs = [celeb2cart['model'].t[0]['data'][img2]]
for i,t in enumerate(lamb):
    print(i)
    z_in = {'mean': zt_mean[i], 'cov': Zshared['cov']}
    pred_img, _ = celeb2cart['model'].img_vae[0].predict(Z=z_in, W=celeb2cart['model'].q_dist.W[0], b=celeb2cart['model'].q_dist.b[0], tau=celeb2cart['model'].q_dist.tau_mean(0))
    imgs.append(pred_img[0])
imgs.append(celeb2cart['model'].t[0]['data'][img1])

imgs.append(img_conditional[img2])
for i,t in enumerate(lamb):
    print(i)
    z_in = {'mean': zt_mean[i], 'cov': Zshared['cov']}
    pred_img, _ = celeb2cart['model'].img_vae[1].predict(Z=z_in, W=celeb2cart['model'].q_dist.W[1], b=celeb2cart['model'].q_dist.b[1], tau=celeb2cart['model'].q_dist.tau_mean(1))
    imgs.append(pred_img[0])
imgs.append(img_conditional[img1])

from mpl_toolkits.axes_grid1 import ImageGrid
fig = plt.figure(figsize=(10,10))
grid = ImageGrid(fig, 111, nrows_ncols=(2,12), axes_pad=(0, 0))
for ax, im in zip(grid, imgs):
    ax.imshow(np.transpose(im, axes=[1,2,0]))
    ax.set(xticks=[], yticks=[])
# plt.savefig("esc3_celeb2cart.png")
plt.show()

######## Latent space experiment over Xs ########

x1 = xout[img_selected[-1]]
x2 = xout[img_selected[3]]
x1c = torch.Tensor(xin[img_selected[-1]])
x2c = torch.Tensor(xin[img_selected[3]])
lamb = np.arange(0.1,0.9,0.1)
x_trans_cart = [l*x2 + (1-l)*x1 for l in lamb]
x_trans_celeb = [l*x2c + (1-l)*x1c for l in lamb]
transition_cart_images = torch.Tensor(10,3,64,64)
transition_celeb_images = torch.Tensor(10,3,64,64)

imgs = [celeb2cart['model'].t[0]['data'][img_selected[-1]]]
for i,t in enumerate(x_trans_celeb):
    transition_celeb_images[i]=celeb2cart['model'].img_vae[0].decoder(t.to(celeb2cart['model'].img_vae[1].device)).data.cpu()
    imgs.append(transition_celeb_images[i])
imgs.append(celeb2cart['model'].t[0]['data'][img_selected[3]])

imgs.append(img_conditional[img_selected[-1]])
for i,t in enumerate(x_trans_cart):
    transition_cart_images[i]=celeb2cart['model'].img_vae[1].decoder(t.to(celeb2cart['model'].img_vae[1].device)).data.cpu()
    imgs.append(transition_cart_images[i])
imgs.append(img_conditional[img_selected[3]])

from mpl_toolkits.axes_grid1 import ImageGrid
fig = plt.figure(figsize=(10,10))
grid = ImageGrid(fig, 111, nrows_ncols=(2,10), axes_pad=(0, 0))
for ax, im in zip(grid, imgs):
    ax.imshow(np.transpose(im, axes=[1,2,0]))
    ax.set(xticks=[], yticks=[])
# plt.savefig("esc3_celeb2cart.png")
plt.show()
