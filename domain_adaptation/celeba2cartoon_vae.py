#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 09:26:30 2020

@author: Alejandro Guerrero-López
"""

# -- coding: utf-8 --
import torch
from torch import nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(0)
np.random.seed(0)

class VAE(nn.Module):
    def __init__(self, channels=1, h=None, w=None, zDim=2, dataset="mnist", latentspace_lr=1):
        super(VAE, self).__init__()

        print(dataset)
        self.latentdim = zDim
        self.channels = channels
        self.dataset=dataset
        self.latentspace_lr = latentspace_lr

        modules = []
        hidden_dims = [64, 128, 256, 512, 1024]
        # CNN Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            channels = h_dim
        # MLP encoder to generate mu and var 
        self.enc = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, zDim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, zDim)
        # CNN decoder
        modules = []
        self.decoder_input = nn.Linear(zDim, hidden_dims[-1] * 4)
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        self.dec = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= self.channels,
                                      kernel_size= 3, padding= 1),
                            nn.Sigmoid())

    def encoder(self, x):
        # CNN
        x = self.enc(x)
        # MLP that generate mu and log_var
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)+1e-6 # Clip the lowerboung of log_var to 1e-6

        return mu, torch.exp(0.5*log_var)

    def reparameterize(self, mu, std):
        #Reparameterization takes in the input mu and std and sample the mu + std * eps
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decoder(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, 1024, 2, 2)
        x = self.dec(x)
        x = self.final_layer(x)
        return x

    def forward(self, x):
        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        mu, std = self.encoder(x)
        z = self.reparameterize(mu, std)
        return self.decoder(z), mu, std, z

class ImgVAE(VAE):

    def __init__(self, dimx=2, channels=1, var_x=0.01, lr=1e-5, h=None, w=None, dataset=None, latentspace_lr=1):
        
        super().__init__(channels=channels, h=h, w=w, zDim=dimx, dataset=dataset, latentspace_lr=latentspace_lr)
        self.lr = lr
        self.dimx=dimx
        self.var_x = var_x
        self.optim = torch.optim.Adam(self.parameters(), self.lr)
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") 
        print(self.device) 
        self.to(self.device)

    def kl_div(self, mu_Q, std_Q, mu_P, std_P):
        # Clamp values lower and upperbound to stabilize
        std_Q = torch.clamp(std_Q, 1e-6, 1e20)
        std_P = torch.clamp(std_P, 1e-6, 1e20)
        mu_Q = torch.clamp(mu_Q, -1e20, 1e20)
        mu_P = torch.clamp(mu_P, -1e20, 1e20)
        # Create the posterior and prior distributions
        q = torch.distributions.Normal(mu_Q, std_Q)
        p = torch.distributions.Normal(mu_P, std_P)
        # Calculate KL(q||p)
        kl = torch.distributions.kl_divergence(q, p)
        return kl.sum(axis=1).mean() # Mean reduction. It has to be equal as reconstruction reduciton

    def gaussian_LL(self, true_images, reconstructed_images):
        D = true_images.shape[1] * true_images.shape[2] * true_images.shape[3]   # Dimension of the image
        true_images = true_images.reshape(-1, D)
        reconstructed_images = reconstructed_images.reshape(-1, D)
        var_x = torch.ones_like(reconstructed_images) * self.var_x
        # Constant term in the gaussian distribution
        cnt = D * np.log(2 * np.pi) + torch.sum(torch.log(var_x), dim=-1)
        # log-likelihood per datapoint
        logp_data = -0.5 * (cnt + torch.sum((true_images - reconstructed_images) * var_x ** -1 * (true_images - reconstructed_images), dim=-1))
        return logp_data.mean()# Mean reduction. It has to be equal as KL reduciton

 
    def trainloop(self, img=None, Z=None, W=None, b=None, tau=None, epochs=20, beta=1, wandb=None):
        if img is not None:
            self.img = torch.tensor(img)
        # Lists to store training evolution
        self.loss_during_training = []
        self.elbo_training = []
        self.reconstruc_during_training = []
        self.KL_during_training = []
        self.KL_QandP = []
        # Prior distribution: N(Z@W.T+b, inv(tau*I))
        Z_sampled = Z['mean'] + np.random.normal()*np.sqrt(np.diag(Z['cov']))
        W_sampled = W['mean'] + np.random.normal()*np.sqrt(np.diag(W['cov']))
        b_sampled = b['mean'] + np.random.normal()*np.sqrt(np.diag(b['cov']))
        # Calculate moments of the Prior
        prior_mean = torch.from_numpy(Z_sampled@W_sampled.T+b_sampled)
        tau = torch.Tensor(np.asarray(tau)).to(self.device)

        # Create the Dataloader to perform batch learning
        dataset = TensorDataset(self.img, prior_mean)
        loader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)

        elbo_checker = -1e20
        self.train()
        for e in range(int(epochs)):
            train_loss = 0
            train_rec = 0
            train_elbo = 0
            train_kl_l = 0
            for images, prior_mu in loader:
                # Move data to GPU
                images = images.to(self.device)
                # Moments of the prior
                mu_P = prior_mu.to(self.device)
                var_P = torch.ones_like(mu_P)*(1/tau)
                std_P = torch.sqrt(var_P)
                # ==================== Gradient calculation ===================
                self.optim.zero_grad()
                # ==================== VAE forward===================
                img_rec, mu_Q, std_Q, latent_sample = self.forward(images)
                # ==================== Loss calculation ===================
                reconstruction_error = -self.gaussian_LL(images, img_rec)
                kl_div = self.kl_div(mu_Q, std_Q, mu_P, std_P)
                loss = reconstruction_error + beta*kl_div
                elbo = -loss
                # ==================== Gradient calculation ===================
                loss.backward()
                self.optim.step()
                # ==================== Save training evolution ===================
                train_loss += loss.data.cpu().numpy()
                train_elbo += elbo.data.cpu().numpy()
                train_rec += reconstruction_error.data.cpu().numpy()
                train_kl_l += kl_div.data.cpu().numpy()
            # Calculate metrics by batch
            self.loss_during_training.append(train_loss/len(loader))
            self.elbo_training.append(train_elbo/len(loader))
            self.reconstruc_during_training.append(train_rec/len(loader))
            self.KL_QandP.append(train_kl_l/len(loader))
            # Overfitter checker: check if the ELBO is better or worse that previous one
            elbo_checker = self.restore_weights(elbo_checker, e)
            # Log the metrics to W&B server
            metrics = {self.dataset+" ELBO": float(self.elbo_training[-1]), self.dataset+" Gaussian LogLikelihood": float(-self.reconstruc_during_training[-1]), 
            self.dataset+" KL(Q||P)": float(self.KL_QandP[-1])}
            if wandb is not None: wandb.log(metrics)
            # Print metric by screen
            if(e%1==0):
                print('Train Epoch: {} \tLoss: {:.3f} ELBO: {:.3f} BCE: {:.3f} KL: {:.3f}'.format(e,self.loss_during_training[-1],self.elbo_training[-1], 
                self.reconstruc_during_training[-1], self.KL_QandP[-1]))
    
        # When convergence is reached, we stop training the VAEs
        keep_train = abs(1-np.mean(self.elbo_training[-10:])/self.elbo_training[-1])>1e-4
        # Delete variables to release GPU memory 
        del prior_mean, tau, images, mu_P, std_P, mu_Q, std_Q, img_rec
        torch.cuda.empty_cache()
        
        return keep_train

    def restore_weights(self, elbo_checker, e):
        # If actual ELBO is better, save a checkpoint
        if self.elbo_training[-1] > elbo_checker:
                checkpoint = {
                    'epoch': e + 1,
                    'elbo': self.elbo_training[-1],
                    'state_dict': self.state_dict(),
                    'optimizer': self.optim.state_dict(),
                }
                # save checkpoint
                torch.save(checkpoint, "./checkpoints/"+self.dataset+"8"+str(self.latentspace_lr)+".pt")
                elbo_checker = self.elbo_training[-1]
        # If actual ELBO is worse, load best checkpoint
        if self.elbo_training[-1] < elbo_checker:
            print("Restoring best training elbo")
            ckp = torch.load("./checkpoints/"+self.dataset+"8"+str(self.latentspace_lr)+".pt")
            self.load_state_dict(ckp['state_dict'])
            self.optim.load_state_dict(ckp['optimizer'])
            elbo_checker = ckp['elbo']
        return elbo_checker


    def update_x(self, img=None):
        # If the update X is with the same data as in training, we shouldn't upload it again as it is already in the GPU from the training phase
        if img is None:
            img = self.img.to(self.device)
        else:
            img = torch.Tensor(img)
        dataset = TensorDataset(self.img)
        loader = DataLoader(dataset=dataset, batch_size=64, shuffle=False)
        mu = np.zeros((img.shape[0], self.latentdim))
        var = np.zeros((img.shape[0], self.latentdim))
        self.eval()
        with torch.no_grad():
            batch_index = 0
            for batch_image in loader:
                batch_image = batch_image[0].to(self.device)
                mu_Qbatch, std_Qbatch = self.encoder(batch_image)
                mu[batch_index:batch_index+batch_image.shape[0]] = mu_Qbatch.data.cpu().numpy()
                var[batch_index:batch_index+batch_image.shape[0]] = std_Qbatch.pow(2).data.cpu().numpy()
                batch_index+=batch_image.shape[0]
        del img, dataset, loader, batch_image, mu_Qbatch, std_Qbatch
        torch.cuda.empty_cache()
        return mu, var

    def reconstruction(self, mean, var):
        # Given mean and var of Q, reconstruct an image
        sample=torch.Tensor(mean+np.random.randn()*np.sqrt(var))
        dataset = TensorDataset(sample)
        loader = DataLoader(dataset=dataset, batch_size=64, shuffle=False)
        img_rec = np.zeros((mean.shape[0], self.img.shape[1], self.img.shape[2], self.img.shape[3]))
        self.eval()
        with torch.no_grad():
            batch_index = 0
            for batch_image in loader:
                batch_image = batch_image[0].to(self.device)
                # We obtain an image using the decoder
                if self.dataset=='mnist':
                    img_rec[batch_index:batch_index+batch_image.shape[0], :, :, :] = self.decoder(batch_image).data.cpu().numpy()
                else:
                    img_rec[batch_index:batch_index+batch_image.shape[0], :, :, :] = self.decoder(batch_image).data.cpu().numpy()
                batch_index+=batch_image.shape[0]
        del batch_image, dataset, loader, sample, batch_index, mean, var
        torch.cuda.empty_cache()
        return img_rec


    def predict(self, Z=None, W=None, b=None, tau=None, mean_x=None, var_x=None):
        #Predict/Generate a new image given N(ZW+b,tau) or N(0,1)
        if mean_x is None:
            Z_sampled = Z['mean'] + np.random.normal()*np.sqrt(np.diag(Z['cov']))
            W_sampled = W['mean'] + np.random.normal()*np.sqrt(np.diag(W['cov']))
            b_sampled = b['mean'] + np.random.normal()*np.sqrt(np.diag(b['cov']))
            # Calculate moments of sshiba latent space
            mean = torch.from_numpy(Z_sampled@W_sampled.T+b_sampled)
            std = torch.sqrt(torch.ones_like(mean)/torch.from_numpy(np.asarray(1/tau)))
        else:
            mean = torch.from_numpy(mean_x)
            std = torch.sqrt(torch.from_numpy(var_x))
        p = torch.distributions.Normal(mean, std)
        rsample = p.rsample().float()
        dataset = TensorDataset(rsample)
        loader = DataLoader(dataset=dataset, batch_size=64, shuffle=False)

        img_aprox = np.zeros((mean.shape[0], self.img.shape[1], self.img.shape[2], self.img.shape[3]))
        self.eval()
        with torch.no_grad():
            batch_index = 0
            for batch_image in loader:
                batch_image = batch_image[0].to(self.device)
                # We obtain an image using the decoder
                if self.dataset=='mnist':
                    img_aprox[batch_index:batch_index+batch_image.shape[0], :, :, :] = self.decoder(batch_image).data.cpu().numpy()
                else:
                    img_aprox[batch_index:batch_index+batch_image.shape[0], :, :, :] = self.decoder(batch_image).data.cpu().numpy()
                batch_index+=batch_image.shape[0]

        # Release GPU memory 
        del batch_image, dataset, loader, batch_index, p, mean, std
        torch.cuda.empty_cache()

        return img_aprox, rsample
