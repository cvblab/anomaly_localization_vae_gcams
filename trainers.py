from models import *
from losses import *
from grad_cams import *
import sklearn
import torch
import numpy as np
import datetime
import cv2
import sklearn.metrics
import os
from timeit import default_timer as timer
import json
from matplotlib import pyplot as plt
import torch.nn.functional as F
from metrics import *
from utils import *
import pandas as pd
import kornia.augmentation as korn
from scipy import ndimage
from kornia.losses import SSIM
import kornia
import imutils


class WSALTrainer:
    def __init__(self, dir_out, item=['flair'], method='proposed', zdim=32, lr=1*1e-4, input_shape=(1, 224, 224),
                 expansion_loss=False, wkl=1, wr=1, wadv=1, wae=1, epochs_to_test=25, load_weigths=False,
                 n_blocks=5, dense=True, log_barrier=True, normalization_cam='sigm', avg_grads=False, t=50,
                 context=False, level_cams=-4, iteration=0, p_activation_cam=1e-3, bayesian=False,
                 loss_reconstruction='cce', expansion_loss_penalty='log_barrier', restoration=False,
                 threshold_on_normal=False):

        # Prepare results folders
        self.dir_results = dir_out + item[0] + '/iteration_' + str(iteration) + str('/')
        if not os.path.isdir(dir_out):
            os.mkdir(dir_out)
        if not os.path.isdir(dir_out + item[0] + '/'):
            os.mkdir(dir_out + item[0] + '/')
        if not os.path.isdir(self.dir_results):
            os.mkdir(self.dir_results)

        # Input properties initialization
        self.dir_out = dir_out
        self.item = item
        self.method = method
        self.zdim = zdim
        self.lr = lr
        self.input_shape = input_shape
        self.expansion_loss = expansion_loss
        self.wkl = wkl
        self.wr = wr
        self.wadv = wadv
        self.wae = wae
        self.epochs_to_test = epochs_to_test
        self.load_weigths = load_weigths
        self.n_blocks = n_blocks
        self.dense = dense
        self.log_barrier = log_barrier
        self.normalization_cam = normalization_cam
        self.avg_grads = avg_grads
        self.t = t
        self.context = context
        self.level_cams = level_cams
        self.iteration = iteration
        self.p_activation_cam = p_activation_cam
        self.bayesian = bayesian
        self.loss_reconstruction = loss_reconstruction
        self.epochs_to_test = epochs_to_test
        self.expansion_loss_penalty = expansion_loss_penalty
        self.restoration = restoration
        self.threshold_on_normal = threshold_on_normal

        # Other properties initialization
        self.init_time = 0
        self.kl_iteration, self.ladv_iteration, self.ldisc_iteration, self.lr_iteration = 0, 0, 0, 0
        self.kl_epoch, self.ladv_epoch, self.ldisc_epoch, self.lr_epoch = 0, 0, 0, 0
        self.kl_lc, self.ladv_lc, self.ldisc_lc, self.lr_lc, self.lperc_lc, self.lae_lc = [], [], [], [], [], []
        self.i_epoch = 0
        self.epochs = 0
        self.i_iteration = 0
        self.iterations = 0
        self.test_dataset = []
        self.train_generator = []
        self.normal_label = 1
        self.generated_label = 0
        self.aucroc_lc, self.auprc_lc = [], []
        self.metrics = {}

        # Set model modules
        self.E = Encoder(self.method, fin=self.input_shape[0], zdim=self.zdim, dense=self.dense, n_blocks=self.n_blocks,
                         spatial_dim=self.input_shape[1]//2**self.n_blocks)
        self.Dec = Decoder(fin=self.zdim, nf0=self.E.backbone.nfeats//2, n_channels=self.input_shape[0],
                           dense=self.dense, n_blocks=self.n_blocks, spatial_dim=self.input_shape[1]//2**self.n_blocks)
        self.Disc = Discriminator(fin=self.E.backbone.nfeats//2**self.n_blocks, n_channels=input_shape[0],
                                  n_blocks=self.n_blocks)

        # Set parameters
        self.params = list(self.E.parameters()) + list(self.Dec.parameters())

        # Set losses
        self.KL = kl_loss
        if self.loss_reconstruction == 'L2':
            self.Lr = torch.nn.MSELoss(reduction='sum')
        elif self.loss_reconstruction == 'ssim':
            self.Lr = SSIM(5)
        else:
            self.Lr = torch.nn.BCEWithLogitsLoss(reduction='sum')
        self.DLoss = torch.nn.BCEWithLogitsLoss()

        # Properties and modules for expansion loss setting
        if self.expansion_loss:
            self.lae_epoch = 0
            self.lae_iteration = 0
            self.lae_lc = []
            self.wae = wae

        # Set optimizers
        self.opt_AL = torch.optim.Adam(self.params, lr=self.lr)
        self.opt_Disc = torch.optim.Adam(self.Disc.parameters(), lr=self.lr*10)

        # Load weights
        if self.load_weigths:
            self.E.load_state_dict(torch.load(self.dir_results + '/encoder_weights.pth'))
            self.Dec.load_state_dict(torch.load(self.dir_results + '/decoder_weights.pth'))
            self.Disc.load_state_dict(torch.load(self.dir_results + '/discriminator_weights.pth'))

    def train(self, train_generator, test_dataset, epochs):
        # Incorporate items to object trainer
        self.iterations = len(train_generator)
        self.train_generator = train_generator
        self.test_dataset = test_dataset
        self.epochs = epochs

        # Move models tu gpu
        self.E.cuda()
        self.Dec.cuda()

        # Train each method
        if self.method == 'ae':
            self.train_ae()
        elif self.method == 'vae':
            self.train_vae()
        elif self.method == 'anoVAEGAN':
            self.Disc.cuda()  # discriminator to gpu
            self.train_ano_vae_gan()
        elif self.method == 'cavga' or self.method == 'proposed':
            self.train_proposed()
        elif self.method == 'f_ano_gan':
            self.Disc.cuda()  # discriminator to gpu
            self.train_f_ano_gan()

    def train_ae(self):

        self.init_time = timer()
        for self.i_epoch in range(self.epochs):

            # init epoch losses
            self.kl_epoch, self.ladv_epoch, self.ldisc_epoch, self.lr_epoch = 0, 0, 0, 0

            # Loop over training dataset
            for self.i_iteration, (x_n, y_n, _, _) in enumerate(self.train_generator):
                if self.context:  # if context option, data augmentation to apply context
                    (x_n_context, _) = augment_input_batch(x_n.copy())
                    x_n_context = torch.tensor(x_n_context).cuda().float()

                # Move tensors to gpu
                x_n = torch.tensor(x_n).cuda().float()

                # Obtain latent space from normal sample via encoder
                if not self.context:
                    z, _, _, _ = self.E(x_n)
                else:
                    z, _, _, _ = self.E(x_n_context)

                # Obtain reconstructed images through decoder
                xhat, _ = self.Dec(z)

                # Calculate criterion
                self.lr_iteration = self.Lr(xhat, x_n) / self.train_generator.batch_size  # Reconstruction loss

                # Init overall losses
                L = self.lr_iteration

                # Update weights
                L.backward()  # Backward
                self.opt_AL.step()  # Update weights
                self.opt_AL.zero_grad()  # Clear gradients

                """
                ON ITERATION/EPOCH END PROCESS
                """

                # Display losses per iteration
                self.display_losses(on_epoch_end=False)

                # Update epoch's losses
                self.lr_epoch += self.lr_iteration.cpu().detach().numpy() / len(self.train_generator)

            # Epoch-end processes
            self.on_epoch_end()

    def train_vae(self):

        self.init_time = timer()
        for self.i_epoch in range(self.epochs):

            # init epoch losses
            self.kl_epoch, self.ladv_epoch, self.ldisc_epoch, self.lr_epoch = 0, 0, 0, 0

            # Loop over training dataset
            for self.i_iteration, (x_n, y_n, _, _) in enumerate(self.train_generator):
                if self.context:
                    (x_n_context, _) = augment_input_batch(x_n.copy())
                    x_n_context = torch.tensor(x_n_context).cuda().float()

                # Move tensors to gpu
                x_n = torch.tensor(x_n).cuda().float()

                # Obtain latent space from normal sample via encoder
                if not self.context:
                    z, z_mu, z_logvar, _ = self.E(x_n)
                else:
                    z, z_mu, z_logvar, _ = self.E(x_n_context)

                # Obtain reconstructed images through decoder
                xhat, _ = self.Dec(z)

                # Calculate criterion
                self.kl_iteration = self.KL(mu=z_mu, logvar=z_logvar) / (self.train_generator.batch_size*self.zdim)  # kl loss
                if self.loss_reconstruction == 'bce':  # Reconstruction loss
                    self.lr_iteration = self.Lr(xhat, x_n) / (self.train_generator.batch_size*self.input_shape[1]*self.input_shape[2])
                elif self.loss_reconstruction == 'L2':
                    self.lr_iteration = self.Lr(torch.sigmoid(xhat), x_n) / (self.train_generator.batch_size * self.input_shape[1] * self.input_shape[2])
                elif self.loss_reconstruction == 'ssim':
                    self.lr_iteration = torch.sum(self.Lr(torch.sigmoid(xhat), x_n)) / (self.train_generator.batch_size*self.input_shape[1]*self.input_shape[2])

                # Init overall losses
                L = self.wr * self.lr_iteration + self.wkl * self.kl_iteration

                # Update weights
                L.backward()  # Backward
                self.opt_AL.step()  # Update weights
                self.opt_AL.zero_grad()  # Clear gradients

                """
                ON ITERATION/EPOCH END PROCESS
                """

                # Display losses per iteration
                self.display_losses(on_epoch_end=False)

                # Update epoch's losses
                self.kl_epoch += self.kl_iteration.cpu().detach().numpy() / len(self.train_generator)
                self.lr_epoch += self.lr_iteration.cpu().detach().numpy() / len(self.train_generator)

            # Epoch-end processes
            self.on_epoch_end()

    def train_ano_vae_gan(self):

        self.init_time = timer()
        for self.i_epoch in range(self.epochs):

            # init epoch losses
            self.kl_epoch, self.ladv_epoch, self.ldisc_epoch, self.lr_epoch, self.lperc_epoch = 0, 0, 0, 0, 0

            # Loop over training dataset
            for self.i_iteration, (x_n, y_n, _, _) in enumerate(self.train_generator):
                if self.context:
                    (x_n_context, _) = augment_input_batch(x_n.copy())
                    x_n_context = torch.tensor(x_n_context).cuda().float()

                # Move tensors to gpu
                x_n = torch.tensor(x_n).cuda().float()

                # Obtain latent space from normal sample via encoder
                if not self.context:
                    z, z_mu, z_logvar, _ = self.E(x_n)
                else:
                    z, z_mu, z_logvar, _ = self.E(x_n_context)

                # Obtain reconstructed images through decoder
                xhat, _ = self.Dec(z)

                # Forward discriminator
                d_x, _ = self.Disc(x_n)
                d_xhat, _ = self.Disc(torch.sigmoid(xhat))

                # Discriminator labels
                d_x_true = torch.tensor(self.normal_label * np.ones((self.train_generator.batch_size, 1))).cuda().float()
                d_f_false = torch.tensor(self.normal_label * np.ones((self.train_generator.batch_size, 1))).cuda().float()
                d_xhat_true = torch.tensor(self.generated_label * np.ones((self.train_generator.batch_size, 1))).cuda().float()

                # ------------D training------------------
                self.ldisc_iteration = 0.5 * F.binary_cross_entropy(d_x, d_x_true) + 0.5 * F.binary_cross_entropy(d_xhat, d_xhat_true)

                self.opt_Disc.zero_grad()
                self.ldisc_iteration.backward(retain_graph=True)
                self.opt_Disc.step()

                # ------------Encoder and Decoder training------------------
                # Discriminator prediction
                d_xhat, _ = self.Disc(torch.sigmoid(xhat))

                # Calculate criterion
                self.kl_iteration = self.KL(mu=z_mu, logvar=z_logvar) / self.train_generator.batch_size  # kl loss
                if self.loss_reconstruction == 'bce':  # Reconstruction loss
                    self.lr_iteration = self.Lr(xhat, x_n) / self.train_generator.batch_size
                else:
                    self.lr_iteration = self.Lr(torch.sigmoid(xhat), x_n) / self.train_generator.batch_size
                self.ladv_iteration = F.binary_cross_entropy(d_xhat, d_f_false) # Adversial loss

                # Init overall losses
                L = self.wr * self.lr_iteration + self.wkl * self.kl_iteration + self.wadv * self.ladv_iteration

                # Update weights
                L.backward()  # Backward
                self.opt_AL.step()  # Update weights
                self.opt_AL.zero_grad()  # Clear gradients

                """
                ON ITERATION/EPOCH END PROCESS
                """

                # Display losses per iteration
                self.display_losses(on_epoch_end=False)

                # Update epoch's losses
                self.kl_epoch += self.kl_iteration.cpu().detach().numpy() / len(self.train_generator)
                self.ladv_epoch += self.ladv_iteration.cpu().detach().numpy() / len(self.train_generator)
                self.ldisc_epoch += self.ldisc_iteration.cpu().detach().numpy() / len(self.train_generator)
                self.lr_epoch += self.lr_iteration.cpu().detach().numpy() / len(self.train_generator)

            # Epoch-end processes
            self.on_epoch_end()

    def train_f_ano_gan(self):

        self.init_time = timer()
        for self.i_epoch in range(self.epochs):

            # init epoch losses
            self.kl_epoch, self.ladv_epoch, self.ldisc_epoch, self.lr_epoch, self.lperc_epoch = 0, 0, 0, 0, 0

            '''
            if self.i_epoch == 100:
                self.opt_AL = torch.optim.Adam(self.E.parameters(), lr=self.lr)
                self.opt_Disc = torch.optim.Adam(self.Disc.parameters(), lr=0)

                
                for g in self.Dec.parameters:
                    g['lr'] = 1e-5
                for g in self.opt_Disc.param_groups:
                    g['lr'] = 1e-5
            '''

            # Loop over training dataset
            for self.i_iteration, (x_n, y_n, _, _) in enumerate(self.train_generator):
                if self.context:
                    (x_n_context, _) = augment_input_batch(x_n.copy())
                    x_n_context = torch.tensor(x_n_context).cuda().float()

                # Move tensors to gpu
                x_n = torch.tensor(x_n).cuda().float()

                # Obtain latent space from normal sample via encoder
                z, _, _, _ = self.E(x_n)

                # Obtain reconstructed images through decoder
                xhat, _ = self.Dec(z)

                # Forward discriminator
                d_x, f_x = self.Disc(x_n)
                d_xhat, _ = self.Disc(torch.sigmoid(xhat))

                # Discriminator labels
                d_x_true = torch.tensor(self.normal_label * np.ones((self.train_generator.batch_size, 1))).cuda().float()
                d_f_false = torch.tensor(self.normal_label * np.ones((self.train_generator.batch_size, 1))).cuda().float()
                d_xhat_true = torch.tensor(self.generated_label * np.ones((self.train_generator.batch_size, 1))).cuda().float()

                # ------------D training------------------

                self.ldisc_iteration = 0.5 * F.binary_cross_entropy(d_x, d_x_true) + 0.5 * F.binary_cross_entropy(d_xhat, d_xhat_true)

                self.opt_Disc.zero_grad()
                self.ldisc_iteration.backward(retain_graph=True)
                self.opt_Disc.step()

                # ------------Encoder and Decoder training------------------
                # Discriminator prediction
                _, f_x = self.Disc(x_n)
                d_xhat, f_xhat = self.Disc(torch.sigmoid(xhat))

                # Calculate criterion
                self.lr_iteration = torch.mean(torch.pow(torch.sigmoid(xhat) - x_n, 2))
                self.ladv_iteration = torch.mean(torch.pow(f_x[-1] - f_xhat[-1], 2))  # Adversial loss

                # Init overall losses
                L = self.wr * self.lr_iteration + self.ladv_iteration

                # Update weights
                L.backward()  # Backward
                self.opt_AL.step()  # Update weights
                self.opt_AL.zero_grad()  # Clear gradients

                """
                ON ITERATION/EPOCH END PROCESS
                """

                # Display losses per iteration
                self.display_losses(on_epoch_end=False)

                # Update epoch's losses
                self.ladv_epoch += self.ladv_iteration.cpu().detach().numpy() / len(self.train_generator)
                self.ldisc_epoch += self.ldisc_iteration.cpu().detach().numpy() / len(self.train_generator)
                self.lr_epoch += self.lr_iteration.cpu().detach().numpy() / len(self.train_generator)

            # Epoch-end processes
            self.on_epoch_end()

    def train_proposed(self):

        self.init_time = timer()
        for self.i_epoch in range(self.epochs):

            # init epoch losses
            self.kl_epoch, self.ladv_epoch, self.ldisc_epoch, self.lr_epoch, self.lae_epoch = 0, 0, 0, 0, 0
            # VAE pre-training iterations
            pre_training_epochs = round(400 / (len(self.train_generator.indexes) / self.train_generator.batch_size))

            # Loop over training dataset
            for self.i_iteration, (x_n, y_n, x_a, y_a) in enumerate(self.train_generator):

                # Move tensors to gpu
                x_n = torch.tensor(x_n).cuda().float()

                # Obtain latent space from normal sample via encoder and noise
                z_n, z_mu, z_logvar, allF = self.E(x_n)

                # Obtain reconstructed images through decoder
                xhat = self.Dec(z_n)[0]

                # Calculate criterion
                self.kl_iteration = self.KL(mu=z_mu, logvar=z_logvar) / (self.train_generator.batch_size*self.zdim)  # kl loss
                if self.loss_reconstruction == 'bce':  # Reconstruction loss
                    self.lr_iteration = self.Lr(xhat, x_n) / (self.train_generator.batch_size*self.input_shape[1]*self.input_shape[2])
                elif  self.loss_reconstruction == 'L2':
                    self.lr_iteration = self.Lr(torch.sigmoid(xhat), x_n) / (self.train_generator.batch_size * self.input_shape[1] * self.input_shape[2])
                elif  self.loss_reconstruction == 'ssim':
                    self.lr_iteration = torch.sum(self.Lr(torch.sigmoid(xhat), x_n)) / (self.train_generator.batch_size*self.input_shape[1]*self.input_shape[2])

                # Init overall losses
                L = self.wkl * self.kl_iteration + self.wr * self.lr_iteration

                # Attention expansion loss
                if self.expansion_loss:
                    # Compute grad-cams
                    gcam = grad_cam(allF[self.level_cams], torch.sum(z_mu), normalization=self.normalization_cam,
                                    avg_grads=self.avg_grads)
                    self.lae_iteration = torch.mean(gcam)

                    if self.i_epoch > pre_training_epochs:
                        # Compute attention expansion loss
                        if self.expansion_loss_penalty == 'l1':
                            lae = torch.sum(torch.abs(-torch.mean(gcam, (1, 2)) + 1 - self.p_activation_cam))  # L1
                        elif self.expansion_loss_penalty == 'l2':
                            lae = torch.sum(torch.sqrt(torch.pow(-torch.mean(gcam, (1, 2)) + 1 - self.p_activation_cam, 2)))  # L2
                        elif self.expansion_loss_penalty == 'log_barrier':
                            z = -torch.mean(gcam, (1, 2)).unsqueeze(-1) + 1
                            lae = log_barrier(z - self.p_activation_cam, t=self.t) / self.train_generator.batch_size

                        # Update overall losses
                        L += self.wae * lae

                # Update weights
                L.backward()  # Backward
                self.opt_AL.step()  # Update weights
                self.opt_AL.zero_grad()  # Clear gradients

                """
                ON ITERATION/EPOCH END PROCESS
                """

                # Display losses per iteration
                self.display_losses(on_epoch_end=False)

                # Update epoch's losses
                self.kl_epoch += self.kl_iteration.cpu().detach().numpy() / len(self.train_generator)
                self.lr_epoch += self.lr_iteration.cpu().detach().numpy() / len(self.train_generator)
                self.lae_epoch += self.lae_iteration.cpu().detach().numpy() / len(self.train_generator)

            # Epoch-end processes
            self.on_epoch_end()

    def inference_data_set_scans(self, X, M):

        # Init variables
        (p, s, c, h, w) = X.shape  # maps dimensions
        Mhat = np.zeros(M.shape)   # Predicted segmentation maps
        Xhat = np.zeros(X.shape)   # Reconstructed images

        for iVolume in np.arange(0, p):
            for iSlice in np.arange(0, s):

                # Take image
                x = X[iVolume, iSlice, :, :, :]

                # Prepare brain eroded mask
                x_mask = 1 - (x == 0).astype(np.int)
                #x_mask = ndimage.binary_closing(x_mask, structure=np.ones((1, 3, 3))).astype(x_mask.dtype)
                x_mask = ndimage.binary_erosion(x_mask, structure=np.ones((1, 6, 6))).astype(x_mask.dtype)

                # Network forward
                z, z_mu, z_logvar, f = self.E(torch.tensor(x).cuda().float().unsqueeze(0))
                xhat = np.squeeze(torch.sigmoid(self.Dec(z)[0]).cpu().detach().numpy())

                # Obtain anomaly activation map
                if self.method == 'ae' or self.method == 'vae' or self.method == 'anoVAEGAN' or self.method == 'f_ano_gan':
                    if self.bayesian:
                        N = 100
                        p_dropout = 0.20
                        mhat = np.zeros((h, w))

                        for i in np.arange(N):
                            if self.method == 'ae':  # apply dropout to z
                                mhat += np.squeeze(np.abs(np.squeeze(torch.sigmoid(
                                    self.Dec(torch.nn.Dropout(p_dropout)(z))[0]).cpu().detach().numpy()) - x)) / N
                            else:  # sample z
                                mhat += np.squeeze(np.abs(np.squeeze(torch.sigmoid(
                                    self.Dec(self.E.reparameterize(z_mu, z_logvar))[0]).cpu().detach().numpy()) - x)) / N
                    elif self.restoration:
                        N = 300
                        step = 1*1e-3
                        x_rest = torch.tensor(x).cuda().float().unsqueeze(0)

                        for i in np.arange(N):
                            # Forward
                            x_rest.requires_grad = True
                            z, z_mu, z_logvar, f = self.E(x_rest)
                            xhat = self.Dec(z)[0]

                            lr = kornia.losses.total_variation(torch.tensor(x).cuda().float().unsqueeze(0) - torch.sigmoid(xhat))
                            L = self.KL(mu=z_mu, logvar=z_logvar) / self.zdim + self.Lr(xhat, torch.tensor(x).cuda().float().unsqueeze(0)) / (h*w) + 5*lr / (h*w)

                            # Get gradients
                            gradients = torch.autograd.grad(L, x_rest, grad_outputs=None, retain_graph=True,
                                                            create_graph=True,
                                                            only_inputs=True, allow_unused=True)[0]

                            x_rest = x_rest - gradients * step
                            #x_rest = torch.relu(x_rest)
                            x_rest = x_rest.clone().detach()
                            #print(str(lr.detach().cpu().numpy().item()) + '/' + str(self.KL(mu=z_mu, logvar=z_logvar).detach().cpu().numpy().item()), end='\r')
                        xhat = np.squeeze(x_rest.cpu().numpy())
                        mhat = np.squeeze(np.abs(x - xhat))
                    else:
                        mhat = np.squeeze(np.abs(x - xhat))

                elif self.method == 'proposed' or self.method == 'cavga':

                    # Compute grad-cams
                    gcam = grad_cam(f[self.level_cams], torch.sum(z_mu), normalization='',
                                    avg_grads=self.avg_grads)

                    # Restore original shape
                    mhat = torch.nn.functional.interpolate(gcam.unsqueeze(0), size=(h, w), mode='bilinear',
                                                           align_corners=True).squeeze().detach().cpu().numpy()
                    # max-normalization
                    mhat = mhat - np.min(mhat[x_mask[0, :, :] == 1])
                    mhat = mhat / np.max(mhat)

                    if self.method == 'cavga':
                        mhat = - mhat + 1

                # Keep only brain region
                mhat[x_mask[0, :, :] == 0] = 0

                Mhat[iVolume, iSlice, :, :, :] = mhat
                Xhat[iVolume, iSlice, :, :, :] = xhat

        return Mhat, Xhat

    def test(self, test_dataset, dir_out, save_maps=False):
        self.opt_AL.zero_grad()
        self.E.eval()
        self.Dec.eval()

        print('[INFO]: Testing...')
        if save_maps:
            if not os.path.isdir(dir_out + 'masks_predicted9/'):
                os.mkdir(dir_out + 'masks_predicted9/')
            if not os.path.isdir(dir_out + 'masks_reference9'):
                os.mkdir(dir_out + 'masks_reference9/')

        if self.threshold_on_normal:

            # take references and inputs from training dataset
            X_training = self.train_generator.dataset.X[self.train_generator.indexes, :, :, :, :]
            M_training = np.expand_dims(self.train_generator.dataset.M[self.train_generator.indexes, :, :, :], 1)
            Mhat_training, _ = self.inference_data_set_scans(X_training, M_training)

            # Threshold on normal images
            prctile = []
            for i in np.arange(0, X_training.shape[0]):
                prctile.append(np.percentile(Mhat_training[i, :, :, :, :].flatten(), 85))
            th_normal = np.median(prctile)

        # Take references and inputs from testing dataset
        X = test_dataset.X
        M = test_dataset.M

        # Predict anomalies heatmaps
        Mhat, Xhat = self.inference_data_set_scans(X, M)

        # Obtain overall metrics and threshold
        AU_ROC = sklearn.metrics.roc_auc_score(M.flatten() == 1, Mhat.flatten())  # au_roc
        AU_PRC, th = au_prc(M.flatten() == 1, Mhat.flatten())  # au_prc
        if self.threshold_on_normal:
            th = th_normal
        self.th = th
        #th = 0.5

        DICE = dice(M.flatten() == 1, (Mhat > th).flatten())  # Dice
        IoU = sklearn.metrics.jaccard_score(M.flatten() == 1, (Mhat > th).flatten())  # IoU

        # Once the threshold is obtained calculate volume-level metrics and plot results
        patient_dice = []
        (p, s, c, h, w) = X.shape  # maps dimensions
        for iVolume in np.arange(0, p):
            patient_dice.append(dice(M[iVolume, :, :, :, :].flatten(), (Mhat[iVolume, :, :, :, :] > th).flatten()))

            if save_maps:  # Save slices' masks
                for iSlice in np.arange(0, s):
                    id = test_dataset.unique_patients[iVolume] + '_' + str(iSlice) + '.jpg'

                    # Obtain heatmaps for predicted and reference
                    m_i = imutils.rotate_bound(np.uint8(M[iVolume, iSlice, 0, :, :] * 255), 270)
                    heatmap_m = cv2.applyColorMap(m_i, cv2.COLORMAP_JET)
                    mhat_i = imutils.rotate_bound(np.uint8((Mhat[iVolume, iSlice, 0, :, :]) * 255), 270)
                    heatmap_mhat = cv2.applyColorMap(mhat_i, cv2.COLORMAP_JET)
                    heatmap_mhat = heatmap_mhat * (np.expand_dims(imutils.rotate_bound(Mhat[iVolume, iSlice, 0, :, :], 270), -1) >0)

                    # Move grayscale image to three channels
                    xh = cv2.cvtColor(np.uint8(np.squeeze(X[iVolume, iSlice, :, :, :]) * 255), cv2.COLOR_GRAY2RGB)
                    xh = imutils.rotate_bound(xh, 270)

                    # Combine original image and masks
                    fin_mask = cv2.addWeighted(xh, 0.7, heatmap_m, 0.3, 0)
                    fin_predicted = cv2.addWeighted(xh, 0.7, heatmap_mhat, 0.3, 0)

                    cv2.imwrite(dir_out + 'masks_predicted9/' + id, heatmap_mhat)
                    cv2.imwrite(dir_out + 'masks_reference9/' + id, fin_mask)

        DICE_mu = np.mean(patient_dice)
        DICE_std = np.std(patient_dice)

        metrics = {'AU_ROC': AU_ROC, 'AU_PRC': AU_PRC, 'DICE': DICE, 'IoU': IoU,
                   'DICE_mu': DICE_mu, 'DICE_std': DICE_std}

        self.E.train()
        self.Dec.train()
        return metrics

    def on_epoch_end(self):

        # Display losses
        self.display_losses(on_epoch_end=True)

        # Update learning curves
        self.kl_lc.append(self.kl_epoch)
        self.ladv_lc.append(self.ladv_epoch)
        self.ldisc_lc.append(self.ldisc_epoch)
        self.lr_lc.append(self.lr_epoch)
        if self.expansion_loss:
            self.lae_lc.append(self.lae_epoch)
        else:
            self.lae_lc.append(0)

        if self.expansion_loss:
            if self.lae_epoch > (1 - self.p_activation_cam):
                if self.log_barrier:
                    self.t = self.t
                    self.wae = self.wae

        # Each x epochs, test models and plot learning curves
        if (self.i_epoch + 1) % self.epochs_to_test == 0:
            # Save weights
            torch.save(self.E.state_dict(), self.dir_results + 'encoder_weights.pth')
            torch.save(self.Dec.state_dict(), self.dir_results + 'decoder_weights.pth')
            torch.save(self.Disc.state_dict(), self.dir_results + 'discriminator_weights.pth')

            # Metrics and segmentation maps
            metrics = self.test(self.test_dataset, self.dir_results, save_maps=True)
            self.metrics = metrics

            # Save metrics as dict
            with open(self.dir_results + 'metrics.json', 'w') as fp:
                json.dump(metrics, fp)
            print(metrics)

            # Plot learning curve
            self.plot_learning_curves()

            # Save learning curves as dataframe
            self.aucroc_lc.append(metrics['AU_ROC'])
            self.auprc_lc.append(metrics['AU_PRC'])
            history = pd.DataFrame(list(zip(self.kl_lc, self.lr_lc, self.lae_lc, self.aucroc_lc, self.auprc_lc)),
                                   columns=['KL', 'Lrec', 'Lae', 'AUCROC', 'AUPRC'])
            history.to_csv(self.dir_results + 'lc_on_direct.csv')

        else:
            self.aucroc_lc.append(0)
            self.auprc_lc.append(0)

    def plot_learning_curves(self):
        def plot_subplot(axes, x, y, y_axis):
            axes.grid()
            axes.plot(x, y, 'o-')
            axes.set_ylabel(y_axis)

        fig, axes = plt.subplots(4, 2, figsize=(20, 15))

        plot_subplot(axes[0, 0], np.arange(self.i_epoch + 1) + 1, np.array(self.kl_lc), "KL loss")
        plot_subplot(axes[0, 1], np.arange(self.i_epoch + 1) + 1, np.array(self.lr_lc), "Reconstruc loss")
        plot_subplot(axes[1, 0], np.arange(self.i_epoch + 1) + 1, np.array(self.ladv_lc), "Gen loss")
        plot_subplot(axes[1, 1], np.arange(self.i_epoch + 1) + 1, np.array(self.ldisc_lc), "Disc loss")
        if self.expansion_loss:
            plot_subplot(axes[2, 1], np.arange(self.i_epoch + 1) + 1, np.array(self.lae_lc), "Expansion loss")

        plt.savefig(self.dir_results + 'learning_curve.png')
        plt.close()

    def display_losses(self, on_epoch_end=False):

        info = "[INFO] Epoch {}/{}  -- Step {}/{}: ".format(self.i_epoch + 1, self.epochs,
                                                            self.i_iteration + 1, self.iterations)

        if on_epoch_end:
            kl = self.kl_epoch
            lr = self.lr_epoch
            ldisc = self.ldisc_epoch
            ladv = self.ladv_epoch
            end = '\n'
        else:
            kl = self.kl_iteration
            lr = self.lr_iteration
            ldisc = self.ldisc_iteration
            ladv = self.ladv_iteration
            end = '\r'

        info += "KL={:.6f}, Reconstruction={:.6f}, Adversial={:.6f}, Discriminator={:.6f}".format(
            kl, lr, ladv, ldisc)

        if not on_epoch_end:
            if self.expansion_loss:
                info += ", Expansion={:.6f}".format(self.lae_iteration)
        else:
            if self.expansion_loss:
                info += ", Expansion={:.6f}".format(self.lae_epoch)

        # Print losses
        et = str(datetime.timedelta(seconds=timer() - self.init_time))
        print(info + ',ET=' + et, end=end)
