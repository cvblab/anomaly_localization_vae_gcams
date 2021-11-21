import torch
import torchvision
import numpy as np


class Resnet(torch.nn.Module):
    def __init__(self, in_channels, n_blocks=4):
        super(Resnet, self).__init__()
        self.n_blocks = n_blocks
        self.nfeats = 512 // (2**(5-n_blocks))

        self.resnet18_model = torchvision.models.resnet18(pretrained=False)
        self.input = torch.nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(7, 7), stride=(2, 2),
                                     padding=(3, 3), bias=False)

        # placeholder for the gradients
        self.gradients = None

    def forward(self, x):
        x = self.input(x)
        F = []
        for iBlock in range(1, self.n_blocks+1):
            x = list(self.resnet18_model.children())[iBlock+2](x)
            F.append(x)

        return x, F


class Encoder(torch.nn.Module):
    def __init__(self, mode, fin=1, zdim=128, dense=False, n_blocks=4, spatial_dim=7):
        super(Encoder, self).__init__()
        self.mode = mode  # Supported modes: ae, vae, cavga, anoVAEGAN
        self.fin = fin
        self.zdim = zdim
        self.dense = dense
        self.n_blocks = n_blocks

        # 1) Feature extraction
        self.backbone = Resnet(in_channels=self.fin, n_blocks=self.n_blocks)
        # 2) Latent space (dense or spatial)
        if not self.dense:  # spatial
            if self.mode == 'ae' or self.mode == 'f_ano_gan':
                self.z = torch.nn.Conv2d(self.backbone.nfeats, zdim, (1, 1))
            else:
                self.mu = torch.nn.Conv2d(self.backbone.nfeats, zdim, (1, 1))
                self.log_var = torch.nn.Conv2d(self.backbone.nfeats, zdim, (1, 1))
        else:  # dense
            if self.mode == 'ae' or self.mode == 'f_ano_gan':
                self.z = torch.nn.Linear(self.backbone.nfeats*spatial_dim**2, zdim)
            else:
                self.mu = torch.nn.Linear(self.backbone.nfeats*spatial_dim**2, zdim)
                self.log_var = torch.nn.Linear(self.backbone.nfeats*spatial_dim**2, zdim)

    def reparameterize(self, mu, log_var):

        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)

        sample = mu + (eps * std)  # sampling
        return sample

    def forward(self, x):

        # 1) Feature extraction
        x, allF = self.backbone(x)

        if self.dense:
            x = torch.nn.Flatten()(x)

        # 2) Latent space
        if self.mode == 'ae' or self.mode == 'f_ano_gan':
            z = self.z(x)
            z_mu, z_logvar = None, None
        else:
            # get `mu` and `log_var`
            z_mu = self.mu(x)
            z_logvar = self.log_var(x)
            # get the latent vector through reparameterization
            z = self.reparameterize(z_mu, z_logvar)

        return z, z_mu, z_logvar, allF


class Decoder(torch.nn.Module):

    def __init__(self, fin=256, nf0=128, n_channels=1, dense=False, n_blocks=4, spatial_dim=7):
        super(Decoder, self).__init__()
        self.n_blocks = n_blocks
        self.dense = dense
        self.spatial_dim = spatial_dim
        self.fin = fin

        if self.dense:
            self.dense = torch.nn.Linear(fin, fin*spatial_dim**2)

        # Set number of input and output channels
        n_filters_in = [fin] + [nf0//2**i for i in range(0, self.n_blocks)]
        n_filters_out = [nf0//2**(i-1) for i in range(1, self.n_blocks+1)] + [n_channels]

        self.blocks = torch.nn.ModuleList()
        for i in np.arange(0, self.n_blocks):
            self.blocks.append(ResBlock(n_filters_in[i], n_filters_out[i]))
        self.out = torch.nn.Conv2d(n_filters_in[-1], n_filters_out[-1], kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x):

        if self.dense:
            x = self.dense(x)
            x = torch.nn.Unflatten(-1, (self.fin, self.spatial_dim, self.spatial_dim))(x)

        for i in np.arange(0, self.n_blocks):
            x = self.blocks[i](x)
        f = x
        out = self.out(f)

        return out, f


class ResBlock(torch.nn.Module):

    def __init__(self, fin, fout):
        super(ResBlock, self).__init__()
        self.conv_straight_1 = torch.nn.Conv2d(fin, fout, kernel_size=(3, 3), padding=(1, 1))
        self.bn_1 = torch.nn.BatchNorm2d(fout)
        self.conv_straight_2 = torch.nn.Conv2d(fout, fout, kernel_size=(3, 3), padding=(1, 1))
        self.bn_2 = torch.nn.BatchNorm2d(fout)
        self.conv_skip = torch.nn.Conv2d(fin, fout, kernel_size=(3, 3), padding=(1, 1))
        self.upsampling = torch.nn.Upsample(scale_factor=(2, 2))
        self.relu = torch.nn.ReLU()

    def forward(self, x):

        x_st = self.upsampling(x)
        x_st = self.conv_straight_1(x_st)
        x_st = self.relu(x_st)
        x_st = self.bn_1(x_st)
        x_st = self.conv_straight_2(x_st)
        x_st = self.relu(x_st)
        x_st = self.bn_2(x_st)

        x_sk = self.upsampling(x)
        x_sk = self.conv_skip(x_sk)

        out = x_sk + x_st

        return out


class ConvBlock(torch.nn.Module):

    def __init__(self, fin, fout):
        super(ConvBlock, self).__init__()

        self.conv = torch.nn.Conv2d(fin, fout, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn = torch.nn.BatchNorm2d(fout)
        self.act = torch.nn.LeakyReLU(0.2, inplace=False)

    def forward(self, x):

        out = self.act(self.bn(self.conv(x)))

        return out


class DeconvBlock(torch.nn.Module):

    def __init__(self, fin, fout):
        super(DeconvBlock, self).__init__()

        self.conv = torch.nn.Conv2d(fin, fout, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn = torch.nn.BatchNorm2d(fout)
        self.act = torch.nn.LeakyReLU(0.2, inplace=False)

    def forward(self, x):

        out = self.act(self.bn(self.conv(x)))

        return out


class Discriminator(torch.nn.Module):
    def __init__(self, fin=32, n_channels=1, n_blocks=4, type='DCGAN'):
        super(Discriminator, self).__init__()

        # Number of feature extractor blocks
        self.n_blocks = n_blocks
        # Set number of input and output channels
        n_filters_in = [n_channels] + [fin*(2**i) for i in range(0, self.n_blocks)]
        n_filters_out = [fin*(2**i) for i in range(0, self.n_blocks+1)]
        # Number of output features
        self.nFeats = n_filters_out[-1]
        # Prepare blocks:
        self.blocks = torch.nn.ModuleList()
        for i in np.arange(0, self.n_blocks + 1):
            if type == 'DCGAN':
                self.blocks.append(ConvBlock(n_filters_in[i], n_filters_out[i]))
        # Output for binary clasification
        self.pool_feats = torch.nn.AdaptiveAvgPool2d((7, 7))
        self.out = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1),
                                       torch.nn.Flatten(),
                                       torch.nn.Linear(self.nFeats, 1),
                                       torch.nn.Sigmoid())

    def forward(self, x):

        for i in np.arange(0, self.n_blocks+1):
            x = self.blocks[i](x)
        f = self.pool_feats(x)

        out = self.out(f)

        return out, f
