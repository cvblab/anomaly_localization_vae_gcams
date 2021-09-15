import os
import numpy as np

from models import *
from PIL import Image
import cv2
from timeit import default_timer as timer
import datetime

# INPUTS
input_z_shape = (100, 1, 1)
input_image_shape = (3, 64, 64)
lr = 2*1e-4
batch_size = 128
epochs = 100
dir_out = '../data/CELEBA/'
dir_input = '../data/CELEBA/img_align_celeba/'
label_true = 1
label_generated = 0


# MAIN

if not os.path.isdir(dir_out):
    os.mkdir(dir_out)

# Set models, optimizers and losses
Dec = Generator(3, input_z_shape[0], 64).cuda()  # Decoder
Disc = Discriminator(3, 64).cuda()                # Discriminator
DLoss = torch.nn.BCEWithLogitsLoss()   # Losses

opt_G = torch.optim.Adam(list(Dec.parameters()), lr=lr)  # Optimizer for G
opt_D = torch.optim.Adam(list(Disc.parameters()), lr=lr)  # Optimizer for D

# Prepare data
images = os.listdir(dir_input)

X = np.zeros((len(images), 3, input_image_shape[-1], input_image_shape[-1]))
c = 0
print('[INFO]: Loading images...')
for iImage in images:
    print(c, end='\r')
    # Load and preprocess image
    x = Image.open(dir_input + iImage)
    x = np.asarray(x)
    x = cv2.resize(x, (input_image_shape[1], input_image_shape[2]), input_image_shape[0])
    x = np.transpose(x, (2, 0, 1))
    x = (x - 127.5) / 127.5
    # Add image
    X[c, :, :, :] = x
    c += 1

indexes = np.arange(X.shape[0])
iterations = X.shape[0] // batch_size

init_time = timer()
fixed_noise = torch.randn((batch_size, input_z_shape[0], input_z_shape[-1], input_z_shape[-1])).cuda().float()
for i_epoch in range(epochs):
    dLoss_epoch = 0
    gLoss_epoch = 0
    for i_iteration in range(iterations):

        # Set inputs
        x = X[i_iteration*batch_size:(i_iteration+1)*batch_size, :, :, :]
        n = np.random.random((batch_size, input_z_shape[0], input_z_shape[-1], input_z_shape[-1]))

        x = torch.tensor(x).cuda().float()
        n = torch.randn((batch_size, input_z_shape[0], input_z_shape[-1], input_z_shape[-1])).cuda().float()

        # Train discriminator
        Dec.zero_grad()
        Disc.zero_grad()

        # Unfreeze D for autoencoder training
        for param in Disc.parameters():
            param.requires_grad = True

        predicted_true = Disc(x)
        predicted_generated = Disc(torch.tanh(Dec(n)))

        n_ref = torch.tensor(label_true * np.ones(predicted_true.shape)).cuda().float()
        a_ref = torch.tensor(label_generated * np.ones(predicted_generated.shape)).cuda().float()

        Ln = DLoss(predicted_true, n_ref)
        Ln.backward()
        Lg = DLoss(predicted_generated, a_ref)
        Lg.backward()
        dL = 0.5 * Ln + 0.5 * Lg

        opt_D.step()
        opt_D.zero_grad()

        # Train generator
        Dec.zero_grad()
        Disc.zero_grad()

        # Freeze D for autoencoder training
        for param in Disc.parameters():
            param.requires_grad = False

        xhat = torch.tanh(Dec(n))
        predicted_generated = Disc(xhat)
        n_ref = torch.tensor(label_true * np.ones(predicted_generated.shape)).cuda().float()

        gLoss = DLoss(predicted_generated, n_ref)

        gLoss.backward()
        opt_G.step()
        opt_G.zero_grad()

        # Update epoch losses
        dLoss_epoch += dL.cpu().detach().numpy() / iterations
        gLoss_epoch += gLoss.cpu().detach().numpy() / iterations
        # Print losses
        et = str(datetime.timedelta(seconds=timer() - init_time))
        print(
            "[INFO] Epoch {}/{}  -- Step {}/{}: Generator={:.6f}, Discriminator={:.6f} ".format(
                i_epoch + 1, epochs, i_iteration + 1, iterations, gLoss.cpu().detach().numpy(), dL.cpu().detach().numpy()) + ',ET=' + et, end='\r')
        if i_iteration % 100 == 0:
            Dec.eval()
            xhat = torch.tanh(Dec(fixed_noise))
            for i in np.arange(8):
                x = xhat[i, :, :, :].detach().cpu().numpy()
                x = np.transpose((x*127.5 + 127.5). astype(int), (1, 2, 0))
                x = x[:, :, [2, 1, 0]]
                cv2.imwrite(dir_out + 'example_' + str(i) + '.jpg', x)
            Dec.train()
    print(
        "[INFO] Epoch {}/{}  -- Step {}/{}: Generator={:.6f}, Discriminator={:.6f} ".format(
            i_epoch + 1, epochs, iterations, iterations, gLoss_epoch,
            dLoss_epoch) + ',ET=' + et, end='\n')

    torch.save(Dec.state_dict(), dir_out + 'encoder_weights.pth')
    torch.save(Disc.state_dict(), dir_out + 'decoder_weights.pth')






