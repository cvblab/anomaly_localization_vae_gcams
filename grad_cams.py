import torch
import numpy as np


def grad_cam(activations, output, normalization='relu_min_max', avg_grads=False, norm_grads=False):
    def normalize(grads):
        l2_norm = torch.sqrt(torch.mean(torch.pow(grads, 2), (-1, -2, -3))) + 1e-5
        l2_norm = l2_norm.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return grads * torch.pow(l2_norm, -1)

    # Obtain gradients
    gradients = torch.autograd.grad(output, activations, grad_outputs=None, retain_graph=True, create_graph=True,
                                    only_inputs=True, allow_unused=True)[0]

    # Normalize gradients
    if norm_grads:
        gradients = normalize(gradients)

    # pool the gradients across the channels
    if avg_grads:
        gradients = torch.mean(gradients, dim=[2, 3])
        gradients = gradients.unsqueeze(-1).unsqueeze(-1)

    # weight activation maps
    if 'relu' in normalization:
        GCAM = torch.sum(torch.relu(gradients * activations), 1)
    else:
        GCAM = torch.sum(gradients * activations, 1)

    # Normalize CAM
    if 'sigm' in normalization:
        GCAM = torch.sigmoid(GCAM)
    if 'abs' in normalization:
        GCAM = torch.abs(GCAM)
    if 'min' in normalization:
        norm_value = torch.min(torch.max(GCAM, -1)[0], -1)[0].unsqueeze(-1).unsqueeze(-1) + 1e-3
        GCAM = GCAM - norm_value
    if 'max' in normalization:
        norm_value = torch.max(torch.max(GCAM, -1)[0], -1)[0].unsqueeze(-1).unsqueeze(-1) + 1e-3
        GCAM = GCAM * norm_value.pow(-1)
    if 'tanh' in normalization:
        GCAM = torch.tanh(GCAM)
    if 'clamp' in normalization:
        GCAM = GCAM.clamp(max=1)

    return GCAM


def grad_cam_liu(activations, output, act='relu', normalization=True):
    """
    Grad-CAM from the paper 'Towards visually explaining Variational autoencoders',
    Liu et al., (2020).
    Obtains a CAM per output and combines them
    """

    b, zdim = output.shape
    _, _, xdim, ydim = activations.shape
    M = torch.zeros((b, zdim, xdim, ydim)).cuda()

    for iZ in np.arange(zdim):
        # Obtain gradients
        gradients = \
        torch.autograd.grad(torch.sum(output[:, iZ]), activations, grad_outputs=None, retain_graph=True,
                            create_graph=True, only_inputs=True, allow_unused=True)[0]

        # pool the gradients across the channels
        gradients = torch.mean(gradients, dim=[2, 3])
        gradients = gradients.unsqueeze(-1).unsqueeze(-1)

        # weight activation maps
        M[:, iZ, :, :] = torch.sum(gradients * activations, 1)

    if act == 'relu':
        M = torch.relu(M)
    elif act == 'sigmoid':
        M = torch.sigmoid(M)
    M = torch.mean(M, 1)

    if normalization:
        M = M / torch.max(M)

    return M