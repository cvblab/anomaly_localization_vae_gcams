import numpy as np
import cv2

from skimage import exposure
from matplotlib import pyplot as plt

def image_normalization(x, shape, norm='max', channels=3, histogram_matching=False, reference_image=None,
                        mask=False, channel_first=True):

    # Histogram matching to reference image
    if histogram_matching:
        # ret2, th2 = cv2.threshold(x, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        x_norm = exposure.match_histograms(x, reference_image)
        x_norm[x == 0] = 0
        x = x_norm

    # image resize with canvas
    x = cv2.resize(x, (shape[1], shape[2]), channels)
    # Grayscale image -- add channel dimension
    if len(x.shape) < 3:
        x = np.expand_dims(x, -1)

    if mask:
        x = (x > 200)

    # channel first
    if channel_first:
        x = np.transpose(x, (2, 0, 1))
    if not mask:
        if norm == 'max':
            x = x / 255.0
        elif norm == 'zscore':
            x = (x - 127.5) / 127.5

    # numeric type
    x.astype('float32')
    return x


def plot_image(x, y=None, denorm_intensity=False, channel_first=True):
    if len(x.shape) < 3:
        x = np.expand_dims(x, 0)
    # channel first
    if channel_first:
        x = np.transpose(x, (1, 2, 0))
    if denorm_intensity:
        if self.norm == 'zscore':
            x = (x*127.5) + 127.5
            x = x.astype(int)

    plt.imshow(x)

    if y is not None:
        y = np.expand_dims(y[0, :, :], -1)
        plt.imshow(y, cmap='jet', alpha=0.1)

    plt.axis('off')
    plt.show()