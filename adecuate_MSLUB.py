import nibabel as nib
import os
import numpy as np
import random
import cv2
from matplotlib import pyplot as plt
import imutils

np.random.seed(42)
random.seed(42)

dir_dataset = '../data/MSLUB/'
dir_out = '../data/MSLUB_5slices/'
partitions = ['test']
nSlices = 5
scan = 'flair'

if not os.path.isdir(dir_out):
    os.mkdir(dir_out)
if not os.path.isdir(dir_out + '/' + scan):
    os.mkdir(dir_out + '/' + scan)

cases = os.listdir(dir_dataset)
c = 0

for iPartition in np.arange(0, len(partitions)):

    if not os.path.isdir(dir_out + '/' + scan + '/' + partitions[iPartition]):
        os.mkdir(dir_out + '/' + scan + '/' + partitions[iPartition])
    if not os.path.isdir(dir_out + '/' + scan + '/' + partitions[iPartition] + '/benign'):
        os.mkdir(dir_out + '/' + scan + '/' + partitions[iPartition] + '/benign')
    if not os.path.isdir(dir_out + '/' + scan + '/' + partitions[iPartition] + '/malign'):
        os.mkdir(dir_out + '/' + scan + '/' + partitions[iPartition] + '/malign')
    if not os.path.isdir(dir_out + '/' + scan + '/' + partitions[iPartition] + '/ground_truth'):
        os.mkdir(dir_out + '/' + scan + '/' + partitions[iPartition] + '/ground_truth')

    for iCase in cases:
        c += 1
        print(str(c) + '/' + str(len(cases)))

        img_path = dir_dataset + iCase + '/' + 'FLAIR_N4_noneck_reduced_winsor_regtoFLAIR_brain_N4_regtoMNI' + '.nii.gz'
        mask_path = dir_dataset + iCase + '/' + 'GOLD_STANDARD_N4_noneck_reduced_winsor_regtoFLAIR_regtoMNI' + '.nii.gz'

        img = nib.load(img_path)
        img = (img.get_fdata())[:, :, :]
        img = (img/img.max())*255
        img = imutils.rotate_bound(img, 180)
        img = img.astype(np.uint8)

        mask = nib.load(mask_path)
        mask = (mask.get_fdata())
        mask[mask > 0] = 255
        mask = imutils.rotate_bound(mask, 180)
        mask = mask.astype(np.uint8)

        for iSlice in np.arange(round(img.shape[-1]/2) - nSlices, round(img.shape[-1]/2) + nSlices):
            filename = iCase + '_' + str(iSlice) + '.jpg'

            i_image = img[:, :, iSlice]
            i_mask = mask[:, :, iSlice]

            if np.any(i_mask == 255):
                label = 'malign'
                cv2.imwrite(dir_out + '/' + scan + '/' + partitions[iPartition] + '/ground_truth/' + filename, i_mask)
            else:
                label = 'benign'

            cv2.imwrite(dir_out + '/' + scan + '/' + partitions[iPartition] + '/' + label + '/' + filename, i_image)


