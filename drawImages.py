import numpy as np
import cv2
import argparse

from skimage import io, color
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb


def runDrawImages(args):

    # Load images as greyscale but make main RGB so we can annotate in colour
    seg_gt  = cv2.imread(os.path.join(args.img_folder,args.img_name+'_reference_binaries.jpg'),cv2.IMREAD_GRAYSCALE)
    seg_ours  = cv2.imread(os.path.join(args.img_folder,args.img_name+'_proposed_binaries.jpg'),cv2.IMREAD_GRAYSCALE)
    seg_CAE = cv2.imread(os.path.join(args.img_folder, args.img_name + '_aecontext_binaries.jpg'), cv2.IMREAD_GRAYSCALE)
    seg_VAE = cv2.imread(os.path.join(args.img_folder, args.img_name + '_vae_binaries.jpg'), cv2.IMREAD_GRAYSCALE)
    seg_fano = cv2.imread(os.path.join(args.img_folder, args.img_name + '_fanogan_binaries.jpg'), cv2.IMREAD_GRAYSCALE)
    cams_ours  = cv2.imread(os.path.join(args.img_folder,args.img_name+'_proposed_cams.jpg'))
    main = cv2.imread(os.path.join(args.img_folder,args.img_name+'_image.jpg'),cv2.IMREAD_GRAYSCALE)
    main = cv2.cvtColor(main,cv2.COLOR_GRAY2BGR)

    dimA = seg_ours.shape
    main = cv2.resize(main, dimA, interpolation=cv2.INTER_AREA)

    # Crop images
    main = main[10:220, 40:180]
    seg_ours = seg_ours[10:220, 40:180]
    seg_CAE = seg_CAE[10:220, 40:180]
    seg_VAE = seg_VAE[10:220, 40:180]
    seg_fano = seg_fano[10:220, 40:180]
    seg_gt = seg_gt[10:220, 40:180]
    cams_ours = cams_ours[10:220, 40:180]

    ## Our segmentation
    seg = np.zeros(seg_ours.shape)  # create a matrix of zeroes of same size as image
    for i in range(seg_ours.shape[0]):
        for j in range(seg_ours.shape[1]):
            if seg_ours[i,j]>125.0:
                seg[i,j] = 1  # Change zeroes to label "1" as per your condition(s)
    #result_image_ours= color.label2rgb(seg, main, colors=[(255, 0, 0), (0, 0, 255)], alpha=0.01, bg_label=0, bg_color=None)
    result_image_ours= color.label2rgb(seg, main, colors=[(0, 149, 66)], alpha=0.01, bg_label=0, bg_color=None)

    ## CAE segmentation
    seg = np.zeros(seg_ours.shape)  # create a matrix of zeroes of same size as image
    for i in range(seg_CAE.shape[0]):
        for j in range(seg_CAE.shape[1]):
            if seg_CAE[i, j] > 125.0:
                seg[i, j] = 1  # Change zeroes to label "1" as per your condition(s)
    # result_image_ours= color.label2rgb(seg, main, colors=[(255, 0, 0), (0, 0, 255)], alpha=0.01, bg_label=0, bg_color=None)
    result_image_CAE = color.label2rgb(seg, main, colors=[(0, 149, 66)], alpha=0.01, bg_label=0, bg_color=None)

    ## VAE segmentation
    seg = np.zeros(seg_ours.shape)  # create a matrix of zeroes of same size as image
    for i in range(seg_VAE.shape[0]):
        for j in range(seg_VAE.shape[1]):
            if seg_VAE[i, j] > 125.0:
                seg[i, j] = 1  # Change zeroes to label "1" as per your condition(s)
    # result_image_ours= color.label2rgb(seg, main, colors=[(255, 0, 0), (0, 0, 255)], alpha=0.01, bg_label=0, bg_color=None)
    result_image_VAE = color.label2rgb(seg, main, colors=[(0, 149, 66)], alpha=0.01, bg_label=0, bg_color=None)

    ## F-anoGAN segmentation
    seg = np.zeros(seg_ours.shape)  # create a matrix of zeroes of same size as image
    for i in range(seg_fano.shape[0]):
        for j in range(seg_fano.shape[1]):
            if seg_fano[i, j] > 125.0:
                seg[i, j] = 1  # Change zeroes to label "1" as per your condition(s)
    # result_image_ours= color.label2rgb(seg, main, colors=[(255, 0, 0), (0, 0, 255)], alpha=0.01, bg_label=0, bg_color=None)
    result_image_fano = color.label2rgb(seg, main, colors=[(0, 149, 66)], alpha=0.01, bg_label=0, bg_color=None)



    # Save result
    if not os.path.exists(args.img_folder_dst):
        os.makedirs(args.img_folder_dst)

    cv2.imwrite(os.path.join(args.img_folder_dst,'orig.jpg'),main)
    cv2.imwrite(os.path.join(args.img_folder_dst,'gt.jpg'),seg_gt)
    cv2.imwrite(os.path.join(args.img_folder_dst,'results_our.jpg'),result_image_ours*255)
    cv2.imwrite(os.path.join(args.img_folder_dst,'results_CAE.jpg'),result_image_CAE*255)
    cv2.imwrite(os.path.join(args.img_folder_dst,'results_VAE.jpg'),result_image_VAE*255)
    cv2.imwrite(os.path.join(args.img_folder_dst,'results_fANO.jpg'),result_image_fano*255)
    cv2.imwrite(os.path.join(args.img_folder_dst,'cams_ours.jpg'),cams_ours)

### 140 and 190


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--img_name",default="",type=str)
    parser.add_argument("--img_folder", default="", type=str)
    parser.add_argument("--img_folder_dst",default="",type=str)

    args=parser.parse_args()
    runDrawImages(args)