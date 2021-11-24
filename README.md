# Constrained Gradient-CAMs in VAEs for Unsupervised Anomaly Segmentation
Paper accepted at BMVC 2021

Code for the paper: "looking at the whole picture: constrained unsupervised anomaly segmentation", freely available at
https://arxiv.org/abs/2109.00482

![This is an image](https://github.com/cvblab/anomaly_localization_vae_gcams/blob/main/figures/brats19_results_qualitative.png)

## Getting Started

###### Minimum requirements

Software minimum requirements:
- torch==1.7.0
- nibabel
- numpy==1.18.4
- cv2==4.2.0


###### Download dataset

In this work, we benchmark the proposed method on unsueprvised anomaly segmentation using the popular Brats19' dataset of brain MRI images. You can find it in the following link:
https://drive.google.com/file/d/1NgHMcIcfVGcoAYWd0ABI6AEZCkpFpvJ8/view?usp=sharing
Download the MRI volumes and allocate then in ./data/ folder.

###### Preprocess data
MRI volumes are preprocessed to satisfy the unsupervied paradigm. The full process is described in the manuscript. We provide the following function to process the previously downloaded dataset, including train/val/test splits.

```
cd code
python adecuate_BRATS.py --dir_datasets ../data/MICCAI_BraTS_2019_Data_Training/ --dir_out ../data/BRATS_5slices/ --scan flair --nSlices 5
```

Preprocessing and further training functions work using the different MRI modalities in Brats. The variable nSlices indicates the number of sliced around the center of each MRI scan used.

## Training

You can train the proposed models for unsupervised anomaly localization as follows:

```
cd code
python main.py --dir_datasets ../data/BRATS_5slices/ --dir_out ../data/results/proposed/ --method proposed
```

Note that baselines (e.g. ae, vae, anovaegan, etc.) used in this paper are also trainable using the main.py file, by inspecting the variable '--method'.

## Contact
For further questions or details, please directly reach out to Julio Silva-Rodríguez
(jjsilva@upv.es)
