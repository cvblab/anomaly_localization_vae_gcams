# Constrained Gradient-CAMs in VAEs for Unsupervised Anomaly Segmentation
Paper accepted at BMVC 2021

Code for the paper: "looking at the whole picture: constrained unsupervised anomaly segmentation", freely available at
https://arxiv.org/abs/2109.00482

![This is an image](https://github.com/cvblab/anomaly_localization_vae_gcams/blob/main/figures/brats19_results_qualitative.png)

## Getting Started

###### Minimum requirements

###### Download dataset

###### Preprocess data

## Training

You can train the proposed models for unsupervised anomaly localization as follows:

```
cd code
python main.py --dir_datasets ../data/BRATS_5slices/ --dir_out ../data/results/proposed/ --method proposed
```

Please, note that baselines (e.g. ae, vae, anovaegan, etc.) used in this paper are also traineble using the main.py file, by inspecting the variable '--method'.

