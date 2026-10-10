# Marine Debris Detection from Sentinel-2 Imagery

A team project at Blekinge Institute of Technology. We segment floating marine debris in Sentinel-2 satellite images from the MARIDA benchmark. We built **DeepSpectralUNet** and compared it with other models. The write-up is in [`marine_debris_report.pdf`](marine_debris_report.pdf).

## What we did
- Classified every pixel of 256 x 256 patches (11 spectral bands) into 10 classes: Marine Debris, Sargassum, Natural Organic Material, Ship, Clouds, Water, Foam, Waves, Cloud Shadows and Wakes. The splits are 694 train, 328 val and 359 test patches.
- Built DeepSpectralUNet: an encoder-decoder with a separate multi-scale branch for four spectral indices (FDI, NDVI, NDWI, FAI) that is fused into every decoder stage.
- Trained against the fact that debris is only 0.45% of labelled pixels, using inverse-frequency loss weights, debris-aware patch sampling, warmup + cosine learning-rate schedule and gradient clipping.
- Trained Random Forest, U-Net, Attention U-Net, DeepLabV3+ and ResAttUNet-CBAM as baselines.

## Results (test set)

| Model | Debris IoU | Debris F1 | Macro F1 | Accuracy |
|-------|-----------:|----------:|---------:|---------:|
| Random Forest | 0.6725 | 0.8042 | 0.5620 | 0.9235 |
| U-Net | 0.1900 | 0.3193 | 0.5113 | 0.8876 |
| Attention U-Net | 0.1996 | 0.3327 | 0.5074 | 0.8728 |
| DeepLabV3+ | 0.0671 | 0.1258 | 0.3631 | 0.7179 |
| ResAttUNet-CBAM | 0.4205 | 0.5921 | 0.7477 | 0.9548 |
| **DeepSpectralUNet** | **0.4264** | **0.5978** | 0.6190 | 0.9350 |

DeepSpectralUNet has the best debris IoU and F1 of the deep-learning models. The Random Forest has the highest debris IoU and F1 overall, and ResAttUNet-CBAM has the best macro F1 and accuracy.

## How it was built
Data loading, models, training and evaluation are all in `marine_debris.ipynb`.

## Tech stack
Python, PyTorch, segmentation-models-pytorch, timm, Albumentations, rasterio, scikit-learn, NumPy, Jupyter.

## Team
- **Sriya Chittaneni:** designed and implemented DeepSpectralUNet (multi-scale spectral branch, attention, residual encoder) and part of the training pipeline
- **Meghashyam Sai Dontha:** preprocessing, spectral features, class imbalance, augmentation, evaluation
- **Saideep Reddy Tamma:** baseline models, hyperparameter tuning, visualisations

## Data
MARIDA: Kikaki et al., *PLOS ONE* 17(1), e0262247, 2022.
