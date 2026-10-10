# Marine debris segmentation on Sentinel-2 (MARIDA)

Per-pixel segmentation of floating marine debris in Sentinel-2 patches (256 x 256, 11 bands) into 10 classes on the real
[MARIDA](https://zenodo.org/records/5151941) benchmark (694 train / 328 val / 359 test patches). Debris is only **0.45%** of the
labelled pixels, so the work is mostly about imbalance.

**DeepSpectralUNet** (this repo's model): a residual U-Net with CBAM and attention-gated skips, plus a branch that sees *only* the four
spectral indices (FDI, NDVI, NDWI, FAI) at every scale and feeds each decoder stage. Training uses inverse-frequency loss weights,
debris-aware patch sampling (10 to 35 times the weight of an empty patch), warm-up and cosine learning rate, and gradient clipping.

## Test-set results (from the course report, not re-run for this cleanup)

| model | debris IoU | debris F1 | macro F1 | accuracy |
|---|---:|---:|---:|---:|
| Random Forest | **0.6725** | **0.8042** | 0.5620 | 0.9235 |
| U-Net | 0.1900 | 0.3193 | 0.5113 | 0.8876 |
| Attention U-Net | 0.1996 | 0.3327 | 0.5074 | 0.8728 |
| DeepLabV3+ | 0.0671 | 0.1258 | 0.3631 | 0.7179 |
| ResAttUNet-CBAM | 0.4205 | 0.5921 | **0.7477** | **0.9548** |
| DeepSpectralUNet | 0.4264 | 0.5978 | 0.6190 | 0.9350 |

DeepSpectralUNet has the best debris IoU among the deep models, by 0.006 over ResAttUNet-CBAM from one run each, which is within what a
different seed could change. **A pixel-wise Random Forest beats every network on debris**, and ResAttUNet-CBAM is better overall (macro F1,
accuracy). With so few debris pixels the honest conclusion is that the spectral branch is a reasonable idea that this dataset is too
small to confirm. Details are in [`marine_debris_report.pdf`](marine_debris_report.pdf).

## Code

```
src/marinedebris/data.py    class merge, spectral indices, normalisation, debris-aware sampler weights, MARIDA loaders
src/marinedebris/model.py   DeepSpectralUNet and its blocks
src/marinedebris/train.py   weighted loss, metrics (debris F1 and IoU, macro F1), training loop
notebooks/marine_debris_full.ipynb   the original notebook with all baselines and figures (outputs kept as evidence)
tests/                      11 tests: index formulas, label merging, sampler weights, model shapes and gradients, loss ignoring, metrics
```

```bash
pip install -e .[dev] && pytest
# training: download MARIDA, set MARIDA_ROOT, then use data.build_patch_lookup / load_split with train.fit (GPU recommended)
```

The numbers above come from the notebook run on a GPU. The packaged code is a refactor of it, tested piece by piece but not retrained here.

## Team and reference

Blekinge Institute of Technology. Sriya Chittaneni: DeepSpectralUNet (spectral branch, attention, residual encoder) and part of the
training pipeline. Meghashyam Sai Dontha: preprocessing, indices, imbalance handling, augmentation, evaluation. Saideep Reddy Tamma:
baselines, tuning, figures. Data: Kikaki et al., "MARIDA", *PLOS ONE* 17(1), 2022.
