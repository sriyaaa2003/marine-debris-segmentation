"""MARIDA patches: class merging, spectral indices, normalisation and the debris-aware sampler.

Expected layout (Kikaki et al. 2022, https://zenodo.org/records/5151941):
    $MARIDA_ROOT/patches/<scene>/S2_*.tif, *_cl.tif (labels), *_conf.tif (annotator confidence)
    $MARIDA_ROOT/splits/{train,val,test}_X.txt
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

B02, B03, B04, B06, B08, B11 = 1, 2, 3, 5, 7, 9      # indices into the 11 bands B01 B02 B03 B04 B05 B06 B07 B08 B8A B11 B12
WL_RED, WL_RE2, WL_NIR, WL_SWIR1 = 665.0, 740.0, 842.0, 1614.0
NUM_CLASSES = 10
CLASS_NAMES = ["Marine Debris", "Sargassum", "Natural Organic Mat.", "Ship", "Clouds", "Water", "Foam", "Waves", "Cloud Shadows", "Wakes"]
INDEX_NAMES = ["FDI", "NDVI", "NDWI", "FAI"]

# MARIDA's 15 raw classes -> 10 study classes (0 = unlabelled, ignored as -1)
CLASS_MERGE = {0: -1, 1: 0, 2: 1, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 5, 9: 6, 10: 5, 11: 5, 12: 7, 13: 8, 14: 9, 15: 5}
CONF_WEIGHTS = {0: 0.0, 1: 0.5, 2: 0.75, 3: 1.0}
MERGE_LUT = np.array([CLASS_MERGE[k] for k in range(16)], dtype=np.int64)
CONF_LUT = np.array([CONF_WEIGHTS[k] for k in range(4)], dtype=np.float32)


def compute_spectral_indices(image: np.ndarray) -> np.ndarray:
    """FDI (Biermann 2020), NDVI, NDWI (Gao 1996) and FAI (Hu 2009) from raw reflectance (11, H, W) -> (4, H, W)."""
    eps = 1e-8
    nir, red, green = (image[i].astype(np.float32) for i in (B08, B04, B03))
    re2, swir1 = image[B06].astype(np.float32), image[B11].astype(np.float32)
    fdi = nir - (re2 + (swir1 - re2) * ((WL_NIR - WL_RE2) / (WL_SWIR1 - WL_RE2)))
    ndvi = (nir - red) / (nir + red + eps)
    ndwi = (green - nir) / (green + nir + eps)
    fai = nir - (red + (swir1 - red) * ((WL_NIR - WL_RED) / (WL_SWIR1 - WL_RED)))
    return np.stack([fdi, ndvi, ndwi, fai], axis=0)


def merge_labels(label_raw: np.ndarray, conf_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Merged label map (-1 = ignore) and per-pixel confidence weight; zero-confidence pixels are ignored."""
    label = MERGE_LUT[np.clip(label_raw, 0, 15)]
    conf = CONF_LUT[np.clip(conf_raw, 0, 3)]
    label[conf == 0.0] = -1
    return label, conf


def build_patch_lookup(patches_dir: Path) -> dict:
    lookup: dict = {}
    for tif in sorted(Path(patches_dir).rglob("*.tif")):
        name = tif.stem
        if name.endswith("_cl"):
            lookup.setdefault(name[:-3], {})["label"] = tif
        elif name.endswith("_conf"):
            lookup.setdefault(name[:-5], {})["conf"] = tif
        else:
            lookup.setdefault(name, {})["image"] = tif
    return {k: v for k, v in lookup.items() if {"image", "label", "conf"} <= set(v)}


def load_split(splits_dir: Path, split: str, lookup: dict) -> list[str]:
    path = Path(splits_dir) / f"{split}_X.txt"
    if not path.exists():
        return []
    return [f"S2_{line.strip()}" for line in path.open() if f"S2_{line.strip()}" in lookup]


def load_patch(paths: dict):
    import rasterio

    with rasterio.open(paths["image"]) as src:
        image = src.read().astype(np.float32)
    with rasterio.open(paths["label"]) as src:
        label_raw = src.read(1).astype(np.int64)
    with rasterio.open(paths["conf"]) as src:
        conf_raw = src.read(1).astype(np.int64)
    label, conf = merge_labels(label_raw, conf_raw)
    return image, label, conf


def stack_channels(image: np.ndarray) -> np.ndarray:
    return np.concatenate([image, compute_spectral_indices(image)], axis=0)


def normalise(full_image: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    return ((full_image - means[:, None, None]) / stds[:, None, None]).astype(np.float32)


def compute_train_stats(lookup: dict, train_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Welford mean and std of each of the 15 channels over labelled training pixels."""
    n_ch = 15
    count, mean, m2 = np.zeros(n_ch), np.zeros(n_ch), np.zeros(n_ch)
    for pid in train_ids:
        image, label, _ = load_patch(lookup[pid])
        valid = label != -1
        if not valid.any():
            continue
        full = stack_channels(image)
        for c in range(n_ch):
            vals = full[c][valid].astype(np.float64)
            count[c] += len(vals)
            delta = vals - mean[c]
            mean[c] += delta.sum() / count[c]
            m2[c] += (delta * (vals - mean[c])).sum()
    std = np.sqrt(m2 / np.maximum(count - 1, 1))
    return mean, np.where(std < 1e-8, 1.0, std)


def save_stats(path: Path, means, stds) -> None:
    Path(path).write_text(json.dumps({"means": list(map(float, means)), "stds": list(map(float, stds))}, indent=2))


def load_stats(path: Path):
    s = json.loads(Path(path).read_text())
    return np.array(s["means"], dtype=np.float32), np.array(s["stds"], dtype=np.float32)


def debris_sample_weight(label: np.ndarray, conf: np.ndarray) -> float:
    """Debris-aware patch weight: patches with few debris pixels are drawn 10 to 25 times more often than empty ones
    (35 times if they hold more than 20 debris pixels); high annotator confidence raises the weight."""
    mask = label == 0
    n = int(mask.sum())
    if n == 0:
        return 1.0
    high = conf[mask].mean() >= 0.75
    if n <= 5:
        return 15.0 if high else 10.0
    if n <= 20:
        return 25.0 if high else 18.0
    return 35.0
