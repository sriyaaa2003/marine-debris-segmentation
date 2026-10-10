import numpy as np
import pytest
import torch

from marinedebris.data import (B03, B04, B06, B08, B11, CLASS_MERGE, NUM_CLASSES, compute_spectral_indices, debris_sample_weight,
                               merge_labels, normalise)
from marinedebris.model import CBAM, DeepSpectralUNet
from marinedebris.train import InvFreqWCE, compute_metrics, warmup_cosine


def flat_image(**bands):
    img = np.full((11, 4, 4), 0.1, dtype=np.float32)
    for name, value in bands.items():
        img[{"green": B03, "red": B04, "re2": B06, "nir": B08, "swir1": B11}[name]] = value
    return img


def test_indices_match_their_definitions():
    idx = compute_spectral_indices(flat_image(green=0.3, red=0.2, re2=0.15, nir=0.5, swir1=0.1))
    ndvi, ndwi = (0.5 - 0.2) / (0.5 + 0.2), (0.3 - 0.5) / (0.3 + 0.5)
    fdi = 0.5 - (0.15 + (0.1 - 0.15) * ((842 - 740) / (1614 - 740)))
    fai = 0.5 - (0.2 + (0.1 - 0.2) * ((842 - 665) / (1614 - 665)))
    assert idx.shape == (4, 4, 4)
    np.testing.assert_allclose(idx[:, 0, 0], [fdi, ndvi, ndwi, fai], rtol=1e-5)


def test_labels_merge_and_zero_confidence_is_ignored():
    raw = np.array([[1, 2, 3], [7, 15, 0]])
    conf = np.array([[3, 3, 2], [1, 0, 3]])
    label, w = merge_labels(raw, conf)
    assert label.tolist() == [[0, 1, 1], [5, -1, -1]]          # debris, sargassum x2, water; zero confidence; unlabelled
    assert w.tolist() == [[1.0, 1.0, 0.75], [0.5, 0.0, 1.0]]
    assert set(CLASS_MERGE.values()) - {-1} == set(range(NUM_CLASSES))


def test_debris_patches_are_sampled_more_often():
    conf = np.ones((8, 8), dtype=np.float32)
    none = np.full((8, 8), 5)
    few = none.copy(); few[0, :3] = 0
    many = none.copy(); many[:4, :] = 0
    assert debris_sample_weight(none, conf) == 1.0
    assert debris_sample_weight(few, conf) == 15.0
    assert debris_sample_weight(many, conf) == 35.0


def test_normalise_uses_channel_statistics():
    x = np.stack([np.full((2, 2), 3.0), np.full((2, 2), 10.0)]).astype(np.float32)
    out = normalise(x, np.array([1.0, 10.0], dtype=np.float32), np.array([2.0, 5.0], dtype=np.float32))
    assert out[0, 0, 0] == pytest.approx(1.0) and out[1, 0, 0] == pytest.approx(0.0)


@pytest.mark.parametrize("size", [32, 40])
def test_model_output_shape_and_gradients(size):
    torch.manual_seed(0)
    model = DeepSpectralUNet(features=(8, 16, 32, 64))
    x = torch.randn(2, 15, size, size)
    y = model(x)
    assert y.shape == (2, NUM_CLASSES, size, size) and torch.isfinite(y).all()
    y.mean().backward()
    assert model.spectral_branch.scales[0][0].weight.grad.abs().sum() > 0       # the index branch really feeds the output


def test_spectral_branch_reads_only_the_index_channels():
    torch.manual_seed(0)
    model = DeepSpectralUNet(features=(8, 16, 32, 64)).eval()
    x = torch.randn(1, 15, 32, 32)
    a = model.spectral_branch(x[:, 11:15])
    x2 = x.clone(); x2[:, :11] += 5.0
    b = model.spectral_branch(x2[:, 11:15])
    assert all(torch.equal(p, q) for p, q in zip(a, b))


def test_cbam_keeps_shape():
    assert CBAM(32)(torch.randn(2, 32, 8, 8)).shape == (2, 32, 8, 8)


def test_weighted_loss_ignores_unlabelled_pixels_and_upweights_debris():
    counts = {i: 100000 for i in range(NUM_CLASSES)}; counts[0] = 450
    crit = InvFreqWCE(counts)
    assert crit.ce.weight[0] > 5 * crit.ce.weight[1]
    logits = torch.randn(1, NUM_CLASSES, 4, 4)
    target = torch.full((1, 4, 4), -1)
    target[0, 0, 0] = 3
    one = crit(logits, target)
    logits2 = logits.clone(); logits2[0, :, 1:, :] += 9.0        # changing ignored pixels must not change the loss
    assert crit(logits2, target).item() == pytest.approx(one.item())


def test_metrics_report_debris_iou_from_f1():
    t = np.array([0, 0, 0, 0, 1, 1, 2, 2])
    p = np.array([0, 0, 0, 1, 1, 1, 2, 0])
    m = compute_metrics(p, t)
    assert m["debris_iou"] == pytest.approx(3 / 5)               # 3 of 5 debris-union pixels overlap
    assert compute_metrics(np.array([0]), np.array([-1]))["macro_f1"] == 0.0


def test_warmup_then_cosine_schedule():
    assert warmup_cosine(0, 50) == pytest.approx(0.2) and warmup_cosine(4, 50) == pytest.approx(1.0)
    assert warmup_cosine(49, 50) < 0.01
