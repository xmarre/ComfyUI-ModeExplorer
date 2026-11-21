from dataclasses import fields
import json
import math
from pathlib import Path

import pytest
import torch

from mode_explorer.mechanism import MechanisticSettings, conditioning_terms, filter_field, paired_measurement, signed_terms
from mode_explorer.mechanism_runtime import estimate_capture_workspace


@pytest.mark.parametrize("h,w", [(9, 15), (8, 16)])
def test_parseval_bands_and_cross_term_on_rectangular_fields(h, w):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    low = torch.cos(2 * torch.pi * x / w)
    high = torch.cos(2 * torch.pi * 3 * y / h)
    a = (2 + low + high)[None]
    b = (2 + 2 * low - high)[None]
    m = paired_measurement(a, b)
    bands = m["bands"]
    assert bands["dc"]["baseline"] == pytest.approx(4 * h * w, rel=2e-6)
    assert bands["low"]["baseline"] == pytest.approx(h * w / 2, rel=2e-6)
    assert bands["high"]["baseline"] == pytest.approx(h * w / 2, rel=2e-6)
    for name, field in (("baseline", a), ("edited", b), ("displacement", b - a)):
        assert sum(v[name] for v in bands.values()) == pytest.approx(float(field.square().sum()), rel=2e-6)
    assert bands["high"]["net_change"] == pytest.approx(0, abs=8e-5)
    assert bands["high"]["displacement"] == pytest.approx(2 * h * w, rel=2e-6)
    assert bands["high"]["cross_term"] == pytest.approx(-2 * h * w, rel=2e-6)
    for v in bands.values():
        assert v["net_change"] == pytest.approx(v["edited"] - v["baseline"], abs=8e-5)
    assert sum(map(sum, m["spatial_tile_energy"])) == pytest.approx(float((b - a).square().sum()))


def test_channel_chunks_and_complementary_windows():
    rng = torch.Generator().manual_seed(8)
    a, b = torch.randn(2, 11, 13, 18, generator=rng)
    first, last = paired_measurement(a, b, 2), paired_measurement(a, b, 11)
    for band in first["bands"]:
        assert first["bands"][band] == pytest.approx(last["bands"][band], rel=1e-5, abs=1e-7)
    for smooth in (0., .15):
        recovered = sum(filter_field(a, band, .12, .3, smooth) for band in ("low", "mid", "high"))
        torch.testing.assert_close(a, recovered, atol=1e-6, rtol=1e-5)


def test_signed_and_factorial_exact_quadratic_and_linear_responses():
    base = torch.full((2, 5, 7), 3.)
    linear = torch.full_like(base, 2.)
    quadratic = torch.full_like(base, .25)
    zero = torch.zeros_like(base)
    plus, minus = base + linear + quadratic, base - linear + quadratic
    odd, even = signed_terms(base, plus, minus)
    assert paired_measurement(linear, odd)["displacement_l2"] == 0
    assert paired_measurement(quadratic, even)["displacement_l2"] == 0
    _, linear_even = signed_terms(base, base + linear, base - linear)
    assert paired_measurement(zero, linear_even)["displacement_l2"] == 0

    c0, c1 = conditioning_terms(base, base + linear, plus, plus + linear)
    assert paired_measurement(c0, c1)["displacement_l2"] == 0
    c0, c1 = conditioning_terms(base, base + linear, plus, plus + linear + quadratic)
    assert paired_measurement(c0, c1)["displacement_l2"] == pytest.approx(.25 * math.sqrt(base.numel()))


def test_patch_and_latent_measure_same_bands_and_resolve_diagonal_support():
    # 32 latent pixels or 16 patches per axis. The scene's diagonal
    # 6/32-frequency modes are above 1/4 radially and inside the common square.
    t = torch.arange(32, dtype=torch.float64)
    y, x = torch.meshgrid(t, t, indexing="ij")
    shared = (torch.cos(2 * torch.pi * 2 * x / 32) +
              torch.cos(2 * torch.pi * 3 * x / 32) +
              torch.cos(2 * torch.pi * 6 * (x + y) / 32))[None].float()
    sub_patch = torch.cos(2 * torch.pi * 10 * x / 32)[None].float()
    pixel = paired_measurement(shared, shared + sub_patch, low=.0625, high=.125, patch_size=2)
    samples = shared[:, ::2, ::2]
    patch = paired_measurement(samples, samples, low=.0625, high=.125, sample_stride=2, patch_size=2)
    for band in ("low", "mid", "high"):
        assert pixel["bands"][band]["baseline_fraction"] == pytest.approx(1 / 3)
        assert patch["bands"][band]["baseline_fraction"] == pytest.approx(1 / 3)
    assert pixel["bands"]["sub_patch"]["displacement_fraction"] == pytest.approx(1)
    assert patch["bands"]["sub_patch"]["frequency_bins"] == 0
    assert sum(b["frequency_bins"] for b in pixel["bands"].values()) == 32**2
    assert sum(b["frequency_bins"] for b in patch["bands"].values()) == 16**2
    torch.testing.assert_close(
        filter_field(shared + sub_patch, "sub_patch", .0625, .125, patch_size=2), sub_patch, atol=1e-6, rtol=1e-5,
    )
    for transition in (0., .05):
        recovered = sum(filter_field(shared + sub_patch, band, .0625, .125, transition, patch_size=2)
                        for band in ("low", "mid", "high", "sub_patch"))
        torch.testing.assert_close(recovered, shared + sub_patch, atol=1e-6, rtol=1e-5)


def test_published_profiles_pass_production_capture_preflight():
    path = Path(__file__).resolve().parents[1] / "tools" / "mechanistic_profiles.json"
    profiles = json.loads(path.read_text())
    spec_fields = {field.name for field in fields(MechanisticSettings)} - {"mode", "alternate"}
    for name, profile in profiles["profiles"].items():
        options = {**profiles["common"], **profile["settings"]}
        spec = MechanisticSettings(mode=options["experiment"], alternate=object(),
                                   **{key: options[key] for key in spec_fields & options.keys()})
        # 1632 × 2464 planner, 8× latent compression, 2× latent patches.
        # Budget the instrumented baseline even when it could be reused.
        captures, reserve = estimate_capture_workspace(
            spec, feature_block=options["feature_block"], grid=(308 // 2, 204 // 2),
            latent_shape=(1, 16, 1, 308, 204), attention_channels=6144, patch_size=2,
            baseline_reused=False,
        )
        assert captures + reserve <= spec.capture_mib * 1024**2, name
        if name == "conditioning_fixed":
            assert spec.mode == "conditioning_fixed"
            assert spec.core_arms == 4
