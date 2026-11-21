import math

import pytest
import torch

from mode_explorer.mechanism import conditioning_terms, filter_field, paired_measurement, signed_terms


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
