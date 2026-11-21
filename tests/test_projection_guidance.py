import math

import pytest
import torch

from mode_explorer.projection_guidance import CHUNK, Guidance, perturb


def apply(output, **options):
    info = {}
    config = Guidance(**options)
    result = perturb(output, (1, output.shape[1] - 1), 1, 1, config, config.amplitude, [12347], info)
    return result, info


@pytest.mark.parametrize("mode", ["gaussian", "token_adaptive", "channel_aware"])
def test_projection_rms_rng_and_native_ownership(mode):
    source = torch.arange(2 * 12 * 8, dtype=torch.float32).reshape(2, 8, 12).transpose(1, 2) / 256
    original = source.clone()
    rng = torch.random.get_rng_state()
    output, info = apply(source, mode=mode, amplitude=.01)
    other, _ = apply(source, mode=mode, amplitude=.01)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert torch.equal(output, other)
    assert torch.equal(source, original)
    assert output.dtype == source.dtype and output.stride() == source.stride() and output.device == source.device
    assert output.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
    assert torch.equal(output[0], source[0])
    assert torch.equal(output[1, (0, -1)], source[1, (0, -1)])
    assert info["status"] == "applied"
    assert info["rows"][0]["generated_rms"] == pytest.approx(.01)
    assert info["rows"][0]["realized_rms"] == pytest.approx(.01, rel=1e-6)


def test_relative_normalization_and_native_rounding():
    source = torch.ones(2, 12, 8, dtype=torch.float16) * 2
    result, info = apply(source, units="relative_rms", amplitude=.001)
    row = info["rows"][0]
    assert result.dtype == source.dtype
    assert row["native_rms"] == 2. and row["generated_rms"] == pytest.approx(.002)
    assert row["realized_rms"] == pytest.approx(float((result[1, 1:-1].float() - 2).square().mean().sqrt()))
    assert 0 < row["changed_fraction"] < 1
    assert row["relative_realization_error"] > 0
    result, info = apply(source, amplitude=1e-8)
    assert result is None and info["reason"] == "zero_native_realization"


def test_zero_energy_outlier_nonfinite_and_native_caps():
    source = torch.zeros(2, 12, 8, dtype=torch.float16)
    assert apply(source, units="relative_rms")[1]["status"] == "unchanged"
    assert apply(source, mode="channel_aware")[1]["status"] == "applied"
    source.fill_(64.)
    native, trace = apply(source, amplitude=.015, max_absolute_rms=.014)
    assert native is None and trace["reason"] == "native_rounded_rms_limit"
    source[1, 2, 0] = torch.inf
    assert apply(source)[1]["reason"] == "nonfinite_projection_or_perturbation"
    source.zero_()
    source[1, 2, 0] = 1000.
    result, info = apply(source, mode="token_adaptive")
    assert result is not None and math.isfinite(info["rows"][0]["native_rms"])
    assert info["rows"][0]["generated_rms"] == pytest.approx(Guidance.amplitude)


def test_chunked_fields(monkeypatch):
    randn = torch.randn
    shapes = []

    def tracked(shape, **kwargs):
        shapes.append(shape)
        return randn(shape, **kwargs)

    monkeypatch.setattr(torch, "randn", tracked)
    source = torch.zeros(2, CHUNK + 11, 8, dtype=torch.float16)
    _, info = apply(source)
    assert shapes == [(CHUNK, 8), (9, 8)] * 2
    assert info["estimated_owned_peak_bytes"] > source.numel() * source.element_size()


def test_explicit_schedule():
    assert Guidance(evaluations=1).strength(1, .99) == 0
    assert Guidance(evaluations=2, schedule="linear_decay").strength(1, .95) == Guidance.amplitude * .5
    sigma_only = Guidance(evaluations=0, min_sigma=.8, max_sigma=1., schedule="linear_decay")
    assert sigma_only.strength(13, .8) == Guidance.amplitude * .5
    assert sigma_only.strength(0, .79) == 0
