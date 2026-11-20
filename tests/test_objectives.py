import pytest
import torch
import torch.nn.functional as F

from mode_explorer.objectives import (
    RIDGE, cell_areas, feature_shift, initial_direction, lift, noise_candidate, objective, pool,
    reference_noise, seed_key, select_candidate, token_pool,
)
from mode_explorer.runtime import (
    BLOCK, FRAME, Frame, Run, Settings, attention_feature, equal_condition_pairs,
    sigma_eligible, token_slice,
)


@pytest.fixture(autouse=True)
def small_torch_threads():
    torch.set_num_threads(1)


@pytest.mark.parametrize("style", ["feature_cosine", "feature_volume"])
def test_analytic_descent_matches_autograd(style):
    x = torch.randn(2, 5, 11, generator=torch.Generator().manual_seed(8), dtype=torch.float64)
    x.requires_grad_()
    group = torch.tensor([0, 0, 1, 1, 1])
    pairs = (group[:, None] == group[None, :]) & ~torch.eye(5, dtype=torch.bool)
    bw = 0.7
    unit = F.normalize(x, dim=-1)
    dots = unit @ unit.transpose(-1, -2)
    if style == "feature_cosine":
        expected = (dots[:, pairs] / 0.5).logsumexp(dim=-1) - pairs.sum().log()
    else:
        kernel = torch.exp((dots - 1) / bw**2).masked_fill(~pairs, 0)
        expected = -torch.linalg.slogdet(kernel + (1 + RIDGE) * torch.eye(5, dtype=x.dtype)).logabsdet
    direction, _, loss = objective(x.detach(), pairs, style, bw)
    torch.testing.assert_close(loss, expected, atol=1e-7, rtol=1e-7)
    gradient, = torch.autograd.grad(expected.sum(), x)
    torch.testing.assert_close(direction, -gradient, atol=1e-11, rtol=1e-10)
    moved = objective(x.detach() + direction, pairs, style, bw)[2]
    assert torch.all(moved < loss)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("style", ["feature_cosine", "feature_volume"])
def test_degenerate_features_and_mixed_precision(dtype, style):
    identical = torch.ones(2, 4, 3, 5, dtype=dtype)
    candidates = torch.cat([identical, identical[:1] + 0.001, torch.zeros_like(identical[:1])])
    pairs = ~torch.eye(len(candidates), dtype=torch.bool)
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        direction, nearest, loss = objective(candidates.flatten(1), pairs, style)
        delta = feature_shift(candidates.float(), pairs, style, 0.5, 0.5, True, 0.7)[0]
        for tensor in (direction, nearest, loss, delta, token_pool(identical.flatten(2).transpose(1, 2), 3, 5)):
            assert torch.isfinite(tensor).all()
        assert torch.count_nonzero(direction[-1]) == 0
        no_direction = objective(identical.flatten(1), torch.zeros(2, 2, dtype=torch.bool), style)[0]
        assert torch.count_nonzero(no_direction) == 0


@pytest.mark.parametrize("height,width", [(1, 1), (7, 11), (8, 16), (2, 12)])
def test_pooling_adjoint_handles_uneven_edges(height, width):
    x = torch.randn(2, 6, height, width)
    z = torch.randn_like(pool(x))
    torch.testing.assert_close((pool(x) * z).sum(), (x * lift(z, height, width, adjoint=True)).sum())


def test_conditional_grouping_uses_text_mask_and_reference():
    context = torch.tensor([[7], [7], [8], [2], [2], [2.]])
    mask = torch.ones(6, 1)
    refs = [torch.tensor([[0], [0], [0], [1], [1], [2.]])]
    want = torch.tensor([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=torch.bool)
    torch.testing.assert_close(equal_condition_pairs(3, 0, context, mask, refs), want)
    torch.testing.assert_close(equal_condition_pairs(3, 3, context, mask, refs), want)
    mask[0] = 0
    assert not equal_condition_pairs(3, 0, context, mask, refs).any()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_hook_preserves_cfg_text_ref_and_input(dtype):
    settings = Settings(adaptive=False, strength=0.8)
    run = Run(settings, 0, 1.0, 1.0, 2, (2, 16, 5, 6), True)
    frame = Frame((3, 3), 2, 2, "sample", run, 0.94, pairs=~torch.eye(2, dtype=torch.bool))
    options = {
        FRAME: frame, "block_type": "single", "block_index": BLOCK,
        "img_slice": [4, 18], "reference_image_num_tokens": [3, 2],
    }
    original = torch.randn(4, 18, 16, generator=torch.Generator().manual_seed(62)).to(dtype)
    retained = original.clone()
    changed = attention_feature(original, options)
    assert torch.equal(original, retained)
    assert torch.equal(changed[:2], retained[:2])
    assert torch.equal(changed[:, :4], retained[:, :4])
    assert torch.equal(changed[:, 13:], retained[:, 13:])
    assert not torch.equal(changed[2:, 4:13], retained[2:, 4:13])
    assert token_slice(options, 18, 9) == (4, 13)
    with pytest.raises(RuntimeError, match="token slice"):
        token_slice(options, 18, 8)


def test_rng_reference_and_priors():
    before = torch.random.get_rng_state()
    template = torch.empty(3, 16, 6, 9)
    original = reference_noise(template, 5).clone()
    assert torch.equal(before, torch.random.get_rng_state())
    assert torch.equal(reference_noise(template, 5), original)
    assert not torch.equal(reference_noise(template, 6), original)
    assert seed_key(1, "x") != seed_key(1 + (1 << 32), "x")
    candidate = noise_candidate(original, reference_noise(template, 6), 0.1)
    assert not torch.equal(original, candidate)
    torch.testing.assert_close(original.square().sum((1, 2, 3)), candidate.square().sum((1, 2, 3)))
    assert ((candidate - original).square().mean((1, 2, 3)).sqrt() <= 0.10001).all()


def test_initialization_accepts_volume_gain_and_rejects_bad_predictions():
    base = reference_noise(torch.empty(1, 16, 12, 12), 0).clone()
    # A toy flow whose first clean estimate is identity, and with two nearby optima anchors.
    refs = (base + 0.28 * reference_noise(base, 1), base + 0.27 * reference_noise(base, 2))
    direction, volume, bandwidth = initial_direction(base, refs)
    candidate = noise_candidate(base, direction, 0.05)
    changed, accepted, gain, _ = select_candidate(base, candidate, base, candidate, refs, volume, bandwidth, 0.6)
    assert accepted.item() and gain.item() > 0
    assert torch.equal(changed, candidate)
    for bad in (base, candidate * 3, candidate * torch.nan):
        noise, accepted, _, _ = select_candidate(base, candidate, base, bad, refs, volume, bandwidth, 0.6)
        assert not accepted.item()
        assert torch.equal(noise, base)


def test_settings_and_sigma_use_absolute_values():
    assert sigma_eligible(0.91, 0.9)
    assert not sigma_eligible(0.89, 0.9)
    assert not sigma_eligible(0.0, 0.5)
    assert not sigma_eligible(float("nan"), 0.5)
    with pytest.raises(ValueError):
        Settings(reference_budget=5)
    with pytest.raises(ValueError):
        Settings(fidelity=float("nan"))


def test_centering_preserves_coarse_mean_and_adaptive_gate():
    gen = torch.Generator().manual_seed(89)
    original = torch.randn(2, 3, 4, 4, generator=gen)
    mask = ~torch.eye(2, dtype=torch.bool)
    shift = feature_shift(original, mask, "feature_cosine", 0.8, 0.4, False, 0.99)[0]
    torch.testing.assert_close(shift.mean(dim=(2, 3)), torch.zeros(2, 3), atol=1e-7, rtol=0)
    assert torch.count_nonzero(shift) > 0
    adaptive = feature_shift(original, mask, "feature_cosine", 0.8, 0.4, True, 0.99)[0]
    assert not torch.count_nonzero(adaptive)


@pytest.mark.parametrize("style,key,low,high", [
    ("feature_cosine", "temperature", 0.15, 1.7),
    ("feature_volume", "bandwidth", 0.25, 1.2),
])
def test_collective_parameters_change_updates_and_pair_stays_baseline(style, key, low, high):
    x = torch.randn(4, 8, 3, 4, generator=torch.Generator().manual_seed(88))
    x[1] = x[0] + 0.4 * x[1]
    x[2] = 0.4 * x[0] - x[2]
    apply = lambda n, value: feature_shift(x[:n], ~torch.eye(n, dtype=torch.bool), style,
                                           0.7, 0.5, False, 0.7, **{key: value})[0]
    torch.testing.assert_close(apply(2, low), apply(2, high), rtol=0, atol=0)
    assert not torch.allclose(apply(4, low), apply(4, high))
    x[:] = x[0].clone()
    assert not torch.count_nonzero(apply(4, low).abs() > 1e-5)


@pytest.mark.parametrize("grid", [2, 4, 6, 8])
def test_verified_shift_preserves_physical_mean_and_improves_measured_loss(grid):
    generator = torch.Generator().manual_seed(763)
    height, width = 9, 17
    image = torch.randn(3, 8, height, width, generator=generator)
    image[1] = image[0] + 0.3 * image[1]
    image[2] = 0.4 * image[0] + image[2]
    x = pool(image, grid)
    areas = cell_areas(height, width, *x.shape[-2:], grid, image.device)
    pairs = ~torch.eye(3, dtype=torch.bool)
    shift, _, loss = feature_shift(x, pairs, "feature_volume", 0.8, 0.4, False, 0.7,
                                   live=1, areas=areas)
    assert torch.count_nonzero(shift[0])
    assert not torch.count_nonzero(shift[1:])
    expanded = lift(shift, height, width, grid=grid)
    torch.testing.assert_close(expanded.sum(dim=(-1, -2)), torch.zeros(3, 8), atol=1e-5, rtol=0)
    assert expanded[0].square().sum() <= 0.08**2 * (x[0].square() * areas).sum() + 1e-6
    y = x + shift
    y = y - (y * areas).sum(dim=(-1, -2), keepdim=True) / areas.sum()
    assert objective(y.flatten(1), pairs, "feature_volume")[2] < loss
    z = torch.randn_like(x)
    torch.testing.assert_close((x * z).sum(), (image * lift(z, height, width, adjoint=True, grid=grid)).sum())
