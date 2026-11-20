import math

import pytest
import torch

from mode_explorer.response import (
    BASES, ResponseSettings, differences, direction_field, effective_tilt_scale, flow_clean,
    orthonormalize, output_projection, projected_operator, select_response, shifted_pair, velocity_for_clean,
)


def test_gaussian_tilt_and_covariance_nonunit_noise():
    sigma, scale = 0.96, 0.7
    alpha, variance = 1 - sigma, (sigma * scale) ** 2
    prior_cov = torch.tensor([[2., 0.6], [0.6, 1.]], dtype=torch.float64)
    cov = torch.linalg.inv(torch.linalg.inv(prior_cov) + alpha**2 / variance * torch.eye(2))
    x = torch.tensor([0.18, -0.28], dtype=torch.float64)
    tilt = torch.tensor([0.01, -0.002], dtype=torch.float64)

    def mean(z):
        return cov @ (alpha / variance * z)

    rho = 1 / effective_tilt_scale(sigma, scale)
    torch.testing.assert_close(mean(x + rho * tilt), mean(x) + cov @ tilt)
    basis = list(torch.eye(2))
    response = [(mean(x + 1e-3 * u) - mean(x - 1e-3 * u)).float() / 2e-3 for u in basis]
    report, _, _ = projected_operator(basis, response)
    torch.testing.assert_close(torch.tensor(report["matrix"], dtype=torch.float64) * rho, cov, atol=1e-6, rtol=1e-6)
    assert report["symmetry_error"] < 1e-7
    assert report["negative_spectral_mass"] == 0
    assert effective_tilt_scale(1., scale) is None
    assert effective_tilt_scale(0., scale) is None


def test_multimodal_posterior_covariance_includes_between_mode_term():
    alpha, sigma, within_var = 0.2, 0.8, 0.03
    means = torch.tensor([-2., 1.5], dtype=torch.float64)
    z = torch.tensor(0.55, dtype=torch.float64)
    var_z = alpha**2 * within_var + sigma**2
    var_post = within_var * sigma**2 / var_z
    rho = sigma**2 / alpha

    def posterior(z, tilt=0):
        loc = means + alpha * within_var / var_z * (z - alpha * means)
        p = torch.softmax(-0.5 * (z - alpha*means)**2 / var_z + tilt * loc, dim=0)
        mu = p @ loc
        return mu + var_post * tilt, var_post + p @ ((loc - mu)**2)

    mu, covariance = posterior(z)
    assert covariance > 20 * var_post
    torch.testing.assert_close(posterior(z, 0.05)[0], posterior(z + rho * 0.05)[0])
    assert mu.item() != 0
    error = []
    for h in (1e-1, 1e-2, 1e-3):
        deriv, _ = differences(posterior(z+h)[0], posterior(z-h)[0], mu, h, h)
        error.append(abs(float(rho * deriv - covariance)))
    assert error[2] < error[1] / 50 < error[0] / 2500


def test_non_symmetric_projection_and_output_basis():
    basis = list(torch.eye(3)[:2])
    responses = [torch.tensor([2., 0., 4.]), torch.tensor([3., -1., 4.])]
    metrics, right, eigen = projected_operator(basis, responses)
    assert metrics["matrix"] == [[2., 3.], [0., -1.]]
    assert metrics["symmetry_error"] > 1
    assert metrics["negative_spectral_mass"] > 0
    assert right.shape == eigen.shape == (2,)
    m = output_projection(responses[0] + responses[1], responses)
    assert m["explained_energy_fraction"] == pytest.approx(1, abs=1e-6)
    assert m["residual_energy"] == pytest.approx(0, abs=1e-4)
    assert output_projection(torch.zeros(3), responses)["explained_energy_fraction"] is None
    assert len(orthonormalize([basis[0], basis[0] * 2, torch.full((3,), math.nan), torch.zeros(3)])) == 1


@pytest.mark.parametrize("basis", BASES)
def test_bases_private_rng_and_unequal_grid(basis):
    value = torch.ones(1, 16, 1, 5, 17)
    rng = torch.random.get_rng_state().clone()
    a = direction_field(value, basis, 4, 9, 42, 0)
    b = direction_field(value, basis, 4, 9, 42, 1)
    assert a.shape == b.shape == value.shape
    assert torch.equal(a, direction_field(value, basis, 4, 9, 42, 0))
    assert not torch.equal(a, b)
    assert torch.equal(rng, torch.random.get_rng_state())
    basis = orthonormalize([a, b])
    torch.testing.assert_close(torch.stack(basis) @ torch.stack(basis).T, torch.eye(2), atol=2e-6, rtol=0)


def test_realized_displacement_unequal_difference_and_original_coordinates():
    x = torch.ones(32, dtype=torch.bfloat16)
    unit = torch.full((32,), 1 / math.sqrt(32))
    assert shifted_pair(x, unit, 1e-5, "rms", 0.25)[2] == "unresolved_native_displacement"
    pair, info, reason = shifted_pair(x, unit, 0.02, "rms", 0.25)
    assert reason is None and info["asymmetry"] > 0
    p, n, a, b = pair
    base = x.float()
    derivative, curvature = differences(p.float()**2, n.float()**2, base**2, a, b)
    torch.testing.assert_close(derivative, 2 * unit, atol=1e-5, rtol=0)
    torch.testing.assert_close(curvature, 2 * unit**2, atol=1e-5, rtol=0)
    goal = flow_clean(p, torch.full((32,), 2.0), 0.96)
    torch.testing.assert_close(flow_clean(x, velocity_for_clean(x, goal, 0.96), 0.96), goal)
    with pytest.raises(ValueError):
        velocity_for_clean(x, goal, 0.0)


def test_candidate_policy_is_deterministic_and_scorer_fails_closed():
    baseline = torch.zeros(4)
    candidates = [(torch.full((4,), x), {}) for x in (0.002, -0.002, 0.4)]
    selected, _ = select_response(candidates, baseline, ResponseSettings(), 0, {})
    assert selected is candidates[0] or selected[1] is candidates[0][1]
    assert candidates[-1][1]["reason"] == "clean_trust_bound"
    selected, _ = select_response([(torch.ones(4) * 0.002, {})], baseline,
                                  ResponseSettings(selection="external"), 0, {})
    assert selected is None
    stronger = [(torch.ones(4) * 0.002, {}), (torch.ones(4) * -0.002, {})]
    selected, score = select_response(stronger, baseline, ResponseSettings(
        selection="external", scorer=lambda candidate, _: float(candidate.mean())), 0, {})
    assert score == 0 and selected[1] is stronger[0][1]


def test_matched_clean_candidate_uses_actual_magnitude():
    from mode_explorer.response import clean_candidate

    baseline = torch.zeros(4)
    cfg = ResponseSettings(candidate_target_rms=0.01, candidate_gain=0.5)
    match = clean_candidate(baseline, torch.full((4,), 0.04), cfg, {})
    assert match[1]["gain"] == pytest.approx(0.25)
    torch.testing.assert_close(match[0], torch.full((4,), 0.01))
    too_weak = clean_candidate(baseline, torch.full((4,), 0.01), cfg, {})
    assert too_weak[1]["reason"] == "target_exceeds_available_response"
    assert select_response([match, too_weak], baseline, cfg, 0.0, {})[0][1] is match[1]
