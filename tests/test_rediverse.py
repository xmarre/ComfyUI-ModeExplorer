"""Numerical contracts of the ReDiverse covariance, schedule, coefficient and archive code."""

import json
import math

import pytest
import torch

from mode_explorer import covariance
from mode_explorer.rediverse import (
    ARCHIVE_KEY, FirstEvaluations, coefficient_seed, coefficients, image_mu, load_adapter, load_directions,
    packed_noise, reference_sigmas, save_directions, tensor_fingerprint,
)


def explicit(X):
    Xc = X.double() - X.double().mean(0)
    cov = Xc.T @ Xc / X.shape[0]
    return torch.linalg.eigvalsh(cov).flip(0).clamp_min(0)


def cohort(n=12, d=40, seed=0, scales=None):
    g = torch.Generator().manual_seed(seed)
    basis, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
    scales = torch.linspace(3, .1, d, dtype=torch.float64) if scales is None else scales
    return (torch.randn(n, d, generator=g, dtype=torch.float64) * scales) @ basis.T + 5.0


def test_gram_spectrum_matches_explicit_covariance_and_chunking():
    X = cohort()
    values, _ = covariance.eigenpairs(covariance.centred_gram(X, chunk=7))
    reference = explicit(X)
    assert torch.allclose(values[:11].clamp_min(0) / 12, reference[:11], rtol=1e-9, atol=1e-12)
    for chunk in (1, 3, 40, 1 << 20):
        assert torch.allclose(covariance.centred_gram(X, chunk=chunk), covariance.centred_gram(X), atol=1e-9)
    stats = covariance.spectrum_stats(values, 12)
    lam = reference
    assert stats["tv"] == pytest.approx(float(lam.sum()), rel=1e-9)
    assert stats["pr"] == pytest.approx(float(lam.sum() ** 2 / lam.square().sum()), rel=1e-9)
    assert stats["conc"] == pytest.approx(stats["tv"] / stats["pr"], rel=1e-12)
    assert stats["pr_cap"] == 11 and stats["normalization"] == "N"
    p = lam[lam > 0] / lam.sum()
    assert stats["entropy_rank"] == pytest.approx(math.exp(float(-(p * p.log()).sum())), rel=1e-9)
    assert stats["cumulative_explained"][-1] == pytest.approx(1.0)


def test_known_spectra_rank_one_isotropic_and_zero_variance():
    n, d = 50, 6
    g = torch.Generator().manual_seed(3)
    direction = torch.randn(d, generator=g, dtype=torch.float64)
    rank_one = torch.randn(n, 1, generator=g, dtype=torch.float64) * direction
    stats = covariance.spectrum_stats(covariance.eigenpairs(covariance.centred_gram(rank_one))[0], n)
    assert stats["pr"] == pytest.approx(1.0, abs=1e-9) and stats["entropy_rank"] == pytest.approx(1.0, abs=1e-6)
    # Exactly isotropic centred cohort: 2d rows +-sqrt(d) e_i scaled to unit covariance.
    rows = torch.cat((torch.eye(d), -torch.eye(d))).double() * math.sqrt(d)
    stats = covariance.spectrum_stats(covariance.eigenpairs(covariance.centred_gram(rows))[0], 2 * d)
    # Covariance is the identity: TV = PR = d and concentration 1.
    assert stats["tv"] == pytest.approx(d, rel=1e-9) and stats["pr"] == pytest.approx(d, rel=1e-9)
    assert stats["conc"] == pytest.approx(1.0, rel=1e-9) and stats["entropy_rank"] == pytest.approx(d, rel=1e-9)
    same = torch.ones(5, d)
    stats = covariance.spectrum_stats(covariance.eigenpairs(covariance.centred_gram(same))[0], 5)
    assert stats["degenerate"] and stats["tv"] == 0 and stats["conc"] is None


def test_rank_is_limited_by_samples():
    X = cohort(n=5, d=40)
    stats = covariance.spectrum_stats(covariance.eigenpairs(covariance.centred_gram(X))[0], 5)
    assert stats["pr"] <= 4 + 1e-9 and stats["pr_cap"] == 4
    assert stats["top_eigenvalues"][4] == pytest.approx(0, abs=1e-9)


def test_principal_basis_reproduces_upstream_formula():
    X = cohort(n=10, d=30).float()
    values, vectors = covariance.eigenpairs(covariance.centred_gram(X))
    U, s, valid = covariance.principal_basis(X, vectors, values, 4, chunk=7, dtype=torch.float32)
    assert valid == 4
    # Upstream rediverse.directions.directions_from_samples on the same cohort.
    Xc = X - X.mean(0, keepdim=True)
    evals, evecs = torch.linalg.eigh((Xc @ Xc.T).double())
    evals, evecs = evals.flip(0), evecs.flip(1)
    for k in range(4):
        u = evecs[:, k].float() @ Xc
        u = u / u.norm()
        assert abs(float(u @ U[k])) == pytest.approx(1.0, abs=1e-5)
        upstream = float((evals[k] ** .5 * evecs[:, k]).std(unbiased=False))
        assert float(s[k]) == pytest.approx(upstream, rel=1e-5)
        assert float(s[k]) == pytest.approx(math.sqrt(float(values[k]) / 10), rel=1e-5)
        # The spread is the teacher's standard deviation along u_k.
        assert float((Xc @ U[k]).std(unbiased=False)) == pytest.approx(float(s[k]), rel=1e-4)
    assert torch.allclose(U @ U.T, torch.eye(4), atol=1e-5)


def test_principal_basis_marks_degenerate_directions():
    X = torch.cat((torch.zeros(1, 8), torch.ones(1, 8), torch.zeros(1, 8)))
    values, vectors = covariance.eigenpairs(covariance.centred_gram(X))
    U, s, valid = covariance.principal_basis(X, vectors, values, 3)
    assert valid == 1 and (s[1:] == 0).all() and U[1:].float().abs().sum() == 0


def test_subspace_comparison_identifies_deficit_and_excess():
    g = torch.Generator().manual_seed(5)
    d, n = 20, 4000
    teacher_scales = torch.tensor([3., 2., 1.] + [.01] * (d - 3), dtype=torch.float64)
    T = torch.randn(n, d, generator=g, dtype=torch.float64) * teacher_scales
    values, vectors = covariance.eigenpairs(covariance.centred_gram(T))
    U, s, _ = covariance.principal_basis(T, vectors, values, 3, dtype=torch.float64)
    # Student: suppressed along the first axis, inflated along the third.
    S = torch.randn(n, d, generator=g, dtype=torch.float64) * torch.tensor([.5, 2., 2.] + [.01] * (d - 3), dtype=torch.float64)
    own_values, own_vectors = covariance.eigenpairs(covariance.centred_gram(S))
    result = covariance.subspace_comparison(s, covariance.project(S, U), own_values, own_vectors, k=3)
    retained = result["retained_variance_ratio"]
    assert retained[0] == pytest.approx(.25 / 9, rel=.15)
    assert retained[2] == pytest.approx(4.0, rel=.15)
    assert result["deficit_eigenvalues"][0] == pytest.approx(9 - .25, rel=.1)
    assert result["excess_eigenvalues"] and max(result["excess_eigenvalues"]) == pytest.approx(3.0, rel=.15)
    assert 0.95 < result["subspace_alignment"] <= 1.0 + 1e-9
    clipped = covariance.positive_part(torch.diag(torch.tensor([1., -2., 0.5])))
    assert torch.allclose(clipped, torch.diag(torch.tensor([1., 0., .5])))


def test_reference_schedules_match_upstream_constants():
    assert image_mu(4096) == pytest.approx(0.90625, abs=1e-9)
    teacher = reference_sigmas(52, 4096)
    student = reference_sigmas(8, mu=1.15)
    # krea2/config.py: forward 5 reads 0.96741..0.96743 and lands at 0.95883; Turbo sigma1 0.95672.
    assert teacher[4] == pytest.approx(0.96743, abs=5e-5)
    assert teacher[5] == pytest.approx(0.95883, abs=5e-5)
    assert student[1] == pytest.approx(0.95672, abs=5e-5)
    assert teacher[0] == 1.0 and teacher[-1] == 0.0 and len(teacher) == 53


def test_first_evaluation_contract():
    plan = FirstEvaluations()
    assert plan.pending(1.0)
    assert plan.admit(1.0) == 0 and plan.admit(1.0) == 0  # CFG branches of one evaluation
    assert not plan.pending(0.9) and plan.admit(0.9) is None
    assert plan.closed and plan.admit(1.0) is None and plan.exit_sigma == 0.9
    two = FirstEvaluations(count=2)
    assert [two.admit(v) for v in (.99, .9, .9, .8)] == [0, 1, 1, None]
    interval = FirstEvaluations("first_interval", boundary=.9)
    assert interval.admit(1.0) == 0 and interval.admit(.95) == 0 and interval.admit(.9) is None and interval.closed
    for bad in (dict(count=0), dict(count=17), dict(start=16), dict(start=-1), dict(schedule="first_interval", start=1)):
        with pytest.raises(ValueError):
            FirstEvaluations(**bad)


SCHEDULE = [0.999968, 0.979327, 0.956724, 0.931906, 0.904531, 0.874183, 0.840349, 0.802392,
            0.759511, 0.710679, 0.654567, 0.589414, 0.512844, 0.421569, 0.310901, 0.173927]


@pytest.mark.parametrize("start,count", [(0, 1), (0, 2), (0, 4), (0, 8), (0, 16), (1, 4), (2, 4), (4, 4), (6, 4), (15, 1)])
def test_window_selects_distinct_sigma_indices(start, count):
    plan = FirstEvaluations(count=count, start=start)
    positions = []
    for sigma in SCHEDULE:
        expected = plan.pending(sigma)
        first = plan.admit(sigma)
        assert (first is not None) == expected
        # A second branch or nested forward at the same sigma shares the slot and the window position.
        assert plan.admit(sigma) == first
        positions.append(first)
    selected = [i for i, value in enumerate(positions) if value is not None]
    assert selected == list(range(start, min(start + count, len(SCHEDULE))))
    assert [positions[i] for i in selected] == list(range(len(selected)))
    assert plan.reached == len(selected) and plan.window_sigmas == SCHEDULE[start:start + count]
    assert plan.seen == SCHEDULE[:min(start + count, len(SCHEDULE))]


def test_short_and_unreachable_windows_are_reported_never_wrapped():
    plan = FirstEvaluations(count=8, start=2)
    positions = [plan.admit(sigma) for sigma in SCHEDULE[:5]]
    assert positions == [None, None, 0, 1, 2] and plan.reached == 3 and not plan.closed
    plan = FirstEvaluations(count=4, start=6)
    assert [plan.admit(sigma) for sigma in SCHEDULE[:4]] == [None] * 4 and plan.reached == 0
    # A fresh invocation starts at index 0 again: no carryover.
    again = FirstEvaluations(count=4, start=6)
    assert again.seen == [] and again.pending(SCHEDULE[0]) is False and again.pending(SCHEDULE[0]) is False


def test_pending_is_a_pure_query_and_tracks_the_next_slot():
    plan = FirstEvaluations(count=2, start=1)
    for _ in range(3):
        assert not plan.pending(1.0)
    assert plan.admit(1.0) is None
    assert plan.pending(1.0) is False  # same sigma as the native opening evaluation
    assert plan.pending(0.9) and plan.admit(0.9) == 0 and plan.pending(0.9)
    assert plan.pending(0.8) and plan.admit(0.8) == 1
    assert not plan.pending(0.7) and plan.admit(0.7) is None and plan.closed


def test_builtin_profiles_have_exact_documented_values():
    from mode_explorer.rediverse import profile_multipliers as profile

    for name in ("constant", "linear_decay", "cosine_decay", "linear_increase", "bell_curve"):
        assert profile(name, 1) == (1.0,)
    assert profile("constant", 4) == (1.0, 1.0, 1.0, 1.0)
    assert profile("linear_decay", 4) == pytest.approx((1.0, 2 / 3, 1 / 3, 0.0))
    assert profile("linear_increase", 4) == pytest.approx((0.0, 1 / 3, 2 / 3, 1.0))
    assert profile("linear_increase", 4)[0] == 0.0 and profile("linear_decay", 4)[-1] == 0.0
    assert profile("cosine_decay", 3) == pytest.approx((1.0, 0.5, 0.0))
    assert profile("cosine_decay", 5) == pytest.approx(tuple((1 + math.cos(math.pi * i / 4)) / 2 for i in range(5)))
    bell = profile("bell_curve", 4)
    assert bell[0] == 0.0 and bell[-1] == 0.0 and bell[1] == bell[2] == pytest.approx(math.sin(math.pi / 3))
    assert max(bell) < 1.0  # even length: no implicit normalization of the peak
    assert profile("bell_curve", 5) == pytest.approx((0.0, math.sin(math.pi / 4), 1.0, math.sin(math.pi / 4), 0.0))
    assert profile("bell_curve", 2) == (0.0, 0.0)
    assert len(profile("cosine_decay", 16)) == 16


def test_custom_profiles_are_parsed_exactly_and_invalid_lists_rejected():
    from mode_explorer.rediverse import profile_multipliers as profile

    assert profile("custom", 4, "0.5, 0.75, 1.0, 0.25") == (0.5, 0.75, 1.0, 0.25)
    assert profile("custom", 4, "2,0,0,0") == (2.0, 0.0, 0.0, 0.0)
    assert profile("custom", 1, " 1.5 ") == (1.5,)
    for bad, count in (("", 4), (" ", 1), ("0.5, 0.5", 4), ("0.5,,0.5,0.5", 4), ("a,b,c,d", 4), ("nan,1,1,1", 4),
                       ("inf,1,1,1", 4), ("-0.1,1,1,1", 4), ("1,1,1,1,1", 4), ("1;1;1;1", 4)):
        with pytest.raises(ValueError):
            profile("custom", count, bad)
    with pytest.raises(ValueError):
        profile("unknown", 4)


def _config(**kwargs):
    from types import SimpleNamespace

    from mode_explorer.rediverse_runtime import ReDiverseConfig

    adapter = SimpleNamespace(describe=lambda: {"file": "a"})
    return ReDiverseConfig(adapter=adapter, **kwargs)


def test_config_scales_profiles_by_global_strength_without_normalization_or_clamping():
    assert _config().strengths() == [1.0] and not _config().temporal and not _config().experimental
    assert _config(strength=.5, evaluations=4, profile="custom", custom="0.5, 0.75, 1.0, 0.25").strengths() == \
        pytest.approx([.25, .375, .5, .125])
    assert _config(strength=1.0, evaluations=4, profile="custom", custom="2, 0, 0, 0").strengths() == [2.0, 0, 0, 0]
    with pytest.raises(ValueError, match="exceed"):
        _config(strength=1.5, evaluations=4, profile="custom", custom="2, 0, 0, 0")
    with pytest.raises(ValueError, match="adapted sigmas"):
        _config(evaluations=4, profile="custom", custom="1, 1")
    with pytest.raises(ValueError):
        _config(evaluations=17)
    with pytest.raises(ValueError, match="first_evaluation"):
        _config(schedule="first_interval", start=1)
    with pytest.raises(ValueError, match="first_evaluation"):
        _config(schedule="first_interval", evaluations=2, profile="linear_decay")
    assert _config(schedule="first_interval", evaluations=3).strengths() == [1.0, 1.0, 1.0]
    sham = _config(strength=0.0, evaluations=4, start=2, profile="cosine_decay")
    assert sham.strengths() == [0.0] * 4 and not sham.adapts and sham.experimental
    described = _config(strength=.5, evaluations=4, start=1, profile="custom", custom="1, 1, .5, 0").describe()
    assert described["start_evaluation"] == 1 and described["planned_evaluations"] == [1, 2, 3, 4]
    assert described["custom_multipliers"] == [1.0, 1.0, 0.5, 0.0]
    assert described["planned_strengths"] == [.5, .5, .25, 0.0]
    assert _config(evaluations=4).describe() != _config(evaluations=4, start=1).describe()
    assert _config(evaluations=4).describe() != _config(evaluations=4, profile="linear_decay").describe()


def test_temporal_ft_options_are_rejected_with_active_tf():
    from types import SimpleNamespace

    from mode_explorer.rediverse_runtime import ReDiverseConfig

    archive = SimpleNamespace(U=torch.zeros(8, 4), describe=lambda: {})
    adapter = SimpleNamespace(describe=lambda: {})
    for options in (dict(start=1), dict(evaluations=4, profile="linear_decay")):
        with pytest.raises(ValueError, match="ReDiverse-TF"):
            ReDiverseConfig(adapter=adapter, directions=archive, band=2.0, **options)
        ReDiverseConfig(adapter=adapter, directions=archive, band=0.0, **options)  # band 0: no steering
    # The constant profile from evaluation 0 keeps the established multi-sigma FT+TF behavior.
    assert ReDiverseConfig(adapter=adapter, directions=archive, band=2.0, evaluations=8).strengths() == [1.0] * 8


def test_coefficients_are_deterministic_and_leave_global_rng_untouched():
    state = torch.get_rng_state()
    first = coefficients(42, 3, 8, 2.0)
    assert torch.equal(torch.get_rng_state(), state)
    assert torch.equal(first, coefficients(42, 3, 8, 2.0))
    assert first.abs().max() <= 2.0 and first.shape == (3, 8)
    assert not torch.equal(first, coefficients(43, 3, 8, 2.0))
    assert not torch.equal(first, coefficients(42, 3, 8, 2.0, trial=1))
    # Rows are independent streams: batch size does not change a row's draw.
    assert torch.equal(coefficients(42, 1, 8, 2.0)[0], first[0])
    assert coefficient_seed(2 ** 64 - 1, 7) < 2 ** 63


def test_packed_noise_matches_reference_packing():
    seed = 27 * 1000 + 3
    packed = torch.randn((4 * 6, 64), generator=torch.Generator("cpu").manual_seed(seed))
    latent = packed_noise(seed, 8, 12)
    # ComfyUI process_img: "c (h ph) (w pw) -> (h w) (c ph pw)".
    repacked = latent.reshape(16, 4, 2, 6, 2).permute(1, 3, 0, 2, 4).reshape(24, 64)
    assert torch.equal(repacked, packed)


def test_fingerprint_ignores_dtype_and_device_but_not_values():
    x = torch.randn(3, 5, dtype=torch.float64).float()
    assert tensor_fingerprint(x) == tensor_fingerprint(x.double())
    assert tensor_fingerprint(x) != tensor_fingerprint(x + 1e-3)
    assert tensor_fingerprint(x) != tensor_fingerprint(x.reshape(5, 3))


def _adapter_tensors(rank=4, blocks=(0, 1), naming="comfy"):
    g = torch.Generator().manual_seed(0)
    tensors = {}
    for b in blocks:
        for module, dims in {"attn.wq": (64, 64), "mlp.down": (128, 64)}.items():
            if naming == "comfy":
                stem = f"diffusion_model.blocks.{b}.{module}"
            else:
                stem = f"transformer.transformer_blocks.{b}.{'attn.to_q' if module == 'attn.wq' else 'ff.down'}"
            tensors[f"{stem}.lora_A.weight"] = torch.randn(rank, dims[0], generator=g)
            tensors[f"{stem}.lora_B.weight"] = torch.randn(dims[1], rank, generator=g)
    return tensors


def test_adapter_parsing_and_rejections(tmp_path):
    from safetensors.torch import save_file

    def write(tensors, name="a.safetensors", metadata=None):
        path = tmp_path / name
        save_file(tensors, str(path), metadata=metadata)
        return str(path)

    comfy = load_adapter(write(_adapter_tensors(), metadata={"rank": "4", "alpha": "4"}))
    diffusers = load_adapter(write(_adapter_tensors(naming="diffusers"), "d.safetensors"))
    assert comfy.rank == 4 and comfy.blocks == [0, 1] and set(comfy.modules[0]) == {"attn.wq", "mlp.down"}
    for block in (0, 1):
        for module in ("attn.wq", "mlp.down"):
            assert all(torch.equal(a, b) for a, b in zip(comfy.modules[block][module][:2], diffusers.modules[block][module][:2]))
            assert comfy.modules[block][module][2] == 1.0
    halved = load_adapter(write(_adapter_tensors(), "h.safetensors", metadata={"rank": "4", "alpha": "2"}))
    assert halved.modules[0]["attn.wq"][2] == .5
    bad = _adapter_tensors()
    del bad["diffusion_model.blocks.0.attn.wq.lora_B.weight"]
    cases = {
        "missing": bad,
        "unknown": {**_adapter_tensors(), "diffusion_model.blocks.0.attn.wx.lora_A.weight": torch.zeros(4, 64)},
        "foreign": {"lora_unet_x.lora_down.weight": torch.zeros(4, 4)},
        "mixed": {**_adapter_tensors(), **{k: v for k, v in _adapter_tensors(rank=2, blocks=(2,)).items()}},
        "nonfinite": {k: (v.fill_(math.nan) if "B" in k else v) for k, v in _adapter_tensors().items()},
        "zero": {k: (torch.zeros_like(v) if "lora_B" in k else v) for k, v in _adapter_tensors().items()},
        "shape": {**_adapter_tensors(), "diffusion_model.blocks.0.attn.wq.lora_B.weight": torch.zeros(64, 3)},
    }
    for name, tensors in cases.items():
        with pytest.raises(ValueError):
            load_adapter(write(tensors, f"{name}.safetensors"))


def _archive(tmp_path, **overrides):
    g = torch.Generator().manual_seed(1)
    grid, width, k = [2, 3], 4, 3
    U = torch.linalg.qr(torch.randn(24, k, generator=g))[0].T.contiguous()
    metadata = {
        "schema": 1, "implementation": "test", "coordinate": "block_output_image_rows", "block": 1,
        "token_grid": grid, "d_model": width, "teacher_model_sha256": "t", "teacher_patches": {},
        "conditioning_sha256": "c", "sigma_in": 1.0, "sigma_out": .9, "teacher_forward": 1, "teacher_steps": 2,
        "guidance": {"cfg": 1.0}, "seeds": [1, 2, 3, 4], "rollouts": 4, "normalization": "N",
        "basis_dtype": "float32", "directions": k, "valid_directions": k, "negative_conditioning_sha256": None,
        "teacher_sigmas": [1.0, .9, 0.0], "initial_noise_sha256": "n", "identity_sha256": "i",
    }
    metadata.update(overrides.pop("metadata", {}))
    tensors = dict(U=U, s=torch.tensor([3., 2., 1.]), eigenvalues=torch.tensor([36., 16., 4., 0.]))
    tensors.update(overrides)
    path = str(tmp_path / "d.safetensors")
    save_directions(path, tensors["U"], tensors["s"], tensors["eigenvalues"], metadata)
    return path


def _nonorthogonal():
    g = torch.Generator().manual_seed(1)
    U = torch.linalg.qr(torch.randn(24, 3, generator=g))[0].T.contiguous()
    U[1] = (U[0] + U[1]) / (U[0] + U[1]).norm()  # unit rows, not orthogonal
    return U


def test_direction_archive_round_trip_and_rejections(tmp_path):
    from safetensors.torch import save_file

    loaded = load_directions(_archive(tmp_path))
    assert loaded.block == 1 and loaded.grid == (2, 3) and loaded.U.shape == (3, 24) and loaded.sha256
    with pytest.raises(ValueError, match="incomplete|missing"):
        save_directions(str(tmp_path / "x.safetensors"), loaded.U, loaded.s, loaded.eigenvalues, {"schema": 1})
    for overrides, message in (
        ({"metadata": {"schema": 2}}, "schema"),
        ({"metadata": {"token_grid": [3, 3]}}, "shapes"),
        ({"metadata": {"basis_dtype": "bfloat16"}}, "shapes"),
        ({"s": torch.tensor([math.inf, 1., 1.])}, "nonfinite"),
        ({"U": torch.ones(3, 24)}, "unit"),
        ({"s": torch.tensor([-1., 1., 1.])}, "usable"),
        # Semantic validation beyond shapes.
        ({"metadata": {"coordinate": "attention_output"}}, "coordinate"),
        ({"metadata": {"valid_directions": 2}}, "valid count"),
        ({"s": torch.zeros(3), "U": torch.zeros(3, 24)}, "usable"),
        ({"s": torch.tensor([3., 2., 0.]), "metadata": {"valid_directions": 2}}, "without spread"),
        ({"U": _nonorthogonal()}, "orthogonal"),
        ({"s": torch.tensor([3., 2., 1.5])}, "sqrt"),
        ({"eigenvalues": torch.tensor([16., 36., 4., 0.]), "s": torch.tensor([2., 3., 1.])}, "descending"),
        ({"eigenvalues": torch.tensor([36., 16., 4.])}, "rollout"),
    ):
        path = _archive(tmp_path, **overrides)
        with pytest.raises(ValueError, match=message):
            load_directions(path)
    foreign = str(tmp_path / "foreign.safetensors")
    save_file({"U": torch.zeros(1)}, foreign)
    with pytest.raises(ValueError, match="not a ModeExplorer"):
        load_directions(foreign)
    malformed = str(tmp_path / "malformed.safetensors")
    save_file({"U": torch.zeros(1)}, malformed, metadata={ARCHIVE_KEY: "{"})
    with pytest.raises(ValueError, match="malformed"):
        load_directions(malformed)
    extra = str(tmp_path / "extra.safetensors")
    save_file({"U": loaded.U, "s": loaded.s, "eigenvalues": loaded.eigenvalues, "payload": torch.zeros(1)}, extra,
              metadata={ARCHIVE_KEY: json.dumps(loaded.metadata)})
    with pytest.raises(ValueError, match="unexpected"):
        load_directions(extra)
    with pytest.raises(ValueError, match="limit"):
        load_directions(_archive(tmp_path), max_bytes=16)
