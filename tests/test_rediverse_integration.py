"""Native ComfyUI contracts of ReDiverse on a tiny real Krea 2 transformer.

These tests establish scheduling, isolation and numerical equivalences. They are not
evidence of image quality or diversity on real checkpoints.
"""

import copy
import json
import logging
import math
import os

import pytest
import torch

comfy = pytest.importorskip("comfy")

from test_comfy_integration import conditions, draw, model, traces  # noqa: F401  (fixture)

import folder_paths
from mode_explorer.nodes import ModeExplorer, ModeExplorerResponse
from mode_explorer.rediverse import COMFY_MODULES, resolve_module
from mode_explorer.rediverse_nodes import (
    ModeExplorerActivationProbe, ModeExplorerReDiverse, ModeExplorerReDiverseCalibrate, ModeExplorerReDiverseDirections,
)
from mode_explorer.runtime import KEY, RUN, STAGE

SIGMAS = torch.tensor([1.0, 0.93, 0.83, 0.62, 0.2, 0.0])
LATENT = torch.zeros(1, 16, 1, 8, 12)


@pytest.fixture
def workspace(tmp_path, model):
    """A loras folder with a deterministic rank-4 adapter on blocks 0-1 and a private output folder."""
    from safetensors.torch import save_file

    loras = tmp_path / "loras"
    loras.mkdir()
    dit = model.get_model_object("diffusion_model")
    g = torch.Generator().manual_seed(17)
    tensors = {}
    for block in (0, 1):
        for name in COMFY_MODULES:
            layer = resolve_module(dit.blocks[block], name)
            tensors[f"diffusion_model.blocks.{block}.{name}.lora_A.weight"] = torch.randn(4, layer.in_features, generator=g) * .1
            tensors[f"diffusion_model.blocks.{block}.{name}.lora_B.weight"] = torch.randn(layer.out_features, 4, generator=g) * .1
    save_file(tensors, str(loras / "rediverse.safetensors"), metadata={"rank": "4", "alpha": "4", "method": "ReDiverse-FT"})
    previous_output = folder_paths.get_output_directory()
    previous_loras = copy.deepcopy(folder_paths.folder_names_and_paths["loras"])
    folder_paths.add_model_folder_path("loras", str(loras))
    folder_paths.set_output_directory(str(tmp_path / "output"))
    try:
        yield tensors
    finally:
        folder_paths.set_output_directory(previous_output)
        folder_paths.folder_names_and_paths["loras"] = previous_loras
        folder_paths.filename_list_cache.clear() if hasattr(folder_paths, "filename_list_cache") else None


def rediverse(model, **kwargs):
    kwargs.setdefault("adapter", "rediverse.safetensors")
    return ModeExplorerReDiverse().apply(model, **kwargs)[0]


def no_hooks(model):
    dit = model.get_model_object("diffusion_model")
    return all(not m._forward_hooks and not m._forward_pre_hooks for m in dit.modules())


def merged(model, tensors, strength=1.0):
    """Independent reference: the adapter merged as native weight-difference patches."""
    clone = model.clone()
    patches = {}
    for key, down in tensors.items():
        if key.endswith("lora_A.weight"):
            stem = key.removesuffix(".lora_A.weight")
            patches[stem + ".weight"] = (strength * tensors[stem + ".lora_B.weight"] @ down,)
    assert len(clone.add_patches(patches, 1.0)) == len(patches)
    return clone


@torch.inference_mode()
def split_reference(first, rest, seed=55, sampler="euler", sigmas=SIGMAS, cfg=1.0):
    """Euler step 1 on ``first``, the remaining steps on ``rest``, from the same noise."""
    x = torch.zeros(1, 16, 1, 7, 11)
    noise = comfy.sample.prepare_noise(x, seed)
    head = comfy.samplers.CFGGuider(first)
    p, n = conditions()
    head.set_conds(p, n)
    head.set_cfg(cfg)
    x1 = head.sample(noise, x, comfy.samplers.ksampler(sampler), sigmas[:2], disable_pbar=True, seed=seed)
    # The guider returns process_latent_out(x / (1 - sigma_last)) for CONST flow.
    x1 = first.model.process_latent_in(x1) * (1 - sigmas[1])
    tail = comfy.samplers.CFGGuider(rest)
    tail.set_conds(p, n)
    tail.set_cfg(cfg)
    # CONST noise scaling at sigma s with an all-zero latent is s * noise.
    return tail.sample(x1 / sigmas[1], x, comfy.samplers.ksampler(sampler), sigmas[1:], disable_pbar=True, seed=seed)


def summaries(caplog):
    return [t["rediverse"] for t in traces(caplog) if "rediverse" in t]


def test_node_registration_and_disabled_paths(model, workspace):
    import asyncio
    from pathlib import Path

    import nodes

    assert asyncio.run(nodes.load_custom_node(str(Path(__file__).resolve().parents[1])))
    for name in ("ModeExplorerReDiverseKrea2", "ModeExplorerReDiverseCalibrateKrea2",
                 "ModeExplorerReDiverseDirectionsKrea2", "ModeExplorerActivationProbeKrea2"):
        assert name in nodes.NODE_CLASS_MAPPINGS
    assert ModeExplorerReDiverse().apply(model, enabled=False)[0] is model
    with pytest.raises(ValueError, match="Select a ReDiverse adapter"):
        ModeExplorerReDiverse().apply(model)
    patched = rediverse(model)
    removed, = ModeExplorerReDiverse().apply(patched, enabled=False)
    assert all(KEY not in by_key for by_key in removed.wrappers.values())
    assert torch.equal(draw(removed), draw(model))


def test_zero_strength_is_bitwise_native(model, workspace):
    assert torch.equal(draw(rediverse(model, strength=0.0)), draw(model))


def test_first_evaluation_matches_merged_first_step_and_leaves_no_state(caplog, model, workspace):
    caplog.set_level(logging.INFO, logger="ModeExplorer")
    native = draw(model)
    reference = split_reference(merged(model, workspace), model)
    patched = rediverse(model, diagnostics=True)
    result = draw(patched)
    assert not torch.allclose(result, native, atol=1e-4)
    assert torch.allclose(result, reference, atol=2e-5), (result - reference).abs().max()
    summary, = summaries(caplog)
    assert summary["admitted_sigmas"] == [1.0]
    assert summary["adapted_evaluations"] == 1 and summary["adapted_rows"] == 1
    assert summary["adapter_module_calls"] == 2 * len(COMFY_MODULES)
    assert summary["forecast_consumed"] == 0 and summary["reasons"] == []
    assert no_hooks(model)
    # The shared model is unchanged: a native draw afterwards is identical.
    assert torch.equal(draw(model), native)
    # Repeated queue executions are deterministic.
    assert torch.equal(draw(patched), result)


def test_strength_scales_adapter_and_upstream_weight_patches_remain(model, workspace):
    half = split_reference(merged(model, workspace, .5), model)
    assert torch.allclose(draw(rediverse(model, strength=.5)), half, atol=2e-5)
    # An upstream weight patch (an identity LoRA, for example) stays active on every step.
    patched = model.clone()
    key = "diffusion_model.blocks.2.attn.wq.weight"
    patched.add_patches({key: (torch.full_like(model.model.state_dict()[key], 1e-2),)}, 1.0)
    upstream = draw(patched)
    assert not torch.equal(upstream, draw(model))
    assert torch.equal(draw(rediverse(patched, strength=0.0)), upstream)
    assert not torch.equal(draw(rediverse(patched)), draw(rediverse(model)))


def test_cfg_branches_share_the_adapted_first_evaluation(caplog, model, workspace):
    caplog.set_level(logging.INFO, logger="ModeExplorer")
    patched = rediverse(model, diagnostics=True)
    reference = split_reference(merged(model, workspace), model, cfg=2.0)
    assert torch.allclose(draw(patched, cfg=2.0), reference, atol=2e-5)
    packed, = summaries(caplog)
    assert packed["adapted_evaluations"] == 1 and packed["adapted_rows"] == 2
    caplog.clear()
    draw(patched, cfg=2.0, split_cfg=True)
    split, = summaries(caplog)
    assert split["adapted_evaluations"] == 2 and split["admitted_sigmas"] == [1.0]


@pytest.mark.parametrize("sampler,calls", [("heun", 1), ("dpmpp_2s_ancestral", 1), ("er_sde", 1)])
def test_multistage_solvers_adapt_only_the_opening_sigma(caplog, model, workspace, sampler, calls):
    caplog.set_level(logging.INFO, logger="ModeExplorer")
    draw(rediverse(model, diagnostics=True), sampler=sampler)
    summary, = summaries(caplog)
    assert summary["adapted_evaluations"] == calls and len(summary["admitted_sigmas"]) == 1
    if sampler == "er_sde":
        # Native ER-SDE evaluates the first CONST timestep slightly below 1.
        assert summary["admitted_sigmas"][0] < 1.0
    caplog.clear()
    draw(rediverse(model, diagnostics=True, schedule="first_interval"), sampler=sampler)
    extended, = summaries(caplog)
    assert extended["config"]["experimental"]
    assert extended["adapted_evaluations"] >= summary["adapted_evaluations"]
    assert all(sigma > .93 for sigma in extended["admitted_sigmas"])


def test_partial_denoise_masks_and_highres_stage_bypass(caplog, model, workspace):
    caplog.set_level(logging.INFO, logger="ModeExplorer")
    patched = rediverse(model)
    partial = torch.tensor([0.83, 0.62, 0.2, 0.0])
    assert torch.equal(draw(patched, sigmas=partial), draw(model, sigmas=partial))
    assert any("partial_denoise" in r.message for r in caplog.records)
    assert torch.equal(draw(patched, stage="highres"), draw(model, stage="highres"))
    mask = torch.ones(1, 1, 1, 7, 11)
    assert torch.equal(draw(patched, mask=mask), draw(model, mask=mask))
    assert no_hooks(model)


def test_exception_and_interrupt_leave_no_hooks(model, workspace):
    import comfy.model_management

    dit = model.get_model_object("diffusion_model")

    def boom(module, args, output):
        raise RuntimeError("cancelled")

    handle = dit.blocks[2].register_forward_hook(boom)
    try:
        with pytest.raises(RuntimeError, match="cancelled"):
            draw(rediverse(model))
    finally:
        handle.remove()
    assert no_hooks(model)
    comfy.model_management.interrupt_current_processing(True)
    try:
        with pytest.raises(comfy.model_management.InterruptProcessingException):
            draw(rediverse(model))
    finally:
        comfy.model_management.interrupt_current_processing(False)
    assert no_hooks(model)
    assert torch.equal(draw(model), draw(rediverse(model, strength=0.0)))


def test_composes_with_modeexplorer_b1_and_response_carries_it(caplog, model, workspace):
    caplog.set_level(logging.INFO, logger="ModeExplorer")
    ft = rediverse(model, diagnostics=True)
    combined, = ModeExplorer().apply(ft, mode="Early Structural (B1)", strength=.03, fidelity=.7, diagnostics=True)
    from comfy.patcher_extension import WrappersMP

    sampler, = combined.get_wrappers(WrappersMP.SAMPLER_SAMPLE, KEY)
    assert sampler.rediverse is not None and sampler.settings.feature_block == 1
    out = draw(combined)
    assert torch.isfinite(out).all()
    trace = traces(caplog)[-1]
    assert trace["feature_reference_dit_calls"] >= 1
    summary = trace["rediverse"]
    # The first evaluation's reference forward also runs on the adapted model.
    assert summary["adapter_module_calls"] == 2 * len(COMFY_MODULES) * 2
    # Order does not matter: ReDiverse after ModeExplorer keeps the same settings.
    other = rediverse(ModeExplorer().apply(model, mode="Early Structural (B1)", strength=.03, fidelity=.7)[0])
    assert torch.equal(draw(other), draw(ModeExplorer().apply(ft, mode="Early Structural (B1)", strength=.03, fidelity=.7)[0]))
    responded, = ModeExplorerResponse().apply(combined, min_sigma=.9, max_sigma=1., directions=1, calibration="off")
    assert responded.get_wrappers(WrappersMP.SAMPLER_SAMPLE, KEY)[0].rediverse is sampler.rediverse
    with pytest.raises(ValueError, match="initialization"):
        ModeExplorer().apply(ft, initialization=True)
    with pytest.raises(ValueError, match="guarded Spectrum"):
        ModeExplorer().apply(ft, full_spectrum=True)
    from mode_explorer.nodes import ModeExplorerProjectionGuidance

    with pytest.raises(ValueError, match="separate MODEL"):
        ModeExplorerProjectionGuidance().apply(ft, enabled=True, amplitude=.01)


def test_adapter_shape_mismatch_is_rejected(model, workspace, tmp_path):
    from safetensors.torch import save_file

    bad = {k: (torch.zeros(4, 3) if k.endswith("blocks.0.attn.wq.lora_A.weight") else v) for k, v in workspace.items()}
    save_file(bad, str(tmp_path / "loras" / "bad.safetensors"))
    with pytest.raises(ValueError, match="expects"):
        rediverse(model, adapter="bad.safetensors")


# ------------------------------------------------------------------ ReDiverse-TF --
@pytest.fixture
def calibrated(model, workspace):
    p, _ = conditions()
    directions, summary = ModeExplorerReDiverseCalibrate().calibrate(
        model, p, {"samples": LATENT}, block=2, teacher_forward=2, teacher_steps=6, rollouts=6, directions=4,
        seed=3, batch=4, label="tiny")
    return directions, json.loads(summary)


def test_calibration_archive_provenance_and_teacher_isolation(model, calibrated):
    directions, summary = calibrated
    state = {k: v.clone() for k, v in model.model.state_dict().items()}
    meta = directions.metadata
    assert meta["block"] == 2 and meta["token_grid"] == [4, 6] and meta["d_model"] == 64
    assert meta["seeds"] == [3100 + j for j in range(6)] and meta["rollouts"] == 6
    assert meta["guidance"]["branches"] == "conditional_only" and meta["teacher_forward"] == 2
    assert meta["sigma_in"] == pytest.approx(meta["sigma_in"]) and 0 < meta["sigma_out"] < meta["sigma_in"] <= 1
    assert meta["valid_directions"] == 4 and directions.U.dtype == torch.bfloat16
    # An EmptyLatentImage-style 4-channel 4D latent resolves to the model's latent layout.
    p, _ = conditions()
    again, text = ModeExplorerReDiverseCalibrate().calibrate(
        model, p, {"samples": torch.zeros(1, 4, 8, 12)}, block=2, teacher_forward=2, teacher_steps=6, rollouts=6,
        directions=4, seed=3, batch=4, label="tiny")
    assert torch.equal(again.U, directions.U) and again.metadata["latent_shape"] == [16, 1, 8, 12]
    # Identical calibration identity; the repeat is preserved next to the first archive.
    assert again.metadata["identity_sha256"] == meta["identity_sha256"]
    assert json.loads(text)["path"] == summary["path"].replace(".safetensors", "-r2.safetensors")
    assert os.path.isfile(summary["path"])
    other, text = ModeExplorerReDiverseCalibrate().calibrate(
        model, p, {"samples": LATENT}, block=2, teacher_forward=2, teacher_steps=6, rollouts=6, directions=4,
        seed=4, batch=4, label="tiny")
    assert json.loads(text)["path"] != summary["path"] and not torch.equal(other.U, directions.U)
    assert torch.allclose(directions.U.float().norm(dim=1), torch.ones(4), atol=1e-2)
    assert os.path.isfile(summary["path"])
    assert no_hooks(model)
    assert all(torch.equal(v, state[k]) for k, v in model.model.state_dict().items())
    loaded, text = ModeExplorerReDiverseDirections().load(os.path.basename(summary["path"]))
    assert json.loads(text)["sha256"] == directions.sha256 and torch.equal(loaded.U, directions.U)
    with pytest.raises(ValueError):
        ModeExplorerReDiverseDirections().load("../escape.safetensors")


def test_calibration_records_the_complete_teacher_trajectory(model, workspace, calibrated):
    from mode_explorer.rediverse import reference_sigmas

    directions, summary = calibrated
    meta = directions.metadata
    assert set(summary["timings"]) == {"rollout_seconds", "pca_seconds", "write_and_validate_seconds", "total_seconds"}
    assert summary["timings"]["total_seconds"] >= summary["timings"]["rollout_seconds"] + summary["timings"]["pca_seconds"]
    assert meta["teacher_sigmas"] == pytest.approx(reference_sigmas(6, 24))
    assert meta["executed_sigmas"] == pytest.approx(reference_sigmas(6, 24)[:3])
    assert meta["negative_conditioning_sha256"] is None and meta["initial_noise_sha256"] and meta["capture_input_sha256"]
    assert "not a full checkpoint hash" in meta["teacher_model_fingerprint"]
    assert set(meta["timings"]) == {"rollout_seconds", "pca_seconds"}
    p, n = conditions()
    node = ModeExplorerReDiverseCalibrate()
    common = dict(block=2, teacher_forward=2, teacher_steps=6, rollouts=6, directions=4, seed=3, batch=4, label="tiny")
    # Same capture sigmas reached through another preceding schedule: a distinct archive identity.
    other = torch.tensor(reference_sigmas(6, 24))
    other[1] = (other[0] + other[1]) / 2
    shifted, _ = node.calibrate(model, p, {"samples": LATENT}, sigmas=other, **common)
    assert shifted.metadata["identity_sha256"] != meta["identity_sha256"]
    for bad in (torch.tensor([1.0, .95, .97, .5]), torch.tensor([1.0, math.nan, .5, .1]),
                torch.tensor([.9, .8, .7, .1]), torch.tensor([1.0, .9])):
        with pytest.raises(ValueError, match="Connected teacher sigmas"):
            node.calibrate(model, p, {"samples": LATENT}, sigmas=bad, **common)
    with pytest.raises(ValueError, match="needs a negative"):
        node.calibrate(model, p, {"samples": LATENT}, cfg=2.0, **common)
    guided, _ = node.calibrate(model, p, {"samples": LATENT}, negative=n, cfg=2.0, **common)
    other_negative, _ = node.calibrate(model, p, {"samples": LATENT}, negative=[[n[0][0] + 1, {}]], cfg=2.0, **common)
    assert guided.metadata["guidance"] == {"cfg": 2.0, "branches": "guided"}
    assert guided.metadata["negative_conditioning_sha256"] != other_negative.metadata["negative_conditioning_sha256"]
    assert guided.metadata["identity_sha256"] != other_negative.metadata["identity_sha256"]
    assert no_hooks(model)


def test_calibration_capture_matches_manual_rollout(model, calibrated):
    """The archived spread equals the PCA of block outputs from a manual Euler rollout."""
    from mode_explorer import covariance
    from mode_explorer.rediverse import packed_noise, reference_sigmas

    directions, _ = calibrated
    dit = model.get_model_object("diffusion_model")
    sigmas = reference_sigmas(6, 24)
    captured = []
    handle = dit.blocks[2].register_forward_hook(
        lambda m, a, k, out: captured.append(out[:, -24:].reshape(out.shape[0], -1).clone()), with_kwargs=True)
    p, _ = conditions()
    try:
        with torch.inference_mode():
            for j in range(6):
                noise = packed_noise(3000 + 100 + j, 8, 12).reshape(1, 16, 1, 8, 12)
                guider = comfy.samplers.CFGGuider(model)
                guider.inner_set_conds({"positive": p})
                guider.set_cfg(1.0)
                guider.sample(noise, torch.zeros_like(noise), comfy.samplers.ksampler("euler"),
                              torch.tensor(sigmas[:3]), disable_pbar=True, seed=0)
    finally:
        handle.remove()
    X = torch.cat(captured[1::2]).to(torch.bfloat16)  # forward 2 of each rollout
    values, vectors = covariance.eigenpairs(covariance.centred_gram(X))
    U, s, _ = covariance.principal_basis(X, vectors, values, 4)
    assert torch.allclose(s, directions.s, rtol=1e-4)
    assert torch.allclose((U.float() * directions.U.float()).sum(1).abs(), torch.ones(4), atol=1e-2)


def test_tf_applies_offset_to_image_rows_only_on_first_evaluation(caplog, model, workspace, calibrated):
    caplog.set_level(logging.INFO, logger="ModeExplorer")
    directions, _ = calibrated
    dit = model.get_model_object("diffusion_model")
    native_rows, steered_rows = [], []

    def capture(store):
        # The final layer's input is the last block's output after every block hook.
        return dit.last.register_forward_pre_hook(lambda m, args: store.append(args[0].clone()))

    handle = capture(native_rows)
    try:
        native = draw(model, latent=LATENT)
    finally:
        handle.remove()
    steered_model, = ModeExplorerReDiverse().apply(model, directions=directions, band=2.0, diagnostics=True)
    handle = capture(steered_rows)
    try:
        steered = draw(steered_model, latent=LATENT)
    finally:
        handle.remove()
    assert not torch.equal(native, steered)
    summary, = summaries(caplog)
    assert summary["tf_applications"] == 1 and summary["adapted_evaluations"] == 1
    from mode_explorer.rediverse import coefficients

    alpha = coefficients(55, 1, 4, 2.0)
    assert summary["coefficients"] == [[round(float(v), 5) for v in alpha[0]]]
    # Hooks fire in registration order, so the capture sees the steered first output.
    first_native, first_steered = native_rows[0], steered_rows[0]
    text = first_native.shape[1] - 24
    assert torch.equal(first_native[:, :text], first_steered[:, :text])
    offset = ((alpha[0] * directions.s).to(directions.U.dtype) @ directions.U).reshape(24, 64)
    assert torch.allclose(first_steered[0, text:] - first_native[0, text:], offset.float(), atol=1e-5)
    # Zero band is the native model; a different seed draws different coefficients.
    zero, = ModeExplorerReDiverse().apply(model, directions=directions, band=0.0)
    assert torch.equal(draw(zero, latent=LATENT), native)
    assert no_hooks(model)


def test_tf_rejects_other_prompts_and_geometries_and_keeps_sampler_rng(model, workspace, calibrated):
    directions, _ = calibrated
    steered, = ModeExplorerReDiverse().apply(model, directions=directions, band=2.0)
    with pytest.raises(RuntimeError, match="token grid"):
        draw(steered, latent=torch.zeros(1, 16, 1, 8, 10))
    p, n = conditions()
    guider = comfy.samplers.CFGGuider(steered)
    guider.set_conds([[p[0][0] * 2, {}]], n)
    guider.set_cfg(1.0)
    with pytest.raises(RuntimeError, match="different positive conditioning"):
        with torch.inference_mode():
            guider.sample(comfy.sample.prepare_noise(LATENT, 1), LATENT, comfy.samplers.ksampler("euler"), SIGMAS,
                          disable_pbar=True, seed=1)
    assert no_hooks(model)
    # The ancestral sampler's noise stream is unaffected by coefficient draws.
    zero, = ModeExplorerReDiverse().apply(model, directions=directions, band=0.0)
    assert torch.equal(draw(zero, latent=LATENT, sampler="euler_ancestral"), draw(model, latent=LATENT, sampler="euler_ancestral"))


def test_tf_adds_no_dit_forwards_and_combines_with_ft(caplog, model, workspace, calibrated):
    caplog.set_level(logging.INFO, logger="ModeExplorer")
    directions, _ = calibrated
    both = rediverse(model, directions=directions, band=4.0, diagnostics=True)
    draw(both, latent=LATENT)
    trace = traces(caplog)[-1]
    assert trace["native_dit_calls"] == len(SIGMAS) - 1
    assert trace["rediverse"]["tf_applications"] == 1
    assert trace["rediverse"]["adapter_module_calls"] == 2 * len(COMFY_MODULES)


def test_cohort_logdet_and_its_norm_matched_control(caplog, model, workspace, calibrated):
    from mode_explorer.rediverse_runtime import activation_norm, cohort_coefficients, expected_random_norm

    caplog.set_level(logging.INFO, logger="ModeExplorer")
    directions, _ = calibrated
    s = directions.s[:4]
    target = expected_random_norm(s, 2.0)
    batch = torch.zeros(3, 16, 1, 8, 12)
    norms = {}
    for mode in ("random", "random_norm_matched", "cohort_logdet"):
        patched, = ModeExplorerReDiverse().apply(model, directions=directions, band=2.0, steering=mode, diagnostics=True)
        draw(patched, latent=batch, batch=3)
        summary = summaries(caplog)[-1]
        alpha = torch.tensor(summary["coefficients"])
        norms[mode] = torch.tensor(summary["tf_offset_norms"])
        assert torch.allclose(norms[mode], activation_norm(alpha, s).float(), rtol=1e-5)
        assert summary["config"]["experimental"] == (mode != "random")
    # Activation-space norms sqrt(sum_k s_k^2 alpha_k^2), not coefficient norms, are matched exactly.
    assert torch.allclose(norms["cohort_logdet"], torch.full((3,), target), rtol=1e-5)
    assert torch.allclose(norms["random_norm_matched"], norms["cohort_logdet"], rtol=1e-5)
    assert not torch.allclose(norms["random"], norms["cohort_logdet"], rtol=1e-3)
    # The uniform draw's RMS activation norm is the matched target.
    from mode_explorer.rediverse import coefficients

    draws = coefficients(5, 4000, 4, 2.0)
    assert float(activation_norm(draws, s).square().mean().sqrt()) == pytest.approx(target, rel=0.03)
    # No silent arm changes: single images and degenerate cohorts are errors.
    patched, = ModeExplorerReDiverse().apply(model, directions=directions, band=2.0, steering="cohort_logdet")
    with pytest.raises(RuntimeError, match="batch of 2 to 8"):
        draw(patched, latent=LATENT)
    assert cohort_coefficients(torch.zeros(2, 24, 64), directions.U.float(), directions.s, 2.0) is None
    assert no_hooks(model)


def _report(text):
    with open(json.loads(text)["path"]) as handle:
        return json.load(handle)


def test_probe_reports_spectra_and_qualified_reference_comparison(model, workspace, calibrated):
    directions, _ = calibrated
    p, _ = conditions()
    probe = ModeExplorerActivationProbe().probe
    report = _report(probe(model, p, {"samples": LATENT}, "0-2", 8, 0, 1.0, 3, "student",
                           reference=directions, workspace="cpu", workspace_mib=1)[0])
    assert set(report["blocks"]) == {"block00", "block01", "block02"}
    entry = report["blocks"]["block02"]
    assert 0 < entry["pr"] <= 7 and entry["conc"] == pytest.approx(entry["tv"] / entry["pr"])
    ref = entry["reference"]
    # Same prompt, but the archive is teacher forward 2 from other noise: an independent population.
    assert ref["comparison"] == report["reference_comparison"] == "same_conditioning_independent_samples"
    assert len(ref["retained_variance_ratio"]) == 4 and 0 <= ref["subspace_alignment"] <= 1 + 1e-9
    assert "mean_displacement_rms" in ref
    other = [[p[0][0] * -1, {}]]
    crossed = _report(probe(model, other, {"samples": LATENT}, "2", 8, 0, 1.0, 8, "crossed", reference=directions)[0])
    assert crossed["reference_comparison"] == "cross_conditioning_exploratory"
    large = {"samples": torch.zeros(1, 16, 1, 40, 40)}
    grouped = _report(probe(model, p, large, "0-2", 8, 0, 1.0, 8, "grouped", workspace="cpu", workspace_mib=4)[0])
    # 409,600 staged bytes per block beside 3,379,200 bytes of float64 Gram chunks and the mean:
    # a 4 MiB workspace holds one staged block per forward group.
    from mode_explorer.rediverse_runtime import research_workspace_bytes

    assert research_workspace_bytes(8, 400 * 64, 0, 0) == 3_379_200
    assert grouped["groups"] == [[0], [1], [2]] and len(grouped["blocks"]) == 3
    with pytest.raises(ValueError, match="research workspace"):
        probe(model, p, large, "0", 8, 0, 1.0, 8, "small", workspace="cpu", workspace_mib=1)
    assert no_hooks(model)


def test_probe_pairs_are_qualified_by_input_evidence(model, workspace):
    from mode_explorer.rediverse_runtime import compare_probes

    p, _ = conditions()
    probe = ModeExplorerActivationProbe().probe
    teacher = _report(probe(model, p, {"samples": LATENT}, "2", 8, 0, 1.0, 4, "teacher")[0])
    # Another batch size executes the same inputs: an exact matched state, ratios exactly one.
    same = _report(probe(model, p, {"samples": LATENT}, "2", 8, 0, 1.0, 8, "student")[0])
    assert same["capture_input_sha256"] == teacher["capture_input_sha256"]
    paired = compare_probes(teacher, same)
    assert paired["comparison"] == "exact_matched_noisy_state" and paired["matched_inputs"]
    assert paired["blocks"]["block02"]["pr_ratio"] == pytest.approx(1.0, rel=1e-6)
    # Equal sigma, samples, prompt and grid but other seeds: never a matched state.
    reseeded = _report(probe(model, p, {"samples": LATENT}, "2", 8, 1, 1.0, 8, "student")[0])
    unmatched = compare_probes(teacher, reseeded)
    assert unmatched["comparison"] == "same_conditioning_independent_samples" and not unmatched["matched_inputs"]
    assert compare_probes(teacher, {**same, "conditioning_sha256": "x"})["comparison"] == "cross_conditioning_exploratory"
    assert compare_probes(teacher, {**same, "capture_input_sha256": "y"})["comparison"] == \
        "matched_noise_state_representation_differs"
    assert compare_probes(teacher, {**same, "token_grid": [2, 3]})["comparison"] == "incompatible_geometry"


def test_probe_archives_have_complete_identities_and_are_never_replaced(model, workspace, calibrated):
    directions, _ = calibrated
    p, _ = conditions()
    probe = ModeExplorerActivationProbe().probe
    base = ("2", 8, 0, 1.0, 4, "id")
    first = json.loads(probe(model, p, {"samples": LATENT}, *base)[0])["path"]
    variants = {
        "seed": ("2", 8, 1, 1.0, 4, "id"), "blocks": ("1-2", 8, 0, 1.0, 4, "id"), "batch": ("2", 8, 0, 1.0, 8, "id"),
    }
    paths = {first}
    for name, args in variants.items():
        path = json.loads(probe(model, p, {"samples": LATENT}, *args)[0])["path"]
        assert path not in paths, name
        paths.add(path)
    referenced = json.loads(probe(model, p, {"samples": LATENT}, *base, reference=directions)[0])["path"]
    guided = json.loads(probe(ModeExplorer().apply(model, mode="Early Structural (B1)", strength=.03)[0],
                              p, {"samples": LATENT}, *base)[0])["path"]
    repeated = json.loads(probe(model, p, {"samples": LATENT}, *base)[0])["path"]
    assert len({*paths, referenced, guided, repeated}) == len(paths) + 3
    assert repeated == first.replace(".json", "-r2.json") and os.path.isfile(first)
    with pytest.raises(ValueError, match="sigma 1"):
        probe(rediverse(model), p, {"samples": LATENT}, "2", 8, 0, 0.9, 4, "adapted")


def test_reference_turbo_schedule_equals_native_simple_scheduler(model):
    from mode_explorer.rediverse_nodes import ModeExplorerReferenceSigmas

    sigmas, = ModeExplorerReferenceSigmas().sigmas({"samples": torch.zeros(1, 16, 1, 128, 128)}, "turbo", 8)
    native = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), "simple", 8)
    assert torch.allclose(sigmas, native.float(), atol=1e-6)
    raw, = ModeExplorerReferenceSigmas().sigmas({"samples": torch.zeros(1, 16, 1, 128, 128)}, "raw", 52)
    assert raw.shape == (53,) and float(raw[4]) == pytest.approx(0.96743, abs=5e-5)


def test_probe_of_adapted_student_measures_the_adapted_first_step(model, workspace):
    p, _ = conditions()
    reports = []
    for candidate in (model, rediverse(model)):
        text, = ModeExplorerActivationProbe().probe(candidate, p, {"samples": LATENT}, "1", 6, 0, 1.0, 6, "ft")
        with open(json.loads(text)["path"]) as handle:
            reports.append(json.load(handle)["blocks"]["block01"])
    # Block 1 output depends on the block 0-1 adapter, so the spectra differ.
    assert reports[0]["tv"] != pytest.approx(reports[1]["tv"], rel=1e-6)
    assert no_hooks(model)


def test_adapter_requires_module_calls_and_restores_force_forward(model, workspace, monkeypatch):
    import torch.nn.functional as F

    mlp = model.get_model_object("diffusion_model").blocks[0].mlp
    seen = []

    def fused(x):
        # A fused path that reads the weight directly would skip the module hook.
        seen.append(getattr(mlp.down, "comfy_force_forward", False))
        return F.linear(F.silu(mlp.gate(x)) * mlp.up(x), mlp.down.weight)

    monkeypatch.setattr(mlp, "forward", fused)
    with pytest.raises(RuntimeError, match="0.mlp.down"):
        draw(rediverse(model))
    # During the adapted evaluation the module-level forward was requested, then restored.
    assert seen[0] is True and "comfy_force_forward" not in mlp.down.__dict__
    assert no_hooks(model)
    monkeypatch.undo()
    assert torch.equal(draw(rediverse(model, strength=0.0)), draw(model))


def test_tf_rejects_multiple_positive_conditionings(model, workspace, calibrated):
    directions, _ = calibrated
    steered = rediverse(model, directions=directions, band=2.0)
    p, n = conditions()
    guider = comfy.samplers.CFGGuider(steered)
    guider.set_conds(p + [[p[0][0] * 2, {}]], n)
    guider.set_cfg(1.0)
    # A combined FT+TF arm never degrades silently into FT alone or the native model.
    with pytest.raises(RuntimeError, match="exactly one plain positive"):
        with torch.inference_mode():
            guider.sample(comfy.sample.prepare_noise(LATENT, 1), LATENT, comfy.samplers.ksampler("euler"), SIGMAS,
                          disable_pbar=True, seed=1)
    assert no_hooks(model)
    with pytest.raises(ValueError, match="exactly one positive"):
        ModeExplorerReDiverseCalibrate().calibrate(model, p + p, {"samples": LATENT}, 2, 2, 6, 6, 4, 3, 4, "")
    # The teacher must be the plain model.
    with pytest.raises(ValueError, match="plain teacher"):
        ModeExplorerReDiverseCalibrate().calibrate(rediverse(model), p, {"samples": LATENT}, 2, 2, 6, 6, 4, 3, 4, "")
    from mode_explorer.rediverse_runtime import start_run

    with pytest.raises(RuntimeError, match="integer noise seed"):
        start_run(steered.get_wrappers("sampler_sample", KEY)[0].rediverse, None, 1, None, SIGMAS)


def test_actual_forward_with_no_adapter_execution_fails(model, workspace, monkeypatch):
    """Every hooked module bypassed (e.g. a fused path) on an actual forward: an error, not a forecast."""
    import mode_explorer.rediverse_runtime as runtime

    detached = {}

    def unexecuted(block, name):
        layer = resolve_module(block, name)
        key = (id(block), name)
        if key not in detached:
            detached[key] = torch.nn.Linear(layer.in_features, layer.out_features, bias=False)
        return detached[key]

    monkeypatch.setattr(runtime, "resolve_module", unexecuted)
    with pytest.raises(RuntimeError, match="not executed through their module call"):
        draw(rediverse(model))
    assert no_hooks(model)


def test_live_omission_is_not_masked_by_auxiliary_forwards(model, workspace, monkeypatch):
    """A reference forward that runs the adapter cannot certify a live forward that skipped it."""
    import torch.nn.functional as F

    from mode_explorer.rediverse_runtime import LIVE

    mlp = model.get_model_object("diffusion_model").blocks[0].mlp
    native = type(mlp).forward

    def live_only_fused(x):
        if LIVE.get() is not None:
            return F.linear(F.silu(mlp.gate(x)) * mlp.up(x), mlp.down.weight)
        return native(mlp, x)

    monkeypatch.setattr(mlp, "forward", live_only_fused)
    combined, = ModeExplorer().apply(rediverse(model), mode="Early Structural (B1)", strength=.03, fidelity=.7)
    with pytest.raises(RuntimeError, match="0.mlp.down"):
        draw(combined)
    assert no_hooks(model)


def test_overlapping_scopes_are_refused_and_hooks_respect_ownership(caplog, model, workspace):
    import contextvars

    import mode_explorer.rediverse_runtime as runtime

    dit = model.get_model_object("diffusion_model")
    runtime._ACTIVE[id(dit)] = object()
    try:
        with pytest.raises(RuntimeError, match="Overlapping execution"):
            draw(rediverse(model))
    finally:
        runtime._ACTIVE.pop(id(dit), None)
    assert no_hooks(model)
    layer = dit.blocks[0].attn.wq
    probe_input = torch.randn(1, 3, layer.in_features)
    seen = {}

    def during(module, args):
        # First forward only: the adapted one. A foreign execution context sees the native layer.
        if not seen:
            seen["foreign"] = contextvars.Context().run(layer, probe_input)
            seen["owner"] = layer(probe_input)

    handle = dit.blocks[1].register_forward_pre_hook(during)
    try:
        draw(rediverse(model))
    finally:
        handle.remove()
    native = torch.nn.functional.linear(probe_input, layer.weight, layer.bias)
    assert torch.equal(seen["foreign"], native) and not torch.allclose(seen["owner"], native)
    assert no_hooks(model)


def _live_rows(model, positive, seeds, sigma=1.0, block=1):
    """Block output image rows of the live sample, from ordinary single-image sampling with these probe seeds."""
    from mode_explorer.rediverse_runtime import _noise_batch

    dit = model.get_model_object("diffusion_model")
    rows = []
    for seed in seeds:
        captured = []
        handle = dit.blocks[block].register_forward_hook(
            lambda m, a, k, out: captured.append(out[:, -24:].reshape(out.shape[0], -1).clone()), with_kwargs=True)
        try:
            noise = _noise_batch([seed], (16, 1, 8, 12))
            guider = comfy.samplers.CFGGuider(model)
            guider.inner_set_conds({"positive": positive})
            guider.set_cfg(1.0)
            with torch.inference_mode():
                guider.sample(noise, torch.zeros_like(noise), comfy.samplers.ksampler("euler"),
                              torch.tensor([sigma, 0.0]), disable_pbar=True, seed=seed)
        finally:
            handle.remove()
        rows.append(captured[-1][0])  # the last forward is the live one; references run first
    return torch.stack(rows)


@pytest.mark.parametrize("arm,batch", [("native", 1), ("b1", 1), ("b1", 2), ("ft_b1", 1)])
def test_probe_captures_live_samples_not_auxiliary_references(model, workspace, arm, batch):
    """ModeExplorer reference forwards must never be captured, counted or fingerprinted by a probe."""
    from mode_explorer import covariance

    probed = model
    if arm == "ft_b1":
        probed = rediverse(model)
    if arm in ("b1", "ft_b1"):
        probed, = ModeExplorer().apply(probed, mode="Early Structural (B1)", strength=.03, fidelity=.7)
    p, _ = conditions()
    samples = 3  # with batch 2 the final chunk is a single image
    report = _report(ModeExplorerActivationProbe().probe(probed, p, {"samples": LATENT}, "1", samples, 0, 1.0,
                                                         batch, f"live-{arm}-{batch}")[0])
    seeds = [1_000_000 + i for i in range(samples)]
    if batch == 1:
        expected = _live_rows(probed, p, seeds)
        values, _ = covariance.eigenpairs(covariance.centred_gram(expected.to(torch.bfloat16)))
        assert report["blocks"]["block01"]["tv"] == pytest.approx(covariance.spectrum_stats(values, samples)["tv"], rel=1e-5)
    native = _report(ModeExplorerActivationProbe().probe(model, p, {"samples": LATENT}, "1", samples, 0, 1.0, batch,
                                                         f"native-{arm}-{batch}")[0])
    # The executed live inputs are the same noisy states as the native probe's.
    assert report["capture_input_sha256"] == native["capture_input_sha256"]
    if arm != "native":
        assert report["blocks"]["block01"]["tv"] != pytest.approx(native["blocks"]["block01"]["tv"], rel=1e-6)
    assert no_hooks(model)


def test_truncated_probe_forwards_verify_the_adapter_prefix(model, workspace, monkeypatch):
    """A probe stopping at block 1 must still prove that blocks 0-1 ran every adapted linear."""
    import torch.nn.functional as F

    mlp = model.get_model_object("diffusion_model").blocks[0].mlp
    monkeypatch.setattr(mlp, "forward", lambda x: F.linear(F.silu(mlp.gate(x)) * mlp.up(x), mlp.down.weight))
    p, _ = conditions()
    with pytest.raises(RuntimeError, match="0.mlp.down"):
        ModeExplorerActivationProbe().probe(rediverse(model), p, {"samples": LATENT}, "1", 3, 0, 1.0, 3, "prefix")
    monkeypatch.undo()
    # Unadapted later blocks are not required before an intentional stop.
    ModeExplorerActivationProbe().probe(rediverse(model), p, {"samples": LATENT}, "0", 3, 0, 1.0, 3, "prefix")
    assert no_hooks(model)


def test_admitted_intervention_without_a_live_forward_is_an_error(caplog, model, workspace):
    """A path that replaces the admitted evaluation (as a forecast would) must not consume the treatment."""
    from comfy.patcher_extension import WrappersMP

    from mode_explorer.rediverse_runtime import CAPTURE_KEY

    def skipping(executor, x, *args, **kwargs):
        options = args[4] if len(args) > 4 else kwargs.get("transformer_options", {})
        sigmas = options.get("sigmas")
        if sigmas is not None and float(sigmas.reshape(-1)[0]) == 1.0:
            return torch.zeros_like(x)  # no transformer forward
        return executor(x, *args, **kwargs)

    def skipped(patched):
        clone = patched.clone()
        clone.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, CAPTURE_KEY, skipping)
        return clone

    with pytest.raises(RuntimeError, match="ran no live transformer forward"):
        draw(skipped(rediverse(model)))
    caplog.set_level(logging.INFO, logger="ModeExplorer")
    # The strength-0 route sham requests no intervention: recorded, not an error.
    draw(skipped(rediverse(model, strength=0.0, diagnostics=True)))
    summary, = summaries(caplog)
    assert summary["forecast_consumed"] == 1 and "admitted_evaluation_not_executed" in summary["reasons"]
    assert no_hooks(model)


def test_tf_reports_realized_per_sample_offsets(caplog, model, workspace, calibrated):
    caplog.set_level(logging.INFO, logger="ModeExplorer")
    directions, _ = calibrated
    steered, = ModeExplorerReDiverse().apply(model, directions=directions, band=2.0, diagnostics=True)
    draw(steered, latent=torch.zeros(2, 16, 1, 8, 12), batch=2, cfg=2.0)
    summary = summaries(caplog)[-1]
    realized, = summary["tf_realized"]
    assert realized["rows"] == 4 and len(realized["realized_offset_norms"]) == 4
    intended = torch.tensor(summary["tf_offset_norms"])
    # Both CFG branches of a sample get the same offset; bf16 basis and native rounding stay within 1 %.
    by_branch = torch.tensor(realized["realized_offset_norms"]).reshape(2, 2)
    assert torch.allclose(by_branch, intended.expand(2, 2), rtol=1e-2)


# ------------------------------------------------------------------ temporal FT --
@torch.inference_mode()
def stepwise_reference(model, tensors, strengths, sigmas=SIGMAS, seed=55, cfg=1.0):
    """Independent trajectory: Euler step i runs a model with the adapter merged at strengths[i] (native for 0)."""
    x = torch.zeros(1, 16, 1, 7, 11)
    noise = comfy.sample.prepare_noise(x, seed)
    p, n = conditions()
    state = None
    for i in range(len(sigmas) - 1):
        stage = merged(model, tensors, strengths[i]) if strengths[i] else model
        guider = comfy.samplers.CFGGuider(stage)
        guider.set_conds(p, n)
        guider.set_cfg(cfg)
        start = noise if state is None else state / sigmas[i]
        out = guider.sample(start, x, comfy.samplers.ksampler("euler"), sigmas[i:i + 2], disable_pbar=True, seed=seed)
        last = float(sigmas[i + 1])
        # The guider returns process_latent_out(x / (1 - sigma_last)) for CONST flow.
        state = out if last == 0 else stage.model.process_latent_in(out) * (1 - last)
    return state


@pytest.mark.parametrize("kwargs,strengths", [
    (dict(adapted_sigmas=3), [1.0, 1.0, 1.0, 0, 0]),
    (dict(adapted_sigmas=4, strength=.5), [.5, .5, .5, .5, 0]),
    (dict(adapted_sigmas=3, start_evaluation=1, strength_profile="custom", custom_profile="0.5, 0, 1.25"), [0, .5, 0, 1.25, 0]),
    (dict(adapted_sigmas=3, strength_profile="linear_increase"), [0, .5, 1.0, 0, 0]),
    (dict(adapted_sigmas=5, strength=.8, strength_profile="cosine_decay"),
     [.8 * (1 + math.cos(math.pi * i / 4)) / 2 for i in range(5)]),
    (dict(adapted_sigmas=2, start_evaluation=3), [0, 0, 0, 1.0, 1.0]),
])
def test_temporal_schedule_matches_independent_per_step_merged_reference(caplog, model, workspace, kwargs, strengths):
    caplog.set_level(logging.INFO, logger="ModeExplorer")
    state = {k: v.clone() for k, v in model.model.state_dict().items()}
    reference = stepwise_reference(model, workspace, strengths)
    result = draw(rediverse(model, diagnostics=True, **kwargs))
    assert torch.allclose(result, reference, atol=2e-5), (result - reference).abs().max()
    temporal = summaries(caplog)[-1]["temporal"]
    start = kwargs.get("start_evaluation", 0)
    slots = temporal["slots"]
    assert [slot["evaluation"] for slot in slots] == list(range(start, start + kwargs["adapted_sigmas"]))
    assert [slot["effective_strength"] for slot in slots] == pytest.approx(strengths[start:start + len(slots)])
    for slot in slots:
        active = slot["effective_strength"] != 0
        assert slot["state"] == ("adapter_active" if active else "zero_strength")
        assert slot["live_forwards"] == 1 and slot["auxiliary_forwards"] == 0
        assert slot["adapter_module_calls"] == (2 * len(COMFY_MODULES) if active else 0)
        assert slot["sigma"] == pytest.approx(float(SIGMAS[slot["evaluation"]]))
    assert temporal["adapter_active_evaluations"] == sum(1 for v in strengths if v)
    assert temporal["native_before_window"] == start
    assert all(torch.equal(v, state[k]) for k, v in model.model.state_dict().items())
    assert no_hooks(model)


def test_legacy_constant_configurations_are_unchanged(caplog, model, workspace):
    """Default controls reproduce the reference recipe; explicit defaults equal omitted fields."""
    legacy = draw(rediverse(model))
    explicit = draw(rediverse(model, start_evaluation=0, strength_profile="constant", custom_profile="", adapted_sigmas=1))
    assert torch.equal(legacy, explicit)
    assert torch.allclose(legacy, split_reference(merged(model, workspace), model), atol=2e-5)
    two = draw(rediverse(model, adapted_sigmas=2))
    assert torch.equal(two, draw(rediverse(model, adapted_sigmas=2, strength_profile="custom", custom_profile="1, 1")))
    # Custom text is ignored by built-in profiles.
    assert torch.equal(two, draw(rediverse(model, adapted_sigmas=2, custom_profile="0.1, 0.2")))
    caplog.set_level(logging.INFO, logger="ModeExplorer")
    draw(rediverse(model, diagnostics=True))
    summary = summaries(caplog)[-1]
    assert summary["admitted_sigmas"] == [1.0] and summary["adapted_evaluations"] == 1
    assert summary["config"]["experimental"] is False and summary["temporal"]["planned_strengths"] == [1.0]


def test_zero_strength_temporal_sham_is_bitwise_native(model, workspace):
    for kwargs in (dict(adapted_sigmas=4, start_evaluation=1), dict(adapted_sigmas=3, strength_profile="cosine_decay")):
        assert torch.equal(draw(rediverse(model, strength=0.0, **kwargs)), draw(model))
    # Zero slots of a nonzero profile run no adapter arithmetic.
    assert torch.equal(draw(rediverse(model, adapted_sigmas=2, strength_profile="custom", custom_profile="0, 0")), draw(model))


def test_truncated_and_unreachable_windows_are_reported(caplog, model, workspace):
    caplog.set_level(logging.INFO, logger="ModeExplorer")
    draw(rediverse(model, diagnostics=True, adapted_sigmas=8, strength=.5))
    temporal = summaries(caplog)[-1]["temporal"]
    assert temporal["window_slots_reached"] == 5 and temporal["unreached_window_slots"] == [5, 6, 7]
    assert "temporal_window_truncated" in summaries(caplog)[-1]["reasons"]
    assert any("temporal_window_truncated" in r.message for r in caplog.records if "ReDiverse:" in r.message)
    caplog.clear()
    result = draw(rediverse(model, diagnostics=True, adapted_sigmas=2, start_evaluation=6))
    summary = summaries(caplog)[-1]
    assert summary["temporal"]["window_slots_reached"] == 0 and "temporal_window_not_reached" in summary["reasons"]
    assert summary["temporal"]["slots"] == [] and torch.equal(result, draw(model))
    caplog.clear()
    draw(rediverse(model, diagnostics=True, adapted_sigmas=2, start_evaluation=2))
    temporal = summaries(caplog)[-1]["temporal"]
    assert temporal["window_entry_sigma"] == pytest.approx(.83) and temporal["window_exit_sigma"] == pytest.approx(.2)
    assert summaries(caplog)[-1]["reasons"] == []


def test_auxiliary_forwards_inherit_their_evaluation_strength(caplog, model, workspace, monkeypatch):
    """ModeExplorer references use the live evaluation's effective strength and never advance the window."""
    import mode_explorer.rediverse_runtime as runtime

    used = []
    original = runtime._lora_hook

    def recording(run, down, up, scale):
        hook = original(run, down, up, scale)

        def wrapped(module, args, output):
            if runtime.SCOPE.get() is run:
                used.append((len(run.slots) - 1, scale, runtime.AUXILIARY.get()))
            return hook(module, args, output)
        return wrapped

    monkeypatch.setattr(runtime, "_lora_hook", recording)
    caplog.set_level(logging.INFO, logger="ModeExplorer")
    ft = rediverse(model, adapted_sigmas=3, strength_profile="custom", custom_profile="1, 0.5, 0.25", diagnostics=True)
    combined, = ModeExplorer().apply(ft, mode="Early Structural (B1)", strength=.03, fidelity=.7, reference_budget=2,
                                     diagnostics=True)
    draw(combined)
    trace = traces(caplog)[-1]
    slots = trace["rediverse"]["temporal"]["slots"]
    assert trace["feature_reference_dit_calls"] == 2
    assert [s["evaluation"] for s in slots] == [0, 1, 2]
    for slot in slots:
        assert slot["adapter_module_calls"] == 2 * len(COMFY_MODULES) * (slot["live_forwards"] + slot["auxiliary_forwards"])
    assert sum(s["auxiliary_forwards"] for s in slots) == 2 and all(s["live_forwards"] == 1 for s in slots)
    by_slot = {}
    for slot, scale, auxiliary in used:
        by_slot.setdefault(slot, set()).add(scale)
    assert by_slot == {0: {1.0}, 1: {0.5}, 2: {0.25}}
    assert {auxiliary for *_, auxiliary in used} == {False, True}
    assert no_hooks(model)


def test_split_cfg_and_multistage_solvers_share_slots_per_sigma(caplog, model, workspace):
    caplog.set_level(logging.INFO, logger="ModeExplorer")
    draw(rediverse(model, adapted_sigmas=2, diagnostics=True), cfg=2.0, split_cfg=True)
    slots = summaries(caplog)[-1]["temporal"]["slots"]
    assert [(s["evaluation"], s["calls"], s["live_forwards"]) for s in slots] == [(0, 2, 2), (1, 2, 2)]
    caplog.clear()
    # Heun: the corrector at sigma 1 and the next predictor at the same sigma share slot 1.
    draw(rediverse(model, adapted_sigmas=2, diagnostics=True), sampler="heun")
    slots = summaries(caplog)[-1]["temporal"]["slots"]
    assert [(s["evaluation"], s["calls"]) for s in slots] == [(0, 1), (1, 2)]
    assert [s["sigma"] for s in slots] == pytest.approx([1.0, .93])
    caplog.clear()
    draw(rediverse(model, adapted_sigmas=3, start_evaluation=1, diagnostics=True), sampler="er_sde")
    temporal = summaries(caplog)[-1]["temporal"]
    assert temporal["evaluation_sigmas"][0] < 1.0 and [s["evaluation"] for s in temporal["slots"]] == [1, 2, 3]


def test_temporal_lifecycle_has_no_stale_state(caplog, model, workspace):
    dit = model.get_model_object("diffusion_model")
    calls = []

    def boom(module, args, output):
        calls.append(1)
        if len(calls) == 3:
            raise RuntimeError("cancelled")

    patched = rediverse(model, adapted_sigmas=4, strength_profile="linear_decay")
    handle = dit.blocks[2].register_forward_hook(boom)
    try:
        with pytest.raises(RuntimeError, match="cancelled"):
            draw(patched)  # stops inside the third adapted evaluation
    finally:
        handle.remove()
    assert no_hooks(model) and all("comfy_force_forward" not in m.__dict__ for m in dit.modules())
    caplog.set_level(logging.INFO, logger="ModeExplorer")
    patched = rediverse(model, adapted_sigmas=4, strength_profile="linear_decay", diagnostics=True)
    first, second = draw(patched), draw(patched)
    assert torch.equal(first, second)
    one, two = summaries(caplog)[-2:]
    assert one["temporal"] == two["temporal"] and [s["evaluation"] for s in one["temporal"]["slots"]] == [0, 1, 2, 3]
