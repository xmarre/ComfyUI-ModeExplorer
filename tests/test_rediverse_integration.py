"""Native ComfyUI contracts of ReDiverse on a tiny real Krea 2 transformer.

These tests establish scheduling, isolation and numerical equivalences. They are not
evidence of image quality or diversity on real checkpoints.
"""

import copy
import json
import logging
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
    assert json.loads(text)["path"] == summary["path"]  # identical calibration identity, one archive
    other, text = ModeExplorerReDiverseCalibrate().calibrate(
        model, p, {"samples": LATENT}, block=2, teacher_forward=2, teacher_steps=6, rollouts=6, directions=4,
        seed=4, batch=4, label="tiny")
    assert json.loads(text)["path"] != summary["path"] and not torch.equal(other.U, directions.U)
    assert torch.allclose(directions.U.float().norm(dim=1), torch.ones(4), atol=1e-2)
    assert os.path.isfile(summary["path"])
    assert no_hooks(model)
    assert all(torch.equal(v, state[k]) for k, v in model.model.state_dict().items())
    loaded, text = ModeExplorerReDiverseDirections().load(os.path.basename(summary["path"]))
    assert json.loads(text)["sha256"] == again.sha256 and torch.equal(loaded.U, directions.U)
    with pytest.raises(ValueError):
        ModeExplorerReDiverseDirections().load("../escape.safetensors")


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


def test_cohort_logdet_is_bounded_and_falls_back_for_single_images(caplog, model, workspace, calibrated):
    from mode_explorer.rediverse_runtime import cohort_coefficients

    caplog.set_level(logging.INFO, logger="ModeExplorer")
    directions, _ = calibrated
    experimental, = ModeExplorerReDiverse().apply(model, directions=directions, band=2.0, steering="cohort_logdet",
                                                  diagnostics=True)
    draw(experimental, latent=torch.zeros(3, 16, 1, 8, 12), batch=3)
    summary = summaries(caplog)[-1]
    alpha = torch.tensor(summary["coefficients"])
    assert alpha.shape == (3, 4) and alpha.abs().max() <= 2.0 + 1e-6
    assert summary["config"]["experimental"]
    draw(experimental, latent=LATENT)
    assert "tf_cohort_requires_batch_random_fallback" in summaries(caplog)[-1]["reasons"]
    hidden = torch.zeros(2, 24, 64)
    assert cohort_coefficients(hidden, directions.U.float(), directions.s, 2.0) is None


def test_probe_reports_spectra_and_reference_comparison(model, workspace, calibrated):
    directions, _ = calibrated
    p, _ = conditions()
    text, = ModeExplorerActivationProbe().probe(model, p, {"samples": LATENT}, "0-2", 8, 0, 1.0, 3, "student",
                                                reference=directions, workspace="cpu", workspace_mib=1)
    summary = json.loads(text)
    with open(summary["path"]) as handle:
        report = json.load(handle)
    assert set(report["blocks"]) == {"block00", "block01", "block02"}
    entry = report["blocks"]["block02"]
    assert 0 < entry["pr"] <= 7 and entry["conc"] == pytest.approx(entry["tv"] / entry["pr"])
    ref = entry["reference"]
    assert ref["coordinate_match"] == "native_trajectory_states" and len(ref["retained_variance_ratio"]) == 4
    assert 0 <= ref["subspace_alignment"] <= 1 + 1e-9
    assert "mean_displacement_rms" in ref
    teacher_text, = ModeExplorerActivationProbe().probe(model, p, {"samples": LATENT}, "2", 8, 0, 1.0, 4, "teacher")
    teacher = json.loads(teacher_text)
    compared, = ModeExplorerActivationProbe().probe(model, p, {"samples": LATENT}, "2", 8, 0, 1.0, 8, "again",
                                                    compare_with=os.path.basename(teacher["path"]))
    ratios = json.loads(compared)["comparison"]
    large = {"samples": torch.zeros(1, 16, 1, 40, 40)}
    grouped, = ModeExplorerActivationProbe().probe(model, p, large, "0-2", 8, 0, 1.0, 8, "grouped",
                                                   workspace="cpu", workspace_mib=1)
    with open(json.loads(grouped)["path"]) as handle:
        grouped_report = json.load(handle)
    # 409,600 staged bytes per block: a 1 MiB workspace holds two blocks per forward group.
    assert grouped_report["groups"] == [[0, 1], [2]] and len(grouped_report["blocks"]) == 3
    # Same model, inputs and batch-independent capture: ratios are exactly one.
    assert ratios["matched_inputs"] and ratios["blocks"]["block02"]["pr_ratio"] == pytest.approx(1.0, rel=1e-6)
    assert no_hooks(model)


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


def test_tf_rejects_multiple_positive_conditionings(caplog, model, workspace, calibrated):
    directions, _ = calibrated
    steered, = ModeExplorerReDiverse().apply(model, directions=directions, band=2.0)
    p, n = conditions()
    guider = comfy.samplers.CFGGuider(steered)
    guider.set_conds(p + [[p[0][0] * 2, {}]], n)
    guider.set_cfg(1.0)
    caplog.set_level(logging.INFO, logger="ModeExplorer")
    with torch.inference_mode():
        guider.sample(comfy.sample.prepare_noise(LATENT, 1), LATENT, comfy.samplers.ksampler("euler"), SIGMAS,
                      disable_pbar=True, seed=1)
    assert any("tf_unsupported_conditioning" in r.message for r in caplog.records)
    with pytest.raises(ValueError, match="exactly one positive"):
        ModeExplorerReDiverseCalibrate().calibrate(model, p + p, {"samples": LATENT}, 2, 2, 6, 6, 4, 3, 4, "")
    # The teacher must be the plain model.
    with pytest.raises(ValueError, match="plain teacher"):
        ModeExplorerReDiverseCalibrate().calibrate(rediverse(model), p, {"samples": LATENT}, 2, 2, 6, 6, 4, 3, 4, "")
