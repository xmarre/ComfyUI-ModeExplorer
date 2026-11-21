"""Native ComfyUI smoke tests using the real Krea 2 implementation at tiny width."""

import asyncio
import json
import logging
from pathlib import Path

import pytest
import torch

comfy = pytest.importorskip("comfy")
from comfy.cli_args import args
args.cpu = True

import comfy.model_management
import comfy.model_patcher
import comfy.sample
import comfy.samplers
import comfy.supported_models
import nodes
from comfy_extras.nodes_custom_sampler import Guider_Basic

from mode_explorer.nodes import ModeExplorer
from mode_explorer.runtime import KEY, RUN, STAGE


@pytest.fixture
def model():
    torch.set_num_threads(1)
    device = torch.device("cpu")
    config = comfy.supported_models.Krea2({
        "image_model": "krea2", "features": 64, "tdim": 32, "txtdim": 32,
        "heads": 2, "kvheads": 1, "multiplier": 2, "layers": 3, "patch": 2,
        "channels": 16, "txtlayers": 2, "txtheads": 2, "txtkvheads": 1,
        "default_ref_method": "index_timestep_zero", "dtype": torch.float32,
    })
    base = config.get_model({}, device=device)
    generator = torch.Generator().manual_seed(4096)
    with torch.no_grad():
        for key, p in base.diffusion_model.named_parameters():
            std = 0.18 if key.endswith("mod.lin") else 0.08
            p.copy_(torch.randn(p.shape, generator=generator) * std)
    return comfy.model_patcher.ModelPatcher(base, device, device)


def conditions(batch=1, *, mixed=False, reference=False, split_cfg=False):
    context = torch.randn(1, 5, 64, generator=torch.Generator().manual_seed(74))
    context = torch.cat((context, context * -1), dim=0) if mixed else context
    meta = {"reference_latents": [torch.ones(1, 16, 1, 4, 6)]} if reference else {}
    negative = torch.zeros_like(context[:, :3] if split_cfg else context)
    return [[context, meta]], [[negative, {}]]


def schedule():
    return torch.tensor([1.0, 0.93, 0.83, 0.62, 0.2, 0.0])


@torch.inference_mode()
def draw(model, seed=55, *, batch=1, cfg=1.0, sigmas=None, mixed=False, reference=False, stage=None,
         zero_noise=False, latent=None, mask=None, guider_basic=False, split_cfg=False, sampler="euler"):
    x = torch.zeros(batch, 16, 1, 7, 11) if latent is None else latent
    p, n = conditions(batch, mixed=mixed, reference=reference, split_cfg=split_cfg)
    guider = Guider_Basic(model) if guider_basic else comfy.samplers.CFGGuider(model)
    if guider_basic:
        guider.set_conds(p)
    else:
        guider.set_conds(p, n)
        guider.set_cfg(cfg)
    if stage:
        guider.model_options = {**guider.model_options, STAGE: stage}
    noise = torch.zeros_like(x) if zero_noise else comfy.sample.prepare_noise(x, seed)
    return guider.sample(noise, x, comfy.samplers.ksampler(sampler), schedule() if sigmas is None else sigmas,
                         denoise_mask=mask, disable_pbar=True, seed=seed)


def traces(caplog):
    return [json.loads(r.message.partition("MODEEXPLORER_TRACE ")[2])
            for r in caplog.records if "MODEEXPLORER_TRACE " in r.message]


def test_discovery_and_clone_ownership(model):
    path = str(Path(__file__).resolve().parents[1])
    assert asyncio.run(nodes.load_custom_node(path))
    assert nodes.NODE_CLASS_MAPPINGS["ModeExplorerKrea2"].RETURN_TYPES == ("MODEL",)
    node = ModeExplorer()
    assert node.INPUT_TYPES()["required"]["model"] == ("MODEL",)
    assert node.INPUT_TYPES()["required"]["full_spectrum"][1]["default"] is False
    assert node.apply(model, enabled=False)[0] is model
    assert node.apply(model, feature="native")[0] is model
    patched, = node.apply(model)
    reattached, = node.apply(patched, feature="feature_volume")
    disabled, = node.apply(reattached, enabled=False)
    assert not model.wrappers
    assert all(len(by_key[KEY]) == 1 for by_key in reattached.wrappers.values())
    assert all(KEY not in by_key for by_key in disabled.wrappers.values())
    assert run_key_absent(patched)
    assert torch.equal(draw(model), draw(disabled))


def test_milli_strength_widget_and_runtime(model, caplog):
    from comfy.patcher_extension import WrappersMP

    node = ModeExplorer()
    kind, spec = node.INPUT_TYPES()["required"]["strength"]
    assert kind == "FLOAT"
    assert spec == {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.001}

    caplog.set_level(logging.INFO, logger="ModeExplorer")
    shifts = []
    for strength in (0.022, 0.024):
        patched, = node.apply(model, strength=strength, adaptive=False, diagnostics=True)
        sampler, = patched.get_wrappers(WrappersMP.SAMPLER_SAMPLE, KEY)
        assert sampler.settings.strength == strength
        draw(patched)
        points = [p for p in traces(caplog)[-1]["points"] if "shift_rms" in p]
        assert points[0]["edit_applied"]
        shifts.append(points[0]["shift_rms"][0])
    assert 0 < shifts[0] < shifts[1]


def run_key_absent(model):
    return RUN not in model.model_options.get("transformer_options", {})


def test_native_trace_parity_and_native_seed(caplog, model):
    caplog.set_level(logging.INFO, logger="ModeExplorer")
    observed, = ModeExplorer().apply(model, feature="native", diagnostics=True)
    first, second = draw(model), draw(model, 56)
    assert not torch.equal(first, second)
    assert torch.equal(first, draw(observed))
    assert traces(caplog)[-1]["native_dit_calls"] == 5
    assert len(traces(caplog)[-1]["points"]) == 6
    state = torch.random.get_rng_state()
    latent = torch.zeros(3, 16, 1, 7, 11)
    noise = comfy.sample.prepare_noise(latent, 7)
    different = comfy.sample.prepare_noise(latent, 8)
    repeated = comfy.sample.prepare_noise(latent, 7, [0, 0, 1])
    assert not torch.equal(noise, different)
    assert torch.equal(noise, comfy.sample.prepare_noise(latent, 7))
    assert torch.equal(repeated[0], repeated[1]) and not torch.equal(repeated[1], repeated[2])
    torch.random.set_rng_state(state)


@pytest.mark.parametrize("feature", ["feature_cosine", "feature_volume"])
@pytest.mark.parametrize("cfg", [1.0, 3.5])
def test_single_native_flow_changes_is_reproducible_and_bounded(caplog, model, cfg, feature):
    caplog.set_level(logging.INFO, logger="ModeExplorer")
    patched, = ModeExplorer().apply(model, feature=feature, adaptive=False, diagnostics=True)
    native = draw(model, cfg=cfg, reference=True)
    first = draw(patched, cfg=cfg, reference=True)
    other = draw(patched, 56, cfg=cfg, reference=True)
    again = draw(patched, cfg=cfg, reference=True)
    assert torch.equal(first, again)
    assert not torch.equal(first, native) and not torch.equal(other, first)
    stats = traces(caplog)[-1]
    assert stats["feature_attempts"] == stats["feature_reference_dit_calls"] == 2
    assert stats["initialization_dit_calls"] == 0
    points = [p for p in stats["points"] if "shift_rms" in p]
    assert len(points) == 2 and min(p["sigma"] for p in points) >= 0.82
    assert torch.equal(native, draw(model, cfg=cfg, reference=True))
    assert run_key_absent(patched)


def test_causal_direction_controls_native_sampling(caplog, model):
    caplog.set_level(logging.INFO, logger="ModeExplorer")
    outputs = {}
    reports = {}
    for direction, sham in (("repel", False), ("reverse", False), ("channel_sign", False), ("repel", True)):
        patched, = ModeExplorer().apply(model, adaptive=False, diagnostics=True,
                                         strength=0.2, feature_direction=direction, sham=sham)
        key = "sham" if sham else direction
        outputs[key] = draw(patched)
        reports[key] = traces(caplog)[-1]
        assert reports[key]["feature_direction"] == direction
        assert reports[key]["feature_reference_dit_calls"] == 2
        assert reports[key]["native_dit_calls"] == 5
        assert torch.equal(outputs[key], draw(patched))
    assert torch.equal(outputs["sham"], draw(model))
    assert not torch.equal(outputs["repel"], outputs["reverse"])
    assert not torch.equal(outputs["repel"], outputs["channel_sign"])
    first = {key: next(p for p in report["points"] if "shift_rms" in p) for key, report in reports.items()}
    assert first["repel"]["shift_rms"] == first["reverse"]["shift_rms"] == first["channel_sign"]["shift_rms"]
    assert first["sham"]["applied_pooled_shift_rms"] == [0.0]
    assert first["repel"]["reference_slots"] == first["channel_sign"]["reference_slots"]
    assert first["repel"]["nearest_cosine"] == first["reverse"]["nearest_cosine"]


def test_same_condition_batch_guidance_and_distinct_prompt_isolation(caplog, model):
    caplog.set_level(logging.INFO, logger="ModeExplorer")
    patched, = ModeExplorer().apply(model, operation="batch", adaptive=False, diagnostics=True)
    assert not torch.equal(draw(model, batch=2), draw(patched, batch=2))
    stats = traces(caplog)[-1]
    assert stats["feature_attempts"] == 3 and stats["feature_reference_dit_calls"] == 0
    assert torch.equal(draw(model, batch=2, mixed=True), draw(patched, batch=2, mixed=True))
    assert torch.equal(draw(model), draw(patched))


def test_initialization_budget_and_native_basic_guider(caplog, model):
    caplog.set_level(logging.INFO, logger="ModeExplorer")
    patched, = ModeExplorer().apply(model, feature="native", initialization=True, diagnostics=True)
    first = draw(patched, batch=2, cfg=3.5, mixed=True)
    assert torch.equal(first, draw(patched, batch=2, cfg=3.5, mixed=True))
    stats = traces(caplog)[-1]
    assert stats["initialization_guider_calls"] == stats["initialization_dit_calls"] == 4
    assert stats["feature_attempts"] == 0
    assert len([p for p in stats["points"] if p["kind"] == "initialization"][0]["accepted"]) == 2
    basic = draw(patched, guider_basic=True)
    assert torch.isfinite(basic).all()
    assert traces(caplog)[-1]["initialization_guider_calls"] == 4


@pytest.mark.parametrize("case", ["partial", "highres", "nonempty", "mask", "no_noise"])
def test_skipped_stages_remain_native(case, model):
    patched, = ModeExplorer().apply(model, initialization=True)
    kwargs = {
        "partial": {"sigmas": schedule()[1:]}, "highres": {"stage": "highres"},
        "nonempty": {"latent": torch.ones(1, 16, 1, 7, 11)}, "mask": {"mask": torch.ones(7, 11)},
        "no_noise": {"zero_noise": True},
    }[case]
    assert torch.equal(draw(model, **kwargs), draw(patched, **kwargs))


def test_unsupported_architecture_and_patch_fail_closed(model):
    model.model.diffusion_model = torch.nn.Identity()
    with pytest.raises(RuntimeError, match="transformer contract"):
        ModeExplorer().apply(model)
    assert ModeExplorer().apply(model, enabled=False)[0] is model


def test_cancellation_does_not_leave_references_or_hooks(caplog, model, monkeypatch):
    caplog.set_level(logging.INFO, logger="ModeExplorer")
    patched, = ModeExplorer().apply(model, initialization=True, diagnostics=True)
    native_interrupt = comfy.model_management.throw_exception_if_processing_interrupted

    def interrupt():
        raise comfy.model_management.InterruptProcessingException()

    with monkeypatch.context() as scoped:
        scoped.setattr(comfy.model_management, "throw_exception_if_processing_interrupted", interrupt)
        with pytest.raises(comfy.model_management.InterruptProcessingException):
            draw(patched)
    assert comfy.model_management.throw_exception_if_processing_interrupted is native_interrupt
    assert traces(caplog)[-1]["completed"] is False
    assert run_key_absent(patched)
    assert torch.isfinite(draw(patched)).all()


@pytest.mark.parametrize("entry", ["ksampler", "advanced", "modular"])
@torch.inference_mode()
def test_public_sampler_nodes_fix_standard_empty_latent(entry, model):
    from comfy_extras.nodes_custom_sampler import Noise_RandomNoise, SamplerCustomAdvanced

    p, n = conditions()
    latent = nodes.EmptyLatentImage().generate(80, 64)[0]
    patched, = ModeExplorer().apply(model, adaptive=False)
    def execute(candidate):
        if entry == "ksampler":
            result, = nodes.KSampler().sample(candidate, 55, 5, 1.0, "euler", "simple", p, n, latent)
        elif entry == "advanced":
            result, = nodes.KSamplerAdvanced().sample(
                candidate, "enable", 55, 5, 1.0, "euler", "simple", p, n, latent, 0, 5, "disable",
            )
        else:
            guider = Guider_Basic(candidate)
            guider.set_conds(p)
            output = SamplerCustomAdvanced.execute(
                Noise_RandomNoise(55), guider, comfy.samplers.ksampler("euler"), schedule(), latent,
            )
            result = output[0]
        return result["samples"]

    first = execute(patched)
    assert first.shape == (1, 16, 1, 8, 10)
    assert not torch.equal(first, execute(model))
    assert torch.equal(first, execute(patched))


def test_audit_examples_against_installed_node_schemas(model, monkeypatch):
    import execution
    import folder_paths
    import os
    from importlib.util import find_spec

    installed = {
        "diffusion_models": ["krea2_turbo_fp8_scaled.safetensors", "krea2_int8_convrot.safetensors"],
        "text_encoders": ["qwen3vl_4b_fp8_scaled.safetensors"],
        "vae": ["qwen_image_vae.safetensors"],
    }
    monkeypatch.setattr(folder_paths, "get_filename_list", lambda key: installed.get(key, []))
    root = Path(__file__).resolve().parents[1]
    assert asyncio.run(nodes.load_custom_node(str(root)))
    spectrum_available = find_spec("comfyui_spectrum_krea2") is not None
    if spectrum_available:
        from comfyui_spectrum_krea2.nodes import NODE_CLASS_MAPPINGS
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "SpectrumApplyKrea2", NODE_CLASS_MAPPINGS["SpectrumApplyKrea2"])
    scale_locked = os.environ.get("MODEEXPLORER_SCALE_LOCKED")
    if scale_locked:
        assert asyncio.run(nodes.load_custom_node(scale_locked))
    for path in sorted((root / "examples").glob("*.json")):
        if path.name in ("spectrum_ab.json", "response_spectrum.json") and not spectrum_available:
            continue
        if (path.name.startswith("projection_") or path.name in ("scale_locked_spectrum.json", "response_scale_locked.json")) and not (
            scale_locked and spectrum_available
        ):
            continue
        graph = json.loads(path.read_text())
        node_map = {node["id"]: node for node in graph["nodes"]}
        assert len(node_map) == len(graph["nodes"])
        api = {}
        for item in node_map.values():
            schema = nodes.NODE_CLASS_MAPPINGS[item["type"]].INPUT_TYPES()
            inputs = schema.get("required", {}) | schema.get("optional", {})
            for port in item["inputs"]:
                assert port["name"] in inputs and inputs[port["name"]][0] == port["type"]
            ports = {port["name"] for port in item["inputs"]}
            values = iter(item.get("widgets_values", []))
            widget_inputs = {}
            for name, spec in inputs.items():
                if name not in ports:
                    widget_inputs[name] = next(values)
                    if name in ("seed", "noise_seed") or (len(spec) > 1 and spec[1].get("control_after_generate")):
                        next(values)
            assert next(values, None) is None, (path.name, item["type"])
            api[str(item["id"])] = {"class_type": item["type"], "inputs": widget_inputs}
        for ident, src, source_slot, dst, dest_slot, value in graph["links"]:
            output = node_map[src]["outputs"][source_slot]
            port = node_map[dst]["inputs"][dest_slot]
            assert port["link"] == ident and ident in output["links"]
            assert value == output["type"] == port["type"]
            api[str(dst)]["inputs"][port["name"]] = [str(src), source_slot]
        if path.name.startswith("projection_"):
            planner, = [item["inputs"] for item in api.values()
                        if item["class_type"] == "Krea2TurboFlowAlignedRegenerate"]
            research, = [item["inputs"] for item in api.values()
                         if item["class_type"] == "ModeExplorerMechanisticKrea2"]
            assert "VAEDecode" not in {item["class_type"] for item in api.values()}
            assert planner["regeneration_mode"] == "planner_final_upscale_only" and planner["handoff_diagnostics"] == "off"
            assert research["analysis_device"] == "cuda" and research["projection_study"] == "observe"
            assert research["projection_activation_check"] is True
            if path.name == "projection_conditioning.json":
                assert research["alternate"] != planner["positive"]
        valid, error, *_ = asyncio.run(execution.validate_prompt(path.name, api, None))
        assert valid, (path.name, error)


def test_spectrum_and_other_numerical_wrappers_are_rejected(model):
    from comfy.patcher_extension import WrappersMP
    for key, match in [("spectrum_krea2", "Spectrum"), ("unknown", "diffusion_model")]:
        alternate = model.clone()
        alternate.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, key, lambda *a: None)
        with pytest.raises(RuntimeError, match=match):
            ModeExplorer().apply(alternate)


def test_native_lora_patch_ownership_is_preserved(model):
    delta = torch.full_like(model.model.diffusion_model.first.weight, 0.02)
    key = "diffusion_model.first.weight"
    lora = model.clone()
    assert key in lora.add_patches({key: (delta,)}, 0.8)
    patched, = ModeExplorer().apply(lora, adaptive=False)
    assert len(patched.patches[key]) == len(lora.patches[key]) == 1
    assert key not in model.patches
    first = draw(patched)
    assert not torch.equal(first, draw(lora))
    assert not torch.equal(draw(lora), draw(model))
    assert torch.equal(first, draw(patched))


def test_split_cfg_accounting_and_combined_path(caplog, model):
    caplog.set_level(logging.INFO, logger="ModeExplorer")
    patched, = ModeExplorer().apply(model, initialization=True, adaptive=False, diagnostics=True)
    native = draw(model, cfg=3.5, split_cfg=True)
    result = draw(patched, cfg=3.5, split_cfg=True)
    assert torch.isfinite(result).all() and not torch.equal(result, native)
    assert torch.equal(result, draw(patched, cfg=3.5, split_cfg=True))
    stats = traces(caplog)[-1]
    assert stats["native_dit_calls"] == 10
    assert stats["feature_reference_dit_calls"] == 2
    assert stats["initialization_guider_calls"] == 4
    assert stats["initialization_dit_calls"] == 8
    assert not stats["reasons"]


def test_full_spectrum_is_an_opt_in_noop_without_spectrum(model):
    node = ModeExplorer()
    native, = node.apply(model, feature="feature_cosine", adaptive=False)
    opted, = node.apply(model, feature="feature_cosine", adaptive=False, full_spectrum=True)
    assert torch.equal(draw(native), draw(opted))
    assert not torch.equal(draw(model), draw(opted))
    with pytest.raises(ValueError, match="full_spectrum must be a boolean"):
        node.apply(model, full_spectrum=1)


@pytest.mark.parametrize("block", [0, 2])
def test_optional_blocks_and_multiple_scoped_references(caplog, model, block):
    caplog.set_level(logging.INFO, logger="ModeExplorer")
    guided, = ModeExplorer().apply(
        model, adaptive=False, diagnostics=True, feature_block=block,
        feature_grid="6", reference_trajectories=2, reference_budget=4,
        reference_source="per_seed", reference_selection="closest",
        reference_continuation="rolling_euler", feature_update="verified",
    )
    result = draw(guided, cfg=3.5, split_cfg=True, reference=True)
    assert torch.equal(result, draw(guided, cfg=3.5, split_cfg=True, reference=True))
    assert not torch.equal(draw(model, cfg=3.5, split_cfg=True, reference=True), result)
    report = traces(caplog)[-1]
    assert report["feature_reference_dit_calls"] == 4
    points = [p for p in report["points"] if "shift_rms" in p]
    assert len(points) == 2 and {p["block"] for p in points} == {block}
    assert len(set(points[0]["reference_slots"])) == 2
    assert points[0]["reference_slots"] == points[1]["reference_slots"]
    with pytest.raises(ValueError, match="native single-stream range"):
        ModeExplorer().apply(model, feature_block=3)
    assert ModeExplorer().apply(model, enabled=False, feature_block=64)[0] is model


def test_optional_sol_isolation(caplog, model):
    sol = pytest.importorskip("sol_h3.krea2")

    caplog.set_level(logging.INFO)
    guided, = ModeExplorer().apply(model, adaptive=False, diagnostics=True, initialization=True)
    joint = sol.install(guided, sol.Settings(), verbose=True)
    assert torch.isfinite(draw(joint)).all()
    last = traces(caplog)[-1]
    assert last["feature_reference_dit_calls"] == 2
    assert last["initialization_dit_calls"] >= 4
    assert any("Sol-Krea2: model_evaluations=5 " in record.message for record in caplog.records)
    assert sol._ACTIVE.get() is None and sol._EXECUTION.get() is None
    with pytest.raises(RuntimeError, match="before the verified Sol"):
        ModeExplorer().apply(sol.install(model, sol.Settings()))
