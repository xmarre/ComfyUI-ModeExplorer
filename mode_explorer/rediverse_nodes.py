"""ComfyUI nodes for ReDiverse on native Krea 2."""

import json
import os
from collections import OrderedDict

from .nodes import ModeExplorer, _remove_modeexplorer_wrappers, check_rediverse_settings
from .rediverse import (
    ANCHOR_BLOCK, DIRECTIONS, PAPER_BAND, ROLLOUTS, SCHEDULES, TEACHER_FORWARD, TEACHER_STEPS, direction_files,
    load_adapter, load_directions, verify_adapter,
)
from .runtime import KEY, SamplerWrapper, Settings, attention_feature, diffusion_wrapper, has_spectrum, verify_patch_graph

DIRECTIONS_TYPE = "MODEEXPLORER_REDIVERSE_DIRECTIONS"
NONE = "none"
_ADAPTERS = OrderedDict()


def _cached_adapter(path):
    stat = os.stat(path)
    key = (os.path.realpath(path), stat.st_mtime_ns, stat.st_size)
    if key in _ADAPTERS:
        _ADAPTERS.move_to_end(key)
        return _ADAPTERS[key]
    adapter = load_adapter(path)
    _ADAPTERS[key] = adapter
    while len(_ADAPTERS) > 2:
        _ADAPTERS.popitem(last=False)
    return adapter


def _adapter_choices():
    try:
        import folder_paths

        return [NONE, *folder_paths.get_filename_list("loras")]
    except Exception:  # Discovery outside a ComfyUI process.
        return [NONE]


class ModeExplorerReDiverse:
    CATEGORY = "model_patches/ModeExplorer"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    DESCRIPTION = ("ReDiverse for native Krea 2 Turbo: a first-evaluation adapter (ReDiverse-FT) and optional "
                   "teacher-direction steering (ReDiverse-TF). Nothing is merged into the model weights.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "enabled": ("BOOLEAN", {"default": True}),
            "adapter": (_adapter_choices(), {"default": NONE, "tooltip": "ReDiverse-FT LoRA from the loras folder, e.g. krea-2-rediverse-comfy.safetensors. Active only on the first evaluation."}),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "Adapter multiplier. 1.0 is the released recipe."}),
            "band": ("FLOAT", {"default": PAPER_BAND, "min": 0.0, "max": 8.0, "step": 0.1, "tooltip": "ReDiverse-TF coefficients alpha ~ U[-band, band]. 2 is the paper setting, 4 the released combined setting. Needs directions."}),
        }, "optional": {
            "directions": (DIRECTIONS_TYPE, {"tooltip": "Teacher directions calibrated for this prompt and latent size."}),
            "schedule": (list(SCHEDULES), {"default": "first_evaluation", "advanced": True, "tooltip": "first_evaluation is the reference contract. first_interval also adapts intermediate solver stages above the second sigma (experimental)."}),
            "adapted_sigmas": ("INT", {"default": 1, "min": 1, "max": 4, "advanced": True, "tooltip": "Number of opening evaluation sigmas. Values above 1 are experimental."}),
            "direction_count": ("INT", {"default": DIRECTIONS, "min": 1, "max": 32, "advanced": True}),
            "coefficient_trial": ("INT", {"default": 0, "min": 0, "max": 65535, "advanced": True, "tooltip": "Independent deterministic coefficient draw for the same noise seed."}),
            "steering": (["random", "cohort_logdet"], {"default": "random", "advanced": True, "tooltip": "random is ReDiverse-TF. cohort_logdet is an experimental teacher-informed batch objective with matched norm."}),
            "diagnostics": ("BOOLEAN", {"default": False, "advanced": True}),
        }}

    def apply(self, model, enabled=True, adapter=NONE, strength=1.0, band=PAPER_BAND, directions=None,
              schedule="first_evaluation", adapted_sigmas=1, direction_count=DIRECTIONS, coefficient_trial=0,
              steering="random", diagnostics=False):
        from comfy.patcher_extension import WrappersMP

        from .rediverse_runtime import ReDiverseConfig

        samplers = model.get_wrappers(WrappersMP.SAMPLER_SAMPLE, KEY)
        existing = samplers[0] if samplers else None
        if existing is not None and type(existing) is not SamplerWrapper:
            raise RuntimeError("ModeExplorer could not verify its sampler wrapper.")
        if not enabled:
            if existing is None or existing.rediverse is None:
                return (model,)
            return (self._attach(model, existing, None),)
        if existing is not None and existing.guidance is not None:
            raise ValueError("Use separate MODEL branches for projection guidance and ReDiverse.")
        if has_spectrum(model.model_options, model.wrappers):
            raise RuntimeError("Apply ModeExplorer ReDiverse before Spectrum.")
        ModeExplorer.verify_model(model, 0)
        _, sol = verify_patch_graph(model.model_options, Settings(feature="native"), model.wrappers)
        if sol is not None:
            raise RuntimeError("Apply ModeExplorer ReDiverse before the verified Sol-Krea2 patch.")
        if existing is not None:
            settings = existing.settings
            check_rediverse_settings(settings.initialization and settings.strength != 0, settings.full_spectrum,
                                     settings.ablation_route)
        parsed = None
        if adapter != NONE:
            import folder_paths

            path = folder_paths.get_full_path("loras", adapter)
            if path is None:
                raise ValueError(f"Adapter {adapter!r} was not found in the loras folder.")
            parsed = _cached_adapter(path)
            verify_adapter(parsed, model.get_model_object("diffusion_model"))
        if directions is not None:
            dit = model.get_model_object("diffusion_model")
            if not 0 <= directions.block < len(dit.blocks) or directions.d_model != dit.blocks[0].attn.wo.out_features:
                raise ValueError("The direction archive does not match the loaded model's blocks or width.")
        if parsed is None and directions is None:
            raise ValueError("Select a ReDiverse adapter, connect calibrated directions, or disable the node.")
        config = ReDiverseConfig(adapter=parsed, strength=strength, schedule=schedule, evaluations=adapted_sigmas,
                                 directions=directions, band=band, count=min(direction_count, directions.U.shape[0])
                                 if directions is not None else direction_count,
                                 trial=coefficient_trial, tf_mode=steering, diagnostics=diagnostics)
        return (self._attach(model, existing, config),)

    @staticmethod
    def _attach(model, existing, config):
        from comfy.patcher_extension import WrappersMP

        clone = model.clone()
        if existing is not None:
            settings, response = existing.settings, existing.response
            clone.remove_wrappers_with_key(WrappersMP.SAMPLER_SAMPLE, KEY)
            keep = (config is not None or response is not None or settings.feature != "native" or settings.initialization
                    or settings.diagnostics or settings.trajectory_label)
            if not keep:
                _remove_modeexplorer_wrappers(clone)
                return clone
            clone.add_wrapper_with_key(WrappersMP.SAMPLER_SAMPLE, KEY, SamplerWrapper(settings, response, rediverse=config))
            return clone
        clone.add_wrapper_with_key(WrappersMP.SAMPLER_SAMPLE, KEY, SamplerWrapper(Settings(feature="native"), rediverse=config))
        clone.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, KEY, diffusion_wrapper)
        # Spectrum's verified ModeExplorer contract requires this marker. Without a
        # feature frame it returns the native attention output unchanged.
        clone.set_model_attn1_output_patch(attention_feature)
        return clone


class ModeExplorerReDiverseCalibrate:
    CATEGORY = "model_patches/ModeExplorer/research"
    RETURN_TYPES = (DIRECTIONS_TYPE, "STRING")
    RETURN_NAMES = ("directions", "summary")
    FUNCTION = "calibrate"
    DESCRIPTION = ("Estimate ReDiverse-TF teacher directions for one prompt and latent size: conditional Krea 2 Raw "
                   "rollouts, captured at the selected block output. Writes a reusable archive.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "teacher": ("MODEL", {"tooltip": "Krea 2 Raw. Apply weight LoRAs upstream if the student uses them."}),
            "positive": ("CONDITIONING",),
            "latent": ("LATENT", {"tooltip": "Empty latent with the generation size. Only its shape is used."}),
            "block": ("INT", {"default": ANCHOR_BLOCK, "min": 0, "max": 63}),
            "teacher_forward": ("INT", {"default": TEACHER_FORWARD, "min": 1, "max": 200}),
            "teacher_steps": ("INT", {"default": TEACHER_STEPS, "min": 1, "max": 200}),
            "rollouts": ("INT", {"default": ROLLOUTS, "min": 2, "max": 256}),
            "directions": ("INT", {"default": DIRECTIONS, "min": 1, "max": 32}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 1 << 40, "tooltip": "Rollout j uses packed noise seed * 1000 + 100 + j."}),
            "batch": ("INT", {"default": 5, "min": 1, "max": 32}),
            "label": ("STRING", {"default": ""}),
        }, "optional": {
            "negative": ("CONDITIONING", {"tooltip": "Research override: guided calibration. The reference is conditional-only."}),
            "cfg": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 20.0, "step": 0.1, "advanced": True}),
            "sigmas": ("SIGMAS", {"tooltip": "Research override of the reference 52-step FlowMatch Euler teacher schedule."}),
            "workspace": (["auto", "cpu"], {"default": "auto", "advanced": True}),
            "workspace_mib": ("INT", {"default": 8192, "min": 64, "max": 98304, "advanced": True}),
        }}

    def calibrate(self, teacher, positive, latent, block, teacher_forward, teacher_steps, rollouts, directions, seed,
                  batch, label, negative=None, cfg=1.0, sigmas=None, workspace="auto", workspace_mib=8192):
        from .rediverse_runtime import calibrate
        from .trajectory import LABEL

        if label and not LABEL.fullmatch(label):
            raise ValueError("Use 1–64 letters, digits, hyphens or underscores for the label.")
        if negative is not None and cfg == 1.0:
            negative = None
        path, archive = calibrate(teacher, positive, latent, negative=negative, guidance=cfg, steps=teacher_steps,
                                  forward=teacher_forward, block=block, rollouts=rollouts, directions=directions,
                                  seed=seed, batch=batch, sigmas=sigmas, label=label, device=workspace,
                                  workspace_mib=workspace_mib)
        return (archive, json.dumps({"path": path, **archive.describe()}, indent=1))


class ModeExplorerReDiverseDirections:
    CATEGORY = "model_patches/ModeExplorer"
    RETURN_TYPES = (DIRECTIONS_TYPE, "STRING")
    RETURN_NAMES = ("directions", "summary")
    FUNCTION = "load"
    DESCRIPTION = "Load a validated ReDiverse-TF direction archive from output/modeexplorer_rediverse/directions."

    @classmethod
    def INPUT_TYPES(cls):
        try:
            files = direction_files() or [NONE]
        except Exception:  # Discovery outside a ComfyUI process.
            files = [NONE]
        return {"required": {"archive": (files,)}}

    @classmethod
    def IS_CHANGED(cls, archive):
        from .rediverse import archive_root

        path = os.path.join(archive_root(), "directions", archive)
        return f"{os.path.getmtime(path)}:{os.path.getsize(path)}" if os.path.isfile(path) else archive

    def load(self, archive):
        from .rediverse import archive_root

        if archive == NONE or os.path.basename(archive) != archive:
            raise ValueError("Select a direction archive.")
        loaded = load_directions(os.path.join(archive_root(), "directions", archive))
        return (loaded, json.dumps(loaded.describe(), indent=1))


class ModeExplorerActivationProbe:
    CATEGORY = "model_patches/ModeExplorer/research"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("summary",)
    FUNCTION = "probe"
    OUTPUT_NODE = True
    DESCRIPTION = ("Directional-collapse probe: total variance, participation ratio, concentration and spectra of "
                   "block-output image rows across seeds at one frozen first-step state (bounded staging).")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "positive": ("CONDITIONING",),
            "latent": ("LATENT",),
            "blocks": ("STRING", {"default": "22-27", "tooltip": "Comma-separated zero-based blocks or inclusive ranges."}),
            "samples": ("INT", {"default": 200, "min": 3, "max": 2000}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 1 << 40, "tooltip": "Sample i uses packed noise 1000000 + seed * 100003 + i."}),
            "sigma": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.0001}),
            "batch": ("INT", {"default": 8, "min": 1, "max": 64}),
            "label": ("STRING", {"default": ""}),
        }, "optional": {
            "reference": (DIRECTIONS_TYPE, {"tooltip": "Teacher directions at one probed block: retained variance, subspace alignment and deficits."}),
            "workspace": (["auto", "cpu"], {"default": "auto", "advanced": True}),
            "workspace_mib": ("INT", {"default": 16384, "min": 64, "max": 98304, "advanced": True}),
            "compare_with": ("STRING", {"default": "", "advanced": True, "tooltip": "Teacher probe report file name in modeexplorer_rediverse/probes for ratio output."}),
        }}

    def probe(self, model, positive, latent, blocks, samples, seed, sigma, batch, label, reference=None,
              workspace="auto", workspace_mib=16384, compare_with=""):
        from .mechanism import MechanisticSettings
        from .rediverse import archive_root
        from .rediverse_runtime import compare_probes, probe
        from .trajectory import LABEL

        if label and not LABEL.fullmatch(label):
            raise ValueError("Use 1–64 letters, digits, hyphens or underscores for the label.")
        path, report = probe(model, positive, latent, blocks=MechanisticSettings(blocks=blocks).block_ids, samples=samples, seed=seed,
                             sigma=sigma, batch=batch, reference=reference, device=workspace,
                             workspace_mib=workspace_mib, label=label)
        summary = {"path": path, "blocks": {name: {key: entry.get(key) for key in ("tv", "pr", "conc", "entropy_rank", "pc1_share")}
                                            for name, entry in report["blocks"].items()}}
        if compare_with:
            if os.path.basename(compare_with) != compare_with:
                raise ValueError("Name a probe report file, not a path.")
            with open(os.path.join(archive_root(), "probes", compare_with)) as handle:
                summary["comparison"] = compare_probes(json.load(handle), report)
        return (json.dumps(summary, indent=1),)


class ModeExplorerReferenceSigmas:
    CATEGORY = "model_patches/ModeExplorer"
    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "sigmas"
    DESCRIPTION = ("Reference Krea 2 FlowMatch Euler schedules: Turbo uses mu 1.15; Raw uses the resolution-dependent "
                   "mu of Krea2Pipeline (0.906 at 1024x1024).")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "latent": ("LATENT",),
            "model": (["turbo", "raw"], {"default": "turbo"}),
            "steps": ("INT", {"default": 8, "min": 1, "max": 1000}),
        }}

    def sigmas(self, latent, model, steps):
        import torch

        from .rediverse import reference_sigmas

        height, width = latent["samples"].shape[-2:]
        tokens = ((height + 1) // 2) * ((width + 1) // 2)
        values = reference_sigmas(steps, mu=1.15) if model == "turbo" else reference_sigmas(steps, tokens)
        return (torch.tensor(values, dtype=torch.float32),)
