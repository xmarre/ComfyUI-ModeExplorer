"""ComfyUI registration and clone ownership."""

from .runtime import (
    FEATURES, FEATURE_GRIDS, FEATURE_UPDATES, KEY, OPERATIONS, REFERENCE_CONTINUATIONS,
    REFERENCE_SELECTIONS, REFERENCE_SOURCES, SamplerWrapper, Settings, attention_feature,
    diffusion_wrapper, has_spectrum, verify_patch_graph,
)


class ModeExplorer:
    CATEGORY = "model_patches/ModeExplorer"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    DESCRIPTION = "Experimental early-flow feature guidance and optional initial-noise optimization for Krea 2."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "enabled": ("BOOLEAN", {"default": True}),
            "feature": (list(FEATURES), {"default": Settings.feature}),
            "initialization": ("BOOLEAN", {"default": Settings.initialization, "tooltip": "Scores up to three noise proposals. Maximum six extra guider calls and twelve actual transformer forwards."}),
            "strength": ("FLOAT", {"default": Settings.strength, "min": 0.0, "max": 1.0, "step": 0.01}),
            "fidelity": ("FLOAT", {"default": Settings.fidelity, "min": 0.0, "max": 1.0, "step": 0.01}),
            "min_sigma": ("FLOAT", {"default": Settings.min_sigma, "min": 0.5, "max": 1.0, "step": 0.01, "tooltip": "Absolute flow sigma. Only the initial empty-latent pass is eligible."}),
            "adaptive": ("BOOLEAN", {"default": Settings.adaptive, "tooltip": "Scale a feature edit by redundancy above the similarity threshold."}),
            "similarity": ("FLOAT", {"default": Settings.similarity, "min": 0.0, "max": 0.99, "step": 0.01, "tooltip": "Used only when Adaptive is enabled."}),
            "operation": (list(OPERATIONS), {"default": Settings.operation}),
            "reference_budget": ("INT", {"default": Settings.reference_budget, "min": 0, "max": 4, "tooltip": "Maximum extra positive transformer forwards for a single image. Batch guidance uses none."}),
            "diagnostics": ("BOOLEAN", {"default": Settings.diagnostics}),
            "full_spectrum": ("BOOLEAN", {"default": Settings.full_spectrum,
                "tooltip": "Experimental: leave Spectrum's forecast policy/history untouched. Forecasted steps cannot receive guidance; cross-regime extrapolation is unvalidated."}),
        }, "optional": {
            "feature_block": ("INT", {"default": Settings.feature_block, "min": 0, "max": 63, "tooltip": "Zero-based native single-stream block. Validated against the loaded model."}),
            "feature_grid": ([str(grid) for grid in FEATURE_GRIDS], {"default": str(Settings.feature_grid), "tooltip": "Maximum spatial cells on each axis."}),
            "cosine_temperature": ("FLOAT", {"default": Settings.cosine_temperature, "min": 0.1, "max": 2.0, "step": 0.05, "tooltip": "Cosine objective only. Changes the weighting for cohorts of at least three valid members."}),
            "volume_bandwidth": ("FLOAT", {"default": Settings.volume_bandwidth, "min": 0.25, "max": 1.25, "step": 0.05, "tooltip": "RBF objective only, for cohorts of at least three. Ignored when median bandwidth is enabled."}),
            "volume_median": ("BOOLEAN", {"default": Settings.volume_median, "tooltip": "Derive collective RBF bandwidth from eligible normalized feature distances."}),
            "reference_source": (list(REFERENCE_SOURCES), {"default": Settings.reference_source}),
            "reference_bank_size": ("INT", {"default": Settings.reference_bank_size, "min": 2, "max": 256, "tooltip": "Bank strategy only. Entries are derived on demand."}),
            "reference_trajectories": ("INT", {"default": Settings.reference_trajectories, "min": 1, "max": 3, "tooltip": "Maximum reference forwards at one actual evaluation, within the total reference budget."}),
            "reference_selection": (list(REFERENCE_SELECTIONS), {"default": Settings.reference_selection, "tooltip": "Use all measured references jointly, or the highest cosine match."}),
            "reference_continuation": (list(REFERENCE_CONTINUATIONS), {"default": Settings.reference_continuation, "tooltip": "Rolling Euler updates the reference with its own latest velocity. Skipped evaluations produce a larger Euler interval."}),
            "feature_update": (list(FEATURE_UPDATES), {"default": Settings.feature_update, "tooltip": "Verified adds an area-weighted mean constraint, an 8% global RMS cap, and pooled objective backtracking."}),
            "initialization_trials": ("INT", {"default": Settings.initialization_trials, "min": 1, "max": 3, "tooltip": "Maximum noise proposals after three baseline/reference guider predictions."}),
        }}

    def apply(self, model, enabled=True, feature=Settings.feature, initialization=Settings.initialization,
              strength=Settings.strength, fidelity=Settings.fidelity, min_sigma=Settings.min_sigma,
              adaptive=Settings.adaptive, similarity=Settings.similarity, operation=Settings.operation,
              reference_budget=Settings.reference_budget, diagnostics=Settings.diagnostics,
              full_spectrum=Settings.full_spectrum, **advanced):
        from comfy.patcher_extension import WrappersMP

        base_options = model.model_options
        wrapper_types = (WrappersMP.SAMPLER_SAMPLE, WrappersMP.DIFFUSION_MODEL)
        attached = any(model.get_wrappers(kind, KEY) for kind in wrapper_types)
        active = enabled and (diagnostics or (strength != 0 and (feature != "native" or initialization)))
        if not active and not attached:
            return (model,)
        if active:
            if "feature_grid" in advanced and isinstance(advanced["feature_grid"], str):
                advanced["feature_grid"] = int(advanced["feature_grid"])
            settings = Settings(feature, initialization, strength, fidelity, min_sigma,
                                adaptive, similarity, operation, reference_budget, diagnostics,
                                full_spectrum, **advanced)
            self.verify_model(model, settings.feature_block)
            if has_spectrum(base_options, model.wrappers):
                raise RuntimeError("Apply ModeExplorer before Spectrum: noise optimization must precede Spectrum's sampling session.")
            verify_patch_graph(base_options, settings, model.wrappers)

        clone = model.clone()
        if attached:
            for kind in wrapper_types:
                clone.remove_wrappers_with_key(kind, KEY)
            patches = clone.model_options["transformer_options"].get("patches", {})
            remaining = [fn for fn in patches.get("attn1_output_patch", ()) if fn is not attention_feature]
            if remaining:
                patches["attn1_output_patch"] = remaining
            else:
                patches.pop("attn1_output_patch", None)
                if not patches:
                    clone.model_options["transformer_options"].pop("patches", None)
        if active:
            clone.add_wrapper_with_key(WrappersMP.SAMPLER_SAMPLE, KEY, SamplerWrapper(settings))
            clone.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, KEY, diffusion_wrapper)
            clone.set_model_attn1_output_patch(attention_feature)
        return (clone,)

    @staticmethod
    def verify_model(patcher, block=Settings.feature_block):
        from comfy.ldm.krea2.model import SingleStreamBlock, SingleStreamDiT
        from comfy.model_base import Krea2

        if not isinstance(patcher.model, Krea2):
            raise RuntimeError("ModeExplorer requires a native Krea 2 Raw or Turbo MODEL.")
        dit = patcher.get_model_object("diffusion_model")
        if (
            type(dit) is not SingleStreamDiT or dit.patch != 2 or dit.channels != 16
        ):
            raise RuntimeError("ModeExplorer could not verify the native Krea 2 transformer contract.")
        if not 0 <= block < len(dit.blocks) or type(dit.blocks[block]) is not SingleStreamBlock:
            raise ValueError(f"ModeExplorer block {block} is outside the native single-stream range [0, {len(dit.blocks) - 1}].")
