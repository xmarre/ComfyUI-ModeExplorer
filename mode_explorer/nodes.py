"""ComfyUI registration and clone ownership."""

from .runtime import (
    FEATURES, KEY, OPERATIONS, SamplerWrapper, Settings, attention_feature,
    diffusion_wrapper, has_spectrum, verify_patch_graph, wrappers,
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
            "feature": (FEATURES, {"default": Settings.feature}),
            "initialization": ("BOOLEAN", {"default": Settings.initialization, "tooltip": "Uses four extra guider evaluations."}),
            "strength": ("FLOAT", {"default": Settings.strength, "min": 0.0, "max": 1.0, "step": 0.01}),
            "fidelity": ("FLOAT", {"default": Settings.fidelity, "min": 0.0, "max": 1.0, "step": 0.01}),
            "min_sigma": ("FLOAT", {"default": Settings.min_sigma, "min": 0.5, "max": 1.0, "step": 0.01, "tooltip": "Absolute flow sigma. Only the initial empty-latent pass is eligible."}),
            "adaptive": ("BOOLEAN", {"default": Settings.adaptive}),
            "similarity": ("FLOAT", {"default": Settings.similarity, "min": 0.0, "max": 0.99, "step": 0.01}),
            "operation": (OPERATIONS, {"default": Settings.operation}),
            "reference_budget": ("INT", {"default": Settings.reference_budget, "min": 0, "max": 4, "tooltip": "Maximum extra positive transformer forwards for a single image. Batch guidance uses none."}),
            "diagnostics": ("BOOLEAN", {"default": Settings.diagnostics}),
        }}

    def apply(self, model, enabled=True, feature=Settings.feature, initialization=Settings.initialization,
              strength=Settings.strength, fidelity=Settings.fidelity, min_sigma=Settings.min_sigma,
              adaptive=Settings.adaptive, similarity=Settings.similarity, operation=Settings.operation,
              reference_budget=Settings.reference_budget, diagnostics=Settings.diagnostics):
        from comfy.patcher_extension import WrappersMP

        base_options = model.model_options
        wrapper_types = (WrappersMP.SAMPLER_SAMPLE, WrappersMP.DIFFUSION_MODEL)
        attached = any(wrappers(base_options).get(kind, {}).get(KEY) for kind in wrapper_types)
        active = enabled and (diagnostics or (strength != 0 and (feature != "native" or initialization)))
        if not active and not attached:
            return (model,)
        if active:
            settings = Settings(feature, initialization, strength, fidelity, min_sigma,
                                adaptive, similarity, operation, reference_budget, diagnostics)
            self.verify_model(model)
            if has_spectrum(base_options):
                raise RuntimeError("ModeExplorer and Spectrum forecasting need separate MODEL branches.")
            verify_patch_graph(base_options, settings)

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
        if active:
            clone.add_wrapper_with_key(WrappersMP.SAMPLER_SAMPLE, KEY, SamplerWrapper(settings))
            clone.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, KEY, diffusion_wrapper)
            clone.set_model_attn1_output_patch(attention_feature)
        return (clone,)

    @staticmethod
    def verify_model(patcher):
        from comfy.ldm.krea2.model import SingleStreamBlock, SingleStreamDiT
        from comfy.model_base import Krea2
        from .runtime import BLOCK

        if not isinstance(patcher.model, Krea2):
            raise RuntimeError("ModeExplorer requires a native Krea 2 Raw or Turbo MODEL.")
        dit = patcher.get_model_object("diffusion_model")
        if (
            type(dit) is not SingleStreamDiT or len(dit.blocks) <= BLOCK
            or type(dit.blocks[BLOCK]) is not SingleStreamBlock or dit.patch != 2 or dit.channels != 16
        ):
            raise RuntimeError("ModeExplorer could not verify the native Krea 2 transformer contract.")
