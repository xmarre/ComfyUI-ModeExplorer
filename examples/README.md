# Example workflows

Open a JSON file in ComfyUI and select your native Krea 2 Turbo diffusion model, Qwen3-VL-4B encoder, and Qwen Image VAE. Standalone examples use Euler, the simple scheduler, CFG 1 and eight initial steps. Change both fixed seeds together in a paired graph.

| Workflow | Purpose |
| --- | --- |
| [Turbo](turbo.json) | Default conservative feature guidance on one sample. |
| [A/B](ab.json) | Native and guided paths at the same seed. |
| [Advanced](advanced.json) | Collective RBF volume, two deterministic references, rolling continuation and verified update. |
| [Initialization](initialization.json) | Initial-noise trust-region search with feature guidance disabled. |
| [Combined](combined.json) | Feature and initialization guidance in the same invocation. |
| [Batch](batch.json) | Four samples under the same prompt with zero reference forwards. |
| [Sequential](sequential.json) | KSampler seed increments on each submission. |
| [Native refinement](multistage.json) | Guided 1024px composition, latent resize, and a native 1536px second pass. |
| [Spectrum A/B](spectrum_ab.json) | Guarded versus Full Spectrum at the same seed. Requires the companion Spectrum overlay. |
| [Spectrum and Scale-Locked](scale_locked_spectrum.json) | Actual Scale-Locked Krea 2 Regenerate with ER-SDE and separate planner and refined outputs. Requires the companion Spectrum overlay and Scale-Locked with stage tags. |

Use a shared upstream weight LoRA for identity comparisons. For Raw, load Raw weights, encode a true negative prompt, and use the Raw Spectrum preset when Spectrum is installed. Begin inside the model's supported resolution range.
