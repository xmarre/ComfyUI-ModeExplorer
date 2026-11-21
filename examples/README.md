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
| [Causal directions](causal_directions.json) | Native, sham, repel, reverse and channel-sign renders at a fixed seed. The low-strength controls use strength 0.025, fidelity 0.70 and block 5. |
| [Latent probe](response_probe.json) | Projected response and local clean/state export at the first matching actual evaluation. |
| [Frozen feature replay](response_feature_replay.json) | Compare latent responses and a cosine edit at the same native state. Includes later-block measurements. |
| [Block/head localization](response_localization.json) | Measure a component-only edit on output head 0. Change the block, cell or matched normalization for ablations. |
| [Response steering](response_steering.json) | Try both signs of the leading projected right singular direction and apply a bounded correction when accepted. |
| [Response-informed site selection](response_site_selection.json) | Compare two supplied block/head sites using projected clean-output candidates. |
| [Candidate review](response_candidate_review.json) | Export feature candidates for local VAE decode and human review. |
| [External quality gate](response_quality_gate.json) | Connect a local validated scorer to enable quality acceptance. An empty scorer input keeps the native path. |
| [Response with Spectrum](response_spectrum.json) | Diagnose otherwise actual evaluations on the Full arm while retaining the paired Guarded arm. Both use the compatibility overlay. |
| [Mechanistic signed smoke](mechanistic_signed.json) | Observational signed response on native velocity and clean latents with aligned frequency bands and exports. |
| [Response with Scale-Locked](response_scale_locked.json) | Planner-only response probes with separate planner and final outputs on the native refinement graph. |

Use a shared upstream weight LoRA for identity comparisons. For Raw, load Raw weights, encode a true negative prompt, and use the Raw Spectrum preset when Spectrum is installed. Begin inside the model's supported resolution range.

Response examples use a broad initial sigma window so they can run on the local checkpoint’s actual schedule. Narrow the window after reading the trace. Pin Reference start sigma from the live arm for paired feature replay. Use Fixed precision for a common displacement across compared runs. See the [response guide](../docs/response.md) for state matching and predicted-clean decoding.
