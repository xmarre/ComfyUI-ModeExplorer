# Example workflows

Open a JSON file in ComfyUI and select your native Krea 2 Turbo diffusion model, Qwen3-VL-4B encoder, and Qwen Image VAE. Standalone examples use Euler, the simple scheduler, CFG 1 and eight initial steps. Change both fixed seeds together in a paired graph.

| Workflow | Purpose |
| --- | --- |
| [Turbo](turbo.json) | Early Structural (B1) at strength 0.03, fidelity 0.70 and fixed seed. |
| [A/B](ab.json) | Native and guided paths at the same seed. |
| [Advanced](advanced.json) | Collective RBF volume, two deterministic references, rolling continuation and verified update. |
| [Initialization](initialization.json) | Initial-noise trust-region search with feature guidance disabled. |
| [Combined](combined.json) | Feature and initialization guidance in the same invocation. |
| [Batch](batch.json) | Four samples under the same prompt with zero reference forwards. |
| [Sequential](sequential.json) | KSampler seed increments on each submission. |
| [Native refinement](multistage.json) | Guided 1024px composition, latent resize, and a native 1536px second pass. |
| [Spectrum A/B](spectrum_ab.json) | Guarded versus Full Spectrum at the same seed. Requires the companion Spectrum overlay. |
| [Spectrum and Scale-Locked](scale_locked_spectrum.json) | Early Structural (B2) with ER-SDE and separate planner and refined outputs. Requires the companion Spectrum overlay and Scale-Locked with stage tags. |
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
| [Projection smoke](projection_smoke.json) | One early frozen-state packed projection replay on the planner. |
| [Projection signed](projection_signed.json) | Three early states with signed, half-amplitude, output-cast and representable-symmetry measurements. |
| [Projection conditioning](projection_conditioning.json) | Four-arm fixed intervention with a connected alternate positive and matched packed projection replay. |
| [Conditioning contrast](conditioning_contrast.json) | One-state matched semantic direction with alternate conditioning, repeat baseline and clean exports. |
| [Weak projection Gaussian](projection_guidance_weak_gaussian.json) | Planner PNG with a 0.000303 RMS projection-output edit and no auxiliary calls. |
| [Higher projection Gaussian](projection_guidance_higher_gaussian.json) | Matched planner PNG at 0.013937 RMS. |
| [Token-adaptive projection Gaussian](projection_guidance_token_adaptive.json) | Matched higher-RMS comparison with bounded native token scales. |
| [Response with Scale-Locked](response_scale_locked.json) | Planner-only response probes with separate planner and final outputs on the native refinement graph. |
| [ReDiverse FT](rediverse_ft.json) | Native Turbo and Turbo with the ReDiverse first-evaluation adapter at the same seed. Needs `krea-2-rediverse-comfy.safetensors` in `models/loras`. |
| [ReDiverse temporal](rediverse_temporal.json) | Experimental temporal adapter schedules at one seed: native, constant 4 × 0.5, the same window delayed to evaluation 2, and a front-loaded custom profile 0.8, 0.6, 0.4, 0.2. Diagnostics log the executed slots. |
| [ReDiverse panel](rediverse_panel.json) | Screening arms A–D at one seed: native, Early Structural (B1), adapter, adapter + B1. |
| [ReDiverse TF](rediverse_tf.json) | Krea 2 Raw calibration of teacher directions for the prompt, then Turbo with TF (band 2) and adapter + TF (band 4). Select your Raw diffusion model. |
| [Activation probe](rediverse_probe.json) | Directional-collapse spectra of blocks 22–27 for Raw, Turbo and Turbo with the adapter, 200 seeds. Research only; long-running. |
| [ReDiverse with Scale-Locked](rediverse_scale_locked.json) | Adapter on the planner stage with guarded Spectrum and ER-SDE, separate planner and final outputs. Requires the companion Spectrum overlay and Scale-Locked with stage tags. |

Use a shared upstream weight LoRA for identity comparisons. For Raw, load Raw weights, encode a true negative prompt, and use the Raw Spectrum preset when Spectrum is installed. Begin inside the model's supported resolution range.

Response examples use a broad initial sigma window so they can run on the local checkpoint’s actual schedule. Narrow the window after reading the trace. Pin Reference start sigma from the live arm for paired feature replay. Use Fixed precision for a common displacement across compared runs. See the [response guide](../docs/response.md) for state matching and predicted-clean decoding.


[Structural discovery native capture](structural_discovery_native.json) and [frozen factorial](structural_discovery_factorial.json) exercise the published workflow contract using sample conditioning and model names. The production pack uses the archival conditioning and model stack. The shared validator checks its seeds, feature recipes, guarded routing, connectivity and archive conditioning fingerprints.

```text
python -m tools.structural_workflows ModeExplorer_Structural_Discovery_Final.zip
```
