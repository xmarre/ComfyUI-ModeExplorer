# Phase IV structural causality

**Status:** This is the historical experimental protocol. The discovery campaign is closed; see [completed findings and evidence inventory](structural-findings.md) for the results and remaining limitations.

The objective is to identify when ModeExplorer changes anatomical coherence and meaningful detail, whether the original direction supplies information beyond a magnitude-matched perturbation, and how the native Krea 2 trajectory propagates the change. Work proceeds from intermediate decoding to temporal ablation, then to gated direction, depth and held-out tests. Existing activation-coding diagnostics and measurements remain archived. Use the same native INT8 ConvRot model, production attention route, VAE, conditioning, seed, ER-SDE schedule, planner geometry and guarded Spectrum settings for matched runs.

## Coordinate audit and review

The Krea 2 model uses ComfyUI's Wan21 latent format. The base model passes a dtype-cast solver input to the transformer, then invokes CONST denoising with the original solver input and a float32 velocity. The native estimate is `solver_input − sigma × float(conditional_velocity)`. ComfyUI applies `process_latent_out` when it returns the planner latent. This applies Wan21's channel standard deviations and means to the estimate. The VAE expects those already-converted coordinates.

The original trajectory schema stores the input seen by the transformer, its conditional velocity and `baseline_clean`, the converted estimate formed using that dtype-cast input. With FP16 inference this estimate can differ from the exact solver estimate by the input cast error. The Phase IV review reconstructs the archived tensor in float32, checks that conversion against the saved tensor bit for bit, then passes the saved tensor directly to ordinary VAE Decode. It performs zero DiT forwards. Use a single coordinate basis for all arms in a sheet.

With a compatible CFG-1 ER-SDE run, the observer now also records `solver_input` at each actual callback and saves `solver_clean` from the exact denoised callback after one output conversion. At the final zero transition, ER-SDE returns the last denoised estimate directly. An actual terminal state is checked against that returned latent in VAE coordinates. With the Scale-Locked planner-only output, the low-resolution planner is a separate output from the upscaled final latent, and the final-output patch-lattice correction applies to the latter. Legacy archives have no separately saved exact solver terminal tensor. The archived final PNG and its pixel hash support an independent check after VAE decoding, with the limits of the cast-input estimate recorded in the review manifest.

The review groups states by exact sigma and solver opportunity and retains each arm's actual index. Forecast-only opportunities are listed without generated replacements. It saves individual native-resolution PNGs and synchronized sheets with fixed pixel crops. Read the noisy early predictions as diagnostic views. Record first clearly visible divergence separately from first interpretable anatomy or detail. Evaluate later consolidation, reversal and any change in pose or identity. A displacement norm cannot answer whether a leg or textured structure is more coherent.

## Temporal application

For a matched ablation, require CFG 1, native ER-SDE and verified guarded Spectrum in the planner. Populate the expected A/F route from the qualifying sham and active archives. The application selector affects only the last step of the feature path. Both reference forwards run at the original opportunities, both proposals are recomputed, and the original guidance eligibility and history resets stay in place. The observer records exact application indices, compact proposal hashes and magnitudes, rounded realized magnitude, baseline-relative proposal RMS, reference slots and continuation, and state hashes.

| Arm | Apply at actual evaluation 0 | Apply at actual evaluation 1 |
| --- | --- | --- |
| Matched sham | No | No |
| First only | Yes | No |
| Second only | No | Yes |
| Original active | Yes | Yes |

The sampler emits an unqualified result when route, reference, proposal, edit or terminal requirements fail. The review then checks archived and new controls together. The first input and proposal must match across the four feature arms. At evaluation 1, sham must match second-only before the edit, and original active must match first-only. Proposals can differ between those pairs after the first edit. The paired checks distinguish total schedule effects from differences in initial conditions and replay provenance. Interpret the four final images as first-only versus sham, second-only versus sham, original active versus those two arms. Record synergy or antagonism as an interaction only for a defined structural outcome in the discovery case.

VAE review costs only decodes. Temporal arms add no DiT forwards beyond the two original reference calls. The trace records wall-clock sampler time and capture/serialization timings. Capture wall time can include waiting for already-enqueued device work. Production elapsed time and peak VRAM must come from actual runs.

## Direction and block tests after temporal review

Freeze the supported application schedule and use its matched sham as control. Start with the original repulsion, the existing `reverse` control, and `channel_sign` trials 0, 1 and 2, fixed before viewing the test images. The implementation negates the already-computed bounded repulsion and uses a seeded per-channel sign broadcast across all spatial cells for the channel controls. Use the same reference budget, route and state fingerprints, and compare the compact and realized norms separately. Record unmatched rounding and geometry rather than forcing a rescale. Validate at identical states for the first applied evaluation. If later arms have already diverged, their recomputed proposals measure total trajectory effects. Compare anatomy, coherent detail, identity and composition with arm labels hidden. Advance additional trials only if the initial results leave a specific ambiguity.

After the direction screen, compare block 1 with block 5 and the middle single-stream block. Use the layer count of the loaded checkpoint to set the middle index before generating active images. The same native attention-output hook supplies each block's own spatially pooled representation, before the attention gate and output projection. The two references must tap that block and retain their original continuation. Record compact and representation-relative proposal RMS, rounded realized RMS, and spatial outcomes before drawing depth conclusions. If scales differ substantially, use a separate calibration seed to select a fixed strength that matches the block-1 realized norm. A native-response RMS can help select examples to inspect. The visual structural scores decide whether the selected depth is useful.

## Held-out validation before active generation

Freeze the selected block, direction, strength, application schedule, routing, decoder and rubric before generating any held-out active images. Use separate seed namespaces for discovery, calibration and held-out validation. Recruit at least two native-only cases per stratum before running active. Use articulated poses, ambiguous hands or overlapping limbs, hair and garments, botanical details, already-correct controls, and cases with strong identity or composition requirements. Keep all admitted cases, including correct natives and failed interventions. Record the admission rule, rejected native candidates and seed list in the evaluation manifest.

Shuffle matched sham and active outputs with a reproducible, hidden key. Rate anatomical topology, pose coherence, organized detail, identity, composition, conditioning fidelity, texture, and introduced artifacts separately. Include a no-change option and a low-confidence option when early images are uninterpretable. Report per-stratum counts, regression counts, time, actual/forecast routes and reference-forward cost. Spectral, similarity and edge metrics are secondary checks. A successful result on either discovery case is insufficient to generalize.
