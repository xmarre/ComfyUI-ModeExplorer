# ReDiverse for native Krea 2

ReDiverse ("Restoring Diversity in Distilled Image and Video Generators by Probing Activations", [arXiv:2611.18550](https://arxiv.org/abs/2611.18550)) attributes part of the seed-to-seed collapse of distilled generators to *directional concentration*: in the first denoising step, the distilled student keeps (or inflates) the activation variance of the teacher at an intermediate block, but spreads it over fewer directions. ModeExplorer provides four pieces built on that finding:

| Piece | Node | Status |
| --- | --- | --- |
| ReDiverse-FT: the released first-step LoRA, active only on the opening evaluation | **ModeExplorer ReDiverse (Krea 2)** with an adapter | Implemented reference method; production use on real checkpoints is not yet qualified |
| ReDiverse-TF: per-prompt teacher principal directions added at a block's residual stream on the opening evaluation | **ModeExplorer ReDiverse Calibrate (Krea 2)** → ReDiverse node | Implemented reference method; research |
| Directional-collapse probe: total variance, participation ratio, concentration and spectra across seeds | **ModeExplorer Activation Probe (Krea 2)** | Research instrumentation |
| Teacher-informed cohort steering (`cohort_logdet`) | ReDiverse node, *steering* | Experimental, not validated |

## Intervention coordinates

These methods act at different places in the transformer. Their strengths and magnitudes are not comparable.

| Method | Coordinate | When |
| --- | --- | --- |
| ModeExplorer feature guidance (B0/B1/B2/B5, Middle) | Image-token *attention output before the native gate and output projection* of one block, pooled to a coarse grid | First eligible actual evaluation (presets) or all eligible ones |
| ReDiverse-FT | Weights of the eight linears (`attn.wq/wk/wv/gate/wo`, `mlp.gate/up/down`) of blocks 0–26, as a low-rank additive path | Opening evaluation |
| ReDiverse-TF | *Complete residual-stream output* of one block (default 26), generated-image rows only | Opening evaluation, live samples only |
| Activation probe | Residual-stream output of selected blocks, image rows | One frozen state |

Teacher directions are never applied to ModeExplorer's attention-space tensors. Text rows and reference-image rows are never modified by TF.

## Upstream sources

| Source | Revision |
| --- | --- |
| ReDiverse code, `krea2/` port | [H-furkan/rediverse](https://github.com/H-furkan/rediverse) at `0a39a7ce38c783038e2394c72d1b877c0129552b` |
| Released adapter | [ReDiverse/Krea-2-ReDiverse](https://huggingface.co/ReDiverse/Krea-2-ReDiverse) at `e13314703bf844e06c8caf42055add96288f480e`, file `krea-2-rediverse-comfy.safetensors` |
| Teacher / student used upstream | `krea/Krea-2-Raw` at `6b0ece7f…`, `krea/Krea-2-Turbo` at `98e0fe11…` |

The reference configuration encoded here comes from that revision's `krea2/config.py`, `estimate_directions.py`, `generate.py`, `convert_weights.py` and the model card:

- Anchor block 26 of 28, image rows of the block output.
- FT: LoRA rank 32, alpha 32 (scale 1), 216 modules on blocks 0–26, 103.5 M fp32 parameters; active in sampling step 1 only. Training targets came from Raw rolled four guided steps (Krea guidance 3.5 = CFG 4.5) followed by teacher forward 5.
- TF: 25 conditional-only Raw rollouts per prompt from latents seeded `i * 1000 + j`, `j = 100..124`; teacher forward 5 of 52 (σ 0.96743 → 0.95883; Turbo's first step lands at 0.95672); top 8 directions; coefficients α ~ U[−2, 2] in the paper, U[−4, 4] in the released combined setting.
- Raw schedule: 52 FlowMatch Euler steps with the resolution-dependent μ of Krea2Pipeline (0.906 at 1024²). Turbo: 8 steps with μ = 1.15. The **ModeExplorer Reference Sigmas (Krea 2)** node produces both; Turbo's equals ComfyUI's native `simple` schedule for Krea 2.

The probe numbers differ between upstream files: the repository README and model card report Turbo/Raw ratios at block 26 of PR 0.49, TV 3.44, Conc 7.05 (N = 200, six prompts), while `config.py` records PR 0.62, TV 3.26, Conc 5.54 (N = 200, twenty prompts). Both are upstream measurements and were not reproduced here.

### Upstream-reported results (not reproduced by ModeExplorer)

GenEval, 100 prompts × 16 seeds, 1024², upstream sampling loop:

| Arm | DINO | Vendi | DreamSim | CLIPScore | Δ HPSv2 | Δ ImageReward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Turbo | 50.8 | 2.47 | 20.0 | 33.7 | 0.00 | 0.000 |
| TF, band 2 | 51.8 | 2.57 | 21.3 | 33.7 | −0.08 | −0.005 |
| FT | 64.1 | 3.48 | 30.3 | 33.3 | −0.38 | −0.103 |
| FT + TF, band 2 | 64.7 | 3.59 | 31.3 | 33.3 | −0.48 | −0.113 |
| FT + TF, band 4 | 66.0 | 3.76 | 33.9 | 33.4 | −0.93 | −0.137 |
| Raw | 65.2 | 4.48 | 37.1 | 33.4 | −2.31 | −0.221 |

Most of the reported gain comes from the adapter; TF alone is small at band 2. The combined arm exceeds Raw on DINO distance but stays below Raw on Vendi and DreamSim, and the preference scores fall relative to Turbo. These numbers do not show full recovery of the teacher distribution or a quality-neutral change, and they do not cover identity-constrained, high-resolution or accelerated workflows.

## Adapter

Download `krea-2-rediverse-comfy.safetensors` from the gated [release](https://huggingface.co/ReDiverse/Krea-2-ReDiverse) after accepting the Krea 2 Community License, and place it in `models/loras`. ModeExplorer does not distribute it. The file uses the original Krea 2 module names (`diffusion_model.blocks.{b}.attn.wq.lora_A.weight`, …), which match ComfyUI's native Krea 2 modules one to one; no conversion is needed. The diffusers file of the same release (`transformer.transformer_blocks.{b}.attn.to_q.lora_A.weight`, …) is also accepted and maps to the same modules, following the upstream converter's key table.

Loading is strict: every tensor must be a `lora_A`/`lora_B` pair of one of the eight supported modules, ranks must agree, values must be finite, shapes must match the loaded model's linear layers, and at least one `lora_B` must be nonzero. Anything else is rejected with an error that names the tensor. The parser was validated against files written with the upstream converter's key layout; the released file itself is gated and was not inspected here.

## ModeExplorer ReDiverse node

Place it on the MODEL path before ModeExplorer feature guidance, Sol-Krea2 and Spectrum. It composes with a ModeExplorer node on the same path in either order and with ModeExplorer Response. It refuses projection guidance, ModeExplorer initialization, Full Spectrum and temporal ablation routes on the same path.

| Control | Meaning |
| --- | --- |
| **Adapter** | ReDiverse-FT file from the loras folder, or `none` for TF only. |
| **Strength** | Multiplier of the low-rank path. 1.0 is the released recipe. 0 applies nothing (see the route-matched sham below). |
| **Band** | TF coefficients α ~ U[−band, band]. Used only with connected directions. 2 is the paper setting, 4 the released combined setting. |
| **Directions** | Output of the Calibrate or Directions node. |
| Schedule, Adapted sigmas (1–16) | Advanced. See the scheduling contract. Anything other than `first_evaluation` with one sigma is experimental. |
| Direction count, Coefficient trial | Advanced. Number of archived directions used; an independent deterministic coefficient draw for the same seed. |
| Steering | Advanced. `random` is ReDiverse-TF; `cohort_logdet` is experimental and `random_norm_matched` is its control. |
| Diagnostics | Logs a `MODEEXPLORER_TRACE` record with admitted sigmas, adapted evaluations, adapter module calls, TF applications, coefficients, the realized TF offset magnitude and the executed temporal schedule. |
| Start evaluation (0–15) | Advanced, experimental. First window slot, as a zero-based distinct evaluation sigma. Earlier evaluations run natively. 0 is the reference. |
| Strength profile, Custom profile | Advanced, experimental. Per-slot multipliers of **Strength** across the window: `constant` (reference), `linear_decay`, `cosine_decay`, `linear_increase`, `bell_curve` or `custom` (exactly *adapted sigmas* comma-separated values). **Strength** stays the global amplitude; the effective strength of slot *i* is Strength × multiplier *i*. The node logs the effective sequence. |

### Scheduling contract

Terms: a *solver step* moves the latent from σᵢ to σᵢ₊₁; a *model evaluation* is one call of the transformer requested by the solver (one per branch group: packed CFG branches form one call, separately batched branches form several calls at the same σ); *auxiliary forwards* are extra transformer calls made by ModeExplorer references, Response probes or initialization; *forecasts* are Spectrum outputs that replace a transformer call.

`first_evaluation` (default) adapts every model evaluation at the first evaluated σ of the sampler invocation, including all CFG branches and the auxiliary forwards nested in those evaluations, and stops at the first evaluation at any other σ. In the solver's terms:

| Sampler | Adapted evaluations | Relation to upstream |
| --- | --- | --- |
| Euler (reference) | The step-1 evaluation | Identical to the upstream rule "adapter active in sampling step 1" |
| Euler ancestral, DPM++ 2M, other single-evaluation solvers | The step-1 evaluation | Same evaluation; the solver update differs from upstream |
| Heun, DPM++ 2S, other multistage solvers | The first stage of step 1 only | Later stages of step 1 run on the native model |
| ER-SDE, SA-Solver | The first evaluation, at ComfyUI's slightly offset first σ | Same evaluation; σ is just below 1 |
| CFG > 1 | Both branches of that evaluation | Upstream Turbo uses no CFG |

`first_interval` (experimental) adapts every evaluation with σ above the schedule's second σ, which includes the intermediate stages of multistage solvers inside step 1. Delayed activation and strength profiles are rejected with it. Neither extension has upstream evidence.

### Temporal schedules (experimental)

Under `first_evaluation`, the evaluations of a sampler invocation are numbered by *distinct evaluation σ*, from 0, in the order they occur:

- An evaluation at the same σ as the previous one belongs to the same slot. This covers another CFG branch, a nested ModeExplorer reference or Response forward, and a same-σ corrector.
- Every new σ opens the next slot.
- Slots are not solver steps, not percentages of denoising progress and not transformer-call counts. With Heun, the corrector at σᵢ₊₁ and the next predictor at σᵢ₊₁ share one slot; with DPM++ 2S the midpoint σ is a slot of its own; ER-SDE's offset first σ is slot 0.

*Start evaluation* S and *adapted sigmas* N select slots S … S + N − 1. Earlier slots run natively, and the schedule closes at the first σ after the window. Numbering always starts at the beginning of the eligible invocation, never at the first nonzero multiplier. Evaluations that Spectrum replaces with forecasts still pass through ModeExplorer's wrapper, so they are numbered too and cannot shift the window. Partial denoise, masks and tagged high-resolution stages bypass ReDiverse entirely, whatever S is: delayed activation schedules an intervention later inside a supported full-denoise invocation, it does not make refinement passes eligible.

Profile multipliers over a window of N slots, with position i = 0 … N − 1 and t = i / (N − 1):

| Profile | Multiplier | Notes |
| --- | --- | --- |
| `constant` | 1 | The reference. With N = 1 this is the upstream recipe. |
| `linear_decay` | 1 − t | Last slot is 0. |
| `cosine_decay` | (1 + cos πt) / 2 | Last slot is 0. |
| `linear_increase` | t | First slot is 0. The profile does not restart at the next slot; the zero slot keeps its position. |
| `bell_curve` | sin πt | Both ends are exactly 0 and the values are mirror-symmetric. For even N the discrete peak is below 1; it is not renormalized. N = 2 gives 0, 0. |
| `custom` | the listed values | Exactly N finite nonnegative values. Malformed lists, NaN, infinity, negatives and wrong lengths are errors. |

Every built-in profile is 1 for N = 1. Nothing is normalized, clamped, padded, truncated or repeated. An effective strength above 2 is an error. Profiles with equal multiplier sums have equal *nominal* cumulative strength, which is not equal activation displacement or intervention energy. Use custom lists to build nominally matched comparisons.

Zero slots, whether from a profile or from Strength 0, run no adapter arithmetic and no adapter hooks. They still belong to the window, so Spectrum routing and the slot's diagnostics are the same as for a nonzero slot.

If the sampler has fewer evaluations than the window needs, only the reachable slots run and the run reports `temporal_window_truncated`. If none is reachable, it reports `temporal_window_not_reached`. A schedule never wraps around, never repeats a finished profile and never carries over into another invocation.

These temporal options apply to ReDiverse-FT only. With active ReDiverse-TF steering (connected directions and a nonzero band), a start above 0 or a non-constant profile is an error: TF keeps its established constant per-evaluation offset, from evaluation 0, for up to 16 slots. Set the band to 0 to use temporal FT options next to a connected archive.

With Diagnostics on, the `temporal` record of the trace lists:

- the requested start, window length, profile, global strength, multipliers and planned strengths;
- every observed evaluation σ, and the number of native evaluations before the window;
- the reached and unreached slots, and the window's entry and exit σ;
- for each slot: its index, σ, multiplier, effective strength, state (`adapter_active`, `zero_strength` or `steering_only`), calls, live and auxiliary forwards, and adapter module calls.

The top-level `adapted_evaluations` keeps its established meaning: admitted window evaluations with a live forward, zero-strength slots and the Strength 0 sham included. `temporal.adapter_active_evaluations` counts distinct window slots whose adapter ran at a nonzero effective strength in a live forward; `temporal.adapter_active_calls` counts the admitted calls behind them, which is larger under split CFG. A probe measures only the opening evaluation, so a probe through a ReDiverse node requires start evaluation 0 and records the probed effective strength. A selected zero slot is not evidence that adapter arithmetic ran. That evidence is a nonzero `adapter_module_calls`, which equals 216 × (live + auxiliary forwards) for the released adapter.

The adapter is bypassed (the sampler runs natively and the reason is logged) for partial denoise (first σ not 1), latent masks, unsupported latent layouts and tagged Scale-Locked high-resolution stages. Projection guidance, multi-GPU and context-window dispatch keep ModeExplorer's existing rejections.

### Implementation

During an admitted evaluation, PyTorch forward hooks are registered on the executing transformer's modules and removed in `finally`, including on exceptions and interrupts. Each adapted linear returns `cast(float32(y) + strength · scale · B(A·float32(x)))`, the same arithmetic as an fp32 peft LoRA on a bf16 base. Nothing is merged into weights, so:

- upstream weight LoRAs (identity LoRAs, for example) remain applied on every step;
- the base linear can be any native implementation, including quantized layers, because the hook only uses its input and output;
- no model reload, weight backup or re-patching occurs between the adapted and native evaluations.

While hooked, the adapted linears are marked `comfy_force_forward`, as ComfyUI's own bypass adapters do, so fused quantized paths fall back to the module call. Every actual transformer forward inside an admitted evaluation is verified on its own: a pre-hook on the first block opens a record, the last block closes it, and the record must contain every adapted linear (and, for a live forward with TF, exactly one steering application). A forward that executes none or only some of the adapted linears stops sampling with an error naming them; an auxiliary reference forward cannot vouch for a live one. A research forward that a probe stops intentionally after block *b* is verified on its executed prefix: every adapted linear in blocks ≤ *b* (and the steering hook, when its block is ≤ *b*). An admitted evaluation that requests an intervention (nonzero strength or band) but runs no live transformer forward is an error, because guarded Spectrum keeps admitted evaluations actual; only the strength-0 route sham records such a case as `admitted_evaluation_not_executed`.

**Execution contract.** Supported execution is ComfyUI's serialized queue: one sampler invocation at a time per loaded transformer. The hooks act only in the execution context that owns the scope (a `ContextVar`), so another context calling the same modules sees native outputs, and a second scope on the same transformer while one is active is refused with an error. The temporary `comfy_force_forward` flag is a module attribute and is visible to any overlapping execution; overlapping or concurrent execution on shared modules is therefore unsupported, not merely untested.

A requested arm is never silently changed. Partial denoise, masks, unsupported latent layouts and tagged high-resolution stages bypass the whole ReDiverse configuration (both adapter and steering) with a logged reason, as documented above. TF requested with unsupported conditioning, without an integer noise seed, with an unverifiable batch layout (for example a batch split under memory pressure), or `cohort_logdet` without a usable cohort is an error; disconnect the directions to run the adapter alone.

The adapter tensors are copied to the evaluation device once per sampler invocation (about 414 MB for the released fp32 adapter) and released when the invocation ends. Outside admitted evaluations no hook exists. With TF, the first adapted evaluation hashes the positive conditioning once (one device-to-host copy) to check the archive; there are no other synchronizations unless Diagnostics is enabled.

### Spectrum

With the [Spectrum ModeExplorer overlay](https://github.com/xmarre/ComfyUI-Spectrum-Krea2/pull/6), the pending ReDiverse evaluation is reported through ModeExplorer's `guidance_pending` contract. Spectrum therefore runs it as an actual evaluation, and the first native step after it starts a new forecast history (`ModeExplorer early-feature regime transition`). Adapted outputs never anchor a forecast of native steps, and a forecast can never consume the adapted evaluation; if an admitted intervention nevertheless runs no live forward, sampling stops with an error.

Without that contract, the pass runs all-actual through Spectrum's native control (`spectrum_all_actual` in the trace). The released Spectrum v0.2.2 refuses to forecast with any ModeExplorer patch, which is also correct but unaccelerated. Full Spectrum is rejected with ReDiverse.

For a temporal window, `guidance_pending` answers whether the *next* evaluation's slot lies inside the window, zero slots included. Spectrum's existing regime rule then gives the conservative routing, with no companion change:

1. Before a delayed window, ordinary guarded forecasting.
2. Entering the window, forecast history is cleared and the first slot is actual.
3. Every slot inside the window is actual, so no forecast is made from one effective strength for another, and native history never feeds an adapted slot.
4. The first evaluation after the window is actual and starts fresh native history, so adapted outputs never anchor native forecasts.
5. Forecasting resumes once Spectrum's own conditions hold again: enough anchors, the extrapolation guard and the actual tail step.

Combined with ModeExplorer feature guidance, each change of the combined pending state is such a boundary. The cost of a window of N slots is N + 1 forced actual evaluations plus Spectrum's history rebuild after it, i.e. fewer forecasts.

Because the boundary changes Spectrum's route, compare a ReDiverse arm with the **route-matched sham**: the same node and the same temporal configuration with Strength 0 (and band 0). It keeps the same window, forced-actual evaluations and fresh history, but applies nothing. Without Spectrum, a Strength 0 sham is bitwise identical to the native model.

### Scale-Locked

ReDiverse applies to the composition-forming planner stage only. Tagged high-resolution stages and partial-denoise refinements bypass it, so the existing composition lock and refinement behavior are unchanged. A temporal window is scheduled, numbered, truncated and closed within the planner invocation. A window that the planner did not complete is reported as truncated; it is never continued or restarted in a later stage. TF directions must be calibrated at the planner's latent size (see below).

## ReDiverse-TF calibration

**ModeExplorer ReDiverse Calibrate (Krea 2)** takes the teacher MODEL (Krea 2 Raw, with the same upstream weight LoRAs as the student when they matter), the positive conditioning and a latent with the generation size. For rollout *j* it starts from the reference packed draw seeded `seed * 1000 + 100 + j`, runs forwards 1 … τ−1 as Euler steps on the teacher schedule, captures the block-output image rows at forward τ and stops the forward there. Defaults reproduce the reference configuration: block 26, τ = 5, 52 steps, 25 rollouts, 8 directions, conditional-only (CFG 1). A negative conditioning with CFG > 1 and a connected SIGMAS schedule are explicit research overrides recorded in the archive.

Directions follow paper Eq. 2 with the unnormalized centred Gram G = XcXcᵀ, eigenpairs (λₖ, vₖ): uₖ = Xcᵀvₖ / ‖Xcᵀvₖ‖ and sₖ = std(√λₖ vₖ) with the biased standard deviation, which equals √(λₖ/N). The Gram is accumulated in float64 over feature chunks, each centred before its product; the basis is stored in bfloat16 as upstream. Directions with eigenvalue ≤ 10⁻¹⁰ of the leading one are stored as zero rows with zero spread and excluded from the valid count.

Guided calibration (CFG above 1) requires a negative conditioning; without one the node refuses instead of recording a guided label for a conditional-only run. A connected SIGMAS schedule must be finite, start at 1, stay in [0, 1], decrease strictly and cover the requested forward.

Cost: 25 × (τ−1) full teacher forwards plus 25 truncated ones, batched by 5, and N × D bf16 staging (1.26 GB at 1024²) on the GPU, or on the CPU with *workspace* `cpu`. The archive records rollout and PCA seconds; the node summary and the `ModeExplorer ReDiverse calibration` log line add archive write/validation and total seconds, which the evaluator records per job. Upstream reports about 60 s per prompt on an RTX PRO 6000 and about 400 MB per archive at 1024². ComfyUI caches the node's output, so later seeds of the same prompt reuse it; changing the prompt, size or teacher recalibrates. Report calibration time with every TF comparison; TF is not free because its teacher forwards run before generation.

### Archive

Archives are safetensors files in `output/modeexplorer_rediverse/directions/` with tensors `U [K, D]`, `s [K]`, `eigenvalues [N]` (unnormalized Gram; covariance eigenvalue = value / N) and the teacher mean, plus JSON metadata: schema and implementation, upstream revision, coordinate and block, token grid and width, latent shape, teacher weight fingerprint, weight-patch fingerprint and a summary of wrappers, patches and attention override, teacher noise scale, positive and negative conditioning fingerprints, the complete teacher schedule and the executed part of it, solver, guidance, seeds and seed rule, batch, a digest of the initial noises and of the executed inputs at the captured forward, rollouts, normalization, staging and basis dtype, direction counts, spectrum summary and timings. The weight and patch fingerprints are *sampled* (fixed slices of named tensors, the first 256 values of each patch tensor) and say so in the archive; they distinguish checkpoints and patch sets in practice but are not full checkpoint hashes. The file name includes a hash of the complete calibration identity (everything except labels, results and timings); a repeated calibration with the same identity is written next to the first as `-r2`, `-r3`, …, never over it.

Loading never unpickles. Rejected: unknown schemas, missing fields, extra tensors, shape or dtype disagreement with the metadata, a coordinate other than the block-output image rows, nonfinite values, negative spreads, a valid-direction count that differs from the number of positive spreads (including all-zero spreads), nonzero rows without spread, basis rows that are not orthonormal within 0.02 (bf16 storage tolerance), eigenvalues that are not a nonnegative descending spectrum of one value per rollout, spreads that differ from √(eigenvalue / rollouts) by more than 0.1 %, and files over 8 GiB. At sampling time a different positive conditioning or token grid is an error: directions are never resized or reused across prompts. The **ModeExplorer ReDiverse Directions (Krea 2)** node reloads an archive by name.

### Inference

During the opening evaluation, the block hook adds Σₖ αₖ sₖ uₖ, computed in the basis dtype and cast to the activation dtype, to the generated-image rows of every live sample (all CFG branches of a sample receive the same offset). ModeExplorer reference shadows and other auxiliary forwards keep the adapter but receive no offset. Coefficients come from an independent CPU generator per sample, seeded `7_000_000 + seed * 1009 + row (+ trial * 1_000_003)` from the noise seed, so the sampler's own random stream is untouched. TF adds no transformer forward; per invocation it moves the K × D basis to the device (≈400 MB at 1024² for K = 8) and forms one batch × D offset.

### Experimental cohort steering

`cohort_logdet` replaces the random draw for an equally conditioned batch of 2–8. In teacher-whitened coordinates zᵦ = (U hᵦ)/s of the student's own block output, the cohort covariance C gives ∇ log det(C + I) ∝ (C + I)⁻¹(zᵦ − z̄); the identity stands for the teacher's whitened covariance, which damps directions where the cohort already matches or exceeds the teacher. The realized offset is Δh = U diag(s) α, whose norm for the orthonormal basis is √(Σₖ sₖ² αₖ²); equal coefficient norms therefore do not give equal displacements when the spreads differ. Each sample's step is rescaled, without clipping, so its activation-space norm equals the RMS norm of the uniform draw, band·√(Σₖ sₖ²/3) (coefficients may then exceed the band). The random arm (E) matches that norm only on average; the norm-matched control is `random_norm_matched`, which uses the same uniform direction as `random` rescaled to exactly the same per-sample activation norm as `cohort_logdet`. The trace records the coefficients and the pre-rounding offset norm ‖Σₖ αₖ sₖ uₖ‖ of every sample; Diagnostics adds the realized displacement norm of every row after the bf16 basis product and native-dtype addition, and its ratio to the row's activation norm, so amplitude matching can be checked on the realized offsets. Single images and degenerate cohorts are errors. There is no acceptance test and no evidence that this is better than random; treat it as a separate research arm.

## Directional-collapse probe

**ModeExplorer Activation Probe (Krea 2)** runs one conditional forward per sample at a frozen state (σ = 1 by default) from the reference packed draws `1_000_000 + seed * 100003 + i`, captures the selected blocks' output image rows and reports per block:

- total variance TV = Σλ, participation ratio PR = (Σλ)²/Σλ², concentration TV/PR, entropy effective rank exp(−Σ p log p), the top 32 eigenvalues, cumulative explained variance and the leading share;
- covariance normalization N (population), with PR capped at N − 1. These are sample-limited spectra, not the intrinsic dimension of the model.

Activations are staged as bf16 (10 GB per block at N = 200, 1024²). The workspace limit (16 GiB default for the probe, 8 GiB for calibration) covers staging *and* the analysis temporaries: two float64 N × 2²⁰ Gram/projection chunks, the float32 column mean and, for calibration, the float32 and bf16 copies of the K × D basis; a reference basis adds K × D bf16. It excludes model weights, activations, cached allocator memory and transfer buffers, so it bounds research allocations, not total device memory. Loading and validating an archive is outside the budget: it holds the stored basis on the host (K × D bf16, about 1.3 GiB for K = 8 at 3.5 MP) plus one float64 K × 2²² chunk for the orthonormality check. Blocks are processed in groups that fit; each group repeats the forwards and stops after its deepest block. No D × D covariance is formed and no pooling is used. The Gram is accumulated exactly in float64, but over the bf16-staged values: activations computed in fp16 or fp32 are rounded to bf16 first, so the spectra are those of the staged values.

At 3.5 MP (about 13,700 image tokens × 6,144 features), one 200-sample block stages about 31.3 GiB, so the default probe workspace refuses that configuration; calibration with 25 rollouts stages about 3.9 GiB plus about 3.8 GiB of basis reconstruction. Fit full-resolution probes to the device explicitly (fewer samples, one block, or a lower resolution), and defer them until the adapter has shown practical value. Reports go to `output/modeexplorer_rediverse/probes/`; `python -m tools.rediverse_probe_report teacher.json student.json` prints student/teacher ratios and the upstream collapsed-block rule (largest step in the concentration ratio). A probe through a ReDiverse node measures the adapted first step.

With a connected direction archive at a probed block and the same token grid, the report adds a comparison in the archive's subspace, using only the N × K projections:

- student variance along each teacher direction and its ratio to sₖ² (retained variance, per direction);
- subspace alignment ‖UₜᵀUₛ‖²_F / K between the teacher directions and the student's own top K;
- eigenvalues of Cₜ − Cₛ in the teacher subspace, split into *deficit* (positive) and *excess* (negative) parts. The difference is indefinite in general and is reported, not injected;
- mean displacement (overall RMS, along each teacher direction and relative to the cohort's spread).

Every comparison is qualified from recorded evidence, never from σ alone. Probe reports record the seed and seed rule, batch, latent shape, noise scale, a digest of the initial noises and a digest of the model inputs actually executed at the captured forward (positive rows, row by row, so batch chunking does not change it); archives record the same digests for the teacher rollouts. The categories, in `comparison` of a teacher/student pair (`tools/rediverse_probe_report.py`) and in `reference_comparison` of a probe against an archive:

| Category | Evidence |
| --- | --- |
| `exact_matched_noisy_state` | Same conditioning and identical executed inputs. Only this sets `matched_inputs`. |
| `matched_noise_state_representation_differs` | Same conditioning, initial noise, σ and noise scale at the first forward, but the executed inputs differ (for example another input dtype). |
| `matched_initial_noise_evolved_trajectories` | Same conditioning and initial noise, captured after different trajectories. |
| `same_conditioning_independent_samples` | Same conditioning, other noise: a population comparison. The default archive (teacher forward 5 from rollout noise) against a probe (σ 1 from probe noise) is in this category. |
| `cross_conditioning_exploratory` | Different or unknown conditioning. |
| `incompatible_geometry` | Coordinate, token grid, width or sample count differ (ratios are still printed). |

Mean shifts, subspace rotations, representation scaling and differing trajectory states can all produce variance differences; the probe separates the first three only within the teacher's K-dimensional span.

Probe report names hash the complete experimental configuration: model and patch fingerprints, model options, attached ReDiverse and ModeExplorer settings, conditioning, latent and token geometry, samples, seed and noise digest, σ and noise scale, requested blocks, batch, reference archive and workspace device. A repeated run of the same configuration is written as `-r2`, `-r3`, …, never over an earlier report. A probe through a ReDiverse node must start at σ 1, because other σ bypass the adapter. A probe through a ModeExplorer feature-guidance node captures, counts and fingerprints only the live samples: the reference, response and initialization forwards that ModeExplorer runs around them are excluded, so a guided probe measures the guided live activations at every batch size.

## Evaluation

Use a dedicated ComfyUI server and the existing evaluator and analyzer. Two separate panels:

**Reference panel** (upstream-style: Euler, reference schedules, Turbo without CFG, Raw at CFG 4.5):

```shell
python tools/evaluate.py --model krea2_turbo.safetensors --teacher-model krea2_raw.safetensors --rediverse-adapter krea-2-rediverse-comfy.safetensors --profiles tools/rediverse_reference_profiles.json --seed-count 8 --out bench-output/rediverse-reference --server-log /path/to/comfy.log
```

Arms: A Turbo, B Turbo + ModeExplorer B1 (strength 0.03, fidelity 0.70), C Turbo + FT, D Turbo + FT + B1, E Turbo + TF (band 2), F Turbo + FT + TF (band 4), G Raw (52 steps). Use a local `--panel` file with eight prompts for the 8 × 8 screening panel (448 images). TF arms calibrate per prompt inside the graph; the run records report the calibration log line of the job that executed it.

**Production panel** (ER-SDE, guarded Spectrum, Scale-Locked planner; arms A–D and the route-matched sham):

```shell
python tools/evaluate.py --model krea2_int8_convrot.safetensors --rediverse-adapter krea-2-rediverse-comfy.safetensors --profiles tools/rediverse_production_profiles.json --sampler er_sde --steps 13 --width 2048 --height 2048 --planner-megapixels 1.0 --seed-count 8 --out bench-output/rediverse-production --server-log /path/to/comfy.log
```

Add `--identity-lora` and `--identity-trigger` for the identity panel; every arm then uses the same identity LoRA and conditioning, and TF arms calibrate on a teacher carrying the same model and identity LoRAs at the same strengths, so the teacher directions do not include identity variation the student is constrained against. Then:

```shell
python tools/analyze.py bench-output/rediverse-reference --baseline A_turbo --embeddings embeddings.json --review
python tools/analyze.py bench-output/rediverse-production --baseline C_route_sham --stage planner --review
```

`--embeddings` takes locally computed DINOv2 or DreamSim vectors keyed by blinded IDs and adds cosine diversity and Vendi per prompt; the analyzer aggregates at the prompt level with paired bootstrap intervals. Review sheets are blinded; the private mapping stays out of the shared archive. Give reviewers the prompt and, for identity panels, authorized reference photographs, and rate composition, framing, pose, anatomy, identity, adherence, detail and artifacts at native size. Count identity drift under an identity LoRA as a regression, not as diversity. Report wall time, calibration time, peak VRAM and the trace counters (actual, forecast and auxiliary evaluations, adapted evaluations, TF applications, realized offset). Decide FT on A–D first; TF needs its own calibration-cost accounting. Confirm any pilot result on a held-out panel before changing defaults.

**Temporal panels** (experimental FT schedules). These are screening panels to run on a small prompt and seed set first; extend them only after production evidence. Spectrum is off in panels A–D, to isolate the adapter's behavior. Add the sampler, steps, planner and LoRA options of the workflow under test on the command line.

| Profiles | Arms |
| --- | --- |
| `tools/rediverse_temporal_duration_profiles.json` (A) | Native; route sham 4 × 0; reference 1 × 1.0; weak 1 × 0.5; constant windows of 2, 4, 6 and 8 at 0.5; 4, 6 and 8 at 1.0; boundary research 16 × 0.5 (not a recommendation) |
| `tools/rediverse_temporal_shape_profiles.json` (B) | Native and six custom profiles with nominal cumulative strength 2.0: constant, front-loaded, increasing, bell, delayed within the window, concentrated opening (2, 0, 0, 0) |
| `tools/rediverse_temporal_delay_profiles.json` (C) | Native; 4 × 0.5 starting at evaluations 0, 1, 2, 4 and 6; isolated single slots at 1.0 starting at 0, 1, 2 and 4 |
| `tools/rediverse_temporal_combination_profiles.json` (D) | Native, B0, reference FT, extended FT 4 × 0.5, and extended FT with B0 and with B1. B0 and B1 use strength 0.30 with Adaptive off and share every other ModeExplorer setting |
| `tools/rediverse_temporal_spectrum_profiles.json` | Guarded Spectrum: native, 4 × 0.5, start 2, front-loaded, with a route-matched sham per window |

```shell
python tools/evaluate.py --model krea2_int8_convrot.safetensors --rediverse-adapter krea-2-rediverse-comfy.safetensors --profiles tools/rediverse_temporal_duration_profiles.json --sampler er_sde --steps 16 --seed-count 4 --out bench-output/rediverse-temporal-duration --server-log /path/to/comfy.log
```

Every run record stores its arm configuration with `$teacher` and `$adapter` resolved to the files of the run, and a hash of it, so arms that differ only in start position or profile are never merged. Judge temporal arms on the same dimensions as above, separately:

- anatomy and structural coherence
- pose, framing and setting variation
- prompt adherence
- identity retention under an identity LoRA
- meaningful detail and introduced artifacts
- runtime and forecast utilization

Paired same-seed differences show trajectory sensitivity; they do not establish population-level diversity. Prompt-level aggregation over several prompts and seeds is required before any temporal configuration is recommended.

The reference panel's Raw arm and the production panel do not reproduce the upstream sampling loop exactly (ComfyUI keeps the latent in float32 between steps, upstream in bf16; text-encoder padding and masking can differ), so their numbers are not reproductions of the upstream table.

## Recommendations

These are starting points for evaluation, not established optima.

| Workflow | Suggestion |
| --- | --- |
| Ordinary Turbo | FT at strength 1.0 on an A/B against native ([FT workflow](../examples/rediverse_ft.json)), then the [A–D panel](../examples/rediverse_panel.json). |
| Identity LoRA | Apply the identity LoRA upstream of ReDiverse. Start FT at strength 0.5–1.0 and judge identity separately from composition. Do not tune TF on identity panels before FT is qualified. |
| Raw | Not applicable: the adapter targets Turbo's first step. Raw is the teacher and the diversity reference. |
| Spectrum | Use guarded Spectrum with the ModeExplorer overlay and compare against the strength-0 route-matched sham. |
| Scale-Locked planner | FT on the planner only ([production workflow](../examples/rediverse_scale_locked.json)). Calibrate TF at the planner latent size if used. |
| Temporal FT (experimental) | Start from the [temporal workflow](../examples/rediverse_temporal.json) and the temporal panels. Keep the reference recipe (start 0, one slot, constant) as the default until panels over several prompts and seeds support a change. |
| Research | [TF workflow](../examples/rediverse_tf.json) and the [probe workflow](../examples/rediverse_probe.json) (Raw, Turbo and Turbo with the adapter on blocks 22–27). |

## First real-checkpoint check

Run [the FT workflow](../examples/rediverse_ft.json) with native Turbo, Euler, 8 steps, `simple`, CFG 1, 1024², no Spectrum, Scale-Locked or feature guidance, the released adapter at strength 1, band 0, directions disconnected and Diagnostics on. The `rediverse` record of the `MODEEXPLORER_TRACE` line should read `admitted_sigmas` `[1.0]`, `adapted_evaluations` 1, `live_forwards` 1, `auxiliary_forwards` 0, `adapter_module_calls` 216 (one call per adapted module), `tf_applications` 0, `forecast_consumed` 0 and empty `reasons`. Any loading error names the offending tensor.

## Verified scope

Established on CPU with ComfyUI's native Krea 2 transformer at tiny width (see `tests/test_rediverse*.py`):

- zero strength and zero band are bitwise native; disabling restores the native graph;
- the adapted run equals an independent reference (first Euler step on merged weights, remaining steps native) within 2·10⁻⁵, for CFG 1 and packed CFG 2, at strengths 1 and 0.5; upstream weight patches remain applied;
- temporal schedules equal an independent per-step reference (each Euler step on weights merged at its effective strength) within 2·10⁻⁵ for constant, delayed, increasing, cosine and custom profiles with zero slots; zero-strength temporal shams are bitwise native; slots follow distinct σ under split CFG, Heun and ER-SDE; references inherit their evaluation's strength without advancing the window; truncated and unreachable windows are reported; with guarded Spectrum a delayed window gets forecasts before it, actual slots and exit, two regime transitions and resumed forecasting, with sham routes equal to treated ones; Scale-Locked windows stay in the planner;
- multistage solvers, ER-SDE's offset σ, split CFG dispatch, partial denoise, masks and high-resolution stages follow the contract above;
- no hook survives a run, an exception or an interrupt; repeated runs are deterministic; the shared model is unchanged;
- a requested intervention whose admitted evaluation runs no live forward fails; truncated probe forwards verify the adapter on their executed prefix;
- probes through ModeExplorer B1 (with and without the adapter, batch 1 and an uneven remainder) capture the live samples, not reference forwards;
- an actual forward that executes none or only some adapted linears fails, also when an auxiliary forward did execute them; overlapping scopes on one transformer are refused, and hooks leave other execution contexts native;
- TF adds exactly Σ αₖ sₖ uₖ to image rows (text rows unchanged) on the first evaluation only, rejects other prompts and grids, adds no transformer call and leaves the sampler's noise stream unchanged;
- calibration reproduces a manual rollout's PCA, does not modify the teacher, and archives round-trip; different preceding schedules and negative prompts give distinct archive identities; repeats never overwrite; malformed and inconsistent archives are rejected; covariance statistics match explicit covariances for known spectra and are invariant to chunking;
- probe pairs with other seeds, prompts or geometries are never certified as matched; probe identities change with seed, blocks, batch, reference archive and ModeExplorer settings;
- `cohort_logdet` and `random_norm_matched` produce identical per-sample activation-space offset norms;
- the evaluator records the effective sampler configuration for KSampler, SamplerCustom (reference schedules) and Scale-Locked graphs through its main recording path;
- with Spectrum PR #6 and Scale-Locked v1.3.2: the adapted step is actual, the next starts fresh history, forecasts resume afterwards, the planner alone is adapted; without the deferred contract the pass is all-actual.

Not established: behavior and timing on real Krea 2 checkpoints, CUDA, native INT8 ConvRot and Comfy Kitchen kernels, torch.compile, memory peaks at production resolutions, and every perceptual property (diversity, anatomy, identity, adherence, detail, artifacts). The code paths are kernel-independent by construction, but that is not evidence of correct quantized CUDA behavior.

## Licensing

The method, reference configuration and seed rules follow the Apache-2.0 [ReDiverse](https://github.com/H-furkan/rediverse) repository; see [third-party notices](../THIRD_PARTY_NOTICES.md). The adapter weights are a Derivative of Krea 2 under the Krea 2 Community License and Acceptable Use Policy; obtain them from the official release and comply with its terms, including its content-filtering requirement for deployments.
