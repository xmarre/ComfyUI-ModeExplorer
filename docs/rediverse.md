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
| Schedule, Adapted sigmas | Advanced. See the scheduling contract. Anything other than `first_evaluation` with one sigma is experimental. |
| Direction count, Coefficient trial | Advanced. Number of archived directions used; an independent deterministic coefficient draw for the same seed. |
| Steering | Advanced. `random` is ReDiverse-TF; `cohort_logdet` is experimental. |
| Diagnostics | Logs a `MODEEXPLORER_TRACE` record with admitted sigmas, adapted evaluations, adapter module calls, TF applications, coefficients and the realized TF offset magnitude. |

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

`first_interval` (experimental) adapts every evaluation with σ above the schedule's second σ, which includes the intermediate stages of multistage solvers inside step 1. *Adapted sigmas* > 1 (experimental) extends `first_evaluation` to the first N distinct evaluation σ. Neither has upstream evidence.

The adapter is bypassed (the sampler runs natively and the reason is logged) for partial denoise (first σ not 1), latent masks, unsupported latent layouts and tagged Scale-Locked high-resolution stages. Projection guidance, multi-GPU and context-window dispatch keep ModeExplorer's existing rejections.

### Implementation

During an admitted evaluation, PyTorch forward hooks are registered on the executing transformer's modules and removed in `finally`, including on exceptions and interrupts. Each adapted linear returns `cast(float32(y) + strength · scale · B(A·float32(x)))`, the same arithmetic as an fp32 peft LoRA on a bf16 base. Nothing is merged into weights, so:

- upstream weight LoRAs (identity LoRAs, for example) remain applied on every step;
- the base linear can be any native implementation, including quantized layers, because the hook only uses its input and output;
- no model reload, weight backup or re-patching occurs between the adapted and native evaluations, and cloned or unrelated workflows never see the hooks.

The adapter tensors are copied to the evaluation device once per sampler invocation (about 414 MB for the released fp32 adapter) and released when the invocation ends. Outside admitted evaluations no hook exists. With TF, the first adapted evaluation hashes the positive conditioning once (one device-to-host copy) to check the archive; there are no other synchronizations unless Diagnostics is enabled.

### Spectrum

With the [Spectrum ModeExplorer overlay](https://github.com/xmarre/ComfyUI-Spectrum-Krea2/pull/6), the pending ReDiverse evaluation is reported through ModeExplorer's `guidance_pending` contract. Spectrum therefore runs it as an actual evaluation, and the first native step after it starts a new forecast history (`ModeExplorer early-feature regime transition`). Adapted outputs never anchor a forecast of native steps, and a forecast can never consume the adapted evaluation; if an admitted evaluation is nevertheless not executed, the trace reports `admitted_evaluation_not_executed`.

Without that contract, the pass runs all-actual through Spectrum's native control (`spectrum_all_actual` in the trace). The released Spectrum v0.2.2 refuses to forecast with any ModeExplorer patch, which is also correct but unaccelerated. Full Spectrum is rejected with ReDiverse.

Because the boundary changes Spectrum's route, compare a ReDiverse arm with the **route-matched sham**: the same node with strength 0 (and band 0) keeps the forced-actual boundary and fresh history but applies nothing. Without Spectrum, strength 0 is bitwise identical to the native model.

### Scale-Locked

ReDiverse applies to the composition-forming planner stage only. Tagged high-resolution stages and partial-denoise refinements bypass it, so the existing composition lock and refinement behavior are unchanged. TF directions must be calibrated at the planner's latent size (see below).

## ReDiverse-TF calibration

**ModeExplorer ReDiverse Calibrate (Krea 2)** takes the teacher MODEL (Krea 2 Raw, with the same upstream weight LoRAs as the student when they matter), the positive conditioning and a latent with the generation size. For rollout *j* it starts from the reference packed draw seeded `seed * 1000 + 100 + j`, runs forwards 1 … τ−1 as Euler steps on the teacher schedule, captures the block-output image rows at forward τ and stops the forward there. Defaults reproduce the reference configuration: block 26, τ = 5, 52 steps, 25 rollouts, 8 directions, conditional-only (CFG 1). A negative conditioning with CFG > 1 and a connected SIGMAS schedule are explicit research overrides recorded in the archive.

Directions follow paper Eq. 2 with the unnormalized centred Gram G = XcXcᵀ, eigenpairs (λₖ, vₖ): uₖ = Xcᵀvₖ / ‖Xcᵀvₖ‖ and sₖ = std(√λₖ vₖ) with the biased standard deviation, which equals √(λₖ/N). The Gram is accumulated in float64 over feature chunks, each centred before its product; the basis is stored in bfloat16 as upstream. Directions with eigenvalue ≤ 10⁻¹⁰ of the leading one are stored as zero rows with zero spread and excluded from the valid count.

Cost: 25 × (τ−1) full teacher forwards plus 25 truncated ones, batched by 5, and N × D bf16 staging (1.26 GB at 1024²) on the GPU, or on the CPU with *workspace* `cpu`. Upstream reports about 60 s per prompt on an RTX PRO 6000 and about 400 MB per archive at 1024². ComfyUI caches the node's output, so later seeds of the same prompt reuse it; changing the prompt, size or teacher recalibrates. Report calibration time with every TF comparison; TF is not free because its teacher forwards run before generation.

### Archive

Archives are safetensors files in `output/modeexplorer_rediverse/directions/` with tensors `U [K, D]`, `s [K]`, `eigenvalues [N]` (unnormalized Gram; covariance eigenvalue = value / N) and the teacher mean, plus JSON metadata: schema and implementation, upstream revision, coordinate and block, token grid and width, latent shape, teacher weight fingerprint and weight-patch fingerprint, conditioning fingerprint, σ in/out, forward and step count, schedule source, guidance, seeds and seed rule, rollouts, normalization, basis dtype, direction counts, spectrum summary and wall time. The file name includes a hash of the calibration identity, so different settings never overwrite each other.

Loading never unpickles. Unknown schemas, missing fields, extra tensors, shape or dtype disagreement with the metadata, nonfinite values, negative spreads, non-unit basis rows and files over 8 GiB are rejected. At sampling time a different positive conditioning or token grid is an error: directions are never resized or reused across prompts. The **ModeExplorer ReDiverse Directions (Krea 2)** node reloads an archive by name.

### Inference

During the opening evaluation, the block hook adds Σₖ αₖ sₖ uₖ, computed in the basis dtype and cast to the activation dtype, to the generated-image rows of every live sample (all CFG branches of a sample receive the same offset). ModeExplorer reference shadows and other auxiliary forwards keep the adapter but receive no offset. Coefficients come from an independent CPU generator per sample, seeded `7_000_000 + seed * 1009 + row (+ trial * 1_000_003)` from the noise seed, so the sampler's own random stream is untouched. TF adds no transformer forward; per invocation it moves the K × D basis to the device (≈400 MB at 1024² for K = 8) and forms one batch × D offset.

### Experimental cohort steering

`cohort_logdet` replaces the random draw for an equally conditioned batch of 2–8. In teacher-whitened coordinates zᵦ = (U hᵦ)/s of the student's own block output, the cohort covariance C gives ∇ log det(C + I) ∝ (C + I)⁻¹(zᵦ − z̄); the identity stands for the teacher's whitened covariance, which damps directions where the cohort already matches or exceeds the teacher. One step is taken per sample, scaled to the RMS norm of a U[−band, band] draw (band·√(K/3)) and clipped to the band, so arm E is its norm-matched control. Single images and degenerate cohorts fall back to the random draw (logged). There is no acceptance test and no evidence that this is better than random; treat it as a separate research arm.

## Directional-collapse probe

**ModeExplorer Activation Probe (Krea 2)** runs one conditional forward per sample at a frozen state (σ = 1 by default) from the reference packed draws `1_000_000 + seed * 100003 + i`, captures the selected blocks' output image rows and reports per block:

- total variance TV = Σλ, participation ratio PR = (Σλ)²/Σλ², concentration TV/PR, entropy effective rank exp(−Σ p log p), the top 32 eigenvalues, cumulative explained variance and the leading share;
- covariance normalization N (population), with PR capped at N − 1. These are sample-limited spectra, not the intrinsic dimension of the model.

Activations are staged as bf16 in a fixed workspace (16 GiB default; 10 GB per block at N = 200, 1024²). Blocks are processed in groups that fit; each group repeats the forwards and stops after its deepest block. The Gram is exact (float64, chunk-centred); no D × D covariance is formed and no pooling is used. Reports go to `output/modeexplorer_rediverse/probes/`; `python -m tools.rediverse_probe_report teacher.json student.json` prints student/teacher ratios and the upstream collapsed-block rule (largest step in the concentration ratio). A probe through a ReDiverse node measures the adapted first step.

With a connected direction archive at a probed block and the same token grid, the report adds a comparison in the archive's subspace, using only the N × K projections:

- student variance along each teacher direction and its ratio to sₖ² (retained variance, per direction);
- subspace alignment ‖UₜᵀUₛ‖²_F / K between the teacher directions and the student's own top K;
- eigenvalues of Cₜ − Cₛ in the teacher subspace, split into *deficit* (positive) and *excess* (negative) parts. The difference is indefinite in general and is reported, not injected;
- mean displacement (overall RMS, along each teacher direction and relative to the cohort's spread).

`coordinate_match` states whether the comparison is a *matched frozen state* (same σ) or between *native trajectory states* (the default archive is teacher forward 5 at σ 0.967, the probe student step 1 at σ 1). Mean shifts, subspace rotations, representation scaling and differing trajectory states can all produce variance differences; the probe separates the first three only within the teacher's K-dimensional span.

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

Add `--identity-lora` and `--identity-trigger` for the identity panel; every arm then uses the same identity LoRA and conditioning. Then:

```shell
python tools/analyze.py bench-output/rediverse-reference --baseline A_turbo --embeddings embeddings.json --review
python tools/analyze.py bench-output/rediverse-production --baseline C_route_sham --stage planner --review
```

`--embeddings` takes locally computed DINOv2 or DreamSim vectors keyed by blinded IDs and adds cosine diversity and Vendi per prompt; the analyzer aggregates at the prompt level with paired bootstrap intervals. Review sheets are blinded; the private mapping stays out of the shared archive. Give reviewers the prompt and, for identity panels, authorized reference photographs, and rate composition, framing, pose, anatomy, identity, adherence, detail and artifacts at native size. Count identity drift under an identity LoRA as a regression, not as diversity. Report wall time, calibration time, peak VRAM and the trace counters (actual, forecast and auxiliary evaluations, adapted evaluations, TF applications, realized offset). Decide FT on A–D first; TF needs its own calibration-cost accounting. Confirm any pilot result on a held-out panel before changing defaults.

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
| Research | [TF workflow](../examples/rediverse_tf.json) and the [probe workflow](../examples/rediverse_probe.json) (Raw, Turbo and Turbo with the adapter on blocks 22–27). |

## Verified scope

Established on CPU with ComfyUI's native Krea 2 transformer at tiny width (see `tests/test_rediverse*.py`):

- zero strength and zero band are bitwise native; disabling restores the native graph;
- the adapted run equals an independent reference (first Euler step on merged weights, remaining steps native) within 2·10⁻⁵, for CFG 1 and packed CFG 2, at strengths 1 and 0.5; upstream weight patches remain applied;
- multistage solvers, ER-SDE's offset σ, split CFG dispatch, partial denoise, masks and high-resolution stages follow the contract above;
- no hook survives a run, an exception or an interrupt; repeated runs are deterministic; the shared model is unchanged;
- TF adds exactly Σ αₖ sₖ uₖ to image rows (text rows unchanged) on the first evaluation only, rejects other prompts and grids, adds no transformer call and leaves the sampler's noise stream unchanged;
- calibration reproduces a manual rollout's PCA, does not modify the teacher, and archives round-trip; covariance statistics match explicit covariances for known spectra and are invariant to chunking;
- with Spectrum PR #6 and Scale-Locked v1.3.2: the adapted step is actual, the next starts fresh history, forecasts resume afterwards, the planner alone is adapted; without the deferred contract the pass is all-actual.

Not established: behavior and timing on real Krea 2 checkpoints, CUDA, native INT8 ConvRot and Comfy Kitchen kernels, torch.compile, memory peaks at production resolutions, and every perceptual property (diversity, anatomy, identity, adherence, detail, artifacts). The code paths are kernel-independent by construction, but that is not evidence of correct quantized CUDA behavior.

## Licensing

The method, reference configuration and seed rules follow the Apache-2.0 [ReDiverse](https://github.com/H-furkan/rediverse) repository; see [third-party notices](../THIRD_PARTY_NOTICES.md). The adapter weights are a Derivative of Krea 2 under the Krea 2 Community License and Acceptable Use Policy; obtain them from the official release and comply with its terms, including its content-filtering requirement for deployments.
