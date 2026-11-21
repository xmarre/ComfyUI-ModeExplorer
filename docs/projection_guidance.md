# Krea 2 projection guidance

**ModeExplorer Projection Guidance (Krea 2)** applies an experimental edit at the native `block.attn.wo` output. The native attention, gate, and projection run once. The returned replacement changes generated-image tokens in the positive CFG rows before the block consumes the projection. The original projection storage, text tokens, reference image tokens, and unconditional rows are preserved. This output-space edit is distinct from editing the input to an INT8 quantizer.

The node defaults to disabled. Disabled and zero-amplitude configurations use the native path with the guidance wrappers absent. Give the original feature-guidance, Response, Mechanistic, and projection-guidance nodes separate MODEL branches. Apply projection guidance before Spectrum or Sol-Krea2.

## Modes and amplitude

Let $A$ be one native generated-image projection row of shape tokens × channels. Let $G$ be independent standard normal draws of the same shape and $s=\operatorname{RMS}(A)$. The proposal is $G\odot w$, normalized by its actual per-image RMS and multiplied by the scheduled target RMS.

| Mode | Weight $w$ |
| --- | --- |
| Gaussian | 1 everywhere |
| Token adaptive | Each token uses its across-channel RMS divided by $\max(s,10^{-12})$, clipped to [0.5, 2] |
| Channel aware | Each channel uses its across-token RMS divided by $\max(s,10^{-12})$, clipped to [0.5, 2] |

Channel scaling is a diagonal activation statistic. Measuring the covariance or reproducing the structure of the dynamic-quantization residual is outside this approximation.

Absolute amplitude is in native projection RMS units. Relative amplitude multiplies $s$. The target RMS is constrained by `max_absolute_rms` and, when positive, `max_relative_rms × s`. The edit is generated in FP32 and added to $A$ before casting back to the native dtype. The trace distinguishes requested, target, generated and realized RMS, the native scale, and the changed-value fraction. An all-zero rounded result is recorded without raising the amplitude. If nearest rounding alone exceeds the limit, the same random field is replayed once for the affected row and overshooting values round toward their baseline. The trace records `rounding_backoff`. The same hard cap applies after the retry. A nonfinite or still-over-limit result keeps the whole original projection for that call. These checks constrain the projection output. Clean-prediction drift and image quality can only be assessed downstream.

## Placement, scheduling, and resources

Block numbers are zero-based. First-actual means `evaluations=1`. First N counts actual positive Krea 2 dispatches in the sampler invocation. A sigma window also applies and `evaluations=0` uses that window alone. Decay falls from 100% to 50% across the first N ordinals or across the sigma window in sigma-only mode. Separate unconditional dispatches do not consume an ordinal. Repeated solver evaluations at the same sigma have different ordinals. An interrupted call does not commit its ordinal. Planner, highres, and standalone scopes follow the existing stage marker. A scope mismatch logs `stage_scope_mismatch` with the actual stage and requested scope. Each sampler invocation owns its hook lifetime.

The generator is local to the inference device. The stream key records the input seed, trial, stage, actual ordinal, exact sigma, block, site, patch grid, channel width, and positive batch-row ordinal. Amplitude and mode share raw Gaussian draws to support controlled comparisons. The expected reproducibility applies to the same device, software stack, batching and evaluation order. Custom conditioning multiplexing, unsupported wrapper graphs, and incompatible CFG callbacks skip the intervention. With the verified Spectrum regime callback on Euler and ER-SDE, the guided evaluations are actual. Spectrum clears forecast history at guided/native boundaries and resumes forecasts from fresh anchors when eligible. The trace reports `guided_regime`. Guided steps and fresh anchors can increase actual calls relative to a Spectrum-only baseline. SA, SA-PECE, and unverified regime contracts retain the scoped `all_actual` fallback. An all-actual native baseline isolates the overhead of the edit itself. A subsequent native failure propagates. Projection-local fallback applies before the replacement is returned.

Each mode adds **zero complete denoiser evaluations, zero feature or initialization shadow evaluations, and zero projection evaluations or projection GEMMs**. It uses neither packed projection replay nor an affine reference. Computation uses one native-layout result clone and token chunks of at most 512 for the FP32 work. The same local Gaussian stream is generated twice for exact per-image normalization without retaining a full-resolution FP32 field. A strided layout is supported when the clone preserves its strides. At 15,708 image tokens, 512 additional packed tokens, and width 6,144, one positive FP16 row has approximately 190 MiB in the output clone and the conservative chunk bound is 96 MiB. The temporary estimate grows with packed rows and width. Ordinary scalar qualification synchronizes once per intervention, with one additional synchronization when rounding backoff is needed. Reading the actual dispatch's scalar sigma is separate. Production overhead is a measurement target, with low single-digit percentage overhead as the goal.

`MODEEXPLORER_TRACE` reports zero auxiliary calls, sampler duration, per-intervention time, status, seed/trial, native-rounded scale, and estimated owned temporary bytes. On CUDA it also reports allocator snapshots before the first intervention and at sampler exit. The allocator peaks are device-wide since the last reset and may include work from other passes or jobs. The node does not reset global peak counters. For an end-to-end comparison, record ComfyUI completion time and, where available, allocator peaks from a fresh process per workflow. Any downstream exception and interrupted sample remain visible in the native execution path. Workspace OOM during candidate construction retains the original projection.

## First planner comparison

Use the same checkpoint, LoRA stack, attention backend, conditioning, seed, resolution, and solver for each arm. The shipped examples run the ER-SDE planner, decode its planner latent to PNG, and use Scale-Locked's planner-only route. No high-resolution denoising takes place.

| Workflow | Requested projection RMS | Mode |
| --- | ---: | --- |
| [Weak Gaussian](../examples/projection_guidance_weak_gaussian.json) | 0.000303 | Gaussian |
| [Higher Gaussian](../examples/projection_guidance_higher_gaussian.json) | 0.013937 | Gaussian |
| [Token adaptive](../examples/projection_guidance_token_adaptive.json) | 0.013937 | Token adaptive |

The weak and higher anchors test whether an independent weak edit has useful downstream effect at the same scale as the much smaller affine component. Compare with a native output. The adaptive profile changes the distribution of the higher-amplitude proposal while matching its RMS. These are comparison settings for block 5 and the first two planner evaluations. A large downstream displacement or different composition is insufficient evidence of quality benefit.

## Optional downstream measurements

Enable `export_clean` when a matched clean comparison is required. It saves the actual guided branch's clean prediction in the existing private response format within the shared 64 MiB per-pass payload limit. It adds CPU transfer, fingerprinting and file I/O and belongs outside the ordinary performance run. Use a same-state Response or Mechanistic export with `baseline_clean`. Following an earlier applied edit, later sampler evaluations normally see different noisy inputs. The comparison tool requires exact input and conditioning fingerprints and the same stage and sigma.

```sh
python -m tools.analyze_guidance GUIDED_EXPORT BASELINE_EXPORT \
  --control affine AFFINE_EXPORT projection_component_clean \
  --control residual RESIDUAL_EXPORT projection_component_clean
```

The tool measures clean RMS, frequency bands and spatial concentration, and optionally alignment with previous clean displacements. Results are in the ComfyUI VAE-latent coordinate system. Repeat under distinct prompts with matched native/guided exports within each prompt to track conditioning dependence. Assess image quality separately. No downstream quantizer measurement or cheap projection-local signed-GEMM mode runs automatically. Later-layer threshold amplification remains a hypothesis that needs targeted frozen-state measurements.

## Heterogeneous screening and confirmation

The [benchmark specification](../tools/projection_guidance_prompts.json) predeclares six varied cases with fixed seeds and a separate six-case confirmation split. Add `--identity TRIGGER LORA_FILENAME` for the optional identity row. That row requires a held reference and the same appropriate LoRA in every arm. Keep it outside the aggregate when no reference is available. Generate planner PNG workflows and a private allocation key.

```sh
python -m tools.projection_guidance_benchmark screening_workflows --blind-key YOUR_PRIVATE_KEY
python -m tools.projection_guidance_benchmark confirmation_workflows --split confirmation \
  --arms native weak_gaussian --blind-key YOUR_PRIVATE_KEY
```

Screen native, the original feature reference at 0.025 strength with repel direction, weak Gaussian, higher Gaussian, and token-adaptive Gaussian. Fix the strongest candidate and the evaluation rubric before opening the confirmation set. Review the masked PNGs without opening their embedded workflow metadata, and reveal the allocation key after scoring. Pair results by prompt and seed. Review anatomy, spatial relationships, prompt adherence, detail and texture coherence, photorealism, lighting, and visible artifacts. Rate identity against the reference when available. On ambiguous differences mark a tie. Record paired wins, ties and losses, sample sizes, notable regressions, and variability by prompt. Report runtime and memory alongside quality. For a small sample, include an interval or exact paired sign test on the non-ties and disclose the tie count. A useful production result requires fewer meaningful failures without systematically losing adherence, identity, or native strengths on confirmation.
