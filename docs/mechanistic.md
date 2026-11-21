# Full-resolution mechanistic experiments

**ModeExplorer Mechanistic (Krea 2)** measures a bounded feature intervention at matched native diffusion inputs. Put it after any production ModeExplorer node and before Sol and Spectrum. It uses the Response wrapper slot, so use one research node per model branch. It leaves the sampler output untouched under the default `component=observe` and `treatment=export_only`. The existing ModeExplorer and ModeExplorer Response nodes keep their controls and defaults. Passive measurements follow Spectrum’s existing route unless you explicitly enable `all_actual`. Active treatment always forces the all-actual path.

## Native sites

| Selector | Tensor captured from the executed Krea 2 forward |
| --- | --- |
| `attention_pre`, `attention_post` | Native attention output at the intervention hook, immediately before and after native-dtype addition |
| `gate_logits` | Output of the attention `gate` linear module |
| `projection_input`, `projection_output` | Actual attention output times sigmoid gate, and the executed output projection result |
| `after_attention` | Residual stream entering the post-attention norm, after the block's modulated attention addition |
| `mlp_input`, `mlp_output` | Actual modulated input to the MLP and its immediate output |
| `after_mlp` | Block return after the modulated MLP residual addition |
| `final_projection` | Final native linear projection before image-token unpatching and cropping |
| `velocity`, `clean` | Cropped native velocity and FP32 clean estimate in diffusion coordinates |
| `sigmoid_gate`, `attention_contribution`, `mlp_contribution` | Unsupported. Exact standalone values require instrumentation inside native functional expressions |

Select zero-based `blocks` with ranges such as `5,8,12-14`. The independent `feature_block` selects the injection site. The global outputs final projection, velocity, and clean use a single site key each. Internal spectra contain only the generated image tokens. Krea's native image slice and reference-token counts define these bounds. The patch grid includes the native padding on odd latent dimensions. Output latent spectra use the cropped grid. A cycle per patch cell and a cycle per latent pixel are different units.

The attention gate can shift energy into higher spatial frequencies between `attention_post` and `projection_input`. Compare these sites and the actual `projection_output` and `after_attention` results before attributing downstream changes to the gate or projection. A growing absolute RMS is compatible with a shrinking baseline-relative response. The residual stream's net band energy distinguishes amplification from changes that cancel existing content.

## Matched arms

| Arm | Conditioning | Intervention |
| --- | --- | --- |
| `D00` | C0, original positive | None |
| `D01` | C0 | Full bounded cosine proposal |
| `D0minus` | C0 | Exact negative of the same requested proposal |
| `D10` | C1, alternate positive | None |
| `D11` | C1 | C0's realized edit field in fixed mode, or independent C1 proposal in recomputed mode |

`signed` uses the first three arms. `conditioning_fixed` and `conditioning_recomputed` use the four factorial arms. `joint_fixed` and `joint_recomputed` add the negative arm. The fixed experiment reapplies C0's actual native-rounded difference in FP32 strips, rounds the resulting C1 activations to native dtype, and reports the discrepancy from C0. The recomputed experiment builds the proposal and reference continuation under C1 with the same noise slots. The signed experiment shares one requested proposal and records the actual opposing-edit mismatch. A comparison exceeding `symmetry_limit` is unqualified. Diagnostic spectra remain available and active even, odd, or fixed-interaction proposals require the relevant match.

Use a single global Krea 2 CONDITIONING entry per prompt. C1 accepts a different embedding sequence length and matching native reference latents and reference mode. Timed, regional, control and custom alternate metadata are outside this contract. Masks are recorded and the pinned Krea 2 forward does not consume them for these attention paths. Text-token activations are outside the FFT comparison, so no alignment is required. All auxiliary arms execute the positive row alone, even when the live CFG path packs conditional and unconditional rows. `sampler_velocity_baseline_rms_difference` reports comparison with the live positive output. At CFG greater than one the live sampler result retains its usual CFG behavior. Active treatment requires CFG 1 and bitwise parity with the live native baseline.

The latest observed actual sampler invocation determines the sigma, actual evaluation index and Sol ordinal. Set `states=3`, `evaluation=-1` and a window including the start to measure the first three eligible distinct actual sigmas. Use a native trajectory for a common baseline or attach Mechanistic after the existing weak guidance node for passive measurements along the guided trajectory. A component-treated trajectory can diverge after the first applied correction. For frozen matching, replay the local export at its exact sigma and matching conditioning, and check the native input hash.

## Reading the evidence

The paired metrics use orthonormal two-dimensional FFTs over each full-resolution channel. Separate `dc`, `low`, `mid` and `high` bands use radial frequency in cycles per grid cell. Unstored negative-horizontal frequencies receive a factor of two, with DC and horizontal Nyquist columns counted once. For each band, with spectra `A` and `δ = FFT(edited − baseline)`, the displacement energy is `Σ|δ|²`. The cross-term is `2ΣRe(conj(A)δ)`. Their sum is the net change in absolute band energy. The trace also reports fractions of total energy and relative net change. Zero denominators are null. Spatial tile energy and the top 1%, 10% and 25% concentration use the sum of squared displacement over channels.

Site comparison keys include `positive`, `negative`, `odd`, `even`, `positive_vs_reflected_negative`, `conditioning_baseline`, `conditioning_edited`, `conditioning_interaction` and `feature_across_conditions`. Odd is `(D01 − D0minus)/2` and even is `(D01 + D0minus)/2 − D00`. Each signed site also reports the fraction of combined odd and even energy that is sign-even, overall and by band. For the factorial interaction the baseline field is `D10 − D00`, the edited field is `D11 − D01`, and their displacement is the interaction. Use `feature_across_conditions` to compare the two feature-induced correction fields directly. Use the energy maps with signed and conditioning contrasts to locate where the effect is concentrated.

`repeat_baseline` adds one exact matched forward and reports the clean repeatability floor. `amplitude_check` adds two signed half-proposal forwards, captures realized injection symmetry again, and compares native clean outputs. Half/full ratios near 0.5 for odd and 0.25 for even are consistent with linear and quadratic scaling at those amplitudes. Neither ratio establishes a derivative or determines the source of the effect. Use the exact captured dtypes, projection operator types, attention override, checkpoint provenance and measured repeatability. Kernel accumulation dtype is unobserved. A different backend with the same weights and input is a separate comparison from dequantization, the original unquantized weights, a fine-tune, or Raw/Turbo. Exact replay requires identical real-valued diffusion input and conditioning. Enable `match_cross_dtype` for a frozen replay across execution dtypes. The saved native latent must be exactly representable and the actual conditioning, mask and reference values must match by value. Use the provenance label that identifies the weight or execution change.

PMD Appendix B relates posterior moments and a conditional prediction derivative under its assumptions. ModeExplorer measures finite-amplitude responses of a learned denoiser and feature interventions are separate from that identity and from PMD training. The existing [Response node](response.md) still provides latent-direction probes. The Mechanistic clean components live in output space. Feeding one into a latent-input probe would require an explicit correspondence and a matched input-space measurement. Neither a small projection into a generic latent basis nor strong high-band displacement establishes semantic quality.

## Applying a component

Select `complete`, `odd`, `even`, `interaction`, or deterministic norm-matched `control`. `band` applies complementary low/mid/high masks to a native clean displacement. Smooth transitions optionally use cosine rolloffs. The low component contains DC. With `cell` enabled the filter is applied first, then the output is restricted to the selected cell, which can introduce boundary frequencies. A nonzero `candidate_target_rms` makes this a separate norm-matched candidate experiment and selects only when the native realized RMS is within the existing tolerance.

Use export-only to inspect `candidate_clean` before selecting `minimum_change` or supplying a local scorer through `external`. The scorer sees VAE-format predicted-clean latents. It can require both conditioning and fidelity scores at least as good as baseline. No correction applies at the zero-sigma endpoint or above the clean trust bound after velocity conversion and rounding. Active use is restricted to the planner or standalone native model with no preexisting feature intervention and with Spectrum in all-actual execution. A spectral component has no built-in quality score. Compare same-seed planner renders and score structure, conditional relation and local detail independently.

## Resource limits and exports

`auxiliary_budget` bounds DiT calls per sampler invocation. Every experiment forward adds one row. A diagnostic with output-only sites can reuse the live positive baseline when it is single-row, unedited and unfrozen. Internal measurements need one instrumented baseline, shared by all compatible arms. Each conditioning-specific shadow reference costs one forward at the measured state and one at the reference start when needed. A previously captured first estimate is reused within a sampler invocation when its native conditioning, reference conditions, sigma, backend, seed, dtype, shape and initial Sol ordinal match. Pooled proposals and shadow references are shared across requested sites.

`memory_mb` limits the estimated extra GPU tensor workspace. `capture_mib` independently bounds retained native-dtype CPU captures plus conservative FFT and output-workspace reserves. Preflight conservatively counts four bytes per activation. It may decline a request that would fit at the observed runtime dtype. With 15,700 patch tokens and 6,144 channels, a single full-width site in one arm occupies about 184 MiB at two bytes per value. Captures are staged to CPU once and reduced in `chunk` channels, then released per site before the optional half-amplitude calibration. The estimate excludes the native forward, allocator and kernel workspaces. A rejected or incomplete observation reports its status without reducing resolution or fabricating zeros.

With `export=true`, use `output/modeexplorer_response/<export_id>/manifest.json` and `predictions.safetensors`. Tensor payloads share the existing 64 MiB per-run export budget. These schema 1 artifacts add a `mechanistic` schema 1 section to the existing Response manifest. `baseline_clean`, `D01_clean`, other named arm cleans, and candidate/applied cleans use the Comfy VAE latent format. `frozen_native_input` and `*_native_delta` remain in native diffusion coordinates. `ModeExplorer Load Response` decodes clean exports and supplies the matched two-step replay schedule. Full activations are staged for measurements and are never written as tensor dumps.

The offline analyzer reads one or two export directories. It includes layer profiles and site spectra, flags input and geometry mismatches, and can compute between-component cosine and normalized spatial-energy overlap from exported component fields.

```shell
python tools/analyze_mechanism.py PATH_TO_EXPORT --comparison conditioning_interaction
python tools/analyze_mechanism.py FIRST_EXPORT SECOND_EXPORT --comparison even --components even_native_delta even_native_delta
```

## GPU sequence on the production planner

Use the [profile settings](../tools/mechanistic_profiles.json) with the original production seed, sampler, scheduler, sigmas, geometry, checkpoint and attention stack. On the pinned Scale-Locked branch use `regeneration_mode=planner_final_upscale_only`, `handoff_diagnostics=off` and decode `lowres_planner`. This dispatches the same planner, then only latent enlargement, with zero target DiT evaluations. Choose the cheap interpolation option if the enlarged target is unused. Preserve the planner's original native-only Spectrum policy. Confirm the Scale-Locked log reports `target denoiser evaluations=0` and the ModeExplorer trace reports planner stage with zero forecasts. Do not use a capture of the enlarged latent as the original planner state.

1. Smoke-test the first eligible state with output-only capture and a matched repeat. Require completed clean metrics, a zero or measured repeat floor, a conforming signed edit and a planner trace. The first state with the reference start observed requires at most five auxiliary calls.
2. On one already informative seed, run the signed spectral profile for the first three actual states. Measure pre/post intervention, product, projection, residuals and clean in the same run. Check whether high-band displacement originates at injection or first grows over an actual measured boundary, and check the net high-band change on the clean latent. Read `signed_realized_edit` and the amplitude check before interpreting even energy.
3. Run the four-arm fixed conditioning profile only if the already known conditioning-sensitive case still needs mechanism localization. Use the saved prompt pair and collect the interaction at the injection block, a later residual, and clean. Use recomputed mode only when the question is whether the intervention algorithm itself changes with C1.
4. Use the sign-even or low-band active profile only after steps 2–3 identify a component worth testing. Compare the decoded planner output with the native planner and inspect `selected_candidate` and `applied_clean_displacement_rms`. Use the corresponding norm-matched control only when necessary to test specificity.
5. If half-amplitude results are unstable or near the repeat floor, change one execution factor on the same frozen export and compare paired site profiles. Keep the distinction between native dtype, dequantized approximation, original weights and a different model.
