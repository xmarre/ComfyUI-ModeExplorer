# Causal investigation of weak feature guidance

ModeExplorer's bounded feature edit can change a fixed seed's composition and local detail. The objective reduces similarity between independently initialized same-conditioned feature representations. It does not measure intra-image anatomical ambiguity, correctness, aesthetics, or texture. Exploratory visual observations are motivation for a mechanism study, not evidence of a general quality improvement.

## Exact intervention and competing explanations

Native Krea 2 exposes the attention-output patch after attention aggregation and before the elementwise sigmoid gate and output projection. The selected generated-image tokens receive spatially pooled channel offsets. The subsequent gated residual and MLP, the remaining transformer blocks, and the solver propagate the intervention. For one cosine reference, if u and v are the normalized centered live and reference vectors and c = u·v, the descent direction is proportional to c u − v. The initial direction is tangent to u. Centering and the existing bounded magnitude rule are then applied. The correction is constant per cell and does not introduce individual-detail texture directly.

A linear flow with x_sigma = (1 − sigma)x0 + sigma z and an optimal squared-error velocity regressor yields x_sigma − sigma E[z − x0 | x_sigma,c] = E[x0 | x_sigma,c]. A multimodal conditional distribution can have a mean that is not a valid realization of either mode. This identity does not establish that the finite-step, distilled Krea 2 prediction is optimal or that visible duplicated objects represent an internal average. Possible explanations include:

- Direction-specific feature dispersion changes a dominant structural representation.
- Coarse early perturbations generally redirect sensitive trajectories, irrespective of reference repulsion.
- Competing textual or learned appearance constraints create unstable layouts; the same mechanism can resolve or worsen a conflict.
- Native rounding, solver updates, or Spectrum route changes contribute to the response.

Related work: [FSG](https://arxiv.org/abs/2606.27371) compares random intermediate feature perturbations with input perturbations and motivates structured feature dispersion; [Dynamical regimes of diffusion models](https://doi.org/10.1038/s41467-024-54281-3) studies idealized symmetry-breaking transitions; [PAG](https://arxiv.org/abs/2403.17377) and [HeadHunter](https://arxiv.org/abs/2506.10978) study attention perturbation for image quality; [Emergence and Evolution of Interpretable Concepts in Diffusion Models](https://arxiv.org/abs/2504.15473) uses time-dependent causal feature interventions. None proves this Krea-specific quality mechanism.

## Direction ablation

Use the bounded update, fixed (non-adaptive) cosine guidance, one bank reference trajectory, budget two, and initialization off. Compare the following arms on the same initial noise, positive/negative conditioning, weights, resolution, native attention backend, solver and sigma schedule.

| Arm | Intervention |
| --- | --- |
| Native | No feature guidance |
| Sham | Compute the normal proposal and reference forwards but do not edit attention |
| Repel | Existing bounded proposal d |
| Reverse | −d |
| Channel sign | S d, where S is a deterministic diagonal channel sign mask |

The sign mask uses an isolated CPU RNG seeded by the generation seed and direction_trial (0–31). One mask is applied across spatial cells and evaluations. It preserves the absolute value of every proposal entry, the norm of each cell, and each channel's spatial power before native dtype addition. For both direction controls the reference forwards, window and local bound are unchanged. Channel-sign controls are not generally tangent or equivalent to isotropic random noise; they can retain some alignment with the original proposal. Reverse is an ascent-direction control, not a guarantee of higher cosine after native rounding. Verify direction and magnitude using the measurements. Multiple masks reduce dependence on one arbitrary control.

At the first intervention, matched inputs imply equal pre-rounding proposal norms. On subsequent steps trajectories differ, so proposals are recomputed and their magnitudes need not match across arms. Actual native rounding can also make applied magnitudes differ at the first step. Non-repulsive controls are restricted to the bounded update; applying them after verified objective descent would invalidate its contract. They are experimental controls, not recommended quality modes.

Use [causal_direction_profiles.json](../tools/causal_direction_profiles.json) with the existing evaluator. This uses 0.025 strength, 0.70 fidelity, block 5 and grid 4, with Spectrum all_actual to remove forecast routing as an explanation. It requires the Spectrum overlay. Use planner-only runs first, and choose solver/steps/resolution appropriate to the checkpoint. A manual production-graph repeat with guarded Spectrum can determine whether the same effect persists there.

```shell
python tools/evaluate.py --model krea2_turbo_fp8_scaled.safetensors --profiles tools/causal_direction_profiles.json --out bench-output/causal-smoke --smoke --diagnostics
python tools/analyze.py bench-output/causal-smoke --baseline sham --trajectory --review
```

Provide a readable server log with --server-log to capture traces; --diagnostics alone only enables logging on the server. Two prompts/two seeds are smoke coverage, not a quality experiment.

## Measurements and falsification

The diagnostics retain pre-edit nearest_cosine and proposed shift_rms. After the native-dtype addition they report applied_pooled_shift_rms, applied_pooled_relative_rms (relative to uncentered pooled activation RMS), nearest_cosine_after and proposal_direction_cosine (alignment with the untransformed bounded proposal; zero for a zero-norm proposal). After-pooling differences measure changes of pooled means, not full token-wise displacement. Sham has zero realized pooled displacement. The extra pooling occurs only when diagnostics record an edit, costs no extra transformer evaluation and can affect runtime. Compare timings with diagnostics disabled.

The analyzer's --trajectory option reports per-sigma direction/magnitude measurements and within-seed relative L2 differences of conditional_x0 fingerprints from matching comparison arms. Fingerprints compress the predicted-clean latent, so they cannot measure the full error or establish anatomical content. Unique actual sigmas rounded to five decimals are matched; duplicate/ambiguous points, incompatible shapes and zero reference norms are excluded. Use identical schedules and verify the routes; do not interpret later divergent predictions as single-forward Jacobian measurements.

Use individual prompt/seed pairs as experimental units. On unselected images, mark anatomy_defect (0/1, missing when no applicable subject), relation_defect (0/1 for visually inconsistent object relations), and artifact (0/1), and separately score detail_quality, prompt_adherence, identity and overall quality on 1–5 scales. Detail quality refers to resolved meaningful structures, not raw high-frequency energy. The report counts recovered, introduced, persistent and both-clear defects, excluding missing scores. Inspect full-size planner and final images separately. The failure-selected examples should be a regression panel, not included as a representative sample. Freeze controls before independent evaluation and treat block or strength search as tuning, not confirmation.

If repel improves structure more often than sham, reverse and several equal-proposal-magnitude sign controls under actual matched routes, the evidence favors a direction-specific mechanism. If reverse and sign controls improve at comparable rates, generic perturbation sensitivity is a stronger explanation. If only guided/full versus guarded routes differ, investigate forecasting first. If the effect disappears under an unambiguous prompt, test prompt constraint conflict separately. None of these tests alone localizes a circuit or proves ambiguity reduction.

For circuit-level investigation after direction ablation, compare the unedited and edited forward at the exact same latent and sigma (with isolated forecasting and RNG), measure full velocity and residual differences, and decode predicted-clean states at common sigmas. Decompose the edit across native attention heads or spatial regions and validate necessity and sufficiency on held-out prompts. Test explicit, mutually exclusive object-relation prompts and independently score both relations. Treat reduced entropy or feature similarity as insufficient evidence of reduced semantic ambiguity. These frozen-input replays and head-level causal interventions are proposed follow-up mechanisms, not implemented features of the current direction controls.
