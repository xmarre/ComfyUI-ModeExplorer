# Validation

## CPU scope

The tests use independent autograd references, a synthetic nonlinear initialization response, and the native ComfyUI backend with randomly initialized tiny Krea 2 weights. They cover sampler entry points, split/combined CFG, reference token slices, weight patches, disabled parity, Euler and ER-SDE, route accounting, staged Scale-Locked isolation, and the actual core Model Attention Backend and KJ diffusion loader overrides. The tests use CPU attention stand-ins for Kitchen and Sage kernels. The workflow tests check saved widget order, links and ComfyUI backend input validation. [CI](../.github/workflows/validation.yml) pins the integration sources.

Population-level full-checkpoint perceptual quality, optimum block and grid, memory peaks, and runtime require systematic production measurements. The pilot observations below are qualitative and are not a representative benchmark.

## Matched panels

Use a dedicated ComfyUI server. Keep precision, checkpoint, CFG, sampler and resolution fixed within each panel. Start with two prompts and two seeds.

```shell
python tools/evaluate.py --model krea2_turbo_fp8_scaled.safetensors --out bench-output/turbo-smoke --smoke
python tools/analyze.py bench-output/turbo-smoke --review
```

The default panel contains 30 prompts and 16 seeds. Use `--methods` to choose native, cosine, volume, initialization, and combined arms. Use `--category` and `--seed-count` to limit the panel. Pair an identity LoRA with `--identity-lora` and `--identity-trigger`, and score likeness against authorized held-out photographs. For Raw, select the Raw weights, CFG, step count and `--spectrum-preset raw`. Compare Raw and Turbo first at matched settings, then in distinct checkpoint-specific panels.

Use `--profiles tools/spectrum_profiles.json` for named arms. A profile contains a method, optional ModeExplorer settings, optional Spectrum settings, and a Spectrum policy of off, guarded, full or all_actual. Change one variable at a time. Use multiple references and the all selection policy to test collective objective controls. For strictly matched routing, disable Spectrum on both arms or use all_actual and confirm reports contain zero forecasts. Full sham can match reference cost, while data-dependent guards can still change the realized routes in Full guidance.

Use `--server-log /path/to/comfy.log` to record INFO traces. Spectrum reports include per-dispatch branch, route, rejection reason, and retained feature-attempt anchors. ModeExplorer includes initialization gain, displacement, trial radius and cost. Compare latent summaries only at common sigmas and equal block/grid settings. Use `--gpu 0` on an idle device for sampled VRAM and measure timing separately from diagnostics. Run repeats with `--repeats 2` and a dedicated server started with `--cache-none`. Only repeats with a witnessed sampler trace enter the decoded-pixel reproducibility statistic.

For end-to-end Scale-Locked runs, specify a larger output size and `--planner-megapixels 1.0` with your preferred sampler. The panel records both planner and final decodes. Run analysis with `--stage planner` and `--stage final` independently to locate changes in anatomy, pose and identity.

## Blinded review

Use group IDs consistently within each prompt for composition, camera, pose and layout. Score adherence, quality and applicable identity from 1 to 5 and mark visible artifacts as 0 or 1. Leave inapplicable entries empty. Feed the completed CSV to `--ratings`. The report measures the fraction of unique groups and pairwise group differences across matched seeds, and bootstraps paired prompt-level differences. Review anatomy and identity at native size. Optional LPIPS and the built-in edge-layout metric respond to texture, crop and contrast, so report them alongside the semantic groups and fidelity scores.

| Provisional preset | Settings |
| --- | --- |
| Low-strength (conservative) | Cosine, strength 0.03, fidelity 0.70, min sigma 0.82, Adaptive off, budget 2, block 1, grid 4, bank 8, one trajectory, all selection, first estimate, bounded update, guarded Spectrum, Initialization off. |
| Adaptive comparison | Cosine, strength 0.30, fidelity 0.70, min sigma 0.85, Adaptive on, similarity 0.70, guarded Spectrum, Initialization off. |
| Balanced comparison | Default cosine, block 1, grid 4, budget 2, first-estimate continuation, guarded Spectrum. Compare Adaptive on and off independently. |
| Experimental collective | Budget 4, two per-seed references, all selection, rolling Euler and verified update. Compare fixed versus median volume bandwidth, then vary block or grid separately. |

The provisional presets are starting configurations. To measure a reference or continuation change, hold the feature update mode and strength fixed. Test the verified pooled update in its own arm because its RMS cap can change the applied edit size.

## Low-strength pilot and follow-up

Qualitative matched full-checkpoint Turbo observations motivate the conservative profile:

- In an anatomical orientation failure case, native and sham outputs retained the defect. Active guidance at strengths 0.03 and 0.05 was reported to correct it; 0.01 and 0.02 were reported to yield less coherent intermediate orientations. At 0.05 the logged pooled feature shift RMS values were approximately 0.00981 and 0.01116. The sham and active 0.05 traces matched the reference slot, two reference forwards, ten actual planner evaluations, three forecast steps and two refinement evaluations. The visual strength transition is not established across other seeds.
- A second matched native/guided image pair at strength 0.03 showed clearer local object semantics and garment construction with largely unchanged framing. The changes were already present in the planner and persisted after refinement.
- A separate pilot reported limb duplication with a four-reference budget and its disappearance with two. Budget and strength affect different aspects of the intervention; this does not isolate the adaptive threshold or establish that four references are universally unsafe.

These are exploratory observations on private material, not a public reproducible image dataset or a statistical quality claim. They motivate an independent, unselected panel. The node defaults remain unchanged.

Run the opt-in native, guided and sham arms of [low_strength_profiles.json](../tools/low_strength_profiles.json) with the existing evaluator:

```shell
python tools/evaluate.py --model krea2_turbo_fp8_scaled.safetensors --profiles tools/low_strength_profiles.json --out bench-output/low-strength-smoke --smoke --diagnostics
python tools/analyze.py bench-output/low-strength-smoke --review
```

Use the same solver and stage settings across arms. The active and sham profiles differ only in the sham flag. Native lacks feature-regime forcing, so compare actual per-dispatch routes before attributing native/active differences to the feature edit. If routes diverge, repeat with Spectrum off or all_actual in all arms. Score recovered defects, newly introduced defects, persistent defects and unaffected outputs separately; assess prompt adherence, composition and identity as applicable. Include seeds whose native output is already good. Inspect planner and final images independently when using refinement.

For dose-response testing, hold fidelity fixed and compare strength values such as 0.01, 0.02, 0.03, 0.05 and 0.10; use 0.001 increments only where finer measurements answer a specific question. Hold reference budget at two for this sweep, then test one versus two versus four references in a separate arm. Do not infer monotonic quality from magnitude or tune a general preset on a single selected failure.
