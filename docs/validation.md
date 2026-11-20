# Validation

## CPU scope

The tests use independent autograd references, a synthetic nonlinear initialization response, and the native ComfyUI backend with randomly initialized tiny Krea 2 weights. They cover sampler entry points, split/combined CFG, reference token slices, weight patches, disabled parity, Euler and ER-SDE, route accounting, and staged Scale-Locked isolation. The workflow tests check saved widget order, links and ComfyUI backend input validation. [CI](../.github/workflows/validation.yml) pins the integration sources.

Full-checkpoint perceptual quality, optimum block and grid, memory peaks, and runtime require production measurements.

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
| Conservative | Cosine, strength 0.30, fidelity 0.70, min sigma 0.85, Adaptive on, similarity 0.70, guarded Spectrum, Initialization off. |
| Balanced comparison | Default cosine, block 1, grid 4, budget 2, first-estimate continuation, guarded Spectrum. Compare Adaptive on and off independently. |
| Experimental collective | Budget 4, two per-seed references, all selection, rolling Euler and verified update. Compare fixed versus median volume bandwidth, then vary block or grid separately. |
