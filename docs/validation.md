# Validation

## Structural checks

The CPU suite uses independent autograd references, a synthetic initialization check, and the real pinned ComfyUI backend with small randomly initialized Krea 2 weights. It checks discovery, sampler-node entry points, native equivalence, CFG branches, same and distinct conditioning, reference token preservation, weight patches, clone ownership, cancellation, and stage boundaries. See [CI](../.github/workflows/validation.yml) for the exact environment.

## Matched image runs

Use a dedicated ComfyUI session, select the same encoder and VAE for every run, keep attention and precision settings fixed, and start with the smoke panel.

```shell
python tools/evaluate.py --model krea2_turbo_fp8_scaled.safetensors --out bench-output/turbo-smoke --smoke
python tools/analyze.py bench-output/turbo-smoke --review
```

Remove `--smoke` for the 30-prompt × 16-seed panel. The five default methods cover native, both feature objectives, initialization and the combination. Each queue request generates one image. Repeat for Raw by choosing your Raw filename, step count and CFG. For a checkpoint attribution experiment, use the same step count, CFG, schedule and seeds, then independently vary step count and guidance. Record an SDA run with the same settings in a separate output directory using `--model-lora <SDA filename> --methods native`. For identity testing, pass `--identity-lora <name> --identity-trigger <token>` and score the added identity prompts against authorized held-out references. Keep LoRA strength and checkpoint constant for that comparison.

For separate trajectory runs, add `--server-log <readable-ComfyUI-INFO-log>` and use `--trajectory` when analyzing. The runner records matchable compact traces, per-run native and extra DiT calls, schedule, initial noise statistics, absolute-sigma feature and conditional predicted-clean distances, and observed update magnitudes. Compare native checkpoints at common sigma values. Shape, pooling and branch semantics must match. Compare the clean-latent summary within each model and conditioning regime. The traces are probes of the hypothesized convergence, and downstream matched-image reviews determine whether the probe is informative.

`--gpu 0` samples total device VRAM via nvidia-smi approximately every 200 ms. Keep the GPU otherwise idle and treat this as an approximation to the device high-water mark. The runner records wall time and ComfyUI execution time when the server publishes timestamps. It does one warm-up per method, excluded from metrics. Use runs without trace collection for timing. The optional `--lpips-device cuda` analysis requires the `lpips` Python package and its AlexNet weights. It computes pairwise learned perceptual distances on reduced-size images.

## Blinded scoring

Contact sheets and a blank CSV are created under `review/`. Score each image at the original size when anatomy or identity is ambiguous. Use one shared set of `composition_group` labels within each prompt, based on camera, spatial layout, subject pose and framing. Rate prompt adherence, image quality and applicable identity on a 1–5 scale. Mark visible artifacts as 0 or 1. Leave inapplicable cells empty. Pass the completed file with `--ratings`. The JSON summary includes matched-prompt means, method-minus-native differences, and prompt-level paired-bootstrap 95% intervals. Layout edge distance and LPIPS are complementary probes. Changes in crop, contrast or residual texture can affect them, so inspect the compositions and quality scores alongside the metrics.

| Result | Current evidence |
| --- | --- |
| Registration and native sampling | Source audit and tiny-transformer CPU smoke |
| Analytic objective and state isolation | Automated CPU tests |
| Raw vs Turbo convergence stage | Unmeasured on production checkpoints |
| Composition, anatomy, prompt adherence and identity | Unmeasured on production checkpoints |
| CUDA timing and device VRAM | Unmeasured |
| Extra DiT calls | Measured on the tiny native path. Production batch and CFG dispatch may differ. |


## Spectrum interoperability

The cross-repository tiny-model CPU suite exercises native Krea 2 sampling
with the actual Spectrum wrapper registration and history-capture logic.
Both states of ModeExplorer's A/B MODEL switch must pass through the same
Spectrum node. Apply ModeExplorer before Spectrum: this isolates initial noise
optimization from Spectrum's sampling session. Auxiliary reference forwards
run without modifying Spectrum counters, capture hooks or history. Early
guidance evaluations are real model forwards; Spectrum clears forecast
history when the guidance regime ends and can resume native forecasting.
These tests do not establish CUDA safety or improved composition diversity
on production checkpoints.
