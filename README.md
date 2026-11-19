# ComfyUI-ModeExplorer

ModeExplorer adds experimental early-flow diversity guidance to native Krea 2 Raw and Turbo workflows. Use it to compare different compositions across independent seeds under the same prompt, including when you generate one image at a time. The CPU tests establish integration and numerical behavior with a small Krea 2 transformer. Composition gains, image quality, identity retention, CUDA runtime, and peak VRAM on production checkpoints are unmeasured.

## Install and use

1. Add this repository in ComfyUI Patcher and apply the implementation PR overlay if using a pull request build. Restart ComfyUI. The node uses the installed PyTorch and ComfyUI runtime.
2. Use the native Krea 2 diffusion loader, a Qwen3-VL-4B encoder loaded as `krea2`, the matching Qwen Image VAE, and an empty latent. Place any standard weight LoRA upstream.
3. Connect **MODEL → ModeExplorer (Krea 2) → KSampler**. Start with the settings in the [Turbo workflow](examples/turbo.json). The [A/B workflow](examples/ab.json) runs the same seed through native and guided branches.
4. Change seeds between matched pairs. Evaluate changes in viewpoint, framing, pose, and layout alongside prompt adherence, anatomy, and identity.

The audited source is [ComfyUI c8abeb00](https://github.com/Comfy-Org/ComfyUI/tree/c8abeb00ad2736b649ca2fceb8395e25072c7ea3). The tests use Python 3.12 and PyTorch 2.8 CPU. The node requires ComfyUI's native Krea 2 transformer and flow-sampling contracts. Other model families are outside the supported scope.

## Node controls

| Control | Behavior |
| --- | --- |
| **Enabled** | Off returns an ordinary input MODEL by identity. On a previously ModeExplorer-patched MODEL, it strips ModeExplorer's callbacks from a clone. |
| **Feature** | `native` disables feature guidance. `feature_cosine` uses a pooled cosine objective inspired by FSG. `feature_volume` uses an RBF Gram determinant on the same features. |
| **Initialization** | Independently enables a constrained noise edit scored by native predicted-clean latents. Uses four extra guider calls. Select `native` under Feature for the initialization-only ablation. |
| **Strength / fidelity** | Controls the update budget and the fidelity limits. Initial settings are 0.45 and 0.55, respectively. |
| **Min sigma** | Applies feature guidance for absolute flow sigma at or above this value. The provisional default is 0.82. |
| **Adaptive / similarity** | Adapts feature magnitude to the largest eligible cosine similarity. The initial threshold is 0.70. Fixed mode still enforces the fidelity cap. |
| **Operation** | `auto` selects sequential mode for one sample and batch mode for 2–8. The explicit settings limit the node to the corresponding batch shape. |
| **Reference budget** | Caps extra positive transformer forwards per sequential image at 0–4. The provisional default is 2. Batch feature guidance uses none. |
| **Diagnostics** | Emits a bounded `MODEEXPLORER_TRACE` JSON log line per sampler invocation. Enable diagnostics with `native` and initialization off to examine the native trajectory. |

Use modes independently before evaluating their combination. If enabled guidance has no effect, inspect `shift_rms` with diagnostics. An eligible trajectory can have redundancy below the configured threshold or a near-zero dispersion gradient. Fixed mode is available for a controlled comparison. Check `reasons` for an unsupported stage or condition and `accepted` for the initialization trust-region decision.

## Compatibility and cost

| Path | Status |
| --- | --- |
| KSampler, KSamplerAdvanced, SamplerCustomAdvanced with a native sampler and BasicGuider or CFGGuider | Exercised through the installed nodes on CPU. Euler is the reference solver. Other native solvers use the actual sigma and keep their native update rules. |
| CFG-free Turbo and CFG Raw | Tiny CPU tests exercise CFG 1 and CFG above 1. The feature hook edits the positive branch. Initialization evaluates the native guided prediction. |
| Batch and reference latents | Batch feature guidance groups equal actual positive text, mask, and reference tensors. Reference and text tokens are excluded from the direct update. |
| LoRA | Standard weight patches remain on the model clone. Weight-patch ownership is tested. Identity consistency needs a production-checkpoint benchmark. |
| Precision | Analytic objective and hook tests cover FP32, FP16, BF16 and CPU autocast. Full-model smoke uses FP32. GPU kernels and quantized production checkpoints need live validation. |
| Multistage | Use the guided MODEL for the composition pass and the original MODEL for refinement. The [two-pass workflow](examples/multistage.json) shows this wiring. ModeExplorer also skips tagged Scale-Locked high-resolution stages and partial or nonempty-latent passes. Scale-Locked's native planner dispatch was source-audited. |
| Spectrum | Use a separate MODEL branch. An active Spectrum forecast may skip the transformer hook; combining both sets of wrappers on one MODEL is rejected. A native composition pass followed by a separately patched Spectrum refiner is the supported graph topology. |
| Other numerical patches | The enabled node requires native attention, token layout and dispatch. Regional/control conditioning, hook-scheduled weights, temporal batches, multi-GPU dispatch and context windows are outside the guided path. Standard native sampling remains available through the master switch. |

The feature modes use no model backpropagation. Batch math uses a sample-sized Gram matrix and at most a 4×4 spatial grid. During a guided positive evaluation the node allocates one copy of the attention output and a small pooled correction. Sequential guidance adds at most the reference budget in full positive DiT passes and retains two latent-sized tensors per active run. Initialization adds four guider evaluations, which may execute one or more DiT passes per call under CFG. An eight-evaluation Turbo Euler run can therefore use up to 10, 12, or 14 DiT passes with default sequential feature, initialization alone, or both, respectively. These are call budgets. Diagnostics synchronize to record compact summaries and can add wall-clock overhead.

See the [research notes](docs/research.md) for the adaptation and [validation guide](docs/validation.md) for matched benchmarks, blinded review, and current evidence.
