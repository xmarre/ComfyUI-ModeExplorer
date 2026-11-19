# Research and implementation notes

Source audit target: [ComfyUI c8abeb00ad2736b649ca2fceb8395e25072c7ea3](https://github.com/Comfy-Org/ComfyUI/tree/c8abeb00ad2736b649ca2fceb8395e25072c7ea3).

## Evidence and limits

Both checkpoints resolve to Krea2 / SingleStreamDiT in ComfyUI. Native CPU noise is seeded per requested seed (or indexed batch noise). Under the standard path, this rules out a constant noise tensor as an explanation; it does not rule out bad workflow wiring, reused seed/index, disabled noise, or rounding/checkpoint/patch effects. The public [Krea technical report](https://www.krea.ai/blog/krea-2-technical-report) and [inference repository](https://github.com/krea-ai/krea-2) identify Turbo's few-step and guidance distillation, and recommend no CFG for Turbo and full CFG for Raw. This makes distillation and CFG separate hypotheses. Fewer/larger solver steps, tightly specified prompts, LoRA weights, and planner/refiner settings also need separate controls. Source inspection cannot establish that any checkpoint collapses, locate the convergence time, or assign causality.

The [SDA LoRA](https://huggingface.co/F16/krea2-turbo-sda) uses a Raw teacher and a learned alignment objective on paired clean predictions. Its fixed-panel numbers evaluate that LoRA only. [Krea Explorations](https://github.com/fblissjr/krea-explorations) documents native/modular sampling, Raw with a Turbo adapter, and a two-resolution pass. [Krea Seed Variance Enhancer](https://github.com/harukimix/KreaSeedVarianceEnhancer) adds seed-dependent noise to text embeddings; that changes conditioning and is an independent baseline.

## Paper mechanisms

| Source | Mechanism and engineering implication |
| --- | --- |
| [FSG](https://arxiv.org/abs/2606.27371), §§3–4, Algorithm 1 and Appendix A | Minimizes log-mean-exp of pairwise cosine similarity in early internal image features, then runs the same block again and blends the two features. The paper's early window and second block are useful candidates; its FLUX block ablation, gradient scale and fidelity projection do not transfer verbatim to Krea 2. |
| [DetNO](https://arxiv.org/abs/2610.22267), §4, Algorithm 1 and appendix, [code](https://github.com/mobled37/DetNO) | Minimizes negative log determinant of an RBF Gram on pooled normalized predicted-clean latents, with a spatial x0 anchor. The SD3 implementation stops gradients through velocity, leaving the identity path from xt to x0. A ridge and bandwidth floor are essential for near duplicates. |
| [DivIn](https://arxiv.org/abs/2606.02453), §4 and Algorithm 1, [code](https://github.com/South7X/divin) | Tilts a Gaussian prior by the negative conditional–unconditional clean-prediction gap, then takes Langevin steps including the Gaussian score and stochastic term. It requires a meaningful unconditional prediction and a derivative through the model. Neither is assumed for Turbo. |
| [Collapse recovery](https://arxiv.org/abs/2601.00090), §3, [code](https://github.com/anneharrington/divgen) | Optimizes initial noise through the complete frozen denoising map using image-space diversity and quality rewards with stop/revert thresholds; sequential optimization depends on past outputs. Its pink-noise option modifies a spectral prior. Neither full trajectory gradients nor mutable output history meet the default runtime goal. |
| [DiverseFlow](https://arxiv.org/abs/2504.07894), §§4–5 | Couples ODE particles using log det(L) minus log det(L+I), applied to predicted endpoints. Source-radius quality supplies a separate constraint. This supports using predicted endpoints and collective volume, but its coupled ODE changes and backward passes are outside the native sampler integration. |

A measure-preserving transform of a standard Gaussian cannot restore population diversity under a fixed deterministic generator. Feature guidance changes the velocity operator; initialization changes the input law through a model-dependent edit with an explicit trust region.
