# ComfyUI-ModeExplorer

Experimental early-flow diversity guidance for native Krea 2 Raw and Turbo. The node takes and returns a standard ComfyUI MODEL. Use it to measure whether camera position, framing, pose and subject arrangement vary more across seeds under the same prompt. Full-checkpoint anatomy, likeness, prompt adherence and composition effects remain to be measured.

## Setup

1. Apply the ModeExplorer PR overlay in ComfyUI Patcher and restart ComfyUI. Apply its companion overlay if using Spectrum.
2. Load native Krea 2 weights, a Qwen3-VL-4B encoder in Krea 2 mode, and the matching Qwen Image VAE. Apply weight LoRAs upstream.
3. Connect **MODEL → ModeExplorer → KSampler**. With additional patches use **MODEL → ModeExplorer → optional Sol-Krea2 → Spectrum → sampler**. Place a native/guided switch upstream of Spectrum so each arm shares the same settings.
4. Load an [example workflow](examples/README.md). Match the prompt, model, seed, conditioning and sigma schedule during comparisons.

The core Model Attention Backend can select PyTorch or Comfy Kitchen upstream. A KJ Diffusion Model Loader can also supply its Sage override. ModeExplorer retains the attention backend set by the last selector on the model path.

## Main controls

| Control | Behavior |
| --- | --- |
| **Enabled** | Off returns the unpatched MODEL or removes an existing ModeExplorer patch from a clone. |
| **Feature** | Native, cosine or RBF volume. Select Native with Initialization enabled for initialization-only. |
| **Strength / Fidelity** | Defaults 0.45 and 0.55. Strength scales the feature and noise proposals; the widget supports 0.001 increments with automatic frontend float precision. Fidelity tightens the activation and predicted-clean trust regions. |
| **Min sigma** | Absolute minimum flow sigma for feature guidance. Applies to the first full-noise empty-latent pass. Tagged Scale-Locked high-resolution passes bypass ModeExplorer. |
| **Adaptive / Similarity** | Adaptive scales the edit from zero at the similarity threshold toward its maximum at cosine 1. Fixed mode keeps the fidelity bound. |
| **Operation** | Auto selects sequential for one image and batch for 2–8. Batch guides equally conditioned positive rows with zero auxiliary feature forwards. |
| **Reference budget** | Zero to four auxiliary positive transformer forwards per sequential sample across the invocation. The budget can run out before reaching Min sigma. |
| **Initialization** | At most six auxiliary guider calls and twelve transformer forwards. Accepts a trial only when measured log-volume increases inside the predicted-clean trust region. The 45-second deadline stops new auxiliary launches and allows an in-flight call to finish. Hitting the deadline can change which initialization candidates are evaluated at a fixed seed. |
| **Diagnostics** | Logs stage, native and auxiliary costs, measured feature shifts and each initialization trial. |
| **Full Spectrum** | Off by default. Guarded mode forces eligible early guidance to use actual forwards and clears history at the feature boundary. Full mode retains Spectrum eligibility and history. With the companion overlay, forecasts spend zero references. If Spectrum rejects the patch graph or a live dispatch, eligible native evaluations still receive ModeExplorer guidance. |

## Optional controls

| Control | Scope |
| --- | --- |
| **Feature block / grid** | Zero-based native block and maximum cells per axis (2, 4, 6, or 8). Defaults 1 and 4. |
| **Cosine temperature / Volume bandwidth** | Change weighting in eligible cohorts of three or more. Pairs retain the default values. Median bandwidth uses observed collective feature distances. |
| **Reference source / bank size** | The default eight-slot bank remains available. Enlarge it or use independently derived per-seed tracks. |
| **Reference trajectories / selection** | Probe one to three tracks per actual step within the shared budget. Use all measured features jointly or the highest cosine match. |
| **Reference continuation** | First estimate uses the first predicted clean state. Rolling Euler uses each shadow's latest velocity. The live sampler retains its native update and RNG. |
| **Feature update / Sham** | Bounded is the lightweight baseline. Verified adds an area-weighted mean constraint, an 8% pooled RMS cap and objective backtracking on pooled features. Sham measures feature proposals without applying them and skips initialization. |
| **Initialization trials** | One to three total proposals after three baseline/reference guider calls. The first uses the original radius. Later proposals back off after rejection. |

See [Research](docs/research.md) for equations and limitations and [Validation](docs/validation.md) for matched comparisons and cost measurements.
