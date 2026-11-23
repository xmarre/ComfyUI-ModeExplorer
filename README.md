# ComfyUI-ModeExplorer

Completed measurements, negative results, reproducibility evidence and the production decision are recorded in [Completed structural-discovery findings](docs/structural-findings.md).

Selectable early-flow feature guidance for native Krea 2 Raw and Turbo. The node takes and returns a standard ComfyUI MODEL. Use the mode dropdown to compare intervention depths in your normal sampling graph. The image-quality effect depends on the prompt, checkpoint, weight patches, seed and sampler.

## Setup

1. Apply the ModeExplorer PR overlay in ComfyUI Patcher and restart ComfyUI. Apply the [Spectrum ModeExplorer overlay](https://github.com/xmarre/ComfyUI-Spectrum-Krea2/pull/6) if using guarded forecasting.
2. Load native Krea 2 weights, a Qwen3-VL-4B encoder in Krea 2 mode, and the matching Qwen Image VAE. Apply weight LoRAs upstream.
3. Connect **MODEL → ModeExplorer → optional Sol-Krea2 → optional Spectrum → sampler**. Place a native/guided switch upstream of Spectrum so each arm shares the same settings. For an opt-in response measurement, place ModeExplorer Response before Sol and Spectrum.
4. Load an [example workflow](examples/README.md). Match the prompt, model, seed, conditioning and sigma schedule during comparisons.

The core Model Attention Backend can select PyTorch or Comfy Kitchen upstream. A KJ Diffusion Model Loader can also supply its Sage override. ModeExplorer retains the attention backend set by the last selector on the model path.

## Mode dropdown

| Mode | Zero-based native single-stream block | Applied feature proposals |
| --- | --- | --- |
| **Original** (default) | 1 by default, or the historical Feature block value | All eligible proposals. An explicit research ablation route can select a fixed actual-evaluation ordinal. |
| **Early Structural (B0)** | 0 | First eligible proposal only |
| **Early Structural (B1)** | 1 | First eligible proposal only |
| **Early Structural (B2)** | 2 | First eligible proposal only |
| **Amplified Structural (B5)** | 5 | First eligible proposal only |
| **Middle Refinement** | Floor of half the loaded block count | First eligible proposal only |
| **Custom** | Feature block, or −1 for the loaded middle | All, first only or second only, as set by Feature application |

The five depth presets set the block, apply the first eligible proposal and disable adaptive similarity scaling. Feature kind, strength, fidelity, reference settings and the optional sham and direction controls use their widget values. Feature block and Feature application are visible in the standard widget group for Custom. A forced research ablation route requires Original mode.

An eligible proposal comes from a positive actual evaluation in the configured sigma window with the required reference or batch inputs. Forecasts never consume a proposal ordinal. Custom with Adaptive enabled can use a scheduled application on a zero shift if the similarity gate is closed. A computed proposal can still be zero if the feature objective produces no shift or the edit rounds to zero in native precision. In guarded Spectrum, reference eligibility and the history transition still follow the reference budget and sigma window. Spectrum determines the route at runtime.

Existing graphs and API prompts that omit Mode use Original. The new widget is appended after the historical fields. When generating workflow JSON, write matching values if supplying both positional and named widgets, and omit obsolete embedded copies of the graph.

## Main controls

| Control | Behavior |
| --- | --- |
| **Enabled** | Off returns the unpatched MODEL or removes an existing ModeExplorer patch from a clone. |
| **Feature** | Native, cosine or RBF volume. Select Native with Initialization enabled for initialization-only. |
| **Strength / Fidelity** | Defaults 0.45 and 0.55. Strength scales the feature and noise proposals in 0.001 increments. Fidelity tightens the feature bound and the initialization clean-latent trust region. Feature magnitude depends on the local activation scale, proposal direction and optional adaptive gate. Equal strength across depths can yield different absolute and relative displacements. |
| **Min sigma** | Absolute minimum flow sigma for feature guidance. Applies to the first full-noise empty-latent pass. Tagged Scale-Locked high-resolution passes bypass ModeExplorer. |
| **Adaptive / Similarity** | Original and Custom can scale the edit from zero at the similarity threshold toward its maximum at cosine 1. The fidelity bound applies in every mode. |
| **Operation** | Auto selects sequential for one image and batch for 2–8. Batch guides equally conditioned positive rows with zero auxiliary feature forwards. |
| **Reference budget** | Zero to four auxiliary positive transformer forwards per sequential sample across the invocation. The budget can run out before reaching Min sigma. |
| **Initialization** | At most six auxiliary guider calls and twelve transformer forwards. Accepts a trial only when measured log-volume increases inside the predicted-clean trust region. The 45-second deadline stops new auxiliary launches and allows an in-flight call to finish. Hitting the deadline can change which initialization candidates are evaluated at a fixed seed. |
| **Diagnostics** | Off by default. Enable to log stage, native and auxiliary costs, measured feature shifts and each initialization trial. A separate advanced Trajectory label opts into a local archive. |
| **Full Spectrum** | Off by default. Guarded mode forces eligible early guidance to use actual forwards and clears history at the feature boundary. Full mode retains Spectrum eligibility and history. With the companion overlay, forecasts spend zero references. If Spectrum rejects the patch graph or a live dispatch, eligible native evaluations still receive ModeExplorer guidance. |

## Low-strength starting configuration

For a sequential comparison, start with **Early Structural (B1)** or **Early Structural (B2)**, cosine guidance, strength **0.03**, fidelity **0.70** and reference budget **2**. Keep grid 4, one bank reference per actual evaluation, first-estimate continuation, bounded updates, initialization off and min sigma 0.82. Use guarded Spectrum when installed.

Compare a fixed seed and prompt with Enabled off, then switch modes while holding the model, conditioning and sampler settings fixed. Use Custom to select another available block or change when the proposal is applied. See the [low-strength profile](tools/low_strength_profiles.json) and [validation guide](docs/validation.md) for matched comparisons.

## Compute and compatibility

With the default two-forward budget and one reference per evaluation, sequential guidance spends up to two extra positive DiT forwards and computes up to two proposals when both evaluations are eligible. The structural presets apply at most one. Original and Custom All can apply both. Batch guidance uses zero extra feature forwards and can compute a proposal on each eligible actual evaluation. Initialization is an independent opt-in cost.

A reference budget of **1** caps sequential guidance at one reference forward and one proposal. Treat it as a separate sampling configuration. Spending the budget sooner can move the guarded Spectrum history transition and change which evaluations forecast.

Each proposal pools image-token attention at one block and computes a shift on the coarse feature grid. Applying it clones the native attention result once and adds bounded cell offsets in native dtype before the model's own gate and projection. Production skips trajectory fingerprints, spectra, spatial measurements and VAE work. After spending the reference budget, the ordinary path releases shadow latents and skips continuation updates for the last reference group. A single reference is reused directly and a one-row sigma check reads one scalar. The guarded route can still alter the count of actual evaluations when Spectrum is active.

Use a full-noise empty unmasked latent for feature guidance. Tagged Scale-Locked high-resolution stages and partial denoise bypass it. Weight patches loaded upstream share the same model on the auxiliary forwards. Unsupported wrapper, dispatch and conditioning configurations use the existing rejection or native fallback paths. Full Spectrum is an experimental opt-in that can retain anchors across the edit boundary.

## Optional controls

| Control | Scope |
| --- | --- |
| **Feature block / grid** | Zero-based native block and maximum cells per axis (2, 4, 6, or 8). Defaults 1 and 4. |
| **Cosine temperature / Volume bandwidth** | Change weighting in eligible cohorts of three or more. Pairs retain the default values. Median bandwidth uses observed collective feature distances. |
| **Reference source / bank size** | The default eight-slot bank remains available. Enlarge it or use independently derived per-seed tracks. |
| **Reference trajectories / selection** | Probe one to three tracks per actual step within the shared budget. Use all measured features jointly or the highest cosine match. |
| **Reference continuation** | First estimate uses the first predicted clean state. Rolling Euler uses each shadow's latest velocity. The live sampler retains its native update and RNG. |
| **Feature update / Sham** | Bounded is the lightweight baseline. Verified adds an area-weighted mean constraint, an 8% pooled RMS cap and objective backtracking on pooled features. Sham measures feature proposals without applying them and skips initialization. |
| **Feature direction / Direction trial** | Experimental causal controls: reverse or channel-sign randomization of the bounded proposal, without changing its pre-rounding cell norms. Default repel. Trial selects an independent deterministic mask. |
| **Initialization trials** | One to three total proposals after three baseline/reference guider calls. The first uses the original radius. Later proposals back off after rejection. |

See [Research](docs/research.md) for equations and limitations and [Validation](docs/validation.md) for matched comparisons and cost measurements. See [Causal investigation](docs/mechanism.md) for the norm-matched direction experiment and its limitations.

## Response experiments

Use [ModeExplorer Mechanistic](docs/mechanistic.md) for full-resolution native-site spectra, signed decomposition, matched conditioning experiments, and opt-in clean component steering. The guide includes planner-only research profiles. See [planner structural discovery](docs/structural-discovery.md) for actual-evaluation trajectory capture and the frozen INT8 activation factorial. Use [local research reports](docs/local-research-reports.md) to qualify and compare complete trajectory campaigns in ComfyUI and export one compact ZIP.

[Projection Guidance](docs/projection_guidance.md) is an opt-in in-forward experiment for native Krea 2. Gaussian, token-adaptive, and channel-aware edits use the existing attention output projection with zero auxiliary DiT or projection calls. The guide includes planner PNG comparisons and a predeclared screening and confirmation protocol.

[Conditioning contrast](docs/mechanistic.md#conditioning-contrast-directional-research) tests a semantic direction derived from two conditional predictions at the same noisy state, with reversed and spatial controls. It uses one extra DiT evaluation per state, or two with an optional repeat, and the existing clean-response trust and acceptance rules. It is an exploratory experiment, not established quality guidance.

The **ModeExplorer Response (Krea 2)** node measures input and feature response at a frozen denoising state. It supports projected Jacobians, head and cell ablations, local predicted-clean exports, and opt-in response-based steering. One default latent probe spends seven auxiliary DiT forwards. Packed CFG processes fourteen auxiliary rows for those seven forwards. Diagnostics follow the existing actual/forecast route unless the all-actual control is enabled.

Start with the [latent probe workflow](examples/response_probe.json). The [response guide](docs/response.md) describes the measurements, Raw/Turbo pairing, candidate acceptance and GPU experiments. The [response profiles](tools/response_profiles.json) use the existing evaluator and blinded analyzer.
