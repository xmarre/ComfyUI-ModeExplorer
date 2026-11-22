# Local Krea 2 research reports

Queue a campaign's **ModeExplorer Research Report** workflow on the ComfyUI installation that generated the experiments. The workflow reads the original trajectory exports and final planner PNGs under the current output directory. It writes one ZIP under `output/modeexplorer_research_reports/`. A default report runs on CPU and uses zero DiT forwards and zero VAE decodes. Full trajectory tensors and full-size PNGs remain on disk.

The campaign JSON widget is intended for workflow authors. It pins the unedited sham and any historical arms by archive ID. New arms are resolved by their deterministic labels, expected settings, intervention schedule and matched sham provenance. More than one compatible label match is allowed only when the recorded state metadata and every safetensors file are byte-equivalent. The resolver then chooses the lexically first archive ID and records the eligible IDs. PNG lookup uses pinned RGB fingerprints for historical images. For a new arm it uses the expected output prefix and the recorded prompt graph contract, including the block and label. Distinct PNGs that both satisfy that contract are reported as ambiguous. PNG qualification describes provenance of the saved image under that graph, and carries its own status.

## Campaign configuration

| Key | Value |
| --- | --- |
| `name`, `budget_mib` | Campaign name and ZIP budget in MiB. The default budget is 15. |
| `cases` | Mapping from case name to seed, reference slot, guarded actual/forecast route, planner resolution in pixels, sham pin and treatment pins. |
| `cases.*.crops` | Named numerical pixel boxes `[x0,y0,x1,y1]`, with exclusive upper bounds. |
| `cases.*.visual_crops` | Optional smaller native-pixel boxes inside the numerical regions, with the same names. Full-image previews have separate geometry. |
| `cases.*.arms` | Arms with expected `label`, `block`, `settings`, and the list of applied actual ordinals. `id` pins an existing archive. `require_exact` requires terminal and captured solver qualification for that arm. |
| `cases.*.arms.*.final` | Local Save Image output prefix, expected prompt hash and optional original decoded-RGB SHA-256. A fresh arm can require the prompt's feature label and block to match the pin. |
| `direction_reference` | Optional case-level arm name for exact initial direction-control verification within one block. |
| `intermediate_steps` | Optional case-level solver-step list of up to two selected actuals. Connect the original VAE. The report decodes only selected qualified states. |
| `regressions` | Optional list of independent numerical references given by name, path into `cases`, expected value and absolute tolerance. |
| `cases.*.allow_partial` | Optional. Preserve matched independent arms when another arm fails. The case and campaign gates still fail. Direction studies and campaigns with positional numerical references use the strict gate. |
| `cases.*.arms.*.require_runtime_telemetry` | Optional. Require recorded feature geometry, post-edit objective, and realization error for each applied proposal. Use a fresh matched trajectory when legacy telemetry is unavailable. |
| `blind_seed` | Optional private seed for reproducible anonymous case, arm and crop IDs. When set, the node creates a shareable visual ZIP and a separate local analysis with the mapping. |

On ModeExplorer, `feature_block=-1` resolves once against the loaded native single-stream block list: zero-based `floor(block_count / 2)`. The trace stores the resolved index and the archive stores the loaded count. A `"middle"` report pin requires both values. A localization arm estimates its proposal from the selected block's own live and reference streams. Feature cosine matrices are grouped by block.

## Blinded visual review

Share only the `blind_review_*.zip` file. It contains anonymous previews, lossless registered crops, optional intermediate panels and a minimal `modeexplorer.blind_review/1` manifest. The manifest gives anonymous IDs, image availability and case-level gate status. Image exports discard source metadata. The same anonymous arm IDs apply across the available visual material for that case.

Keep the campaign workflow, ordinary named report, and `output/modeexplorer_research_reports/private/<review_id>.json` local until ratings are locked. The private JSON contains the matching review ID, case/arm/crop mapping, full named numerical report, trajectory provenance and rejection reasons. Treatment metadata, sham-relative metrics and CSV are reserved for this local analysis because they can identify the controls. Blinding removes identifying metadata. It cannot hide visible differences or prior reviewer knowledge of the images.

## ZIP and JSON schema

The ZIP contains `report.json` (schema `modeexplorer.research_report/1`), `evaluations.csv`, `SUMMARY.md` and qualified visual evidence under `visual/`. The JSON is capped at 1 MiB, uses full floating-point serialization and records both uncompressed inventory byte counts and the exact ZIP size. JPEG previews and panels use one width and quality per report. Registered crops preserve native RGB pixels as lossless WebP or PNG. Image displacement metrics are derived from the original decoded RGB pixels before compression. Report writing is staged, size checked and published atomically. Interruption removes any staging file.

| Field | Semantics |
| --- | --- |
| `qualification`, `cases.*.qualification`, `regression` | Campaign and matched-case gates, plus independent numerical reference checks. The default gate suppresses comparisons for an incomplete case. For an independent-arm case with partial comparisons enabled, `comparison_status=qualified_subset` identifies surviving comparisons while the full case remains unqualified. Any rejected arm is excluded from every evaluation, response matrix and image panel. A failed sham prevents comparisons. A campaign with a reference mismatch needs numerical review before interpretation. |
| `cases.*.arms` | Pins, selected archive IDs, per-arm checks and reasons, actual/forecast ledger, performance telemetry and per-state checks. Exported FP32 bytes have their own SHA-256. The original native-input fingerprint is rechecked after an exact dtype round trip. |
| `cases.*.clean_coordinate` | `solver_clean` when the entire case is solver-certified. Otherwise all arms are compared using `baseline_clean`, the prediction using the cast transformer input converted once through Wan21. Available exact solver tensors are still verified independently. Legacy terminal and continuation evidence is explicitly unavailable where unrecorded. The chosen basis stays fixed if a case later excludes an arm. |
| `cases.*.evaluations` | Matched solver-step, opportunity and exact-sigma results. Each entry contains clean, conditional-velocity and input displacement, immediate RMS and L2 gains, amplification, alignment, proposal geometry, energy maps, and low/mid/high spectral measurements. |
| `proposal` | Compact and expanded intended norms are recomputed. Realized native-rounded RMS, objective and cosine telemetry come from the run. Realized L2 is derived from the exact RMS and the expanded element count. The full realized direction is unavailable. Feature geometry is recomputed from the proposal and latent shape, with a separate source label indicating whether recorded runtime geometry agrees or was unavailable. |
| `response_geometry` | Pairwise response cosine and distances, Gram, singular values and entropy effective rank for the sampled vectors. Each of these describes the finite set of measured responses at the applied amplitudes. |
| `regions`, `energy_map`, `spectral` | ROI and complement energies use the registered Wan21 8× mapping, rounding the box outward to cover intersecting cells. The 8×8 energy map gives a non-overlapping partition summed over channels. Radial bands follow the mechanistic latent-pixel boundaries at 0.0625 and 0.125 cycles per latent pixel. Spectral net change is displacement energy plus the baseline cross-term. |
| `signed_decomposition` | Initial odd and even clean response RMS and L2 for a verified matched proposal and its true reverse. |
| `visual_evidence`, `final_image` | Visual completeness and per-arm PNG checks. Final exact pixel metrics use saved originals. Compare structure, conditioning, identity and artifacts in the registered imagery. |
| `performance` | Recorded sampler, capture and serialization time, actual/forecast/reference and auxiliary counts, and peak memory only if measured. Report wall-time fields and optional VAE time are separate. Missing evidence uses null or an explicit unavailable status. |

Use the preserved visual evidence for qualitative conclusions. Norm gains, spectral changes and the sampled-response effective rank do not establish generalized anatomical improvement or identify the full model Jacobian.
