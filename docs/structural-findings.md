# Completed structural-discovery findings (Phases I–IV)

## Status and scope

The structural-discovery campaign is closed. Production exposes the tested depths as opt-in presets; this is not a claim of generally improved anatomical coherence, identity, prompt adherence or detail. Two retrospectively selected discovery cases ("anatomy" and "detail") are not an unselected benchmark. The user-reported corrections and restricted independent visual review are qualitative evidence, not population-level estimates. No new research gate is required for opt-in production use.

This record complements [research and numerical contracts](research.md), the [causal controls](mechanism.md), [frozen projection study](structural-discovery.md), [temporal protocol](phase-iv-structural-causality.md), and [report schema](local-research-reports.md). The protocol documents explain how to repeat the measurements, not a requirement to continue the campaign. Research implementation was preserved at the reviewed production head `a0295593a7d490b660d0811b23cd2b9c5134ab70`; the pre-production report implementation was `ea85f01eb02430edf9393decfc8ca0593b34d100`.

## 1. Earlier discovery and negative results

The historical independent research handoff records the following (not newly reconstructed from raw tensors here):

- Phase I–II: native repeats were bit-exact. A matched sham was necessary because reference eligibility itself changes guarded Spectrum history and can change the actual/forecast route. The original active arm differed from the matched sham by applied edits. Low-strength pilot observations are retained in [validation.md](validation.md#low-strength-pilot-and-follow-up).
- Phase III: the same-stored-weight continuous-activation counterfactual showed that INT8 activation coding was not necessary for a numerical feature response at the tested projections. This does not rule out quantization effects elsewhere or establish a perceptual effect. Downstream FP64 factorials were computationally impractical and were suspended; their code and methodology remain available. No general quantization-dependent explanation was established.
- Phase IV-A/B: comparison of sham, first-only, second-only and original active retained both reference forwards and both proposals. First-only at block 1 reportedly reproduced the useful anatomical orientation correction; second-only did not. This is case-specific qualitative sufficiency, not a general anatomical correction guarantee. The subsequent compact report directly confirms substantial downstream numerical divergence without a second applied edit.

## 2. Matched configuration and coordinates

Native Krea 2 INT8 ConvRot, planner 1632 × 2464, latent-only enlargement to 4256 × 6400 (zero target-resolution denoiser evaluations), CFG 1, 13-step ER-SDE with max_stage 1, eta 1, s_noise 1. Guarded Spectrum route `AAAAAAFAFAFAA`: 10 actual live evaluations, 3 forecasts, two positive reference forwards at actual ordinals 0 and 1, and two computed feature proposals. Active first-only arms apply ordinal 0 only. Sham computes both proposals without an edit. This route is a historical matching contract, not a production route requirement.

Common feature configuration: cosine, fidelity 0.70, min sigma 0.82, adaptive off, grid 4, bank 8, one reference trajectory, all selection, first_estimate continuation, bounded update, initialization off. Anatomy: strength 0.05, seed 354937070410882, reference slot 7. Detail: strength 0.03, seed 896275366471460, slot 8. The native architecture had 28 single-stream blocks; the middle sentinel resolved to zero-based index 14.

All compared arms use `baseline_clean` (cast transformer input, float velocity, one Wan21 conversion), because historical shams lack the exact solver input. Available `solver_clean` tensors were validated separately. Do not mix coordinate bases. Expanded feature dimension 96,509,952, clean latent dimension 1,005,312. RMS gain is clean-delta RMS / realized feature RMS; L2 gain uses the corresponding L2 norms. Neither is a Jacobian singular value. Region concentration is displacement energy fraction divided by registered latent area fraction. Spectral displacement is distinct from signed net band energy.

## 3. Phase IV-C — directional specificity and retrospective validation

The compact report qualified 14/14 arms and 140 actual states. Its 20 independent numerical references all passed; worst absolute error relative to its tolerance was approximately 0.165. The initial native input, reference and route were matched within each case; the directional control transforms were certified against the stored compact proposal. Historical clean reconstruction passed 140/140 checks; exact solver-clean reconstruction passed 100/100 available checks and 10/10 available terminal certifications were bit-exact. These are report-certified checks; the compact archive does not contain the original tensors.

Immediate clean displacement and realized native-rounded feature RMS:

| Direction | Anatomy clean RMS | Anatomy realized RMS | Detail clean RMS | Detail realized RMS |
| --- | ---: | ---: | ---: | ---: |
| Repulsion | 0.170286831 | 0.009801196 | 0.053746288 | 0.005476345 |
| Reverse | 0.277280755 | 0.009801188 | 0.047006998 | 0.005476336 |
| Channel sign 0 | 0.094987134 | 0.009801186 | 0.037978506 | 0.005476342 |
| Channel sign 1 | 0.123194942 | 0.009801175 | 0.043290696 | 0.005476344 |
| Channel sign 2 | 0.061520401 | 0.009801214 | 0.038417477 | 0.005476324 |

Relative range of realized RMS: 3.9909e-6 (anatomy), 3.9115e-6 (detail), approximately 0.0004%. Magnitude alone cannot explain the differing responses.

Anatomy channel-sign 0/1 injected proposal cosine was -0.007232323. Their immediate clean response cosine was 0.877763417, at actual 1: 0.978546498, at actual 2: 0.989568904, and at actual 9: 0.964530190. These responses became aligned before a subsequent solver update and alignment was strengthened during later evaluations. The end-to-end response is state-dependent and anisotropic; this does not identify full Jacobian rank or prove attractor basins.

Signed response (R+ and R- are initial clean displacements relative to sham; odd = (R+ - R-)/2, even = (R+ + R-)/2):

| Case | Odd RMS | Even RMS |
| --- | ---: | ---: |
| Anatomy | 0.216069294 | 0.079089639 |
| Detail | 0.032517595 | 0.038623395 |

Both cases are finite-amplitude responses, particularly detail where the even component exceeds the odd. The comparison controls the compact negation, not exact native-rounded vector negation.

### Independent regression references

| Metric | Expected | Absolute tolerance | Measured |
| --- | ---: | ---: | ---: |
| Anatomy sign 0/1 proposal cosine | -0.007232 | 0.000003 | -0.007232323 |
| Anatomy sign 0/1 clean cosine actual 0 | 0.877763 | 0.000040 | 0.877763417 |
| Anatomy sign 0/1 clean cosine actual 1 | 0.978546 | 0.000040 | 0.978546498 |
| Anatomy sign 0/1 clean cosine actual 2 | 0.989569 | 0.000040 | 0.989568904 |
| Anatomy sign 0/1 clean cosine actual 9 | 0.964530 | 0.000040 | 0.964530190 |
| Anatomy repulsion clean RMS | 0.17029 | 0.000030 | 0.170286831 |
| Anatomy reverse clean RMS | 0.27728 | 0.000030 | 0.277280755 |
| Anatomy sign 0 clean RMS | 0.09499 | 0.000030 | 0.094987134 |
| Anatomy sign 1 clean RMS | 0.12319 | 0.000030 | 0.123194942 |
| Anatomy sign 2 clean RMS | 0.06152 | 0.000030 | 0.061520401 |
| Detail repulsion clean RMS | 0.05375 | 0.000030 | 0.053746288 |
| Detail reverse clean RMS | 0.04701 | 0.000030 | 0.047006998 |
| Detail sign 0 clean RMS | 0.03798 | 0.000030 | 0.037978506 |
| Detail sign 1 clean RMS | 0.04329 | 0.000030 | 0.043290696 |
| Detail sign 2 clean RMS | 0.03842 | 0.000030 | 0.038417477 |
| Anatomy initial ROI concentration | 1.499 | 0.002 | 1.498695849 |
| Anatomy odd RMS | 0.216069 | 0.000040 | 0.216069294 |
| Anatomy even RMS | 0.079090 | 0.000040 | 0.079089639 |
| Detail odd RMS | 0.032518 | 0.000040 | 0.032517595 |
| Detail even RMS | 0.038623 | 0.000040 | 0.038623395 |

### First-only evolution

| Solver step (actuals only) | Anatomy clean RMS | Detail clean RMS |
| ---: | ---: | ---: |
| 0 | 0.170286831 | 0.053746288 |
| 1 | 0.701760531 | 0.150768703 |
| 2 | 0.717236529 | 0.368071634 |
| 3 | 0.621272850 | 0.400520986 |
| 4 | 0.582154474 | 0.385943883 |
| 5 | 0.572106585 | 0.372446752 |
| 7 | 0.536629255 | 0.333811056 |
| 9 | 0.516573250 | 0.311041912 |
| 11 | 0.501716393 | 0.298129118 |
| 12 | 0.499257838 | 0.297702308 |

Terminal/initial amplification 2.932x and 5.539x; peak/initial about 4.21x and 7.45x. Terminal response cosine to its own initial response -0.004741 and +0.094921. The trajectory reorganizes the change rather than simply scaling the initial latent residual. This includes ER-SDE dynamics, not just the transformer. The first-only and original two-edit arms share the initial clean fingerprint but differ by terminal inter-arm RMS about 0.258474 and 0.216046. They are not interchangeable.

Anatomy lower-body ROI covers 31.66% of latent area; displacement energy fraction increases from 47.45% to 65.40%, concentration from 1.499 to 2.066. Non-ROI displacement remains substantial. Detail terminal hair/garment/environment concentrations are 0.190/1.314/0.342. These are localization diagnostics, not quality scores.

Energy-based entropy effective rank, recalculated on the same five nonredundant direction arms (repulsion, reversal, three sign controls) rather than repeated sham/original proposals: anatomy proposal 3.787, response 1.580; detail proposal 3.786, response 3.818. Both sampled immediate response Gram matrices have five positive eigenvalues. The anatomy compression is case-dependent and does not mean the full response mapping is rank two.

### Frequency evidence

| Band | Anatomy displacement share | Anatomy net band change | Detail displacement share | Detail net band change |
| --- | ---: | ---: | ---: | ---: |
| DC | 0.528% | -0.776% | 3.972% | -1.001% |
| Low | 95.108% | -2.570% | 77.925% | +1.342% |
| Mid | 2.031% | -8.336% | 7.388% | +2.017% |
| High | 1.192% | -4.530% | 5.144% | +1.003% |
| Sub-patch | 1.142% | -2.049% | 5.570% | +0.827% |

This is the initial repulsion response. Global high-frequency boosting or sharpening is unsupported; a coherent detail change can arise downstream of low-frequency structural displacement.

## 4. Phase IV-D — block localization and refresh

All 8 new depth runs qualified; each localization report has 12 arms and 120 actual states (100 available exact solver states, 10 terminal certificates). The original and refreshed localization reports retain the same sham and eight additional-depth arms. Refreshed block 1 matches its historical first-only counterpart in each case: 10/10 native inputs, solver inputs, conditional velocities, baseline-clean and exact solver-clean fingerprints; 2/2 compact proposal fingerprints; terminal solver and final decoded RGB fingerprint. The refresh fills missing runtime geometry, objective-after and rounding-error evidence. Historical shams still lack exact solver coordinates.

| Case | Block | Realized feature RMS | Relative proposal RMS | Initial clean RMS | Initial RMS gain | Terminal clean RMS | Terminal/initial |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Anatomy | 0 | 0.053198982 | 0.015652826 | 0.398828714 | 7.496924 | 0.567261212 | 1.422318 |
| Anatomy | 1 | 0.009801196 | 0.014997937 | 0.170286831 | 17.374087 | 0.499257838 | 2.931864 |
| Anatomy | 2 | 0.009448851 | 0.013077950 | 0.092574165 | 9.797400 | 0.421955764 | 4.558029 |
| Anatomy | 5 | 0.010066222 | 0.010601210 | 0.052700316 | 5.235362 | 0.492370489 | 9.342838 |
| Anatomy | 14 | 0.016861610 | 0.011885156 | 0.018091915 | 1.072965 | 0.149069871 | 8.239585 |
| Detail | 0 | 0.030258657 | 0.008447574 | 0.089882277 | 2.970465 | 0.308252555 | 3.429514 |
| Detail | 1 | 0.005476345 | 0.008429179 | 0.053746288 | 9.814262 | 0.297702308 | 5.539030 |
| Detail | 2 | 0.004342947 | 0.006069789 | 0.048348106 | 11.132557 | 0.158861953 | 3.285795 |
| Detail | 5 | 0.006286691 | 0.006682221 | 0.031742200 | 5.049111 | 0.199586837 | 6.287744 |
| Detail | 14 | 0.010835705 | 0.007335697 | 0.018543088 | 1.711295 | 0.130903925 | 7.059446 |

Immediate displacement decreases monotonically with depth in these two cases. Block 0 receives 5.43–5.53x the absolute edit of block 1 while representation-relative magnitude is similar. Depth and edit scale are confounded. Blocks 1/2 lead per-absolute-RMS gain; block 5 in anatomy begins at 31% of block-1 response but reaches 99% of its terminal RMS. A large amplification ratio can simply reflect a small denominator.

### Terminal spatial metrics and initial feature telemetry

| Block | Anatomy ROI concentration | Anatomy complement RMS | Detail hair concentration | Detail garment concentration | Detail environment concentration |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1.679921 | 0.469477 | 0.206868 | 1.448149 | 0.126900 |
| 1 | 2.065571 | 0.355233 | 0.189623 | 1.313667 | 0.341981 |
| 2 | 2.254495 | 0.273043 | 0.148068 | 1.409135 | 0.126859 |
| 5 | 1.310803 | 0.455539 | 0.317482 | 1.821297 | 0.413134 |
| 14 | 0.761648 | 0.157086 | 0.100998 | 1.572210 | 0.663894 |

Anatomy block 2 has the greatest concentration and lower complement RMS than blocks 0, 1 or 5, not the absolute minimum of all blocks (14 has weak overall response). Detail block 5 has greatest garment concentration. Neither metric proves correction.

| Block | Anatomy pooled reference cosine before → after | Detail pooled reference cosine before → after | Anatomy rounding error / intended RMS | Detail rounding error / intended RMS | Anatomy initial DC+low share | Detail initial DC+low share |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 0 | 0.403388 → 0.080775 | 0.399743 → 0.204843 | 1.101% | 1.942% | 96.871% | 84.215% |
| 1 | 0.605672 → 0.554563 | 0.609969 → 0.582378 | 2.492% | 4.467% | 95.636% | 81.898% |
| 2 | 0.860702 → 0.836504 | 0.873700 → 0.863721 | 1.817% | 3.890% | 93.261% | 83.700% |
| 5 | 0.947149 → 0.937198 | 0.938369 → 0.932185 | 2.440% | 3.921% | 86.821% | 75.841% |
| 14 | 0.904333 → 0.894320 | 0.910938 → 0.905163 | 2.664% | 4.218% | 70.553% | 67.562% |

Block-1 absolute realization-error RMS: 0.000244204 / 0.000244390. Intended RMS: 0.009797837 / 0.005471303. Objective displacement: -0.102219 / -0.055182. Block 0 achieves far greater reference-cosine repulsion without demonstrated quality advantage. Rounding is not an established principal explanation for depth sensitivity. Higher-frequency displacement share at greater depth does not imply sharpening.

### Response geometry

Anatomy early block pairs (0,1), (1,2), (0,2) have immediate cosine approximately 0.904, 0.892, 0.765, and terminal 0.755, 0.734, 0.637. Block 5 is initially negatively aligned with early blocks (1,5) = -0.365, (0,5) = -0.441, (2,5) = -0.259; terminal values become positive. A change in sampled response geometry lies between blocks 2 and 5; the transition has not been localized more precisely. Detail (0,1) alignment increases from 0.589 to 0.846, with weaker terminal alignment for block 2. This is case-dependent, not a fixed universal structural direction.

## 5. Blinded visual review and limitations

Review ID `blind_review_29a0e3d278de9b27379b6f30`. Ratings were supplied before unblinding. Independent reviewer inspected 12 previews (560 × 845), 24 native-resolution crops, two cases, six arms each. Its scope excluded some primary anatomical regions, and prompts/identity references and designated baseline were absent. No absolute conditioning or identity rating was possible. This is a single restricted qualitative review. Treatment mapping was recovered from the separate private manifest; the refreshed outputs match the original localization outputs by recorded fingerprints.

| Anonymous arm | Case 01 (detail) | Case 02 (anatomy) |
| --- | --- | --- |
| arm_01 | Sham | B5 |
| arm_02 | B2 | B2 |
| arm_03 | B0 | Sham |
| arm_04 | B14 | B1 |
| arm_05 | B5 | B14 |
| arm_06 | B1 | B0 |

All detail arms scored 8.0 structure, 8.5 meaningful detail, 8.0 artifact avoidance on the inspected elements. Anatomy sham, B0, B1, B2 and B14 received those same scores. Anatomy B5 received 7.0 / 8.0 / 7.5 and was a structural outlier. Across 10 active-versus-sham comparisons there were nine ties, one lower rating and no higher ratings on this rubric. Ties do not prove perceptual equivalence. Primary anatomical correction could not be comprehensively rated. No generalized quality gain is established.

Independently identified geometry clusters: detail B0/B1 versus sham/B2/B5/B14, anatomy B0/B1, sham/B14, and distinct B2/B5 configurations. These agree qualitatively with the numerical geometry. B5 supplies a counterexample to amplification as a quality proxy in the evaluated scope. The result does not rule out its utility in other cases.

## 6. Evidence inventory and reproducibility

The original private evidence is retained outside the public repository. Do not publish source images, resolved conditioning, private workflow graphs or raw tensors by default. Archive names and SHA-256 identify the exact evidence bytes; original files can be kept offline or in a private archive. The full report JSON, CSV, summary, regression references, per-state fingerprints, qualification checks, visual inventory, ROI definitions, and recorded timings are retained inside the report ZIPs.

| Archive | SHA-256 | Scope |
| --- | --- | --- |
| phase_ivc_retrospective_to4iux_j(1).zip | 2d4590cfa43631cc61eb1eb1d1f2f0e6c7c0113097c7d8711de70ee9536de5f9 | 47 entries; directional and temporal report |
| phase_ivd_localization_p5u202l1.zip | 8c9d1559936fd28c4398720ddb14a8c4a0ac0837f0cba5247c917ab7851cab15 | 41 entries; original localization |
| phase_ivd_localization_with_block_1_refresh_qq6vn0we.zip | d1f0bbe249eb329443951124372d0eb7bfc1bf9a1d3a627cea907e9169261141 | 41 entries; refreshed localization |
| modeexplorer_phase_ivd.zip | 20932a8235b5db19a760d84dac59a072c5c1025a5ed04b58f703444ccba39778 | Original eight generation PNGs |
| ModeExplorer_Phase_IV_D_Reviewed_Workflows.zip | 520d6954635df982a922c5048ee480355677877df4271bd88035b286fd847072 | Original campaign definitions and graphs |
| blind_review_29a0e3d278de9b27379b6f30_vt_tw2_n (2).zip | b3699af94c3819b8fcb96c4cf179c71b7d44f64d63488db05edcfec608780369 | Outer ZIP with anonymous images and private mapping |

The 3 reports passed ZIP CRC checks. Their report computation before writing was 16.505 s, 14.780 s and 11.703 s respectively, with zero additional DiT forwards and zero VAE decodes. This is offline reporting cost, not sampler or GPU performance. Report-only transfer is roughly 12–15 MiB per campaign; raw trajectories remain separate.

Trajectory pins (all have prefix `trajectory-`):

| Arm | Anatomy | Detail |
| --- | --- | --- |
| Sham | t2wbtfrd | lf5brxh0 |
| Original active | f5h29mfx | r_5041dp |
| First-only B1 historical | icpob67x | zde846u4 |
| Second-only (historical handoff) | 8z59h5o4 | dknv6eup |
| Reverse | 6a2z6hvb | _0ijnz3f |
| Channel sign 0 | pyzb8si7 | cttonsa_ |
| Channel sign 1 | rf3kq653 | 6ra2luny |
| Channel sign 2 | e6q4ub_2 | uyfbgzco |
| B0 | wqjtpq9z | afbzks4i |
| B2 | p0p65j2o | _944a76k |
| B5 | wa88it7c | acgagdvi |
| Middle B14 | 9wdwoy1m | lib5bh73 |
| First-only B1 refresh | 7k7xv_50 | ho6b0wdv |

The compact reports certify checks from their original raw tensors; they do not independently provide those tensors, a comprehensive effective-weight fingerprint, a cryptographic binding between terminal latent and final PNG, or full production CUDA timing/peak memory. Independent algebraic checks of the reports found norm/Gram, ROI and energy conservation consistent to floating-point tolerance (spectral displacement relative error about 1.25e-7); small net spectral differences are subject to subtraction cancellation. These checks are not evidence of perceptual quality.

## 7. Production decision and open questions

- Retain Original and Custom plus first-only B0, B1, B2, B5 and dynamic Middle presets. B1 is the reproduced discovery configuration; B2 is a promising spatially selective alternative. B0, B5 and Middle expose distinct behaviors rather than validated quality rankings. Do not remove B5 based on a single restricted review.
- Suggested starting strength 0.03, fidelity 0.70, bounded cosine, grid 4, two-reference budget and guarded Spectrum. Presets disable adaptive gating to retain the investigated feature recipe. Scale remains user-controlled.
- Keep the two-reference/proposal eligibility policy separate from first-only edit application. A one-reference configuration changes the history boundary and may alter Spectrum routing; it is not numerically equivalent by assumption.
- No automatic quality selection from repulsion, RMS, amplification, spectrum, effective rank or spatial concentration. None is a validated perceptual objective.
- Production integration is opt-in. New held-out campaigns, exhaustive depth/strength sweeps and FP64 investigations are suspended. Any future statistical validation must report recovered and newly introduced defects, preserve baseline quality and condition/identity context, and use unselected cases. It is not a prerequisite to use the node.
- Remaining uncertainty: generalization, direction selection, strength calibration, CUDA throughput/VRAM and reliable assessment of structural quality under complex production conditioning.

The reviewed production implementation passed exact-head CPU validation (248 passed, 14 skipped) and pinned companion validation (282 passed, 5 skipped) in [run 45085113146](https://github.com/xmarre/ComfyUI-ModeExplorer/actions/runs/45085113146). This is engineering evidence at the cited head, not full-checkpoint CUDA or image-quality validation.

## 8. Post-campaign production observation: high-strength block 0 (2026-11-23)

This observation was collected during ordinary use, not as a new controlled quality study. The operator reported a complex, difficult scene in which the unpatched generation already had coherent anatomy. Several low-strength production modes produced different, visually conflicting body arrangements; block 0 preserved the originally coherent configuration but initially seemed to reduce background detail. Increasing the edit strength at block 0 through 0.15, 0.30, 0.50, 0.75 and 1.00 reportedly produced substantially more varied body/object interactions and facial expressions while retaining coherent anatomy and recovering environmental detail in the reviewed outputs. Higher-strength block 2 reportedly retained more of the original composition; block 5 continued to exhibit a conflicting anatomical configuration. These are **operator visual observations**, not independently inspected or blind-scored images. The earlier impression of slightly enhanced detail from block 2 and Middle Refinement also remains a case-specific subjective observation.

The two available diagnostic traces cover block 0 at strengths 0.75 and 1.00. Each used the same seed, resolved conditioning, initial noise, sigma schedule, reference slot and native single-stream block. The trace does not itself serialize the strength widget; strength identification follows the operator's run order and the near-exact 4:3 ratio of realized edits.

| Metric | Strength 0.75 | Strength 1.00 |
| --- | ---: | ---: |
| Initial sigma | 0.999968350 | 0.999968350 |
| Initial reference cosine | 0.491066158 | 0.491066158 |
| Initial objective | 0.982132316 | 0.982132316 |
| Initial proposal RMS | 0.759044647 | 1.012059450 |
| Applied pooled edit RMS | 0.759042382 | 1.012063980 |
| Applied pooled relative RMS | 0.229006797 | 0.305344641 |
| Reference cosine after edit | -0.756117702 | -0.787354290 |
| Second proposal computed | Yes | Yes |
| Second proposal applied | No | No |
| Planner live/reference/forecast forwards | 10 / 2 / 3 | 10 / 2 / 3 |
| Planner trace time (s) | 19.444983 | 19.578846 |

Both used the same `AAAAAAFAFAFAA` route, first-only feature application, grid 4, positive reference slot 7, and one native-rounded edit at the first positive actual evaluation. The stage-tagged high-resolution pass then used two actual evaluations without an additional ModeExplorer edit. Their original planner attention representation before the first edit matched in the recorded summaries. At the second proposal (not applied), the reference cosine had already diverged (0.506958 vs. 0.467242), consistent with a changed downstream state. Conditional-clean summaries after the first applied edit differed.

The increase from 0.75 to 1.00 scaled realized edit RMS by 1.3333× (up 33.33%) and pooled relative RMS from 22.90% to 30.53%. The cosine after the intervention was strongly negative in both configurations. This is a substantially more aggressive operating regime than the earlier 0.03–0.05 discovery settings. The bounded objective has no image-quality guarantee; a larger negative reference cosine does not imply greater perceptual quality.

Interpretation: the depth-0 intervention can act as a deterministic composition/structure variation control at a fixed seed and prompt. It does not introduce new stochastic noise. Multiple distinct outcomes when **changing strength** do not prove improved **between-seed** diversity at a fixed strength. The observation suggests that feature repulsion can redirect structure both toward and away from coherent arrangements, depending on the native state and selected intervention. No evidence establishes a targeted anatomical-correction objective, universal high-strength safety or guaranteed identity/conditioning retention. The reported recovery of detail and anatomical coherence cannot be independently established without the images.

The original 99-line diagnostic capture is retained privately under the evidence library, identified by SHA-256 `4961fc97a90b2077bd7e2d2a9ac6067649ae6984dd478d88c43b169625359f57`. It contains both ModeExplorer traces, Spectrum route summaries, Scale-Locked high-resolution handoff telemetry and production warnings. Private conditioning, images and local output paths are intentionally not reproduced here.
