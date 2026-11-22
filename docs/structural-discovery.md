# Planner structural discovery

Two discovery outcomes have separate acceptance criteria. For anatomy, track the visibility, connection and orientation of the same limb or joint and record defects introduced elsewhere. For detail, track structure at comparable object scale and position. Use full-resolution crops of the same hair region, seam or fold, plant or mushroom structure, and surface texture. Score coherent structure, loss of structure, sharpening, noise, and duplication separately. Use ties and unjudgeable regions. Independently confirm the outcomes on an untouched, unselected split after the discovery controls have reproduced them.

This campaign uses the original production checkpoint and planner settings. Conditioning contrast remains available in the Mechanistic node. The reproduction workflows use native, native repeat, sham and the original feature intervention. Native repeat establishes a numerical floor. Sham computes the proposal and incurs reference forwards with zero applied field. It can still exercise different Spectrum decisions from an unpatched run. Hold the resolved conditioning, seed, model, attention backend, adapters, solver and VAE constant. Compare guarded Spectrum routes before interpreting any differences. The companion Scale-Locked planner-only mode produces the same planner and zero target-resolution evaluations.

## Trajectory capture and comparison

Set `trajectory_label` on the ModeExplorer node. An empty label disables capture. For the native arm, leave the node enabled with `feature=native` and initialization off. Use distinct labels for the repeat and treatment arms so that ComfyUI evaluates them separately. Capture takes place in the planner and standalone sampler stages, downstream of the actual positive-row dispatch. The observer preserves the existing solver and Spectrum forecast policy. Forecasts appear in the route ledger and produce no actual-state tensors. For a 13-step ER-SDE run with forecasts, there may be fewer than 13 states.

The observer performs zero auxiliary transformer forwards. Existing feature guidance retains its ordinary reference cost. Each state contains the noisy transformer input and conditional velocity in native flow coordinates, and the clean prediction converted through `process_latent_out` for the same VAE. On an active arm, the velocity includes the feature edit. Captures retain only FP32 CPU latent tensors, capped at 512 MiB per invocation and 32 states. For a 1632×2464 planner, one state uses about 11.5 MiB and 13 states use about 150 MiB. Two small compact proposal fields add roughly 0.75 MiB to typical guided runs. Transfers occur during capture and serialization occurs after denoising. The manifest records capture and serialization time. It also records exact sigma, actual ordinal, solver step, native or forecast route, available patch-coupled ER-SDE noise cursor, model coordinates, hashes of conditioning and inputs, the proposed compact shift and the realized edit after rounding.

The cell Gram matrix and neighbor cosines describe the spatially pooled **pre-gate attention** at the selected block. They describe variation among pooled cells after aggregation. Raw token covariance, anatomical region correspondence and attention routing require a separate targeted measurement after the visual regions and time of first difference are known.

Each invocation writes its own folder under `output/modeexplorer_trajectory`. Use the per-state ID with the existing Load Response node. The `tools.trajectory_report` utility builds a decoder-only graph using the original VAE filename and compares saved runs offline.

```text
python -m tools.trajectory_report decode --vae ORIGINAL_VAE.safetensors --output Decode.json NATIVE/manifest.json ACTIVE/manifest.json
python -m tools.trajectory_report compare NATIVE/manifest.json ACTIVE/manifest.json Comparison.json
```

An optional `--roi NAME Y0 Y1 X0 X1` adds band measurements for a visually registered shared region in the latent grid. Run the same comparison for native versus native repeat first. The report pairs exact sigma, solver step and denoiser opportunity and shows missing pairs and route disagreement. A visually aligned crop can still have a different latent coordinate if the object moved. Quantify composition and object scale before making a shared-grid comparison. Use the exported VAE clean tensors for decoding and compare visually at their native resolution. Each band reports baseline energy, displacement energy, cross-term, and net energy change. High-frequency displacement can represent sharper coherent structure, a position change, an artifact or noise. Establish those distinctions in the crops.

Float32 snapshots, seed and solver provenance support reproduction from the start. The observer does not serialize restart state for ER-SDE or Spectrum. Compare divergent trajectories descriptively. Use the frozen test for a matched-state causal estimate.

## Frozen activation factorial

After obtaining the native trajectory, select a native actual early evaluation. Keep the same checkpoint, model stack, conditioning, backend and route. Set `frozen_state` to the native trajectory state ID and `evaluation` to its actual ordinal. Pin the first eligible feature sigma that was observed in the live configuration. Select `projection_study=activation_factorial`, `projection_activation_check=true`, `component=observe`, `treatment=export_only`, and enable export. The node can evaluate a projection in a different block from the feature injection. CFG 1 with only the native positive row is required.

| Arm | Feature edit | Target projection |
| --- | --- | --- |
| A | Off | Native packed INT8 |
| B | Original proposal at the configured feature block | Native packed INT8 |
| C | Off | Continuous activations on generated-image tokens |
| D | Same proposal as B | Continuous activations on generated-image tokens |

All four arms share the exact same frozen noisy input. The baseline must reproduce the archived native velocity bit for bit. Both packed projection replays and the selected Kitchen quantizer and epilogue emulation must reproduce native output bit for bit across all packed tokens for both native arms. The report records Kitchen version, candidate GEMM route and reproduced quantizer route. A GEMM candidate records the audited dispatch contract and possible fallback, without instrumenting the native kernel launch. CUDA requires the audited CUDA quantizer. A CPU eager run can test the implementation.

The replacement keeps the stored weight codes, weight scales, native activation rotation, measured per-row scales, output dtype, and qualified epilogue. It removes activation coding on generated-image tokens in the target projection. Text and reference tokens retain the verified packed native output. The result propagates through the remainder of the same transformer. Reconstructed affine references and genuine original unquantized checkpoint weights are separate execution conditions. The factorial uses the stored INT8 checkpoint in every arm and does not require a BF16 checkpoint.

The exported arms are `baseline_clean`, `D01_clean`, `continuous_native_clean`, and `continuous_feature_clean`. Two reconstruction references, two native arms and two replacement arms cost at most six auxiliary transformer forwards for one frozen state. The existing CUDA workspace preflight and OOM handling bound the diagnostic path. The main planner always receives its original prediction. A failed qualification produces no C or D export. Check `mechanistic.projection.status`, `activation_counterfactual`, the per-arm packed replay, the recorded dispatch, the timings and `factorial.production_cuda` before interpreting C or D. This gated experiment qualifies an isolated downstream prediction. Live solver continuation requires additional production evidence.

Estimate the feature effect under each precision condition and the interaction `(D − C) − (B − A)`. Compare C versus A and D versus B visually and account for the repeatability floor. A visible C-versus-A correction at this state supports a local precision contribution. A comparable feature effect under both routes weakens the claim that activation coding at this projection is required for the local effect. A large spectral interaction with no visible improvement supports numerical sensitivity without establishing a beneficial mechanism.

## Interpretation and deferred controls

| Hypothesis | Discriminating evidence |
| --- | --- |
| Early structural redirection | A matched-state edit changes early geometry and a correctly coupled continuation retains the change. |
| Later loss or corruption | The native clean sequence contains an earlier acceptable structure and loses it while the active sequence retains it. |
| Quantization effects | The qualified four-arm test changes the relevant visual outcome at equal noisy input, proposal and effective stored weights. |
| Generic trajectory sensitivity | Multiple magnitude-matched directions give differing visual outcomes without consistent direction-specific improvement. |
| Solver or Spectrum coupling | The effect tracks changed solver increments, actual versus forecast routes, or qualified solver continuation. |
| Composition or selection | The apparent gain shrinks in scale-registered crops or fails a blinded independent confirmation split. |

First localize the visual change. Then test direction specificity with the sign-reversed proposal, multiple matched controls, and the measured native versus continuous precision correction, using realized edit magnitude and output displacement as separate measurements. A production-qualified live continuation must preserve solver noise increments, solver memory and Spectrum state. Restarting from a frozen latent plus a new schedule cannot establish that coupling.

### Research context

| Supplied work | Relevance and limit |
| --- | --- |
| [DiverseFlow](https://arxiv.org/abs/2504.07894) | Learns more diverse flow sampling through an amortized training procedure. Its coverage objective motivates a future population benchmark. |
| [It’s Never Too Late](https://arxiv.org/abs/2601.00090) | Optimizes noise in groups of samples during inference, including late recovery. It motivates testing when a trajectory retains the capacity to change without attributing an anatomical defect to mode collapse. |
| [Initialization is Half the Battle](https://arxiv.org/abs/2606.02453) | Samples noise using a guidance-potential posterior. It motivates a distinct initialization study with equal compute and multiple seeds. |
| [Beyond Pairwise Repulsion](https://arxiv.org/abs/2610.22267) | Uses determinant maximization to account for collective diversity. It motivates group-level coverage metrics when comparing seeds. |
| [Don’t Settle at the Mode!](https://arxiv.org/abs/2606.27371) | Uses inter-sample feature repulsion for flow diversity. This motivates the existing coarse feature intervention. Within-image structure and quality require their own controls. |
| [Posterior Matching Distillation](https://arxiv.org/abs/2611.26629) | Matches a denoising posterior in student training and discusses expressive conditional outputs. The posterior identities do not identify why a particular Krea 2 output has an anatomical defect. |
