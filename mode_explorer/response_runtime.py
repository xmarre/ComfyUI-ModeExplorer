"""Run response experiments inside the existing ModeExplorer diffusion owner."""

from contextlib import contextmanager
from dataclasses import dataclass, field, replace
import hashlib
import math
import re
import shutil
from pathlib import Path
import tempfile

import torch

from .objectives import cell_size, seed_key
from .response import (
    MAX_EXPORT_BYTES, differences, direction_field, effective_tilt_scale, flow_clean,
    velocity_for_clean,
    orthonormalize, output_projection, projected_operator, rms, select_response, shifted_pair,
)


@dataclass
class ResponseRun:
    config: object
    process_latent_out: object
    eligibility: str | None = None
    forwards: int = 0
    evaluation: int = -1
    states: int = 0
    exports: int = 0
    points: list = field(default_factory=list)
    last_sigma: float | None = None
    cfg_scale: float | None = None


class ResponseBudget(RuntimeError):
    pass


def feature_delta(shift, current, output, view, sy, sx, frame):
    """Edit only the selected generated-token slice of a replay forward."""
    cfg, stats = frame.replay
    shift = shift[:frame.batch]
    if cfg.feature_direction in ("tangent", "spatial_random"):
        generator = torch.Generator(device="cpu").manual_seed(seed_key(frame.run.seed, f"feature-response-{cfg.trial}"))
        direction = torch.randn(shift.shape, generator=generator, device="cpu").to(shift)
        if cfg.feature_direction == "tangent":
            direction -= direction.mean(dim=(-1, -2), keepdim=True)
            live = (current[:frame.batch] - current[:frame.batch].mean(dim=(-1, -2), keepdim=True)).flatten(1)
            q = live / torch.linalg.vector_norm(live, dim=1, keepdim=True).clamp_min(1e-12)
            flat = direction.flatten(1)
            flat -= (flat * q).sum(dim=1, keepdim=True) * q
            shift = direction * (torch.linalg.vector_norm(shift) / torch.linalg.vector_norm(direction).clamp_min(1e-12))
        else:
            shift = direction * (torch.linalg.vector_norm(shift, dim=1, keepdim=True)
                                 / torch.linalg.vector_norm(direction, dim=1, keepdim=True).clamp_min(1e-12))

    height, width = view.shape[1:3]
    attn = frame.replay_attention
    heads, head_width = attn.heads, attn.headdim
    if output.ndim != 3 or output.shape[-1] != heads * head_width:
        raise ValueError("unverified_attention_output_layout")
    if cfg.head >= heads:
        raise ValueError("head_out_of_range")
    cells = shift.shape[-2] * shift.shape[-1]
    if cfg.cell >= cells:
        raise ValueError("cell_out_of_range")

    # Measure proposal energy on the actual rectangular token grid.
    from .objectives import cell_areas

    areas = cell_areas(height, width, *shift.shape[-2:], cfg.feature_grid, output.device)
    selector = torch.ones_like(shift, dtype=torch.bool)
    if cfg.head >= 0:
        selector[:, :cfg.head * head_width] = False
        selector[:, (cfg.head + 1) * head_width:] = False
    if cfg.cell >= 0:
        cell_mask = torch.arange(cells, device=output.device).reshape_as(areas) == cfg.cell
        selector &= cell_mask
    full_energy = (shift.square() * areas).sum()
    if cfg.localization.startswith("only"):
        shift = shift.masked_fill(~selector, 0)
    elif cfg.localization.startswith("remove"):
        shift = shift.masked_fill(selector, 0)
    component_energy = (shift.square() * areas).sum()
    if cfg.localization.endswith("matched") and component_energy > 0:
        shift = shift * (full_energy / component_energy).sqrt()
    stats.update(
        output_heads=heads, head_width=head_width, token_grid=[height, width],
        cell_grid=list(areas.shape), head=cfg.head, cell=cfg.cell, localization=cfg.localization,
        original_proposal_l2=float(full_energy.sqrt()),
        applied_proposal_l2=float((shift.square() * areas).sum().sqrt()),
    )
    energy = torch.zeros((), device=output.device)
    for iy in range(shift.shape[-2]):
        delta = shift[:, :, iy].to(output.dtype).repeat_interleave(sx, dim=-1)
        row = view[:, iy * sy:(iy + 1) * sy]
        for strip in row.split(2, dim=1):
            before = strip.to(dtype=torch.float32, copy=True)
            strip.add_(delta[:, :, :width].transpose(1, 2)[:, None])
            before.sub_(strip).square_()
            energy += before.sum()
    stats["realized_edit_l2"] = float(energy.sqrt())
    stats["realized_edit_rms"] = float((energy / view.numel()).sqrt())


def spectrum_all_actual(spectrum):
    owner = spectrum[0]
    from sys import modules

    module = modules[type(owner).__module__]
    flag = getattr(module, "FORCE_NATIVE_ALL_SAMPLERS", None)
    if flag is None or getattr(type(owner).sampler, "__globals__", {}).get("FORCE_NATIVE_ALL_SAMPLERS") is not flag:
        return None

    @contextmanager
    def force():
        token = flag.set(True)
        try:
            yield
        finally:
            flag.reset(token)

    return force()


def _tensor_hash(tensor):
    if tensor is None:
        return None
    data = tensor.detach().contiguous().view(torch.uint8).cpu().numpy()
    return hashlib.sha256(data).hexdigest()


def read_export(export_id):
    from json import loads
    import folder_paths

    if not re.fullmatch(r"response-[a-z0-9_]{8}", export_id):
        raise ValueError("invalid_response_export_id")
    root = (Path(folder_paths.get_output_directory()) / "modeexplorer_response").resolve()
    folder = (root / export_id).resolve()
    if folder.parent != root:
        raise ValueError("invalid_response_export_path")
    manifest = (folder / "manifest.json").resolve()
    data = (folder / "predictions.safetensors").resolve()
    if manifest.parent != folder or data.parent != folder:
        raise ValueError("invalid_response_export_files")
    if manifest.stat().st_size > 512 * 1024 or data.stat().st_size > MAX_EXPORT_BYTES + 65536:
        raise ValueError("response_export_size_limit")
    info = loads(manifest.read_text())
    if not isinstance(info, dict) or info.get("schema") != 1:
        raise ValueError("unknown_response_export_schema")
    return info, data


def _export(response, tensors, point):
    if not response.config.export:
        return
    size = sum(t.numel() * t.element_size() for t in tensors.values())
    if size + response.exports > MAX_EXPORT_BYTES:
        point["export_status"] = "export_budget"
        return
    run_dir = None
    try:
        import folder_paths
        from safetensors.torch import save_file
        import json

        directory = Path(folder_paths.get_output_directory()) / "modeexplorer_response"
        directory.mkdir(parents=True, exist_ok=True)
        payload = {}
        for name, tensor in tensors.items():
            output = name.endswith("_clean")
            value = response.process_latent_out(tensor) if output else tensor
            payload[name] = value.detach().to(device="cpu", dtype=torch.float32, copy=True).contiguous()
        size = sum(t.numel() * t.element_size() for t in payload.values())
        if size + response.exports > MAX_EXPORT_BYTES:
            point["export_status"] = "export_budget"
            return
        # A private unique directory prevents accidental overwrite across renders.
        run_dir = Path(tempfile.mkdtemp(prefix="response-", dir=directory))
        save_file(payload, run_dir / "predictions.safetensors",
                  metadata={"modeexplorer_response": "1", "clean_format": "comfy_vae_latent"})
        (run_dir / "manifest.json").write_text(json.dumps(point, allow_nan=False, indent=2) + "\n")
        point["export_status"] = "completed"
        point["export_id"] = run_dir.name
        response.exports += size
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        if run_dir is not None:
            shutil.rmtree(run_dir, ignore_errors=True)
        point["export_status"] = f"failed_{type(exc).__name__}"


def run_response(executor, run, frame, arguments, kwargs, velocity, actual):
    from comfy.model_management import throw_exception_if_processing_interrupted
    from .runtime import FRAME, Frame, Run, isolated_reference, shadow_forward

    response = run.response
    if response is None or not actual or response.eligibility is not None:
        return velocity
    cfg = response.config
    response.evaluation += 1
    index = response.evaluation
    sigma = frame.sigma
    same_sigma = response.last_sigma == sigma
    response.last_sigma = sigma
    if (response.states >= cfg.states or (same_sigma and cfg.evaluation < 0)
            or (cfg.evaluation >= 0 and cfg.evaluation != index)
            or not cfg.min_sigma <= sigma <= cfg.max_sigma):
        return velocity
    response.states += 1
    point = {
        "schema": 1, "status": "skipped", "evaluation": index, "sigma": sigma,
        "coordinate": "native_krea2_const_diffusion_input", "stage": run.stage,
        "mode": cfg.mode, "basis": cfg.basis, "trial": cfg.trial, "seed": run.seed,
        "probe_seed": run.seed, "clean_trust_rms": cfg.clean_limit, "candidate_gain": cfg.candidate_gain,
        "response_direction": cfg.response_direction, "calibration": cfg.calibration,
        "requested_directions": cfg.directions, "probe_grid": cfg.probe_grid, "displacement_norm": cfg.norm,
        "checkpoint_label": cfg.checkpoint_label,
        "noise_scale": run.noise_scale, "effective_tilt_per_unit_displacement": effective_tilt_scale(sigma, run.noise_scale),
        "selection": cfg.selection, "cfg_scale": response.cfg_scale,
        "feature_settings": {
            "block": cfg.feature_block, "grid": cfg.feature_grid, "strength": cfg.feature_strength,
            "reference_source": run.settings.reference_source if run.feature_active else "bank",
        },
    }
    response.points.append(point)
    x, timesteps, context, mask, refs, options = arguments
    row = slice(frame.offset, frame.offset + 1)
    latent = x[row]
    wants_latent = cfg.mode in ("latent_probe", "joint_probe", "latent_steer") or cfg.selection == "response_alignment"
    wants_feature = cfg.mode in ("feature_replay", "joint_probe", "feature_steer")
    before = response.forwards

    # Account for retained latent-size fields plus temporary packed latents and
    # the one extra attention-output clone used by a feature replay.
    dim = executor.class_obj.blocks[cfg.feature_block].attn.wo.in_features if wants_feature else 0
    token_count = math.prod(frame.grid) + context.shape[1]
    token_count += sum(math.prod(math.ceil(n / executor.class_obj.patch) for n in ref.shape[-2:]) for ref in refs or ())
    captures = len(executor.class_obj.blocks) * cfg.feature_grid**2 * dim * 2 if cfg.downstream else 0
    size = 4 * (latent.numel() * (18 + 4 * cfg.directions) + 6 * x.numel()
                + token_count * x.shape[0] * dim + 8 * frame.grid[1] * dim + captures)
    point["memory_scope"] = "research_tensor_workspace_excluding_native_forward"
    point["estimated_working_bytes"] = size
    if size > cfg.memory_mb * 1024**2:
        point["reason"] = "research_memory_budget"
        return velocity
    if cfg.steers and (sigma < 1e-5 or cfg.scorer is None and cfg.selection == "external"):
        point["reason"] = "score_unavailable" if sigma >= 1e-5 else "flow_conversion_endpoint"
        return velocity

    clean_options = {k: v for k, v in options.items() if k != FRAME}
    override = options.get("optimized_attention_override")
    backend = getattr(override, "__qualname__", type(override).__qualname__) if override is not None else "native_selector"
    point.update(input_dtype=str(x.dtype), velocity_dtype=str(velocity.dtype),
                 timestep=float(timesteps.flatten()[frame.offset]),
                 backend=backend,
                 conditioning_sha256=_tensor_hash(context[row]),
                 attention_mask_sha256=_tensor_hash(mask),
                 reference_latent_sha256=[_tensor_hash(ref) for ref in refs or ()],
                 native_input_sha256=_tensor_hash(latent), branches=list(options.get("cond_or_uncond", ())))
    if cfg.scorer is not None and cfg.selection == "external":
        point["scorer"] = getattr(cfg.scorer, "__qualname__", type(cfg.scorer).__qualname__)

    def spend(count):
        throw_exception_if_processing_interrupted()
        if response.forwards + count > cfg.auxiliary_budget:
            raise ResponseBudget
        response.forwards += count

    def predict(new_input, replay_frame=None):
        spend(1)
        packed = x
        if cfg.frozen_state or new_input is not latent:
            packed = x.clone()
            packed[row] = new_input
        override = clean_options if replay_frame is None else {**clean_options, FRAME: replay_frame}
        with torch.no_grad(), isolated_reference(run):
            value = executor(packed, timesteps, context, mask, refs, override, **kwargs)
        return value[row]

    probe_seed = run.seed
    try:
        if cfg.frozen_state:
            from safetensors import safe_open

            try:
                saved, data = read_export(cfg.frozen_state)
                matched = ("sigma", "timestep", "conditioning_sha256", "attention_mask_sha256",
                           "reference_latent_sha256", "branches", "noise_scale", "cfg_scale")
                if any(saved.get(key) != point.get(key) for key in matched):
                    raise ValueError("frozen_sigma_or_conditioning_mismatch")
                with safe_open(data, framework="pt", device="cpu") as package:
                    saved_latent = package.get_tensor("frozen_native_input")
                if saved_latent.shape != latent.shape:
                    raise ValueError("frozen_shape_mismatch")
                latent = saved_latent.to(device=latent.device, dtype=latent.dtype)
                if not torch.equal(latent.float().cpu(), saved_latent):
                    raise ValueError("frozen_input_rounding_mismatch")
                probe_seed = saved.get("probe_seed", saved.get("seed"))
                if type(probe_seed) is not int:
                    raise ValueError("frozen_probe_seed_missing")
                point["probe_seed"] = probe_seed
                point["frozen_export_id"] = cfg.frozen_state
                point["native_input_sha256"] = _tensor_hash(latent)
            except (OSError, ValueError, KeyError, RuntimeError) as exc:
                point.update(status="rejected", reason=str(exc) if isinstance(exc, ValueError) else "frozen_input_unavailable")
                return velocity

        baseline_taps = {} if wants_feature and cfg.downstream else None
        baseline_frame = Frame(frame.grid, 1, frame.offset, "response_capture", run, sigma, taps=baseline_taps,
                               tap_from=cfg.feature_block, tap_grid=cfg.feature_grid)
        needs_baseline = frame.attempted or bool(cfg.frozen_state) or baseline_taps is not None
        base_velocity = predict(latent, baseline_frame if baseline_taps is not None else None) if needs_baseline else velocity[row]
        base_clean = flow_clean(latent, base_velocity, sigma)
        if not bool(torch.isfinite(base_clean).all()):
            point.update(status="rejected", reason="nonfinite_baseline")
            return velocity
        noise_floor = 0.0
        if cfg.calibration != "off":
            repeat = flow_clean(latent, predict(latent), sigma)
            noise_floor = rms(repeat - base_clean)
            if not math.isfinite(noise_floor):
                point.update(status="rejected", reason="nonfinite_repeat")
                return velocity
        point.update(baseline_reused=not needs_baseline, repeat_rms=noise_floor if cfg.calibration != "off" else None,
                     differences=[], feature=[], candidates=[])
        tensors = {"frozen_native_input": latent.float(), "baseline_clean": base_clean} if cfg.export else {}
        basis, responses = [], []
        coefficients = None
        if wants_latent:
            proposals = orthonormalize(
                direction_field(latent, cfg.basis, cfg.probe_grid, probe_seed, cfg.trial, i)
                for i in range(cfg.directions)
            )
            for i, unit in enumerate(proposals):
                if response.forwards + 2 > cfg.auxiliary_budget:
                    point["differences"].append({"index": i, "status": "skipped", "reason": "auxiliary_forward_budget"})
                    break
                pair, metrics, reason = shifted_pair(latent, unit, cfg.displacement, cfg.norm, 0.25)
                metrics.update(index=i, status="rejected" if reason else "completed", reason=reason)
                point["differences"].append(metrics)
                if pair is None:
                    continue
                p, n, a, b = pair
                plus, minus = (flow_clean(p, predict(p), sigma), flow_clean(n, predict(n), sigma))
                derivative, curvature = differences(plus, minus, base_clean, a, b)
                if not bool(torch.isfinite(derivative).all() & torch.isfinite(curvature).all()):
                    metrics.update(status="rejected", reason="nonfinite_response")
                    continue
                signal = rms(plus - minus)
                metrics.update(first_order_rms=rms(derivative), curvature_rms=rms(curvature),
                               odd_signal_rms=signal * 0.5, even_signal_rms=rms((plus - base_clean) + (minus - base_clean)) * 0.5,
                               cancellation_or_repeat_limited=signal <= max(4 * noise_floor, 1e-8))
                if cfg.calibration != "off" and not basis and response.forwards + 2 <= cfg.auxiliary_budget:
                    half, half_metrics, half_reason = shifted_pair(latent, unit, cfg.displacement * 0.5, cfg.norm, 0.25)
                    metrics["half_calibration"] = {**half_metrics, "reason": half_reason}
                    if half is not None:
                        hp, hn, ha, hb = half
                        other, _ = differences(flow_clean(hp, predict(hp), sigma),
                                               flow_clean(hn, predict(hn), sigma), base_clean, ha, hb)
                        stability = rms(other - derivative) / max(rms(derivative), 1e-15)
                        metrics["half_calibration"]["relative_first_order_change"] = stability
                if cfg.steers and metrics["cancellation_or_repeat_limited"]:
                    metrics.update(status="rejected", reason="response_below_repeat_floor")
                    continue
                basis.append(unit)
                responses.append(derivative.flatten())
            if basis:
                op, right, eig = projected_operator(basis, responses)
                point["projected_response"] = op
                coefficients = right if cfg.response_direction == "singular" else eig

        candidates = []
        if cfg.mode == "latent_steer" and coefficients is not None:
            unit = sum((float(coef) * direction for coef, direction in zip(coefficients, basis)),
                       torch.zeros_like(basis[0]))
            # Fix the arbitrary SVD/eigenvector sign using the largest coefficient.
            if coefficients[coefficients.abs().argmax()] < 0:
                unit = -unit
            pair, metrics, reason = shifted_pair(latent, unit, cfg.displacement, cfg.norm, 0.25)
            point["candidate_latent_displacement"] = metrics
            if reason is not None:
                point["candidates"].append({"reason": reason})
            elif response.forwards + 2 <= cfg.auxiliary_budget:
                for label, candidate in zip(("positive", "negative"), pair[:2]):
                    clean = flow_clean(candidate, predict(candidate), sigma)
                    candidates.append((base_clean + cfg.candidate_gain * (clean - base_clean), {"arm": label}))
            else:
                point["candidates"].append({"reason": "auxiliary_forward_budget"})

        if wants_feature:
            directions = (cfg.feature_direction,)
            if cfg.mode == "feature_steer" and cfg.feature_direction in ("repel", "reverse"):
                directions = ("repel", "reverse")
            replay_cfg = replace(
                run.settings, feature="feature_cosine", feature_block=cfg.feature_block,
                feature_grid=cfg.feature_grid, strength=cfg.feature_strength, fidelity=0.70,
                adaptive=False, diagnostics=False, reference_continuation="first_estimate",
                reference_selection="all", feature_update="bounded", sham=False,
                feature_direction="repel", direction_trial=cfg.trial % 32,
            )
            replay_run = Run(replay_cfg, probe_seed, run.start, run.noise_scale, 1, tuple(latent.shape),
                             True, spectrum=run.spectrum, sol=run.sol, stage=run.stage)
            replay_frame = Frame(frame.grid, 1, frame.offset, "response", replay_run, sigma)
            if response.forwards + 1 + len(directions) <= cfg.auxiliary_budget:
                spend(1)
                with torch.no_grad():
                    replay_frame.reference = shadow_forward(executor, replay_run, replay_frame, x, timesteps,
                                                            context, mask, refs, clean_options, kwargs, 0)
                point["feature_reference_slots"] = replay_frame.reference_slots
                for direction in directions:
                    info = {"arm": direction, "block": cfg.feature_block, "grid": cfg.feature_grid}
                    replay_frame.replay = (replace(cfg, feature_direction=direction), info)
                    replay_frame.replay_attention = executor.class_obj.blocks[cfg.feature_block].attn
                    replay_frame.seen = False
                    replay_frame.taps = {} if baseline_taps is not None else None
                    replay_frame.tap_from, replay_frame.tap_grid = cfg.feature_block, cfg.feature_grid
                    try:
                        v = predict(latent, replay_frame)
                    except ValueError as exc:
                        info.update(status="rejected", reason=str(exc))
                        point["feature"].append(info)
                        continue
                    if not replay_frame.seen:
                        info.update(status="rejected", reason="attention_hook_not_observed")
                    else:
                        dv = v.float() - base_velocity.float()
                        clean = flow_clean(latent, v, sigma)
                        delta = clean - base_clean
                        info.update(status="completed", velocity_rms=rms(dv), clean_rms=rms(delta))
                        edit = info.get("realized_edit_l2", 0.0)
                        info["clean_per_edit_l2"] = float(torch.linalg.vector_norm(delta)) / edit if edit else None
                        if baseline_taps is not None:
                            info["later_block_pooled_attention_rms"] = {
                                str(block): rms(value - baseline_taps[block])
                                for block, value in replay_frame.taps.items() if block in baseline_taps
                            }
                        if responses:
                            info.update(output_projection(delta, responses))
                        if cfg.mode == "feature_steer" and direction != "sham":
                            candidates.append((base_clean + cfg.candidate_gain * delta, {
                                "arm": direction, "response_alignment": info.get("explained_energy_fraction"),
                            }))
                        if cfg.export and bool(torch.isfinite(clean).all()):
                            tensors[f"feature_{direction}_clean"] = clean
                    point["feature"].append(info)
            else:
                point["feature"].append({"status": "skipped", "reason": "auxiliary_forward_budget"})
        realized = {}
        if cfg.steers:
            for index, (clean, info) in enumerate(candidates):
                corrected = velocity_for_clean(latent, clean, sigma).to(velocity.dtype)
                realized[info["arm"]] = corrected
                candidates[index] = (flow_clean(latent, corrected, sigma), info)
        selected, score = select_response(candidates, base_clean, cfg, noise_floor, point)
        point["baseline_external_score"] = score
        point["candidates"].extend(info for _, info in candidates)
        point["status"] = "completed" if basis or any(a.get("status") == "completed" for a in point["feature"]) else "rejected"
        if not cfg.steers or selected is None:
            point["applied"] = False
        else:
            corrected = realized[selected[1]["arm"]]
            actual_change = rms(sigma * (corrected.float() - base_velocity.float()))
            point["applied_clean_displacement_rms"] = actual_change
            point["applied"] = 1e-8 < actual_change <= cfg.clean_limit
            if not point["applied"]:
                selected[1]["reason"] = "native_output_rounding_or_trust_bound"
        if cfg.export:
            tensors.update({f"candidate_{info['arm']}_clean": clean for clean, info in candidates
                            if bool(torch.isfinite(clean).all())})
            if point["applied"]:
                tensors["applied_clean"] = flow_clean(latent, corrected, sigma)
            from .runtime import finite_json

            export_point = finite_json(point)
            export_point["auxiliary_dit_calls"] = response.forwards - before
            _export(response, tensors, export_point)
            for key in ("export_status", "export_id"):
                if key in export_point:
                    point[key] = export_point[key]
        if point["applied"]:
            result = velocity.clone()
            result[row] = corrected
            return result
    except ResponseBudget:
        point.update(status="skipped", reason="auxiliary_forward_budget")
    finally:
        point["auxiliary_dit_calls"] = response.forwards - before
    return velocity
