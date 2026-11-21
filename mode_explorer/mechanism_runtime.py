"""Scoped native Krea 2 measurement and matched experiment scheduling."""

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import replace
import math

import torch

from .mechanism import UNAVAILABLE, conditioning_terms, filter_field, paired_measurement, signed_terms
from .objectives import cell_size, feature_shift, seed_key, token_pool
from .response import clean_candidate, flow_clean, latent_cell_bounds, rms, select_response, velocity_for_clean
from .runtime import FRAME, Frame, Run, isolated_reference, shadow_forward, spatial_image, token_slice

_CAPTURE = ContextVar("modeexplorer_mechanistic_capture", default=None)
_GLOBAL = {"final_projection", "velocity", "clean"}


class CaptureLimit(ValueError):
    pass


class Workspace:
    def __init__(self, limit):
        self.limit = limit
        self.retained = self.peak = 0

    def stage(self, value):
        size = value.numel() * value.element_size()
        if self.retained + size > self.limit:
            raise CaptureLimit("mechanistic_capture_limit")
        result = value.detach().to("cpu", copy=True, memory_format=torch.contiguous_format)
        self.retained += size
        self.peak = max(self.peak, self.retained)
        return result

    def release(self, value):
        self.retained -= value.numel() * value.element_size()


class Capture:
    def __init__(self, cfg, grid, workspace, block, references=None, proposal=None, scale=0, fixed=None):
        self.cfg, self.grid, self.workspace, self.block = cfg, grid, workspace, block
        self.references, self.proposal, self.scale, self.fixed = references, proposal, scale, fixed
        self.fields, self.operators = {}, {}
        self.bounds = None
        self.edit = None
        self.wants_edit = references is not None or proposal is not None or fixed is not None

    def store(self, site, tensor, force=False):
        if site[1] not in self.cfg.site_names and not force:
            return
        if site in self.fields or self.bounds is None:
            raise ValueError("mechanistic_capture_geometry")
        lo, hi = self.bounds
        if tensor.ndim != 3 or tensor.shape[0] != 1 or hi > tensor.shape[1]:
            raise ValueError("mechanistic_capture_geometry")
        tokens = tensor[0, lo:hi]
        self.fields[site] = self.workspace.stage(tokens.T.unflatten(-1, self.grid))

    def attention(self, output, options, frame):
        block = options["block_index"]
        edited = block == self.block and self.wants_edit
        lo, hi = token_slice(options, output.shape[1], math.prod(self.grid))
        self.bounds = lo, hi
        measure = block in self.cfg.block_ids
        if measure or edited:
            self.store((block, "attention_pre"), output, force=edited)
        result = output
        if edited:
            if self.edit is not None:
                raise ValueError("mechanistic_duplicate_intervention")
            if self.proposal is None and self.fixed is None:
                settings = frame.run.settings
                current = token_pool(output[:, lo:hi], *self.grid, settings.feature_grid)
                joined = torch.cat((current, self.references))
                pairs = ~torch.eye(len(joined), device=output.device, dtype=torch.bool)
                shift, _, _ = feature_shift(
                    joined, pairs, settings.feature, settings.strength, settings.fidelity, False,
                    settings.similarity, temperature=settings.cosine_temperature,
                    bandwidth=settings.volume_bandwidth, median_bandwidth=settings.volume_median, live=1,
                )
                self.proposal = shift[:1]
            result = output.clone()
            height, width = self.grid
            view = result[:, lo:hi].unflatten(1, self.grid)
            if self.fixed is not None:
                before, after = self.fixed
                if before.shape != (output.shape[-1], *self.grid) or before.dtype != output.dtype:
                    raise ValueError("mechanistic_fixed_edit_geometry")
                for y in range(0, height, 2):
                    delta = (after[:, y:y + 2].float() - before[:, y:y + 2].float()).to(output.device)
                    strip = view[:, y:y + 2]
                    strip.copy_(strip.float() + delta.permute(1, 2, 0))
            else:
                sy, sx = cell_size(height, width, frame.run.settings.feature_grid)
                shift = self.proposal * self.scale
                for y in range(shift.shape[-2]):
                    delta = shift[:, :, y].to(output.dtype).repeat_interleave(sx, dim=-1)
                    view[:, y * sy:(y + 1) * sy].add_(delta[:, :, :width].transpose(1, 2)[:, None])
            self.edit = (block, "attention_pre"), (block, "attention_post")
        if measure or edited:
            self.store((block, "attention_post"), result, force=edited)
        return result

    @contextmanager
    def hooks(self, dit):
        handles = []
        token = _CAPTURE.set(self)

        def geometry(module, args, kwargs):
            if _CAPTURE.get() is self:
                self.bounds = token_slice(kwargs["transformer_options"], args[0].shape[1], math.prod(self.grid))

        def attach(module, site, pre=False):
            if site[1] not in self.cfg.site_names:
                return

            def record(module, args, output=None):
                if _CAPTURE.get() is self:
                    self.operators[str(site)] = f"{type(module).__module__}.{type(module).__qualname__}"
                    self.store(site, args[0] if pre else output)

            register = module.register_forward_pre_hook if pre else module.register_forward_hook
            handles.append(register(record))

        try:
            for i in self.cfg.block_ids:
                block = dit.blocks[i]
                handles.append(block.register_forward_pre_hook(geometry, with_kwargs=True))
                for name, module, pre in (
                    ("gate_logits", block.attn.gate, False),
                    ("projection_input", block.attn.wo, True), ("projection_output", block.attn.wo, False),
                    ("after_attention", block.postnorm, True), ("mlp_input", block.mlp, True),
                    ("mlp_output", block.mlp, False), ("after_mlp", block, False),
                ):
                    attach(module, (i, name), pre)
            attach(dit.last.linear, (None, "final_projection"))
            yield
        finally:
            for handle in reversed(handles):
                handle.remove()
            _CAPTURE.reset(token)


def _alternate(alt, base, latent, context, mask, refs, kwargs, dit):
    """Use native Krea conditioning conversion for a single global conditioning entry."""
    if len(alt) != 1 or not isinstance(alt[0], (list, tuple)) or len(alt[0]) != 2:
        raise ValueError("mechanistic_requires_single_global_alternate")
    raw, meta = alt[0]
    metadata = {"reference_latents", "reference_latents_method", "attention_mask", "pooled_output"}
    if not isinstance(meta, dict) or any(key not in metadata for key in meta):
        raise ValueError("mechanistic_unsupported_alternate_metadata")
    if not torch.is_tensor(raw) or raw.ndim != 3 or raw.shape[0] != 1 or raw.shape[-1] != context.shape[-1]:
        raise ValueError("mechanistic_alternate_embedding_geometry")
    extra = base.extra_conds(**meta, cross_attn=raw, device=latent.device, noise=latent)
    if set(extra) - {"c_crossattn", "ref_latents", "ref_latents_method", "attention_mask"}:
        raise ValueError("mechanistic_unsupported_native_conditioning")
    c1 = extra["c_crossattn"].process_cond(batch_size=1).cond.to(context)
    alternate_mask = extra["attention_mask"].process_cond(batch_size=1).cond if "attention_mask" in extra else None
    if alternate_mask is not None:
        alternate_mask = alternate_mask.to(device=latent.device, dtype=mask.dtype if mask is not None else alternate_mask.dtype)
    if (mask is None) != (alternate_mask is None):
        raise ValueError("mechanistic_alternate_mask_contract")
    alternate_refs = extra["ref_latents"].process_cond(batch_size=1).cond if "ref_latents" in extra else ()
    if len(alternate_refs) != len(refs or ()) or any(
        not torch.equal(new.to(old), old) for new, old in zip(alternate_refs, refs or ())
    ):
        raise ValueError("mechanistic_alternate_reference_mismatch")
    new_method = extra["ref_latents_method"].cond if "ref_latents_method" in extra else dit.default_ref_method
    if new_method != kwargs.get("ref_latents_method", dit.default_ref_method):
        raise ValueError("mechanistic_alternate_reference_method_mismatch")
    raw_mask = meta.get("attention_mask")
    if raw_mask is not None and not torch.is_tensor(raw_mask):
        raise ValueError("mechanistic_alternate_mask_contract")
    return c1, alternate_mask, raw_mask


def _edit_pair(capture):
    if capture.edit is None:
        raise ValueError("mechanistic_intervention_hook_not_observed")
    return tuple(capture.fields[key] for key in capture.edit)


def _edit_compare(left, right, sign, chunk):
    sums = [0., 0., 0.]
    for i in range(0, left[0].shape[0], chunk):
        sl = slice(i, i + chunk)
        a = left[1][sl].float() - left[0][sl].float()
        b = right[1][sl].float() - right[0][sl].float()
        for j, field in enumerate((a, b, a - sign * b)):
            sums[j] += float(field.square().sum(dtype=torch.float64))
    norm, other, residual = map(math.sqrt, sums)
    return {"left_l2": norm, "right_l2": other, "mismatch_l2": residual,
            "relative_mismatch": residual / max(norm, other) if max(norm, other) else None}


def _site_metrics(fields, cfg):
    comparisons = {}

    def measure(name, base, edit):
        comparisons[name] = paired_measurement(base, edit, cfg.chunk, cfg.low_cut, cfg.high_cut)

    d00, d01 = fields["D00"], fields["D01"]
    measure("positive", d00, d01)
    if "D0minus" in fields:
        minus = fields["D0minus"]
        measure("negative", d00, minus)
        odd, even = signed_terms(d00, d01, minus)
        zero = ((0., d00),)
        measure("odd", zero, odd)
        measure("even", zero, even)
        measure("positive_vs_reflected_negative", ((1., d00), (-1., minus)), ((1., d01), (-1., d00)))
    if "D11" in fields:
        d10, d11 = fields["D10"], fields["D11"]
        c0, cdelta = conditioning_terms(d00, d10, d01, d11)
        measure("conditioning_baseline", d00, d10)
        measure("conditioning_edited", d01, d11)
        measure("conditioning_interaction", c0, cdelta)
        measure("feature_across_conditions", ((1., d01), (-1., d00)), ((1., d11), (-1., d10)))
    return comparisons


def _amplitude(full_plus, full_minus, half_plus, half_minus, base):
    odd = (full_plus - full_minus) * .5
    even = (full_plus + full_minus) * .5 - base
    odd_half = (half_plus - half_minus) * .5
    even_half = (half_plus + half_minus) * .5 - base
    return {"status": "measured", "odd_half_over_full": rms(odd_half) / rms(odd) if rms(odd) else None,
            "even_half_over_full": rms(even_half) / rms(even) if rms(even) else None,
            "odd_scaled_relative_change": rms(2 * odd_half - odd) / rms(odd) if rms(odd) else None,
            "even_quadratic_relative_change": rms(4 * even_half - even) / rms(even) if rms(even) else None}


def research_response(executor, run, frame, latent, x, timesteps, context, mask, refs, options,
                      kwargs, native_velocity, point, spend, reference_sigma, initial_ordinal, ordinal, probe_seed):
    from .response_runtime import _tensor_hash, _tensor_value_hash

    cfg = run.response.config
    spec = cfg.mechanistic
    dit = executor.class_obj
    stage = point["mechanistic"] = {
        "schema": 1, "mode": spec.mode, "checkpoint_provenance": spec.checkpoint_provenance,
        "injection_block": cfg.feature_block, "blocks": list(spec.block_ids), "requested_sites": list(spec.site_names),
        "chunk_channels": spec.chunk, "frequency_cycles_per_grid_cell": [spec.low_cut, spec.high_cut],
        "capture_budget_bytes": spec.capture_mib * 1024**2, "cfg_scope": "isolated_positive_row",
        "trajectory": "frozen_export" if cfg.frozen_state else "live_feature" if run.feature_active else "native_or_component",
        "sites": {}, "arms": [], "component": spec.component, "band": spec.band,
        "cell": spec.cell, "filter_transition": spec.transition, "symmetry_limit": spec.symmetry_limit,
        "amplitude_check": spec.amplitude_check, "match_cross_dtype": spec.match_cross_dtype,
        "attention_mask_consumed_by_native_krea": False,
    }
    if reference_sigma is None:
        raise ValueError("feature_reference_start_not_observed")
    if frame.sigma > reference_sigma:
        raise ValueError("feature_reference_start_after_probe")
    row = slice(frame.offset, frame.offset + 1)

    def pick(value):
        return value[row] if value is not None and value.shape[0] == x.shape[0] else value

    context, mask, timesteps = context[row], pick(mask), timesteps[row]
    refs = [pick(ref) for ref in refs] if refs is not None else None
    options = {**options, "cond_or_uncond": [0]}
    if "uuids" in options:
        options["uuids"] = options["uuids"][row]
    if torch.is_tensor(options.get("sigmas")) and options["sigmas"].ndim:
        options["sigmas"] = pick(options["sigmas"])
    branches = {"C0": (context, mask)}
    metadata_masks = {"C0": run.response.positive_metadata_mask}
    if spec.conditioning:
        alt_context, alt_mask, metadata_masks["C1"] = _alternate(
            spec.alternate, run.response.base_model, latent, context, mask, refs, kwargs, dit,
        )
        branches["C1"] = alt_context, alt_mask
    stage["conditioning"] = {
        key: {"sha256": _tensor_hash(ctx), "mask_sha256": _tensor_hash(msk),
              "metadata_mask_sha256": _tensor_hash(metadata_masks[key]), "tokens": ctx.shape[1],
              "context_values_sha256": _tensor_value_hash(ctx), "mask_values_sha256": _tensor_value_hash(msk)}
        for key, (ctx, msk) in branches.items()
    }
    stage["kernel_contract"] = {
        "latent_device": str(latent.device), "latent_dtype": str(latent.dtype),
        "projection_operator": f"{type(dit.blocks[cfg.feature_block].attn.wo).__module__}.{type(dit.blocks[cfg.feature_block].attn.wo).__qualname__}",
        "torch": str(torch.__version__), "attention_override": point["backend"],
        "accumulation_dtype": "unobserved",
    }
    if latent.is_cuda:
        stage["kernel_contract"]["device_name"] = torch.cuda.get_device_name(latent.device)
    block_sites = [(b, s) for b in spec.block_ids for s in spec.site_names if s not in _GLOBAL]
    keys = block_sites + [(None, s) for s in spec.site_names if s in _GLOBAL]
    captured_blocks = set(spec.block_ids) | {cfg.feature_block}
    if any(block >= len(dit.blocks) for block in captured_blocks):
        raise ValueError("mechanistic_block_out_of_range")
    channels = dit.blocks[cfg.feature_block].attn.wo.in_features
    points = math.prod(frame.grid)
    core_arms = 2 + int(spec.signed) + 2 * int(spec.conditioning)
    rebuild = reference_sigma != frame.sigma
    recomputed = spec.mode.endswith("recomputed")
    ref_branches = ("C0", "C1") if recomputed else ("C0",)
    response = run.response
    reference_keys = {
        branch: (
            reference_sigma, initial_ordinal, probe_seed, str(latent.dtype), tuple(latent.shape),
            stage["conditioning"][branch]["sha256"], stage["conditioning"][branch]["mask_sha256"],
            tuple(point["reference_latent_sha256"]), point["reference_method"], point["backend"],
        ) for branch in ref_branches
    }
    reused_anchors = {branch for branch, key in reference_keys.items()
                      if rebuild and (cached := response.mechanistic_anchors.get(branch)) is not None
                      and cached[0] == key}
    reference_calls = cfg.feature_references * (len(ref_branches) + (len(ref_branches) - len(reused_anchors)) * int(rebuild))
    stage["first_estimate_reused_for"] = sorted(reused_anchors)
    repeat = cfg.calibration != "off"
    native_capture = any(site not in {"velocity", "clean"} | UNAVAILABLE for site in spec.site_names)
    can_reuse = not (native_capture or cfg.frozen_state or frame.attempted or x.shape[0] != 1)
    required = core_arms - int(can_reuse) + reference_calls + 2 * int(spec.amplitude_check) + int(repeat)
    stage["baseline_reused"] = can_reuse
    stage.update(auxiliary_calls_planned=required, feature_reference_calls_planned=reference_calls,
                 feature_reference_calls=0,
                 feature_reference_origin="reconstructed_first_estimate" if rebuild else "initial_shadow")
    if run.response.forwards + required > cfg.auxiliary_budget:
        raise ValueError("auxiliary_forward_budget")
    # Float32 is conservative for native capture on current supported kernels.
    # Reserve space for paired FFT reductions, clean fields, and two edited captures
    # per arm even if neither attention measurement was selected.
    site_channels = sum(
        dit.channels * dit.patch**2 if s == "final_projection" else channels
        for _, s in keys if s not in UNAVAILABLE and s not in {"velocity", "clean"}
    )
    captured_arms = core_arms - int(can_reuse)
    edited_arms = 1 + int(spec.signed) + int(spec.conditioning)
    extra_edit_sites = 2 - (sum(s in spec.site_names for s in ("attention_pre", "attention_post"))
                            if cfg.feature_block in spec.block_ids else 0)
    fft_bytes = max(points, math.prod(latent.shape[-2:])) * (128 * spec.chunk + 128)
    output_bytes = latent.numel() * 4 * (10 + (4 * core_arms) + 4 * int(spec.amplitude_check))
    main_site_channels = site_channels * captured_arms + channels * extra_edit_sites * edited_arms
    site_bytes = 4 * points * max(main_site_channels, 4 * channels * int(spec.amplitude_check))
    stage["capture_preflight_bytes"] = fft_bytes + output_bytes + site_bytes
    if stage["capture_preflight_bytes"] > stage["capture_budget_bytes"]:
        raise CaptureLimit("mechanistic_capture_preflight")
    workspace = Workspace(stage["capture_budget_bytes"] - fft_bytes - output_bytes)

    # GPU-side allowance covers the cloned attention output (including non-image
    # tokens), references and latent fields. Full-resolution captures live on CPU.
    max_text = max(ctx.shape[1] for ctx, _ in branches.values())
    ref_tokens = sum(math.prod(math.ceil(n / dit.patch) for n in ref.shape[-2:])
                     * (ref.shape[2] if ref.ndim == 5 else 1) for ref in refs or ())
    gpu_bytes = 4 * (channels * (points + max_text + ref_tokens + 4 * frame.grid[1])
                     + latent.numel() * (12 + core_arms * 2)
                     + channels * cfg.feature_grid**2 * cfg.feature_references * 3)
    # First-estimate anchors retain native FP32 clean and noise per reference.
    gpu_bytes += 8 * latent.numel() * cfg.feature_references * len(ref_branches)
    point["estimated_working_bytes"] = max(gpu_bytes, point["estimated_working_bytes"])
    if point["estimated_working_bytes"] > cfg.memory_mb * 1024**2:
        raise CaptureLimit("research_memory_budget")

    replay_settings = replace(run.settings, feature="feature_cosine", feature_block=cfg.feature_block,
                              feature_grid=cfg.feature_grid, strength=cfg.feature_strength, fidelity=.70, adaptive=False,
                              diagnostics=False, reference_continuation="first_estimate", reference_selection="all",
                              feature_update="bounded", sham=False, feature_direction="repel")
    replay_run = Run(replay_settings, probe_seed, run.start, run.noise_scale, 1, tuple(latent.shape),
                     True, spectrum=run.spectrum, sol=run.sol, stage=run.stage)

    def references(branch):
        ctx, msk = branches[branch]
        if branch in reused_anchors:
            shadows = dict(response.mechanistic_anchors[branch][1])
        else:
            shadows = {}
        shadow_run = replace(replay_run, shadows=shadows)
        ref_frame = Frame(frame.grid, 1, 0, "response", shadow_run, frame.sigma)
        initial_frame = Frame(frame.grid, 1, 0, "response", shadow_run, reference_sigma)
        initial_options = {**options, "sigmas": torch.full_like(options["sigmas"], reference_sigma)}
        pooled = []
        if rebuild and branch not in reused_anchors:
            response.mechanistic_anchors.pop(branch, None)
            for i in range(cfg.feature_references):
                spend(1)
                stage["feature_reference_calls"] += 1
                shadow_forward(executor, shadow_run, initial_frame, latent, torch.full_like(timesteps, reference_sigma),
                               ctx, msk, refs, initial_options, kwargs, i, sol_evaluations=initial_ordinal)
            response.mechanistic_anchors[branch] = reference_keys[branch], dict(shadows)
        for i in range(cfg.feature_references):
            spend(1)
            stage["feature_reference_calls"] += 1
            pooled.append(shadow_forward(executor, shadow_run, ref_frame, latent, timesteps, ctx, msk, refs,
                                         options, kwargs, i, sol_evaluations=ordinal))
        if not rebuild:
            response.mechanistic_anchors[branch] = reference_keys[branch], dict(shadows)
        stage.setdefault("reference_slots", {})[branch] = ref_frame.reference_slots
        return torch.cat(pooled)

    captures, velocities, clean = {}, {}, {}

    def predict(name, branch="C0", capture=None):
        ctx, msk = branches[branch]
        arm_frame = Frame(frame.grid, 1, 0, "response_capture", replay_run, frame.sigma, mechanistic_capture=capture)
        opts = {**options, FRAME: arm_frame} if capture is not None else {**options}
        scope = capture.hooks(dit) if capture is not None else nullcontext()
        spend(1)
        with torch.no_grad(), scope, isolated_reference(run, sol_evaluations=ordinal):
            value = executor(latent, timesteps, ctx, msk, refs, opts, **kwargs)
        if not bool(torch.isfinite(value).all()):
            raise ValueError("mechanistic_nonfinite_velocity")
        if capture is not None:
            captures[name] = capture
        velocities[name] = value
        clean[name] = flow_clean(latent, value, frame.sigma)
        stage["arms"].append(name)

    def capture(**edit):
        return Capture(spec, frame.grid, workspace, cfg.feature_block, **edit)

    if can_reuse:
        captures["D00"] = capture()
        velocities["D00"] = native_velocity[row]
        clean["D00"] = flow_clean(latent, velocities["D00"], frame.sigma)
        stage["arms"].append("D00")
    else:
        predict("D00", capture=capture())
    if repeat:
        predict("D00_repeat")
        stage["repeat_clean_rms"] = rms(clean.pop("D00_repeat") - clean["D00"])
        velocities.pop("D00_repeat")
    refs0 = references("C0")
    plus = capture(references=refs0, scale=1)
    predict("D01", capture=plus)
    plus_edit = _edit_pair(plus)
    stage["proposal"] = {"C0_sha256": _tensor_hash(plus.proposal), "C0_compact_rms": rms(plus.proposal)}
    stage["realized_positive_l2"] = _edit_compare(plus_edit, plus_edit, 1, spec.chunk)["left_l2"]
    stage["realized_positive_rms"] = stage["realized_positive_l2"] / math.sqrt(plus_edit[0].numel())
    if spec.signed:
        minus = capture(proposal=plus.proposal, scale=-1)
        predict("D0minus", capture=minus)
        symmetry = _edit_compare(plus_edit, _edit_pair(minus), -1, spec.chunk)
        symmetry["status"] = "matched" if (symmetry["relative_mismatch"] is not None
                                            and symmetry["relative_mismatch"] <= spec.symmetry_limit) else "unqualified"
        stage["signed_realized_edit"] = symmetry
    if spec.conditioning:
        predict("D10", "C1", capture())
        alt = capture(references=references("C1"), scale=1) if recomputed else capture(fixed=plus_edit)
        predict("D11", "C1", alt)
        match = _edit_compare(plus_edit, _edit_pair(alt), 1, spec.chunk)
        match["status"] = ("independent_proposals" if recomputed else "matched"
                           if match["relative_mismatch"] is not None and match["relative_mismatch"] <= spec.symmetry_limit
                           else "unqualified")
        stage["conditioning_realized_edit"] = match
        alt.fixed = None
        if recomputed:
            stage["proposal"]["C1_sha256"] = _tensor_hash(alt.proposal)
            stage["proposal"]["C1_compact_rms"] = rms(alt.proposal)

    del plus_edit
    stage["sampler_velocity_baseline_rms_difference"] = rms(velocities["D00"].float() - native_velocity[row].float())
    stage["operators"] = captures["D00"].operators
    main_arms = tuple(captures)
    for key in keys:
        block, site = key
        name = site if block is None else f"b{block}.{site}"
        info = {"site": site, "block": block, "coordinate": "native_diffusion_latent" if site in {"velocity", "clean"}
                else "generated_image_patch_grid"}
        if site in UNAVAILABLE:
            info.update(status="unsupported", reason="no_exact_native_module_boundary")
        else:
            if site == "clean" or site == "velocity":
                values = clean if site == "clean" else velocities
                fields = {arm: spatial_image(values[arm])[0].detach().cpu() for arm in main_arms}
            else:
                fields = {arm: captures[arm].fields.get(key) for arm in main_arms}
            if any(value is None for value in fields.values()):
                info.update(status="incomplete", missing_arms=[arm for arm, value in fields.items() if value is None])
            else:
                comparisons = _site_metrics(fields, spec)
                info.update(status="completed", comparisons=comparisons)
                if spec.signed:
                    odd_power, even_power = (comparisons[part]["displacement_l2"]**2 for part in ("odd", "even"))
                    info["even_energy_fraction"] = even_power / (odd_power + even_power) if odd_power + even_power else None
                    fraction = {}
                    for band, stats in comparisons["even"]["bands"].items():
                        energy = stats["displacement"]
                        total = energy + comparisons["odd"]["bands"][band]["displacement"]
                        fraction[band] = energy / total if total else None
                    info["even_energy_fraction_by_band"] = fraction
            del fields
        stage["sites"][name] = info
        for arm in main_arms:
            if key in captures[arm].fields:
                workspace.release(captures[arm].fields.pop(key))
    for arm in main_arms:
        for value in captures[arm].fields.values():
            workspace.release(value)
        captures[arm].fields.clear()
    # Release even the loop variable before staging the calibration pair.
    value = None
    if spec.amplitude_check:
        light = replace(spec, sites="")
        for name, sign in (("D0half_plus", .5), ("D0half_minus", -.5)):
            predict(name, capture=Capture(light, frame.grid, workspace, cfg.feature_block,
                                          proposal=plus.proposal, scale=sign))
        half_symmetry = _edit_compare(_edit_pair(captures["D0half_plus"]), _edit_pair(captures["D0half_minus"]), -1, spec.chunk)
        stage["half_realized_edit"] = half_symmetry
        stage["amplitude"] = _amplitude(clean["D01"], clean["D0minus"],
                                        clean["D0half_plus"], clean["D0half_minus"], clean["D00"])
        for name in ("D0half_plus", "D0half_minus"):
            captures.pop(name)
            velocities.pop(name)
            clean.pop(name)
    stage["captured_bytes"] = workspace.peak
    point["status"] = "completed" if all(s["status"] != "incomplete" for s in stage["sites"].values()) else "incomplete"
    point.update(applied=False, selected_candidate="baseline", candidates=[{"arm": "baseline", "selected": True}])
    base = clean["D00"]
    delta = clean["D01"] - base
    components = {"complete": delta}
    if spec.signed:
        negative_delta = clean["D0minus"] - base
        components["odd"] = (delta - negative_delta) * .5
        components["even"] = (delta + negative_delta) * .5
    if spec.conditioning:
        components["interaction"] = (clean["D11"] - clean["D10"]) - delta

    corrected = None
    candidate = None
    if spec.component != "observe":
        component = components["complete" if spec.component == "control" else spec.component]
        component = filter_field(component, spec.band, spec.low_cut, spec.high_cut, spec.transition)
        bounds = latent_cell_bounds(component.shape, cfg.feature_grid, spec.cell) if spec.cell >= 0 else None
        if bounds is not None:
            y0, y1, x0, x1 = bounds
            def localize(value):
                target = torch.zeros_like(value)
                target[..., y0:y1, x0:x1] = value[..., y0:y1, x0:x1]
                return target
            component = localize(component)
        if spec.component == "control":
            rng = torch.Generator(device="cpu").manual_seed(seed_key(probe_seed, f"mechanism-control-{cfg.trial}"))
            noise = torch.randn(component.shape, generator=rng, device="cpu").to(component)
            noise = filter_field(noise, spec.band, spec.low_cut, spec.high_cut, spec.transition)
            if bounds is not None:
                noise = localize(noise)
            component = noise * (rms(component) / max(1e-12, rms(noise)))
        info = {"arm": spec.component, "isolated_component_rms": rms(component)}
        if spec.component in ("odd", "even") and stage["signed_realized_edit"]["status"] != "matched":
            info["reason"] = "unqualified_signed_intervention"
        elif spec.component == "interaction" and stage["conditioning_realized_edit"]["status"] == "unqualified":
            info["reason"] = "unqualified_fixed_intervention"
        elif cfg.steers and not torch.equal(native_velocity[row], velocities["D00"]):
            info["reason"] = "live_baseline_mismatch"
        proposal, info = clean_candidate(base, component, cfg, info)
        if frame.sigma >= 1e-5:
            corrected = velocity_for_clean(latent, proposal, frame.sigma).to(native_velocity.dtype)
            candidate = flow_clean(latent, corrected, frame.sigma)
        else:
            candidate = proposal
            info["reason"] = "flow_conversion_endpoint"
        selected, score = select_response([(candidate, info)], base, cfg, stage.get("repeat_clean_rms", 0.),
                                          point, run.response.process_latent_out)
        point["candidates"].append(info)
        point["candidates"][0]["external_scores"] = score
        if cfg.steers and selected is not None and point["status"] == "completed":
            actual = rms(frame.sigma * (corrected.float() - native_velocity[row].float()))
            point["applied_clean_displacement_rms"] = actual
            if 1e-8 < actual <= cfg.clean_limit:
                point.update(applied=True, selected_candidate=spec.component)
                point["candidates"][0]["selected"] = False
            else:
                info["reason"] = "native_output_rounding_or_trust_bound"
        components["selected_component"] = component
    tensors = {}
    if cfg.export:
        tensors = {"frozen_native_input": latent, "baseline_clean": base}
        tensors.update({f"{arm}_clean": clean[arm] for arm in main_arms if arm != "D00"})
        tensors.update({f"{name}_native_delta": value for name, value in components.items()})
        if candidate is not None:
            tensors["candidate_clean"] = candidate
        if point["applied"]:
            tensors["applied_clean"] = candidate
    if point["applied"]:
        result = native_velocity.clone()
        result[row] = corrected
        return result, tensors
    return native_velocity, tensors
