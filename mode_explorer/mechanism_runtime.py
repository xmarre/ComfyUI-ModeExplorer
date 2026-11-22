"""Scoped native Krea 2 measurement and matched experiment scheduling."""

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import replace
import math
from time import perf_counter

import torch

from .mechanism import UNAVAILABLE, filter_field, measure_comparisons
from .projection import ProjectionCapture, ProjectionUnavailable, bitwise_equal, geometry as projection_geometry, study_projection
from .objectives import cell_size, feature_shift, seed_key, token_pool
from .response import clean_candidate, flow_clean, latent_cell_bounds, rms, select_response, velocity_for_clean
from .runtime import FRAME, Frame, Run, isolated_reference, shadow_forward, spatial_image, token_slice

_CAPTURE = ContextVar("modeexplorer_mechanistic_capture", default=None)
_PACKED_PROJECTION = "final_projection"
_UNPATCHED_PROJECTION = "final_projection_unpatched"
_LATENT_GRID = {_UNPATCHED_PROJECTION, "velocity", "clean"}
_GLOBAL = _LATENT_GRID | {_PACKED_PROJECTION}


def _unpatch_projection(value, patch, spatial_shape):
    """Match Krea's (channel, patch-y, patch-x) unpatching and native crop."""
    _, h, w = value.shape
    y, x = spatial_shape
    return (value.reshape(-1, patch, patch, h, w).permute(0, 3, 1, 4, 2)
            .reshape(-1, h * patch, w * patch)[:, :y, :x])


class CaptureLimit(ValueError):
    pass


class ResearchTiming:
    def __init__(self):
        self.elapsed = dict.fromkeys(("auxiliary_dit", "capture_copy", "capture_transfer", "output_transfer", "spectral", "edit_validation",
                                      "projection_native_replay", "projection_reference", "projection_spectral",
                                      "projection_activation_emulation", "projection_activation_reference", "factorial_clean_spectral"), 0.)
        self.events = []

    @contextmanager
    def measure(self, name, device):
        if device.type == "cuda":
            with torch.cuda.device(device):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                try:
                    yield
                finally:
                    end.record()
                    self.events.append((name, start, end))
        else:
            start = perf_counter()
            try:
                yield
            finally:
                self.elapsed[name] += (perf_counter() - start) * 1000

    def finish(self):
        elapsed = self.elapsed.copy()
        if self.events:
            # All events were recorded on the inference device's current stream.
            self.events[-1][2].synchronize()
            for name, start, end in self.events:
                elapsed[name] += start.elapsed_time(end)
        elapsed["auxiliary_dit"] = max(0., elapsed["auxiliary_dit"] - elapsed["capture_copy"]
                                      - elapsed["capture_transfer"])
        elapsed["capture_transfer"] += elapsed.pop("output_transfer")
        elapsed["export"] = 0.
        return elapsed


class Workspace:
    def __init__(self, limit, device=torch.device("cpu"), timing=None):
        self.limit, self.device, self.timing = limit, device, timing
        self.retained = self.peak = self.transfer_bytes = 0

    def stage(self, value, *, inside_forward=True, preserve_layout=False):
        size = value.numel() * value.element_size()
        if self.retained + size > self.limit:
            raise CaptureLimit("mechanistic_capture_limit")
        transfer = value.device != self.device
        if transfer and self.device.type == "cuda":
            raise ValueError("mechanistic_capture_device_mismatch")
        label = ("capture_transfer" if inside_forward else "output_transfer") if transfer else "capture_copy"
        scope = self.timing.measure(label, value.device) if self.timing else nullcontext()
        with scope:
            layout = torch.preserve_format if preserve_layout else torch.contiguous_format
            result = value.detach().to(self.device, copy=True, memory_format=layout)
        if transfer:
            self.transfer_bytes += size
        self.retained += size
        self.peak = max(self.peak, self.retained)
        return result

    def release(self, value):
        self.retained -= value.numel() * value.element_size()


class Capture:
    def __init__(self, cfg, grid, workspace, block, references=None, proposal=None, scale=0, fixed=None,
                 projection=False, projection_override=None, replace_projection=False):
        self.cfg, self.grid, self.workspace, self.block = cfg, grid, workspace, block
        self.references, self.proposal, self.scale, self.fixed = references, proposal, scale, fixed
        self.fields, self.operators = {}, {}
        self.bounds = None
        self.projection_block = cfg.projection_block if cfg.projection_block >= 0 else block
        self.edit = None
        self.want_projection = projection
        self.projection = None
        self.projection_override = projection_override
        self.replace_projection = replace_projection
        self.projection_realized_energy = None
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
                    delta = after[:, y:y + 2].float() - before[:, y:y + 2].float()
                    if delta.device != output.device:
                        scope = (self.workspace.timing.measure("capture_transfer", output.device)
                                 if self.workspace.timing else nullcontext())
                        with scope:
                            delta = delta.to(output.device)
                        self.workspace.transfer_bytes += delta.numel() * delta.element_size()
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

        def attach(module, site, pre=False, force=False):
            if site[1] not in self.cfg.site_names and not force:
                return

            def record(module, args, output=None):
                if _CAPTURE.get() is self:
                    self.operators[str(site)] = f"{type(module).__module__}.{type(module).__qualname__}"
                    self.store(site, args[0] if pre else output, force=force)

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
            if self.want_projection or self.projection_override is not None:
                block = dit.blocks[self.projection_block]
                if self.projection_block not in self.cfg.block_ids:
                    handles.append(block.register_forward_pre_hook(geometry, with_kwargs=True))

                def capture_input(module, args):
                    if _CAPTURE.get() is not self:
                        return
                    x, = args
                    if self.projection_override is not None:
                        original, _ = self.projection_override
                        if self.bounds != original.bounds or projection_geometry(x) != original.input_geometry or not bitwise_equal(x, original.inputs):
                            raise ProjectionUnavailable("downstream_packed_projection_input_mismatch")
                    else:
                        if self.projection is not None:
                            raise ProjectionUnavailable("packed_projection_called_more_than_once")
                        self.projection = ProjectionCapture(
                            self.bounds, self.workspace.stage(x, preserve_layout=True), projection_geometry(x),
                        )

                def capture_output(module, args, output):
                    if _CAPTURE.get() is not self:
                        return
                    if self.projection_override is not None:
                        original, delta = self.projection_override
                        if projection_geometry(output) != original.output_geometry or not bitwise_equal(output, original.output):
                            raise ProjectionUnavailable("downstream_packed_projection_baseline_mismatch")
                        lo, hi = self.bounds
                        result = output.clone()
                        energy = torch.zeros((), dtype=torch.float64, device=output.device)
                        for first in range(lo, hi, self.cfg.projection_token_chunk):
                            stop = min(first + self.cfg.projection_token_chunk, hi)
                            before = output[0, first:stop].float()
                            replacement = delta[first-lo:stop-lo]
                            result[0, first:stop] = replacement if self.replace_projection else before + replacement
                            energy.add_((result[0, first:stop].float() - before).square().sum())
                        self.projection_realized_energy = energy
                        return result
                    self.projection.output = self.workspace.stage(output, preserve_layout=True)
                    self.projection.output_geometry = projection_geometry(output)

                handles.append(block.attn.wo.register_forward_pre_hook(capture_input))
                handles.append(block.attn.wo.register_forward_hook(capture_output))
            attach(dit.last.linear, (None, _PACKED_PROJECTION), force=_UNPATCHED_PROJECTION in self.cfg.site_names)
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
    sums = torch.zeros(3, dtype=torch.float64, device=left[0].device)
    for i in range(0, left[0].shape[0], chunk):
        sl = slice(i, i + chunk)
        a = left[1][sl].float() - left[0][sl].float()
        b = right[1][sl].float() - right[0][sl].float()
        sums.add_(torch.stack((a.square().sum(), b.square().sum(), (a - sign * b).square().sum())))
    norm, other, residual = map(math.sqrt, sums.tolist())
    return {"left_l2": norm, "right_l2": other, "mismatch_l2": residual,
            "relative_mismatch": residual / max(norm, other) if max(norm, other) else None}


def _site_metrics(fields, cfg, sample_stride, patch, chunk):
    # Transform small intervention deltas directly. Subtracting spectra of
    # full activations first would lose precision when the response is weak.
    d00, d01 = fields["D00"], fields["D01"]
    basis = {"base": ((1., d00),), "plus": ((1., d01), (-1., d00))}
    base, plus = ((1., "base"),), ((1., "plus"),)
    comparisons = {"positive": (base, plus)}
    if "D0minus" in fields:
        basis["minus"] = ((1., fields["D0minus"]), (-1., d00))
        comparisons.update({
            "negative": (base, ((1., "minus"),)),
            "odd": ((), ((.5, "plus"), (-.5, "minus"))),
            "even": ((), ((.5, "plus"), (.5, "minus"))),
            "positive_vs_reflected_negative": (((-1., "minus"),), ((1., "plus"), (1., "minus"))),
        })
    if "D11" in fields:
        d10, d11 = fields["D10"], fields["D11"]
        basis.update(conditioning=((1., d10), (-1., d00)), alternate_edit=((1., d11), (-1., d10)))
        conditioning = ((1., "conditioning"),)
        interaction = ((1., "alternate_edit"), (-1., "plus"))
        comparisons.update({
            "conditioning_baseline": (base, conditioning),
            "conditioning_edited": (base + plus, conditioning + interaction),
            "conditioning_interaction": (conditioning, interaction),
            "feature_across_conditions": (plus, interaction),
        })
    return measure_comparisons(basis, comparisons, chunk, cfg.low_cut, cfg.high_cut,
                               sample_stride=sample_stride, patch_size=patch)


def _amplitude(full_plus, full_minus, half_plus, half_minus, base):
    odd = (full_plus - full_minus) * .5
    even = (full_plus + full_minus) * .5 - base
    odd_half = (half_plus - half_minus) * .5
    even_half = (half_plus + half_minus) * .5 - base
    return {"status": "measured", "odd_half_over_full": rms(odd_half) / rms(odd) if rms(odd) else None,
            "even_half_over_full": rms(even_half) / rms(even) if rms(even) else None,
            "odd_scaled_relative_change": rms(2 * odd_half - odd) / rms(odd) if rms(odd) else None,
            "even_quadratic_relative_change": rms(4 * even_half - even) / rms(even) if rms(even) else None}


def estimate_capture_workspace(spec, *, feature_block, grid, latent_shape, attention_channels, patch_size,
                               baseline_reused, chunk=None, packed_tokens=None):
    """Return conservative bytes for staged captures and spectral/output workspace."""
    points = math.prod(grid)
    projection_channels = latent_shape[1] * patch_size**2
    block_sites = sum(site not in (_GLOBAL | UNAVAILABLE) for site in spec.site_names)
    site_channels = attention_channels * len(spec.block_ids) * block_sites
    if {_PACKED_PROJECTION, _UNPATCHED_PROJECTION}.intersection(spec.site_names):
        site_channels += projection_channels

    captured_arms = spec.core_arms - int(baseline_reused)
    edited_arms = 1 + int(spec.signed) + int(spec.conditioning)
    # Even when unrequested, each edit captures its input and output for validation.
    extra_edit_sites = 2
    if feature_block in spec.block_ids:
        extra_edit_sites -= sum(site in spec.site_names for site in ("attention_pre", "attention_post"))
    main_channels = site_channels * captured_arms + attention_channels * extra_edit_sites * edited_arms
    # Calibration captures run after the main sites are released.
    capture_bytes = 4 * points * max(main_channels, 4 * attention_channels * int(spec.amplitude_check))

    chunk = spec.chunk if chunk is None else chunk
    # Cached casts, real basis fields, a packed batch, complex transforms,
    # comparison scratch, FFT work space, band masks and spatial accumulators.
    fft_bytes = max(points, math.prod(latent_shape[-2:])) * ((192 + 40 * spec.core_arms) * chunk + 128)
    output_bytes = math.prod(latent_shape) * 4 * (10 + 4 * spec.core_arms + 4 * int(spec.amplitude_check))
    if _UNPATCHED_PROJECTION in spec.site_names:
        output_bytes += 4 * points * projection_channels * captured_arms
    if spec.projection_study != "off":
        packed_tokens = points if packed_tokens is None else packed_tokens
        replay_arms = spec.core_arms + 2 * int(spec.amplitude_check)
        # Packed native operands and outputs survive until qualification,
        # including the additional two arms at half amplitude.
        capture_bytes += 8 * packed_tokens * attention_channels * replay_arms
        # Replay result + component fields and spectral scratch. Effective FP64
        # weight, INT8 codes, weight-equality temporaries, and rotation/GEMM
        # scratch are budgeted independently of the captured activations.
        output_bytes += (12 * attention_channels**2 + (1024**2) * 8
                         + (16 * packed_tokens + (28 + 4 * replay_arms) * points) * attention_channels
                         + 128 * spec.projection_token_chunk * attention_channels)
        if spec.projection_symmetric_check:
            output_bytes += (8 * packed_tokens + 20 * points) * attention_channels
        if spec.projection_activation_check:
            # Bitwise qualification uses temporary packed rotation, codes and
            # scales. Retain only generated-token output stages and a baseline
            # for the optional symmetric probe.
            output_bytes += (2 * attention_channels**2 + (16 * packed_tokens + 8 * replay_arms * points) * attention_channels
                             + 96 * spec.projection_token_chunk * attention_channels)
            if spec.projection_symmetric_check:
                output_bytes += 20 * points * attention_channels
    return capture_bytes, fft_bytes + output_bytes


def _conditioning_contrast(executor, run, frame, latent, x, timesteps, context, mask, refs, options,
                           kwargs, native_velocity, point, spend, ordinal, timing):
    """Matched-state semantic contrast, with no feature replay or spectral capture."""
    from .response_runtime import _tensor_hash, _tensor_value_hash

    cfg, spec = run.response.config, run.response.config.mechanistic
    stage = point["mechanistic"]
    if (x.shape[0] != 1 or frame.offset != 0 or list(options.get("cond_or_uncond", ())) != [0]
            or run.response.cfg_scale != 1.0 or run.feature_active or run.settings.initialization
            or frame.attempted or cfg.frozen_state):
        raise ValueError("contrast_requires_live_single_positive_native_cfg_one")
    if not 1e-5 <= frame.sigma:
        raise ValueError("flow_conversion_endpoint")
    required = 1 + int(cfg.calibration != "off")
    stage.update(analysis_device=str(latent.device), baseline_reused=True,
                 auxiliary_calls_planned=required, feature_reference_calls=0,
                 captured_bytes=0, capture_transfer_bytes=0, arms=["D00"],
                 direction_scope="conditional_clean_prediction")
    if run.response.forwards + required > cfg.auxiliary_budget:
        raise ValueError("auxiliary_forward_budget")
    alt, alt_mask, raw_mask = _alternate(spec.alternate, run.response.base_model, latent,
                                          context, mask, refs, kwargs, executor.class_obj)
    stage["conditioning"] = {
        key: {"sha256": _tensor_hash(ctx), "mask_sha256": _tensor_hash(msk),
              "metadata_mask_sha256": _tensor_hash(metadata_mask), "tokens": ctx.shape[1],
              "context_values_sha256": _tensor_value_hash(ctx), "mask_values_sha256": _tensor_value_hash(msk)}
        for key, ctx, msk, metadata_mask in (("C0", context, mask, run.response.positive_metadata_mask),
                                             ("C1", alt, alt_mask, raw_mask))
    }

    def predict(ctx, msk):
        spend(1)
        with torch.no_grad(), isolated_reference(run, sol_evaluations=ordinal):
            with timing.measure("auxiliary_dit", latent.device):
                value = executor(latent, timesteps, ctx, msk, refs, options, **kwargs)
        if not bool(torch.isfinite(value).all()):
            raise ValueError("contrast_nonfinite_velocity")
        return flow_clean(latent, value, frame.sigma)

    base = flow_clean(latent, native_velocity, frame.sigma)
    if not bool(torch.isfinite(base).all()):
        raise ValueError("nonfinite_baseline")
    noise_floor = 0.
    if cfg.calibration != "off":
        noise_floor = rms(predict(context, mask) - base)
        if not math.isfinite(noise_floor):
            raise ValueError("nonfinite_repeat")
    stage["repeat_clean_rms"] = noise_floor if cfg.calibration != "off" else None
    other = predict(alt, alt_mask)
    stage["arms"].append("D10")
    delta = base - other
    stage["contrast_rms"] = rms(delta)
    if not math.isfinite(stage["contrast_rms"]):
        raise ValueError("nonfinite_contrast")
    point.update(status="completed", applied=False, selected_candidate="baseline",
                 candidates=[{"arm": "baseline", "selected": True}])
    tensors = {"frozen_native_input": latent, "baseline_clean": base, "D10_clean": other,
               "contrast_native_delta": delta} if cfg.export else {}
    if spec.component == "observe":
        return native_velocity, tensors
    if (stage["conditioning"]["C0"]["context_values_sha256"] == stage["conditioning"]["C1"]["context_values_sha256"]
            and stage["conditioning"]["C0"]["mask_values_sha256"] == stage["conditioning"]["C1"]["mask_values_sha256"]):
        point["candidates"].append({"arm": spec.component, "reason": "identical_native_conditioning"})
        return native_velocity, tensors
    if stage["contrast_rms"] <= max(2 * noise_floor, 1e-8):
        point["candidates"].append({"arm": spec.component, "reason": "contrast_below_repeat_floor"})
        return native_velocity, tensors
    direction = delta
    if spec.component == "contrast_reverse":
        direction = -delta
    elif spec.component == "contrast_spatial":
        h, w = delta.shape[-2:]
        if h == w == 1:
            raise ValueError("contrast_spatial_requires_spatial_extent")
        shifts = (h // 2, w // 2)
        direction = torch.roll(delta, shifts=shifts, dims=(-2, -1))
        stage["control_shift_yx"] = list(shifts)
    direction_rms = rms(direction)
    denom = stage["contrast_rms"] * direction_rms
    stage["direction_rms"] = direction_rms
    stage["direction_cosine"] = float((direction * delta).mean()) / denom if denom else None
    proposal, info = clean_candidate(base, direction, cfg, {"arm": spec.component})
    corrected = velocity_for_clean(latent, proposal, frame.sigma).to(native_velocity.dtype)
    candidate = flow_clean(latent, corrected, frame.sigma)
    selected, baseline_score = select_response([(candidate, info)], base, cfg, noise_floor,
                                                point, run.response.process_latent_out)
    point["candidates"].append(info)
    point["candidates"][0]["external_scores"] = baseline_score
    if cfg.export and bool(torch.isfinite(candidate).all()):
        tensors["candidate_clean"] = candidate
    if cfg.steers and selected is not None:
        actual = rms(frame.sigma * (corrected.float() - native_velocity.float()))
        point["applied_clean_displacement_rms"] = actual
        if (1e-8 < actual <= cfg.clean_limit
                and (not cfg.candidate_target_rms
                     or abs(actual - cfg.candidate_target_rms) <= .1 * cfg.candidate_target_rms)):
            point.update(applied=True, selected_candidate=spec.component)
            point["candidates"][0]["selected"] = False
            if cfg.export:
                tensors["applied_clean"] = candidate
            return corrected, tensors
        info["reason"] = "native_output_rounding_or_trust_bound"
    return native_velocity, tensors


def research_response(executor, run, frame, latent, x, timesteps, context, mask, refs, options,
                      kwargs, native_velocity, point, spend, reference_sigma, initial_ordinal, ordinal, probe_seed, timing):
    from .response_runtime import _tensor_hash, _tensor_value_hash

    cfg = run.response.config
    spec = cfg.mechanistic
    dit = executor.class_obj
    analysis_device = latent.device if spec.analysis_device == "cuda" else torch.device("cpu")
    chunk = spec.cuda_chunk if spec.analysis_device == "cuda" else spec.chunk
    budget_mib = spec.capture_vram_mib if spec.analysis_device == "cuda" else spec.capture_mib
    stage = point["mechanistic"] = {
        "schema": 2, "mode": spec.mode, "checkpoint_provenance": spec.checkpoint_provenance,
        "injection_block": cfg.feature_block, "blocks": list(spec.block_ids), "requested_sites": list(spec.site_names),
        "analysis_device_requested": spec.analysis_device,
        "analysis_device": None if spec.analysis_device == "cuda" and not latent.is_cuda else str(analysis_device),
        "chunk_channels": chunk, "spectral_dtype": "torch.float32", "summary_accumulation_dtype": "torch.float64",
        "frequency_cycles_per_latent_pixel": [spec.low_cut, spec.high_cut],
        "patch_nyquist_per_axis": .5 / dit.patch,
        "capture_budget_bytes": budget_mib * 1024**2, "cfg_scope": "isolated_positive_row",
        "trajectory": "frozen_export" if cfg.frozen_state else "live_feature" if run.feature_active else "native_or_component",
        "sites": {}, "arms": [], "component": spec.component, "band": spec.band,
        "cell": spec.cell, "filter_transition": spec.transition, "symmetry_limit": spec.symmetry_limit,
        "amplitude_check": spec.amplitude_check, "match_cross_dtype": spec.match_cross_dtype,
        "attention_mask_consumed_by_native_krea": False,
    }
    if spec.mode == "conditioning_contrast":
        return _conditioning_contrast(executor, run, frame, latent, x, timesteps, context, mask, refs, options,
                                      kwargs, native_velocity, point, spend, ordinal, timing)
    projection_enabled = spec.projection_study != "off"
    factorial = spec.projection_study == "activation_factorial"
    projection_downstream = spec.projection_study not in ("off", "observe", "activation_factorial")
    if factorial and (not cfg.frozen_state or not cfg.export or cfg.steers
                      or run.response.cfg_scale != 1.0 or point["branches"] != [0]
                      or run.response.frozen_velocity is None):
        raise ValueError("factorial_requires_native_cfg_one_trajectory_and_export_only")
    if projection_enabled and analysis_device != latent.device:
        raise CaptureLimit("packed_projection_replay_requires_inference_device")
    if spec.analysis_device == "cuda" and not latent.is_cuda:
        raise CaptureLimit("mechanistic_cuda_requires_cuda_inference")
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
    projection_block = spec.projection_block if spec.projection_block >= 0 else cfg.feature_block
    captured_blocks = set(spec.block_ids) | {cfg.feature_block, projection_block}
    if any(block >= len(dit.blocks) for block in captured_blocks):
        raise ValueError("mechanistic_block_out_of_range")
    channels = dit.blocks[cfg.feature_block].attn.wo.in_features
    if projection_enabled and channels != dit.blocks[projection_block].attn.wo.in_features:
        raise ValueError("projection_and_feature_width_mismatch")
    points = math.prod(frame.grid)
    core_arms = spec.core_arms
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
    can_reuse = not (projection_enabled or native_capture or cfg.frozen_state or frame.attempted or x.shape[0] != 1)
    required = (core_arms - int(can_reuse) + reference_calls + 2 * int(spec.amplitude_check)
                + int(repeat) + int(projection_downstream) + 2 * int(factorial))
    stage["baseline_reused"] = can_reuse
    stage.update(auxiliary_calls_planned=required, feature_reference_calls_planned=reference_calls,
                 feature_reference_calls=0,
                 feature_reference_origin="reconstructed_first_estimate" if rebuild else "initial_shadow")
    if run.response.forwards + required > cfg.auxiliary_budget:
        raise ValueError("auxiliary_forward_budget")
    max_text = max(ctx.shape[1] for ctx, _ in branches.values())
    reference_tokens = sum(math.prod(math.ceil(n / dit.patch) for n in ref.shape[-2:])
                           * (ref.shape[2] if ref.ndim == 5 else 1) for ref in refs or ())
    packed_tokens = points + max_text + reference_tokens
    capture_bytes, reserve_bytes = estimate_capture_workspace(
        spec, feature_block=cfg.feature_block, grid=frame.grid, latent_shape=latent.shape,
        attention_channels=channels, patch_size=dit.patch, baseline_reused=can_reuse, chunk=chunk,
        packed_tokens=packed_tokens,
    )
    stage.update(capture_preflight_bytes=capture_bytes + reserve_bytes, capture_reserve_bytes=reserve_bytes)
    if stage["capture_preflight_bytes"] > stage["capture_budget_bytes"]:
        raise CaptureLimit("mechanistic_capture_preflight")
    workspace = Workspace(stage["capture_budget_bytes"] - reserve_bytes, analysis_device, timing)

    # The cloned attention output, references and latent fields are separate
    # from diagnostic captures.
    gpu_bytes = 4 * (channels * (packed_tokens + 4 * frame.grid[1])
                     + latent.numel() * (12 + core_arms * 2)
                     + channels * cfg.feature_grid**2 * cfg.feature_references * 3)
    # First-estimate anchors retain native FP32 clean and noise per reference.
    gpu_bytes += 8 * latent.numel() * cfg.feature_references * len(ref_branches)
    point["estimated_working_bytes"] = max(gpu_bytes, point["estimated_working_bytes"])
    if point["estimated_working_bytes"] > cfg.memory_mb * 1024**2:
        raise CaptureLimit("research_memory_budget")

    if analysis_device.type == "cuda":
        free, _ = torch.cuda.mem_get_info(analysis_device)
        cached = torch.cuda.memory_reserved(analysis_device) - torch.cuda.memory_allocated(analysis_device)
        stage["cuda_available_bytes"] = free + max(0, cached)
        if stage["capture_preflight_bytes"] + gpu_bytes > stage["cuda_available_bytes"]:
            raise CaptureLimit("mechanistic_cuda_headroom")

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
                with timing.measure("auxiliary_dit", latent.device):
                    shadow_forward(executor, shadow_run, initial_frame, latent, torch.full_like(timesteps, reference_sigma),
                                   ctx, msk, refs, initial_options, kwargs, i, sol_evaluations=initial_ordinal)
            response.mechanistic_anchors[branch] = reference_keys[branch], dict(shadows)
        for i in range(cfg.feature_references):
            spend(1)
            stage["feature_reference_calls"] += 1
            with timing.measure("auxiliary_dit", latent.device):
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
            with timing.measure("auxiliary_dit", latent.device):
                value = executor(latent, timesteps, ctx, msk, refs, opts, **kwargs)
        if not bool(torch.isfinite(value).all()):
            raise ValueError("mechanistic_nonfinite_velocity")
        if capture is not None:
            captures[name] = capture
        velocities[name] = value
        clean[name] = flow_clean(latent, value, frame.sigma)
        stage["arms"].append(name)

    def capture(**edit):
        return Capture(spec, frame.grid, workspace, cfg.feature_block, projection=projection_enabled, **edit)

    if can_reuse:
        captures["D00"] = capture()
        velocities["D00"] = native_velocity[row]
        clean["D00"] = flow_clean(latent, velocities["D00"], frame.sigma)
        stage["arms"].append("D00")
    else:
        predict("D00", capture=capture())
    if factorial and not bitwise_equal(velocities["D00"].float().cpu(), run.response.frozen_velocity):
        raise ValueError("factorial_frozen_native_velocity_mismatch")
    if repeat:
        predict("D00_repeat")
        stage["repeat_clean_rms"] = rms(clean.pop("D00_repeat") - clean["D00"])
        velocities.pop("D00_repeat")
    refs0 = references("C0")
    plus = capture(references=refs0, scale=1)
    predict("D01", capture=plus)
    plus_edit = _edit_pair(plus)
    stage["proposal"] = {"C0_sha256": _tensor_hash(plus.proposal), "C0_compact_rms": rms(plus.proposal)}
    def compare_edits(left, right, sign):
        with timing.measure("edit_validation", analysis_device):
            return _edit_compare(left, right, sign, chunk)

    if spec.signed:
        minus = capture(proposal=plus.proposal, scale=-1)
        predict("D0minus", capture=minus)
        symmetry = compare_edits(plus_edit, _edit_pair(minus), -1)
        symmetry["status"] = "matched" if (symmetry["relative_mismatch"] is not None
                                            and symmetry["relative_mismatch"] <= spec.symmetry_limit) else "unqualified"
        stage["signed_realized_edit"] = symmetry
    if spec.conditioning:
        predict("D10", "C1", capture())
        alt = capture(references=references("C1"), scale=1) if recomputed else capture(fixed=plus_edit)
        predict("D11", "C1", alt)
        match = compare_edits(plus_edit, _edit_pair(alt), 1)
        match["status"] = ("independent_proposals" if recomputed else "matched"
                           if match["relative_mismatch"] is not None and match["relative_mismatch"] <= spec.symmetry_limit
                           else "unqualified")
        stage["conditioning_realized_edit"] = match
        alt.fixed = None
        if recomputed:
            stage["proposal"]["C1_sha256"] = _tensor_hash(alt.proposal)
            stage["proposal"]["C1_compact_rms"] = rms(alt.proposal)

    if factorial:
        positive_norm = compare_edits(plus_edit, plus_edit, 1)["left_l2"]
    else:
        positive_norm = symmetry["left_l2"] if spec.signed else match["left_l2"]
    stage["realized_positive_l2"] = positive_norm
    stage["realized_positive_rms"] = positive_norm / math.sqrt(plus_edit[0].numel())
    del plus_edit
    stage["sampler_velocity_baseline_rms_difference"] = rms(velocities["D00"].float() - native_velocity[row].float())
    stage["operators"] = captures["D00"].operators
    main_arms = tuple(captures)
    for key in keys:
        block, site = key
        name = site if block is None else f"b{block}.{site}"
        stride = 1 if site in _LATENT_GRID else dit.patch
        coordinate = "native_diffusion_latent" if stride == 1 else "generated_image_patch_grid"
        info = {"site": site, "block": block, "coordinate": coordinate, "latent_pixels_per_sample": stride}
        if site in UNAVAILABLE:
            info.update(status="unsupported", reason="no_exact_native_module_boundary")
        else:
            if site == "clean" or site == "velocity":
                values = clean if site == "clean" else velocities
                fields = {arm: spatial_image(values[arm])[0].detach() for arm in main_arms}
                if latent.device != analysis_device:
                    fields = {arm: workspace.stage(value, inside_forward=False) for arm, value in fields.items()}
            elif site == _UNPATCHED_PROJECTION:
                fields = {}
                for arm in main_arms:
                    packed = captures[arm].fields.get((None, _PACKED_PROJECTION))
                    fields[arm] = _unpatch_projection(packed, dit.patch, latent.shape[-2:]) if packed is not None else None
                del packed
            else:
                fields = {arm: captures[arm].fields.get(key) for arm in main_arms}
            if any(value is None for value in fields.values()):
                info.update(status="incomplete", missing_arms=[arm for arm, value in fields.items() if value is None])
            else:
                with timing.measure("spectral", analysis_device):
                    comparisons = _site_metrics(fields, spec, stride, dit.patch, chunk)
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
            if site in {"clean", "velocity"} and latent.device != analysis_device:
                for value in fields.values():
                    workspace.release(value)
                value = None
            del fields
        stage["sites"][name] = info
        capture_key = (None, _PACKED_PROJECTION) if site in {_PACKED_PROJECTION, _UNPATCHED_PROJECTION} else key
        if capture_key == (None, _PACKED_PROJECTION) and any(
            requested in spec.site_names and requested not in stage["sites"]
            for requested in (_PACKED_PROJECTION, _UNPATCHED_PROJECTION)
        ):
            continue
        for arm in main_arms:
            if capture_key in captures[arm].fields:
                workspace.release(captures[arm].fields.pop(capture_key))
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
                                          proposal=plus.proposal, scale=sign, projection=projection_enabled))
        half_symmetry = compare_edits(_edit_pair(captures["D0half_plus"]), _edit_pair(captures["D0half_minus"]), -1)
        stage["half_realized_edit"] = half_symmetry
        stage["amplitude"] = _amplitude(clean["D01"], clean["D0minus"],
                                        clean["D0half_plus"], clean["D0half_minus"], clean["D00"])
        for name in ("D0half_plus", "D0half_minus"):
            for value in captures[name].fields.values():
                workspace.release(value)
            captures[name].fields.clear()
            value = None
            velocities.pop(name)
            clean.pop(name)

    downstream_clean = None
    factorial_clean = {}
    if projection_enabled:
        projection_report = stage["projection"] = {"schema": 1, "mode": spec.projection_study, "block": projection_block,
                                                   "effect": spec.projection_effect, "packed_tokens_upper_bound": packed_tokens}
        snapshots = {name: cap.projection for name, cap in captures.items()}
        try:
            if any(snapshot is None for snapshot in snapshots.values()):
                raise ProjectionUnavailable("packed_projection_capture_incomplete")
            if any(snapshot.inputs.shape[1] > packed_tokens for snapshot in snapshots.values()):
                raise ProjectionUnavailable("packed_projection_exceeds_token_estimate")
            projection_report["retained_packed_bytes"] = sum(
                tensor.numel() * tensor.element_size()
                for snapshot in snapshots.values() for tensor in (snapshot.inputs, snapshot.output) if tensor is not None
            )
            delta = study_projection(dit.blocks[projection_block].attn.wo, snapshots, frame.grid, spec, dit.patch,
                                     chunk, projection_report, timing,
                                     seed_key(probe_seed, f"projection-control-{cfg.trial}"))
            if factorial:
                # Both reruns must reproduce the complete native packed inputs
                # and outputs before replacement on generated tokens only.
                light = replace(spec, sites="")
                for name, source in (("continuous_native", "D00"), ("continuous_feature", "D01")):
                    patched = Capture(light, frame.grid, workspace, cfg.feature_block,
                                      proposal=plus.proposal if source == "D01" else None,
                                      scale=1, projection_override=(snapshots[source], delta[source]),
                                      replace_projection=True)
                    try:
                        predict(name, capture=patched)
                        if patched.projection_realized_energy is None:
                            raise ProjectionUnavailable("factorial_projection_hook_not_observed")
                    finally:
                        for value in patched.fields.values():
                            workspace.release(value)
                        patched.fields.clear()
                        patched.projection_override = None
                a, b, c, d = (clean[name] for name in ("D00", "D01", "continuous_native", "continuous_feature"))
                fields = {key: ((1., spatial_image(value)[0]),) for key, value in zip("ABCD", (a, b, c, d))}
                comparisons = {
                    "feature_native": (((1., "A"),), ((1., "B"), (-1., "A"))),
                    "feature_continuous": (((1., "C"),), ((1., "D"), (-1., "C"))),
                    "precision_native": (((1., "A"),), ((1., "C"), (-1., "A"))),
                    "precision_feature": (((1., "B"),), ((1., "D"), (-1., "B"))),
                    "interaction": (((1., "A"),), ((1., "D"), (-1., "C"), (-1., "B"), (1., "A"))),
                }
                with timing.measure("factorial_clean_spectral", latent.device):
                    stats = measure_comparisons(fields, comparisons, chunk, spec.low_cut, spec.high_cut, patch_size=dit.patch)
                projection_report["factorial"] = {
                    "scope": "frozen_same_checkpoint_generated_tokens_only", "proposal_sha256": stage["proposal"]["C0_sha256"],
                    "comparisons": {key: value for key, value in stats.items()},
                    "production_cuda": latent.is_cuda,
                }
                factorial_clean = {name: clean[name] for name in ("continuous_native", "continuous_feature")}
            if projection_downstream:
                if delta is None:
                    raise ProjectionUnavailable("projection_component_below_numerical_floor")
                if spec.projection_effect in ("odd", "even") and stage["signed_realized_edit"]["status"] != "matched":
                    raise ProjectionUnavailable("signed_intervention_unqualified")
                if spec.projection_effect == "conditioning_interaction" and stage["conditioning_realized_edit"]["status"] != "matched":
                    raise ProjectionUnavailable("fixed_conditioning_intervention_unqualified")
                original = snapshots["D00"]
                if not projection_report["native_replay"]["D00"]["packed_bit_exact"]:
                    raise ProjectionUnavailable("downstream_requires_bit_exact_baseline")
                # Keep the actual projected baseline for all tokens and add a
                # component only at generated tokens in one native baseline arm.
                patched = Capture(spec, frame.grid, workspace, cfg.feature_block,
                                  projection_override=(original, delta))
                predict("projection_component", capture=patched)
                if patched.projection_realized_energy is None:
                    raise ProjectionUnavailable("downstream_projection_hook_not_observed")
                projection_report["downstream_requested_projection_l2"] = float(torch.linalg.vector_norm(delta))
                projection_report["downstream_realized_projection_l2"] = float(patched.projection_realized_energy.sqrt())
                if not math.isfinite(projection_report["downstream_realized_projection_l2"]):
                    raise ProjectionUnavailable("downstream_projection_displacement_nonfinite")
                downstream_clean = clean["projection_component"]
                projection_report["downstream_clean_rms"] = rms(downstream_clean - clean["D00"])
                projection_report["downstream_output"] = "generated_token_displacement_on_native_baseline"
                patched.projection_override = None
                del delta, original
        except ProjectionUnavailable as exc:
            projection_report.update(status="unqualified", reason=str(exc))
        finally:
            for capture_state in captures.values():
                snapshot = capture_state.projection
                if snapshot is not None:
                    workspace.release(snapshot.inputs)
                    if snapshot.output is not None:
                        workspace.release(snapshot.output)
                    capture_state.projection = None
            snapshots.clear()
            snapshot = None

    stage["captured_bytes"] = workspace.peak
    stage["capture_transfer_bytes"] = workspace.transfer_bytes
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
        if projection_downstream and downstream_clean is not None:
            component = downstream_clean - base
        component = filter_field(component, spec.band, spec.low_cut, spec.high_cut, spec.transition, patch_size=dit.patch)
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
            noise = filter_field(noise, spec.band, spec.low_cut, spec.high_cut, spec.transition, patch_size=dit.patch)
            if bounds is not None:
                noise = localize(noise)
            component = noise * (rms(component) / max(1e-12, rms(noise)))
        info = {"arm": spec.component, "isolated_component_rms": rms(component)}
        if projection_downstream and downstream_clean is None:
            info["reason"] = "projection_component_unqualified"
        elif spec.component in ("odd", "even") and stage["signed_realized_edit"]["status"] != "matched":
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
        tensors.update({f"{name}_clean": value for name, value in factorial_clean.items()})
        if downstream_clean is not None:
            tensors["projection_component_clean"] = downstream_clean
        if candidate is not None and (not projection_downstream or downstream_clean is not None):
            tensors["candidate_clean"] = candidate
        if point["applied"]:
            tensors["applied_clean"] = candidate
    if point["applied"]:
        result = native_velocity.clone()
        result[row] = corrected
        return result, tensors
    return native_velocity, tensors
