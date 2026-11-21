"""In-forward guidance at the native Krea 2 attention output projection."""

import json
import math
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from time import perf_counter

import torch

from .objectives import seed_key

MODES = ("gaussian", "token_adaptive", "channel_aware")
SCOPES = ("planner", "highres", "standalone", "all")
SCHEDULES = ("constant", "linear_decay")
UNITS = ("absolute_rms", "relative_rms")
SITE = "attention_output_projection"
CHUNK = 512
TRACE_LIMIT = 64
_FRAME_KEY = "mode_explorer_krea2.projection_guidance_frame"
_ACTIVE = ContextVar(_FRAME_KEY, default=None)


@dataclass(frozen=True)
class Guidance:
    mode: str = "gaussian"
    units: str = "absolute_rms"
    amplitude: float = .000303
    block: int = 5
    evaluations: int = 2
    min_sigma: float = .82
    max_sigma: float = 1.00001
    schedule: str = "constant"
    scope: str = "planner"
    trial: int = 0
    max_absolute_rms: float = .02
    max_relative_rms: float = 0.
    export_clean: bool = False

    def __post_init__(self):
        for key, choices in (("mode", MODES), ("units", UNITS), ("scope", SCOPES), ("schedule", SCHEDULES)):
            if getattr(self, key) not in choices:
                raise ValueError(f"Unknown projection guidance {key}")
        for key, maximum in (("block", 63), ("evaluations", 64), ("trial", 2**31 - 1)):
            value = getattr(self, key)
            if type(value) is not int or not 0 <= value <= maximum:
                raise ValueError(f"Invalid projection guidance {key}")
        if not (0 <= self.amplitude <= 1 and 0 < self.max_absolute_rms <= 1
                and 0 <= self.max_relative_rms <= 1 and 0 <= self.min_sigma <= self.max_sigma <= 1.00001):
            raise ValueError("Invalid projection guidance amplitude, limit or sigma window")

    def strength(self, ordinal, sigma):
        if (self.evaluations and ordinal >= self.evaluations) or not self.min_sigma <= sigma <= self.max_sigma:
            return 0.
        if self.schedule == "constant":
            return self.amplitude
        if self.evaluations:
            progress = ordinal / max(1, self.evaluations - 1)
        else:
            progress = (self.max_sigma - sigma) / max(1e-8, self.max_sigma - self.min_sigma)
        return self.amplitude * (1. - .5 * progress)


def _norm2(values):
    flat = values.flatten()
    return torch.linalg.vector_norm(flat).square()


def _scale_statistics(row, mode):
    tokens, channels = row.shape
    scale2 = torch.zeros(channels if mode == "channel_aware" else tokens, device=row.device, dtype=torch.float32)
    for start in range(0, tokens, CHUNK):
        part = row[start:start + CHUNK].float()
        if mode == "channel_aware":
            scale2.add_(part.square().sum(dim=0))
        else:
            scale2[start:start + CHUNK] = part.square().sum(dim=1)
    rms = (scale2.sum() / (tokens * channels)).sqrt()
    if mode == "gaussian":
        return rms, None
    weights = (scale2 / (tokens if mode == "channel_aware" else channels)).sqrt()
    return rms, (weights / rms.clamp_min(1e-12)).clamp_(.5, 2.)


def _noise(row, generator, weights, mode):
    for start in range(0, row.shape[0], CHUNK):
        span = slice(start, start + CHUNK)
        delta = torch.randn(row[span].shape, generator=generator, device=row.device, dtype=torch.float32)
        if weights is not None:
            delta.mul_(weights if mode == "channel_aware" else weights[span, None])
        yield span, delta


def perturb(output, bounds, offset, batch, cfg, strength, seeds, info):
    """Return a validated replacement or None, leaving native storage untouched."""
    lo, hi = bounds
    if output.layout != torch.strided or output.ndim != 3 or output.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        info.update(status="skipped", reason="unsupported_projection_output")
        return None
    if not (0 <= lo < hi <= output.shape[1] and 0 <= offset < offset + batch <= output.shape[0]
            and output.shape[2] > 0):
        info.update(status="skipped", reason="unverified_projection_geometry")
        return None

    result = output.clone(memory_format=torch.preserve_format)
    if result.stride() != output.stride():
        info.update(status="skipped", reason="unsupported_projection_layout")
        return None
    n = (hi - lo) * output.shape[2]
    summary = []
    generator = torch.Generator(device=output.device)
    for index, seed in enumerate(seeds):
        row = output[offset + index, lo:hi]
        target_row = result[offset + index, lo:hi]
        native_rms, weights = _scale_statistics(row, cfg.mode)
        requested = native_rms * strength if cfg.units == "relative_rms" else torch.full_like(native_rms, strength)
        cap = torch.full_like(native_rms, cfg.max_absolute_rms)
        if cfg.max_relative_rms:
            cap = torch.minimum(cap, native_rms * cfg.max_relative_rms)
        target = torch.minimum(requested, cap)
        total = torch.zeros((), device=output.device, dtype=torch.float32)
        generator.manual_seed(seed)
        for _, delta in _noise(row, generator, weights, cfg.mode):
            total.add_(_norm2(delta))
        factor = target / (total / n).sqrt().clamp_min(1e-12)
        generator.manual_seed(seed)
        generated, realized, changed = (torch.zeros((), device=output.device, dtype=torch.float32) for _ in range(3))
        for span, delta in _noise(row, generator, weights, cfg.mode):
            delta.mul_(factor)
            generated.add_(_norm2(delta))
            before = row[span].float()
            target_row[span].copy_(before + delta)
            rounded = target_row[span].float()
            changed.add_((rounded != before).sum())
            realized.add_(_norm2(rounded - before))
        summary.append(torch.stack((native_rms, requested, target, (generated / n).sqrt(),
                                    (realized / n).sqrt(), changed / n, cap)))

    # A single compact synchronization qualifies the entire conditional batch
    # before the caller can expose any replacement to the next native block.
    values = torch.stack(summary).tolist()
    rows = []
    for native, requested, target, generated, realized, changed, cap in values:
        rows.append(dict(native_rms=native, requested_rms=requested, target_rms=target, generated_rms=generated,
                         realized_rms=realized, changed_fraction=changed, allowed_rms=cap,
                         relative_realization_error=abs(realized - generated) / generated if generated else None))
    info.update(rows=rows, generated_image_tokens=hi - lo,
                estimated_owned_peak_bytes=(output.numel() * output.element_size()
                    + 32 * min(hi - lo, CHUNK) * output.shape[-1] + 12 * ((hi - lo) + output.shape[-1])))
    if not all(math.isfinite(value) for row in values for value in row):
        info.update(status="rejected", reason="nonfinite_projection_or_perturbation")
    elif any(row[4] > row[6] for row in values):
        info.update(status="rejected", reason="native_rounded_rms_limit")
    elif not any(row[5] for row in values):
        info.update(status="unchanged", reason="zero_native_realization")
    else:
        info["status"] = "applied"
        return result
    return None


@dataclass
class GuidanceRun:
    config: Guidance
    seed: int
    stage: str
    process_latent_out: object
    ordinal: int = 0
    points: list = field(default_factory=list)
    reason: str | None = None
    exports: int = 0
    cuda_device: torch.device | None = None
    cuda_before: dict | None = None

    def record(self, point):
        if len(self.points) < TRACE_LIMIT:
            self.points.append(point)

    def report(self):
        report = {"schema": 1, "mode": self.config.mode, "block": self.config.block, "site": SITE,
                "trial": self.config.trial, "positive_actual_evaluations": self.ordinal,
                "auxiliary_dit_calls": 0, "auxiliary_projection_calls": 0, "auxiliary_projection_gemms": 0,
                "reason": self.reason or ("no_matching_actual_evaluation" if not self.points else None),
                "points": self.points}
        if self.cuda_device is not None:
            report["cuda_allocator"] = {"scope": "device_lifetime_peak_since_last_reset",
                                        "before_first_intervention": self.cuda_before,
                                        "at_sampler_exit": _cuda_snapshot(self.cuda_device)}
        return report


def _cuda_snapshot(device):
    return {"allocated_bytes": torch.cuda.memory_allocated(device), "reserved_bytes": torch.cuda.memory_reserved(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}


@contextmanager
def _hooks(dit, grid, offset, batch, packed_rows, state, strength, point):
    from .runtime import token_slice

    block = dit.blocks[state.config.block]
    handles = []
    marker = object()
    token = _ACTIVE.set(marker)
    bounds = None
    entered = False
    width = block.attn.heads * block.attn.headdim

    def geometry(module, args, kwargs):
        nonlocal bounds, entered
        if _ACTIVE.get() is not marker:
            return
        options = kwargs.get("transformer_options", {})
        if options.get(_FRAME_KEY) is not marker:
            bounds = None
            return
        if entered:
            raise RuntimeError("Projection guidance received a duplicate native block entry")
        entered = True
        try:
            if args[0].ndim != 3 or args[0].shape[0] != packed_rows or args[0].shape[-1] != width:
                raise ValueError("packed block shape")
            bounds = token_slice(options, args[0].shape[1], math.prod(grid))
        except (KeyError, IndexError, TypeError, ValueError, RuntimeError):
            point.update(status="skipped", reason="unverified_generated_token_slice")

    def edit(module, args, output):
        if _ACTIVE.get() is not marker or bounds is None:
            return
        if point["status"] != "pending":
            raise RuntimeError("Projection guidance received a duplicate projection call")
        if (not torch.is_tensor(output) or not args or output.shape != args[0].shape
                or output.device != args[0].device or output.shape[:2] != (packed_rows, args[0].shape[1])
                or output.shape[-1] != width):
            point.update(status="skipped", reason="unverified_projection_geometry")
            return
        cfg = state.config
        stream = f"projection:{state.stage}:{point['evaluation']}:{point['sigma'].hex()}:{cfg.block}:{SITE}:{grid}:{width}:{cfg.trial}"
        seeds = [seed_key(state.seed, f"{stream}:{row}") for row in range(batch)]
        point["stream_seeds"] = seeds
        if output.is_cuda and state.cuda_device is None:
            state.cuda_device = output.device
            state.cuda_before = _cuda_snapshot(output.device)
        begin = perf_counter()
        try:
            return perturb(output, bounds, offset, batch, cfg, strength, seeds, point)
        except torch.OutOfMemoryError:
            point.update(status="rejected", reason="projection_workspace_oom")
            return None
        finally:
            point["guidance_ms"] = (perf_counter() - begin) * 1000

    try:
        handles.append(block.register_forward_pre_hook(geometry, with_kwargs=True))
        handles.append(block.attn.wo.register_forward_hook(edit))
        yield marker
    finally:
        for handle in reversed(handles):
            handle.remove()
        _ACTIVE.reset(token)


def dispatch(forward, arguments, dit, run):
    """Count completed actual positive dispatches, excluding unconditional calls."""
    from .runtime import spatial_image

    x, _, _, _, _, options = arguments
    state = run.guidance
    cfg = state.config
    branches = options.get("cond_or_uncond", ())
    if not branches or any(value not in (0, 1) for value in branches):
        state.reason = "unverified_condition_dispatch"
        return forward(arguments)
    if 0 not in branches:
        return forward(arguments)
    if branches.count(0) != 1 or len(set(branches)) != len(branches) or x.shape[0] != len(branches) * run.batch:
        state.reason = "unverified_condition_dispatch"
        return forward(arguments)
    if spatial_image(x) is None or tuple(x.shape[1:]) != run.shape[1:]:
        state.reason = "unsupported_latent_shape"
        return forward(arguments)
    ordinal = state.ordinal
    forecasts = run.spectrum_forecasts

    def count_actual(result):
        state.ordinal += int(run.spectrum_forecasts == forecasts)
        return result

    if cfg.evaluations and ordinal >= cfg.evaluations:
        return count_actual(forward(arguments))
    sigmas = options.get("sigmas")
    if not torch.is_tensor(sigmas) or sigmas.shape != (run.batch,):
        state.reason = "unverified_sigma_dispatch"
        return forward(arguments)
    low, high = torch.stack((sigmas.amin(), sigmas.amax())).tolist()
    if not math.isfinite(low) or low != high:
        state.reason = "nonuniform_sigma"
        return forward(arguments)
    strength = cfg.strength(ordinal, low)
    if not strength:
        return count_actual(forward(arguments))
    offset = branches.index(0) * run.batch
    grid = tuple(math.ceil(size / dit.patch) for size in x.shape[-2:])
    point = dict(status="pending", evaluation=ordinal, sigma=low, grid=grid, branches=list(branches),
                 strength=strength, units=cfg.units)
    with _hooks(dit, grid, offset, run.batch, x.shape[0], state, strength, point) as marker:
        result = forward((*arguments[:-1], {**options, _FRAME_KEY: marker}))
    if run.spectrum_forecasts != forecasts:
        state.reason = "unexpected_spectrum_forecast"
        return result
    count_actual(result)
    if point["status"] == "pending":
        point.update(status="skipped", reason="projection_hook_not_reached")
    state.record(point)
    if cfg.export_clean and point["status"] in ("applied", "unchanged"):
        try:
            _export_clean(state, x, result, offset, run, point, arguments[2:5])
        except torch.OutOfMemoryError:
            point["export_status"] = "diagnostic_export_oom"
    return result


def _export_clean(state, x, velocity, offset, run, point, conditioning):
    from types import SimpleNamespace

    from .response_runtime import MAX_EXPORT_BYTES, _export, _tensor_value_hash

    rows = slice(offset, offset + run.batch)
    if x[rows].numel() * torch.float32.itemsize > MAX_EXPORT_BYTES - state.exports:
        point["export_status"] = "export_budget"
        return
    source = x[rows].float()
    clean = source - point["sigma"] * velocity[rows].float()
    context, mask, references = conditioning
    point.update(schema=1, stage=state.stage, seed=state.seed, trial=state.config.trial,
                 mode=state.config.mode, block=state.config.block, site=SITE,
                 native_input_fp32_sha256=_tensor_value_hash(source))
    point["native_condition_values"] = {
        "context": _tensor_value_hash(context[rows]),
        "mask": _tensor_value_hash(mask[rows] if mask is not None and mask.ndim >= 3 and mask.shape[0] == x.shape[0] else mask),
        "references": [_tensor_value_hash(ref[rows] if ref.shape[0] == x.shape[0] else ref) for ref in references or ()],
    }
    export = SimpleNamespace(exports=state.exports, process_latent_out=state.process_latent_out,
                             config=SimpleNamespace(export=True, mechanistic=None))
    _export(export, {"guided_clean": clean}, point)
    state.exports = export.exports


def sample_guidance(config, settings, executor, model_wrap, sigmas, extra_args, noise, sample):
    from comfy.model_sampling import CONST
    from comfy.samplers import CFGGuider, KSAMPLER

    from .response_runtime import spectrum_all_actual
    from .runtime import (CONDITION_FLAGS, KEY, LOGGER, MAX_BATCH, RUN, STAGE, Run, finite_json, spatial_image,
                          verify_patch_graph, wrappers)

    options = extra_args["model_options"]
    stage = options.get(STAGE, getattr(model_wrap, "model_options", {}).get(STAGE, "standalone"))
    if config.scope != "all" and config.scope != stage:
        return sample()
    seed = extra_args.get("seed")
    model = model_wrap.inner_model
    batch = noise.shape[0]
    state = GuidanceRun(config, seed, stage, model.process_latent_out)
    run = None
    completed = False
    begin = perf_counter()

    def eligible():
        if stage not in SCOPES[:-1]:
            return "unsupported_sampler_stage"
        if type(seed) is not int:
            return "missing_seed"
        if type(executor.class_obj) is not KSAMPLER or type(model_wrap).predict_noise is not CFGGuider.predict_noise:
            return "unsupported_sampler_or_guider"
        if options.get("multigpu_clones") or options.get("context_handler") is not None:
            return "unsupported_sampler_dispatch"
        if spatial_image(noise) is None or not 1 <= batch <= MAX_BATCH or len(sigmas) < 2:
            return "unsupported_latent_or_schedule"
        sampling = model.model_sampling
        if not isinstance(sampling, CONST) or any(getattr(type(sampling), key) is not getattr(CONST, key)
                for key in ("calculate_input", "calculate_denoised", "noise_scaling")):
            return "unverified_flow_sampling"
        if any(options.get(key) for key in ("sampler_cfg_function", "sampler_pre_cfg_function",
                                            "sampler_post_cfg_function")):
            return "unsupported_cfg_callbacks"
        conds = model_wrap.conds
        if not conds.get("positive") or any(group and (
                len(group) != 1 or CONDITION_FLAGS.intersection(group[0])) for group in conds.values()):
            return "unsupported_conditioning"
        return None

    try:
        state.reason = eligible()
        spectrum = sol = None
        if state.reason is None:
            try:
                spectrum, sol = verify_patch_graph(options, settings)
            except RuntimeError as exc:
                state.reason = str(exc)
        force_native = nullcontext()
        if state.reason is None and spectrum is not None:
            order = tuple(wrappers(options).get("sampler_sample", {}))
            if KEY not in order or order.index(KEY) > order.index("spectrum_krea2"):
                state.reason = "projection_guidance_requires_outer_sampler_scope"
            else:
                force_native = spectrum_all_actual(spectrum)
            if state.reason is None and force_native is None:
                state.reason = "missing_spectrum_all_actual_contract"
        if state.reason is None:
            run = Run(settings, seed, 0., 1., batch, tuple(noise.shape), False,
                      spectrum=spectrum, sol=sol, stage=stage, guidance=state)
            patched = {**options, "transformer_options": {**options.get("transformer_options", {}), RUN: run}}
            with force_native:
                result = sample({**extra_args, "model_options": patched})
        else:
            result = sample()
        completed = True
        return result
    finally:
        payload = {"seed": seed, "stage": stage, "completed": completed, "batch": batch,
                   "seconds": perf_counter() - begin, "native_dit_calls": run.native_dit_calls if run else None,
                   "spectrum_forecast_calls": run.spectrum_forecasts if run else None,
                   "spectrum_policy": "all_actual" if run and run.spectrum else "none",
                   "projection_guidance": state.report()}
        LOGGER.info("MODEEXPLORER_TRACE %s", json.dumps(finite_json(payload), separators=(",", ":"), allow_nan=False))
