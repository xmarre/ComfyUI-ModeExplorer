"""Clone-local ComfyUI integration. All tensor state belongs to a sampler invocation."""

import json
import logging
import math
import sys
import time
from contextlib import contextmanager
from functools import partial
from dataclasses import dataclass, field

import torch

from .objectives import (
    cell_size,
    feature_shift,
    initial_direction,
    noise_candidate,
    pool,
    reference_noise,
    seed_key,
    select_candidate,
    token_pool,
)

LOGGER = logging.getLogger("ModeExplorer")
KEY = "mode_explorer_krea2"
STAGE = "spectrum_krea2_stage"
RUN = f"{KEY}.run"
FRAME = f"{KEY}.frame"
BLOCK = 1  # second single-stream block, zero-indexed
MAX_BATCH = 8
MAX_POINTS = 12
MAX_INITIALIZATION_DIT = 12
MAX_INITIALIZATION_SECONDS = 45
MAX_INITIALIZATION_NOISE_BYTES = 32 * 1024**2
FEATURES = ("native", "feature_cosine", "feature_volume")
OPERATIONS = ("auto", "sequential", "batch")
CONDITION_FLAGS = frozenset({
    "area", "mask", "control", "hooks", "gligen", "start_percent", "end_percent",
    "timestep_start", "timestep_end", "default",
})


@dataclass(frozen=True)
class Settings:
    feature: str = "feature_cosine"
    initialization: bool = False
    strength: float = 0.45
    fidelity: float = 0.55
    min_sigma: float = 0.82
    adaptive: bool = True
    similarity: float = 0.70
    operation: str = "auto"
    reference_budget: int = 2
    diagnostics: bool = False
    # Experimental opt-in to the unmodified Spectrum native/forecast policy.
    # Forecasted evaluations cannot receive attention guidance.
    full_spectrum: bool = False
    initialization_trials: int = 3

    def __post_init__(self):
        if self.feature not in FEATURES or self.operation not in OPERATIONS:
            raise ValueError("ModeExplorer received an unknown feature or batch mode")
        for key, low, high in (
            ("strength", 0, 1), ("fidelity", 0, 1), ("min_sigma", 0.5, 1),
            ("similarity", 0, 0.99),
        ):
            if not low <= getattr(self, key) <= high:
                raise ValueError(f"ModeExplorer {key} must be in [{low}, {high}]")
        if type(self.reference_budget) is not int or not 0 <= self.reference_budget <= 4:
            raise ValueError("ModeExplorer reference_budget must be an integer in [0, 4]")
        if type(self.initialization_trials) is not int or not 1 <= self.initialization_trials <= 3:
            raise ValueError("ModeExplorer initialization_trials must be an integer in [1, 3]")
        if type(self.full_spectrum) is not bool:
            raise ValueError("ModeExplorer full_spectrum must be a boolean")


@dataclass
class Run:
    settings: Settings
    seed: int
    start: float
    noise_scale: float
    batch: int
    shape: tuple[int, ...]
    feature_active: bool
    phase: str = "sample"
    reference_calls: int = 0
    initialization_guider_calls: int = 0
    initialization_dit_calls: int = 0
    initialization_deadline: float = math.inf
    native_dit_calls: int = 0
    feature_attempts: int = 0
    spectrum_forecasts: int = 0
    spectrum: object = None
    shadow_noise: torch.Tensor | None = None
    shadow_clean: torch.Tensor | None = None
    shadow_start: float | None = None
    points: list[dict] = field(default_factory=list)
    reasons: set[str] = field(default_factory=set)


@dataclass
class Frame:
    grid: tuple[int, int]
    batch: int
    offset: int
    phase: str
    run: Run
    sigma: float
    pairs: torch.Tensor | None = None
    reference: torch.Tensor | None = None
    pooled: torch.Tensor | None = None
    metrics: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    seen: bool = False


def sigma_eligible(sigma: float, minimum: float) -> bool:
    return math.isfinite(sigma) and minimum <= sigma <= 1.00001


def wrappers(options):
    return options.get("transformer_options", {}).get("wrappers", {})


def has_spectrum(options, registered=None):
    return any(entries.get("spectrum_krea2") for tree in (wrappers(options), registered or {})
               for entries in tree.values())



def verified_spectrum(tree):
    """Verify the actual registered Spectrum owner without importing duplicate state."""
    kinds = ("sampler_sample", "predict_noise", "diffusion_model")
    found = {kind: tree.get(kind, {}).get("spectrum_krea2", ()) for kind in kinds}
    if not any(found.values()):
        return None
    if any(len(found[kind]) != 1 for kind in kinds):
        raise RuntimeError("ModeExplorer requires the complete Spectrum wrapper set.")
    sampler, predictor, diffusion = (found[kind][0] for kind in kinds)
    owner = getattr(sampler, "__self__", None)
    module_name = type(owner).__module__
    module = sys.modules.get(module_name)
    if (not module_name.endswith("comfyui_spectrum_krea2.sampling")
            or module is None or type(owner) is not getattr(module, "Binding", None)
            or getattr(sampler, "__func__", None) is not type(owner).sampler
            or getattr(predictor, "__self__", None) is not owner
            or getattr(predictor, "__func__", None) is not type(owner).predict
            or not isinstance(diffusion, partial)
            or len(diffusion.args) != 1 or diffusion.args[0] is not owner
            or diffusion.keywords):
        raise RuntimeError("ModeExplorer could not verify Spectrum wrapper ownership.")
    prefix = module_name.rsplit(".", 1)[0]
    adapter = sys.modules.get(prefix + ".adapter")
    runtime = sys.modules.get(prefix + ".runtime")
    if (adapter is None or runtime is None
            or diffusion.func is not getattr(adapter, "diffusion", None)
            or not all(callable(getattr(var, method, None))
                       for var in (getattr(runtime, "CURRENT", None), getattr(adapter, "CAPTURE", None))
                       for method in ("get", "set", "reset"))):
        raise RuntimeError("ModeExplorer could not verify Spectrum's scoped state contract.")
    return owner, runtime.CURRENT, adapter.CAPTURE


@contextmanager
def isolated_spectrum_reference(run):
    """Auxiliary DiT passes cannot advance Spectrum or enter its feature history."""
    if run.spectrum is None:
        yield
        return
    owner, current, capture = run.spectrum
    session = current.get()
    if session is not None and session.binding is not owner:
        raise RuntimeError("ModeExplorer encountered a foreign Spectrum session.")
    token_session = current.set(None)
    token_capture = capture.set(None)
    try:
        yield
    finally:
        capture.reset(token_capture)
        current.reset(token_session)


def spectrum_forecast_count(run):
    if run.spectrum is None:
        return None
    owner, current, _ = run.spectrum
    session = current.get()
    return session.counts["forecast"] if session is not None and session.binding is owner else None


def verify_patch_graph(options, settings, registered=None):
    spectrum = None
    for tree in (wrappers(options), registered or {}):
        verified = verified_spectrum(tree)
        if verified is not None:
            if spectrum is not None and verified[0] is not spectrum[0]:
                raise RuntimeError("ModeExplorer found inconsistent Spectrum bindings.")
            spectrum = verified
        for name in ("sampler_sample", "predict_noise", "calc_cond_batch", "apply_model", "diffusion_model"):
            for key, value in tree.get(name, {}).items():
                if value and key != KEY and not (key == "spectrum_krea2"
                    and name in ("sampler_sample", "predict_noise", "diffusion_model")):
                    raise RuntimeError(f"ModeExplorer needs the native {name} path except for verified Spectrum.")
    tr = options.get("transformer_options", {})
    if options.get("model_function_wrapper") or options.get("sampler_calc_cond_batch_function"):
        raise RuntimeError("ModeExplorer needs the native model and conditioning dispatch.")
    if tr.get("patches_replace") or tr.get("patches", {}).get("post_input"):
        raise RuntimeError("ModeExplorer needs the native Krea 2 image-token layout.")
    if settings.initialization and any(
        options.get(key) for key in ("sampler_cfg_function", "sampler_pre_cfg_function", "sampler_post_cfg_function")
    ):
        raise RuntimeError("ModeExplorer initialization requires native CFG callbacks.")
    if tr.get("patches") and any(
        fn is not attention_feature for functions in tr["patches"].values() for fn in functions
    ):
        raise RuntimeError("ModeExplorer needs the native attention path. Use a separate MODEL branch for other transformer patches.")
    return spectrum


def token_slice(options: dict, total: int, count: int) -> tuple[int, int]:
    start, end = options["img_slice"]
    reference_count = sum(options.get("reference_image_num_tokens", ()))
    if not 0 <= start < end == total or start + count + reference_count != end:
        raise RuntimeError("ModeExplorer could not verify the Krea 2 generated-image token slice.")
    return start, start + count


def equal_condition_pairs(batch: int, offset: int, context, attention_mask, ref_latents) -> torch.Tensor:
    """Use the actual tensors reaching the positive Krea call as equivalence keys."""
    rows = context.shape[0]
    pairs = ~torch.eye(batch, device=context.device, dtype=torch.bool)
    for tensor in (context, attention_mask, *(ref_latents or [])):
        if tensor is None or (tensor.ndim > 0 and tensor.shape[0] == 1):
            continue
        if tensor.ndim == 0 or tensor.shape[0] != rows:
            return torch.zeros_like(pairs)
        if tensor.stride(0) == 0:
            continue
        selected = tensor[offset:offset + batch].flatten(1)
        for i in range(batch - 1):
            same = (selected[i + 1:] == selected[i]).all(dim=1)
            pairs[i, i + 1:] &= same
            pairs[i + 1:, i] &= same
    return pairs


def attention_feature(output, options):
    frame = options.get(FRAME)
    if frame is None or options.get("block_type") != "single" or options.get("block_index") != BLOCK:
        return output
    if frame.seen:
        raise RuntimeError("ModeExplorer received the same Krea 2 attention callback twice in one forward.")
    frame.seen = True
    height, width = frame.grid
    start, end = token_slice(options, output.shape[1], height * width)
    numbers = frame.offset, frame.offset + frame.batch
    image = output[numbers[0]:numbers[1], start:end]
    current = token_pool(image, height, width)
    if frame.phase == "anchor":
        frame.pooled = current
        return output

    settings = frame.run.settings
    if settings.diagnostics and len(frame.run.points) < MAX_POINTS:
        frame.pooled = current
    if frame.pairs is None:
        return output
    if frame.reference is not None:
        current = torch.cat((current, frame.reference), dim=0)
    shift, redundancy, loss = feature_shift(
        current, frame.pairs, settings.feature, settings.strength, settings.fidelity,
        settings.adaptive, settings.similarity,
    )
    if settings.diagnostics and len(frame.run.points) < MAX_POINTS:
        frame.metrics = (
            redundancy[:frame.batch], loss,
            shift[:frame.batch].square().mean(dim=(1, 2, 3)).sqrt(),
        )
    result = output.clone()
    view = result[numbers[0]:numbers[1], start:end].unflatten(1, (height, width))
    sy, sx = cell_size(height, width)
    # Broadcasting per cell avoids materializing a second full attention tensor.
    for iy in range(shift.shape[-2]):
        for ix in range(shift.shape[-1]):
            delta = shift[:frame.batch, :, iy, ix].to(output.dtype)[:, None, None, :]
            view[:, iy * sy:(iy + 1) * sy, ix * sx:(ix + 1) * sx].add_(delta)
    return result


def as_list(values):
    return values.detach().float().cpu().tolist()


def finite_json(value):
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def summary(values):
    # Keep spatial cell order and channel groups. Export no full activations.
    values = values[:MAX_BATCH].movedim(1, -1)
    groups = [group.mean(dim=-1) for group in values.chunk(8, dim=-1)]
    return as_list(torch.stack(groups, dim=-1).flatten(1))


def spatial_image(latent):
    if latent.ndim == 4:
        return latent
    if latent.ndim == 5 and latent.shape[2] == 1:
        return latent.squeeze(2)
    return None


def record(frame, x, velocity):
    if frame.pooled is None or len(frame.run.points) >= MAX_POINTS:
        return
    rows = slice(frame.offset, frame.offset + frame.batch)
    x0 = x[rows].float() - frame.sigma * velocity[rows].float()
    point = {
        "kind": "trajectory", "sigma": frame.sigma,
        "attention": summary(frame.pooled - frame.pooled.mean(dim=(-1, -2), keepdim=True)),
        "conditional_x0": summary(pool(spatial_image(x0))),
    }
    if frame.metrics is not None:
        cosine, loss, rms = frame.metrics
        point.update(nearest_cosine=as_list(cosine), objective=as_list(loss), shift_rms=as_list(rms))
    frame.run.points.append(point)


def shadow_forward(executor, run, frame, x, timestep, context, mask, refs, options, kwargs):
    from comfy.model_management import throw_exception_if_processing_interrupted

    throw_exception_if_processing_interrupted()
    row = slice(frame.offset, frame.offset + 1)
    if run.shadow_noise is None:
        slot = seed_key(run.seed, "single-reference-assignment") % 8
        run.shadow_noise = reference_noise(x[row], slot + 2).float() * run.noise_scale
        run.shadow_start = frame.sigma
    shadow = frame.sigma * run.shadow_noise
    if run.shadow_clean is not None:
        shadow = shadow + (1 - frame.sigma / run.shadow_start) * run.shadow_clean
    shadow = shadow.to(x.dtype)
    anchor = Frame(frame.grid, 1, 0, "anchor", run, frame.sigma)
    positive_index = frame.offset  # single-sample frames have one row per CFG chunk
    probe_options = {
        **options, FRAME: anchor, "cond_or_uncond": [0],
        "uuids": options.get("uuids", [None] * x.shape[0])[positive_index:positive_index + 1],
    }

    def row_of(tensor):
        return tensor[row] if tensor is not None and tensor.shape[0] == x.shape[0] else tensor

    run.reference_calls += 1
    with isolated_spectrum_reference(run):
        velocity = executor(
            shadow, timestep[row], context[row], row_of(mask),
            [row_of(ref) for ref in refs] if refs is not None else None, probe_options, **kwargs,
        )
    if not anchor.seen:
        raise RuntimeError("ModeExplorer did not reach the expected Krea 2 attention hook during the reference forward.")
    if run.shadow_clean is None:
        run.shadow_clean = shadow.float() - frame.sigma * velocity.float()
    return anchor.pooled


def diffusion_wrapper(executor, x, timesteps, context, attention_mask=None, ref_latents=None,
                      transformer_options=None, **kwargs):
    options = transformer_options or {}
    run = options.get(RUN)
    arguments = (x, timesteps, context, attention_mask, ref_latents, options)
    if run is None:
        return executor(*arguments, **kwargs)
    if run.phase == "initialization":
        check_initialization_budget(run)
        run.initialization_dit_calls += 1
        return executor(*arguments, **kwargs)
    run.native_dit_calls += 1
    before_forecast = spectrum_forecast_count(run)

    def forward(args):
        result = executor(*args, **kwargs)
        after = spectrum_forecast_count(run)
        if before_forecast is not None and after is not None and after > before_forecast:
            run.spectrum_forecasts += after - before_forecast
            run.native_dit_calls -= 1
        return result

    if not (run.feature_active or run.settings.diagnostics) or spatial_image(x) is None:
        return forward(arguments)
    branches = options.get("cond_or_uncond", ())
    if 0 not in branches:
        if not branches:
            run.reasons.add("missing_condition_dispatch")
        return forward(arguments)
    if branches.count(0) != 1 or x.shape[0] != run.batch * len(branches) or tuple(x.shape[1:]) != run.shape[1:]:
        run.reasons.add("unverified_condition_dispatch")
        return forward(arguments)
    trace = run.settings.diagnostics and len(run.points) < MAX_POINTS
    may_steer = run.feature_active and (run.batch > 1 or run.reference_calls < run.settings.reference_budget)
    if not trace and not may_steer:
        return forward(arguments)
    sigmas = options.get("sigmas")
    if sigmas is None or sigmas.shape != (run.batch,):
        run.reasons.add("unverified_sigma_dispatch")
        return forward(arguments)
    # All native solver rows share the same scalar. Verify custom schedules
    # defensively so guidance never couples distinct timesteps.
    low, high = as_list(torch.stack((sigmas.amin(), sigmas.amax())))
    if not math.isfinite(low) or high != low:
        run.reasons.add("nonuniform_sigma")
        return forward(arguments)
    active = may_steer and sigma_eligible(low, run.settings.min_sigma)
    if not active and not trace:
        return forward(arguments)
    offset = branches.index(0) * run.batch
    patch = executor.class_obj.patch
    grid = tuple(math.ceil(size / patch) for size in x.shape[-2:])
    frame = Frame(grid, run.batch, offset, "sample", run, low)
    if active and run.batch == 1 and run.reference_calls < run.settings.reference_budget:
        # Use the first eligible actual evaluation even if the solver adjusts its
        # initial sigma. Do not extrapolate the shadow above its start.
        if run.shadow_start is None or low <= run.shadow_start:
            frame.reference = shadow_forward(
                executor, run, frame, x, timesteps, context, attention_mask,
                ref_latents, options, kwargs,
            )
            frame.pairs = ~torch.eye(2, device=x.device, dtype=torch.bool)
    elif active and run.batch > 1:
        frame.pairs = equal_condition_pairs(run.batch, offset, context, attention_mask, ref_latents)
    if frame.pairs is not None:
        run.feature_attempts += 1
    if frame.pairs is None and not trace:
        return forward(arguments)
    result = forward((*arguments[:-1], {**options, FRAME: frame}))
    if not frame.seen:
        if before_forecast is not None and spectrum_forecast_count(run) > before_forecast:
            return result
        raise RuntimeError("ModeExplorer did not reach the expected Krea 2 attention hook.")
    if trace:
        record(frame, x, result)
    return result


class InitializationLimit(RuntimeError):
    """An auxiliary forward would exceed the initialization budget."""


def check_initialization_budget(run):
    if run.initialization_dit_calls >= MAX_INITIALIZATION_DIT:
        raise InitializationLimit("initialization_dit_limit")
    if time.perf_counter() >= run.initialization_deadline:
        raise InitializationLimit("initialization_time_limit")


def optimize_noise(model_wrap, sigmas, options, noise, latent_image, run, evaluation_sigma=None):
    from comfy.model_management import throw_exception_if_processing_interrupted

    # Four base-sized FP32 predictions (baseline, two references and one
    # candidate) are retained at most, with a constant number of scratch latents.
    if 4 * noise.numel() > MAX_INITIALIZATION_NOISE_BYTES:
        run.reasons.add("initialization_latent_limit")
        return noise
    sampling = model_wrap.inner_model.model_sampling
    sigma = sigmas[0]
    sigma_batch = (sigma if evaluation_sigma is None else evaluation_sigma).expand(run.batch)

    def estimate(raw_noise):
        throw_exception_if_processing_interrupted()
        check_initialization_budget(run)
        run.initialization_guider_calls += 1
        x = sampling.noise_scaling(sigma, raw_noise, latent_image, True)
        with isolated_spectrum_reference(run):
            return model_wrap(x, sigma_batch, model_options=options, seed=run.seed).float()

    selected = noise
    run.phase = "initialization"
    run.initialization_deadline = time.perf_counter() + MAX_INITIALIZATION_SECONDS
    accepted = torch.zeros(run.batch, device=noise.device, dtype=torch.bool)
    attempts = []
    limit = 0.32 - 0.27 * run.settings.fidelity
    broadcast = (run.batch,) + (1,) * (noise.ndim - 1)
    try:
        clean = spatial_image(estimate(noise))
        refs = tuple(spatial_image(estimate(reference_noise(noise, i))) for i in range(2))
        direction, volume, bandwidth = initial_direction(clean, refs)
        valid = torch.stack([torch.isfinite(t).flatten(1).all(dim=1) for t in (clean, *refs)]).all(dim=0)
        valid &= clean.flatten(1).square().mean(dim=1).sqrt() > 1e-7
        radius = torch.full_like(volume, run.settings.strength * 0.18)
        for _ in range(run.settings.initialization_trials):
            active = valid & ~accepted & (radius > 1e-7)
            if not active.any():
                break
            candidate = noise_candidate(noise, direction.reshape_as(noise), radius[:, None])
            changed = (candidate != noise).flatten(1).any(dim=1) & active
            if not changed.any():
                run.reasons.add("initialization_no_representable_direction")
                break
            candidate = torch.where(changed.reshape(broadcast), candidate, noise)
            before = run.initialization_dit_calls
            candidate_clean = spatial_image(estimate(candidate))
            _, passed, gain, relative = select_candidate(
                spatial_image(noise), spatial_image(candidate), clean, candidate_clean,
                refs, volume, bandwidth, run.settings.fidelity,
            )
            newly_accepted = changed & passed
            selected = torch.where(newly_accepted.reshape(broadcast), candidate, selected)
            accepted |= newly_accepted
            if run.settings.diagnostics:
                statuses = [
                    "inactive" if not attempted else
                    "accepted" if ok else
                    "nonfinite" if not (math.isfinite(d) and math.isfinite(g)) else
                    "outside_trust_region" if d > limit else "insufficient_volume_gain"
                    for attempted, ok, d, g in zip(
                        changed.tolist(), newly_accepted.tolist(), as_list(relative), as_list(gain))
                ]
                attempts.append({
                    "rms_budget": as_list(radius), "volume_gain": as_list(gain),
                    "relative_clean_shift": as_list(relative), "status": statuses,
                    "guider_calls": 1, "dit_calls": run.initialization_dit_calls - before,
                })
            # Use measured overshoot to propose an interior radius. Shrink
            # geometrically on a rejected objective or nonfinite prediction.
            ratio = (0.8 * limit / relative).clamp(0.0625, 0.5)
            shrink = torch.where(torch.isfinite(relative) & (relative > limit), ratio, 0.5)
            radius = torch.where(changed, radius * shrink, 0)
            del candidate_clean
        return selected
    except InitializationLimit as exc:
        run.reasons.add(str(exc))
        return selected
    finally:
        run.phase = "sample"
        if run.settings.diagnostics:
            run.points.append({
                "kind": "initialization", "accepted": accepted.tolist(), "attempts": attempts,
                "relative_clean_limit": limit, "guider_calls": run.initialization_guider_calls,
                "dit_calls": run.initialization_dit_calls,
            })


class SamplerWrapper:
    def __init__(self, settings):
        self.settings = settings

    def __call__(self, executor, model_wrap, sigmas, extra_args, callback, noise,
                 latent_image=None, denoise_mask=None, disable_pbar=False):
        from comfy.model_sampling import CONST
        from comfy.samplers import CFGGuider, KSAMPLER

        def sample(args=extra_args, initial=noise):
            return executor(model_wrap, sigmas, args, callback, initial, latent_image, denoise_mask, disable_pbar)

        settings = self.settings
        model_options = extra_args["model_options"]
        if model_options.get(STAGE) == "highres" or getattr(model_wrap, "model_options", {}).get(STAGE) == "highres":
            return sample()
        spectrum = verify_patch_graph(model_options, settings)
        if type(executor.class_obj) is not KSAMPLER:
            raise RuntimeError("ModeExplorer requires the native ComfyUI KSAMPLER container.")
        if model_options.get("multigpu_clones") or model_options.get("context_handler") is not None:
            raise RuntimeError("ModeExplorer does not support multi-GPU dispatch or context windows.")
        sampling = model_wrap.inner_model.model_sampling
        if not isinstance(sampling, CONST) or any(
            getattr(type(sampling), key) is not getattr(CONST, key)
            for key in ("calculate_input", "calculate_denoised", "noise_scaling")
        ):
            raise RuntimeError("ModeExplorer requires Krea 2's native flow prediction and noise scaling.")
        if len(sigmas) <= 1:
            return sample()
        seed = extra_args.get("seed")
        start = float(sigmas[0])
        batch = noise.shape[0]
        active = settings.strength > 0 and (settings.feature != "native" or settings.initialization)
        reasons = set()
        if active:
            if spatial_image(noise) is None or not 1 <= batch <= MAX_BATCH:
                reasons.add("unsupported_latent_shape")
            if not math.isclose(start, 1.0, abs_tol=1e-5):
                reasons.add("partial_denoise")
            if denoise_mask is not None or latent_image is None or torch.count_nonzero(latent_image):
                reasons.add("requires_empty_unmasked_latent")
            if not reasons and not torch.count_nonzero(noise):
                reasons.add("noise_disabled")
            if not isinstance(seed, int):
                reasons.add("missing_seed")
            if type(model_wrap).predict_noise is not CFGGuider.predict_noise:
                reasons.add("unsupported_guider")
            conds = model_wrap.conds
            if not conds.get("positive") or any(
                len(group) != 1 or CONDITION_FLAGS.intersection(group[0])
                for group in conds.values() if group is not None
            ):
                reasons.add("unsupported_conditioning")
        if reasons:
            LOGGER.info("ModeExplorer skipped %s (seed=%s)", ", ".join(sorted(reasons)), seed)
        operation = settings.operation
        batch_allowed = (batch == 1 and operation != "batch") or (batch > 1 and operation != "sequential")
        feature_active = active and not reasons and settings.feature != "native" and batch_allowed
        initialize = active and not reasons and settings.initialization and batch_allowed
        if active and not reasons and not batch_allowed:
            reasons.add("batch_mode_mismatch")
            LOGGER.info("ModeExplorer skipped batch_mode_mismatch (seed=%s)", seed)
        if feature_active and batch == 1 and settings.reference_budget == 0:
            LOGGER.info("ModeExplorer sequential feature guidance has zero reference budget (seed=%s)", seed)
            feature_active = False
            reasons.add("zero_reference_budget")
        if not (feature_active or initialize or settings.diagnostics):
            return sample()

        scale = getattr(sampling, "noise_scale", 1.0)
        if not math.isfinite(scale) or scale <= 0:
            raise RuntimeError("ModeExplorer requires a finite positive flow noise scale.")
        run = Run(settings, int(seed or 0), start, scale, batch, tuple(noise.shape), feature_active,
                  spectrum=spectrum, reasons=reasons)
        options = {**model_options, "transformer_options": {**model_options.get("transformer_options", {}), RUN: run}}
        begin = time.perf_counter()
        if settings.diagnostics:
            run.points.append({
                "kind": "noise", "mean": float(noise.float().mean()), "rms": float(noise.float().square().mean().sqrt()),
                "schedule": as_list(sigmas[:128]), "schedule_length": len(sigmas),
            })
        completed = False
        try:
            initial = noise
            if initialize:
                from comfy.k_diffusion import sampling as native_solvers

                solver = executor.class_obj.sampler_function
                evaluation_sigma = sigmas[0]
                if solver in (getattr(native_solvers, name, None) for name in
                              ("sample_er_sde", "sample_sa_solver", "sample_sa_solver_pece")):
                    evaluation_sigma = native_solvers.offset_first_sigma_for_snr(sigmas, sampling)[0]
                if solver is native_solvers.sample_euler and executor.class_obj.extra_options.get("s_churn", 0):
                    run.reasons.add("initialization_euler_churn")
                else:
                    initial = optimize_noise(model_wrap, sigmas, options, noise, latent_image, run, evaluation_sigma)
            result = sample({**extra_args, "model_options": options}, initial)
            completed = True
            return result
        finally:
            run.shadow_noise = run.shadow_clean = None
            if settings.diagnostics:
                payload = {
                    "seed": run.seed, "completed": completed, "batch": batch,
                    "seconds": time.perf_counter() - begin, "reasons": sorted(run.reasons),
                    "native_dit_calls": run.native_dit_calls, "feature_reference_dit_calls": run.reference_calls,
                    "initialization_dit_calls": run.initialization_dit_calls,
                    "initialization_guider_calls": run.initialization_guider_calls,
                    "feature_attempts": run.feature_attempts, "spectrum_forecast_calls": run.spectrum_forecasts,
                    "spectrum_policy": ("full" if settings.full_spectrum else "guarded")
                        if spectrum is not None else "none",
                    "points": run.points,
                }
                LOGGER.info("MODEEXPLORER_TRACE %s", json.dumps(finite_json(payload), separators=(",", ":"), allow_nan=False))
