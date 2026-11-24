"""Scoped ReDiverse execution on native Krea 2.

Inference: during the admitted opening evaluations of a sampler invocation, PyTorch
forward hooks are registered on the executing transformer's modules and removed in
``finally``. Adapter weights are never merged, so quantized base weights, upstream
weight LoRAs and the loaded model stay untouched; outside the admitted evaluations
the model runs without any hook.

* ReDiverse-FT adds strength * scale * B(A x) in float32 to each adapted linear
  output and casts back, as peft does for an fp32 LoRA on a bf16 base.
* ReDiverse-TF adds sum_k alpha_k s_k u_k (computed in the basis dtype) to the
  generated-image rows of the selected block's residual-stream output, on live
  solver evaluations only. Auxiliary forwards keep the adapter but not the offset.

Research: the same block-output coordinate is captured for teacher calibration and
for activation covariance probes, through a dedicated diffusion-model wrapper.
"""

import contextlib
import hashlib
import json
import logging
import math
import os
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass, field

import torch

from . import covariance
from .rediverse import (
    ANCHOR_BLOCK, DIRECTIONS, PAPER_BAND, ROLLOUTS, ROLLOUT_SAMPLE_BASE, SCHEDULES, TEACHER_FORWARD,
    TEACHER_STEPS, UPSTREAM_REVISION, FirstEvaluations, coefficients, model_fingerprint, packed_noise,
    patch_fingerprint, reference_sigmas, resolve_module, save_directions, tensor_fingerprint,
)

LOGGER = logging.getLogger("ModeExplorer")
CAPTURE_KEY = "modeexplorer_rediverse_capture"
TF_MODES = ("random", "random_norm_matched", "cohort_logdet")
IMPLEMENTATION = "modeexplorer-rediverse-2"
_MISSING = object()
# Set on the live solver evaluation; auxiliary forwards clear it.
LIVE = ContextVar("modeexplorer_rediverse_live", default=None)
SCOPE = ContextVar("modeexplorer_rediverse_scope", default=None)
# True inside ModeExplorer auxiliary forwards (references, response probes, initialization
# estimates); set by runtime.isolated_reference. Research capture ignores them.
AUXILIARY = ContextVar("modeexplorer_auxiliary_forward", default=False)
# Transformers with an admitted ReDiverse scope. Hooks live on shared modules, so a
# second, overlapping scope on the same transformer is refused instead of mixed.
_ACTIVE = {}
_ACTIVE_LOCK = threading.Lock()


@dataclass(frozen=True)
class ReDiverseConfig:
    adapter: object = None
    strength: float = 1.0
    schedule: str = "first_evaluation"
    evaluations: int = 1
    directions: object = None
    band: float = PAPER_BAND
    count: int = DIRECTIONS
    trial: int = 0
    tf_mode: str = "random"
    diagnostics: bool = False

    def __post_init__(self):
        if self.adapter is None and self.directions is None:
            raise ValueError("ReDiverse needs an adapter, a direction archive or both.")
        if self.schedule not in SCHEDULES or type(self.evaluations) is not int or not 1 <= self.evaluations <= 4:
            raise ValueError("Unknown ReDiverse schedule.")
        if not (math.isfinite(self.strength) and 0 <= self.strength <= 2):
            raise ValueError("ReDiverse adapter strength must be in [0, 2].")
        if not (math.isfinite(self.band) and 0 <= self.band <= 8):
            raise ValueError("ReDiverse band must be in [0, 8].")
        if self.tf_mode not in TF_MODES or type(self.count) is not int or not 1 <= self.count <= 32:
            raise ValueError("Unknown ReDiverse steering configuration.")
        if self.directions is not None and self.count > self.directions.U.shape[0]:
            raise ValueError(f"The archive holds {self.directions.U.shape[0]} directions.")

    @property
    def adapts(self):
        return self.adapter is not None and self.strength != 0

    @property
    def steers(self):
        return self.directions is not None and self.band != 0

    @property
    def experimental(self):
        return self.schedule != "first_evaluation" or self.evaluations != 1 or (
            self.steers and self.tf_mode != "random")

    def describe(self):
        out = {"schedule": self.schedule, "evaluations": self.evaluations, "experimental": self.experimental}
        if self.adapter is not None:
            out.update(adapter=self.adapter.describe(), strength=self.strength)
        if self.directions is not None:
            out.update(directions=self.directions.describe(), band=self.band, count=self.count,
                       trial=self.trial, tf_mode=self.tf_mode)
        return out


@dataclass
class ReDiverseRun:
    config: ReDiverseConfig
    seed: int | None
    batch: int
    conditioning: object
    schedule: FirstEvaluations
    reasons: set = field(default_factory=set)
    adapted_calls: int = 0
    adapted_rows: int = 0
    module_calls: int = 0
    tf_applications: int = 0
    tf_rows: int = 0
    live_forwards: int = 0
    auxiliary_forwards: int = 0
    forecast_consumed: int = 0
    conditioning_checked: bool = False
    device_weights: dict = field(default_factory=dict)
    offsets: object = None
    coefficient_values: object = None
    offset_norms: object = None
    realized: list = field(default_factory=list)
    steer: bool = True
    forwards: list = field(default_factory=list)
    verify_prefix: object = None

    def pending(self, sigma):
        # Also with zero strength or band: that configuration is the route-matched
        # sham, keeping Spectrum's forced-actual boundary without an intervention.
        return self.schedule.pending(sigma)

    def release(self):
        self.device_weights.clear()
        self.offsets = None

    def summary(self):
        out = {
            "config": self.config.describe(), "seed": self.seed, "admitted_sigmas": self.schedule.seen,
            "adapted_evaluations": self.adapted_calls, "adapted_rows": self.adapted_rows,
            "adapter_module_calls": self.module_calls, "tf_applications": self.tf_applications,
            "tf_rows": self.tf_rows, "live_forwards": self.live_forwards, "auxiliary_forwards": self.auxiliary_forwards,
            "forecast_consumed": self.forecast_consumed, "reasons": sorted(self.reasons),
        }
        if self.coefficient_values is not None:
            out["coefficients"] = [[round(float(v), 5) for v in row] for row in self.coefficient_values]
            # Pre-rounding activation-space norm ||sum_k alpha_k s_k u_k|| per sample.
            out["tf_offset_norms"] = [round(float(v), 6) for v in self.offset_norms]
        if self.realized:
            out["tf_realized"] = self.realized
        return out


def start_run(config, seed, batch, conditioning, sigmas):
    """Per-invocation state. A requested TF arm that cannot be applied is an error, never a silent untreated arm."""
    boundary = float(sigmas[1]) if len(sigmas) > 1 else None
    run = ReDiverseRun(config, seed, batch, conditioning,
                       FirstEvaluations(config.schedule, config.evaluations, boundary))
    run.steer = config.steers
    if config.steers and not isinstance(seed, int):
        raise RuntimeError("ReDiverse steering needs the sampler's integer noise seed for its coefficients.")
    if config.steers and config.tf_mode == "cohort_logdet" and batch < 2:
        raise RuntimeError("cohort_logdet steering needs an equally conditioned batch of 2 to 8 images.")
    return run


def _sigma(options):
    sigmas = options.get("sigmas")
    if not torch.is_tensor(sigmas) or sigmas.numel() < 1:
        return None
    values = sigmas.detach().float().reshape(-1)
    low, high = (float(values[0]),) * 2 if values.numel() == 1 else [float(v) for v in torch.stack((values.amin(), values.amax()))]
    return low if math.isfinite(low) and low == high else None


def _lora_hook(run, down, up, scale):
    def hook(module, args, output):
        if SCOPE.get() is not run:
            return None  # an execution that does not own this scope
        run.module_calls += 1
        if run.forwards:
            run.forwards[-1]["fired"].add(id(module))
        x = args[0]
        delta = (x.to(torch.float32) @ down.T) @ up.T
        return (output.to(torch.float32) + delta * scale).to(output.dtype)
    return hook


def _image_rows(options, total, grid):
    start, end = options["img_slice"]
    count = grid[0] * grid[1]
    references = sum(options.get("reference_image_num_tokens", ()))
    if not 0 <= start < end == total or start + count + references != end:
        raise RuntimeError("ReDiverse could not verify the Krea 2 generated-image token slice.")
    return start, start + count


def activation_norm(alpha, s):
    """Pre-rounding norm of sum_k alpha_k s_k u_k for an orthonormal basis: sqrt(sum_k s_k^2 alpha_k^2)."""
    return (alpha.double().cpu().square() * s.double().cpu().square()).sum(-1).sqrt()


def expected_random_norm(s, band):
    """Root-mean-square activation norm of the uniform U[-band, band] draw: band * sqrt(sum_k s_k^2 / 3)."""
    return band * math.sqrt(float(s.double().cpu().square().sum()) / 3.0)


def match_activation_norm(direction, s, band):
    """Rescale coefficient rows so every sample's activation norm equals the random draw's RMS norm."""
    norms = activation_norm(direction, s)
    if not bool(torch.isfinite(norms).all()) or float(norms.min()) <= 1e-12:
        return None
    return (direction.double().cpu() * (expected_random_norm(s, band) / norms)[:, None]).float()


def _tf_offsets(run, device, hidden=None):
    """Per-sample block offsets [batch, tokens, d_model] in the basis dtype."""
    cfg, archive = run.config, run.config.directions
    key = ("basis", device)
    if key not in run.device_weights:
        run.device_weights[key] = (archive.U[:cfg.count].to(device), archive.s[:cfg.count].to(device))
    U, s = run.device_weights[key]
    if cfg.tf_mode == "cohort_logdet":
        if hidden is None:
            raise RuntimeError("cohort_logdet steering needs the positive branch in the first adapted evaluation.")
        alpha = cohort_coefficients(hidden, U, s, cfg.band)
        if alpha is None:
            raise RuntimeError("cohort_logdet steering found a degenerate cohort; use random steering for this batch.")
    else:
        alpha = coefficients(run.seed, run.batch, cfg.count, cfg.band, cfg.trial)
        if cfg.tf_mode == "random_norm_matched":
            # Control for cohort_logdet: random directions, identical activation norm per sample.
            alpha = match_activation_norm(alpha, s, cfg.band)
            if alpha is None:
                raise RuntimeError("The direction archive has no usable spread for a norm-matched control.")
    run.coefficient_values = alpha.cpu()
    run.offset_norms = activation_norm(alpha, s)
    weights = (alpha.to(device) * s).to(U.dtype)
    run.offsets = (weights @ U).reshape(run.batch, *archive.grid, archive.d_model).flatten(1, 2)
    return run.offsets


def cohort_coefficients(hidden, U, s, band):
    """Experimental: one log-det ascent direction for an equally conditioned cohort.

    hidden [B, tokens, d] is the student's block output, U [k, D] the teacher basis and
    s [k] its spreads. In teacher-whitened coordinates z_b = (U h_b) / s, the cohort
    covariance C has gradient of log det(C + I) proportional to (C + I)^-1 (z_b - mean z).
    The identity is the teacher's own whitened covariance, so directions in which the
    cohort already matches or exceeds the teacher are damped. Each sample's step is
    rescaled, without clipping, so its activation-space norm sqrt(sum_k s_k^2 alpha_k^2)
    equals the RMS norm of the uniform U[-band, band] draw; ``random_norm_matched`` is
    the control with exactly the same norms. Returns None for a degenerate cohort.
    """
    batch = hidden.shape[0]
    flat = hidden.reshape(batch, -1)
    usable = s > 0
    z = (flat.to(U.dtype) @ U.T).float()
    z = torch.where(usable, z / s.clamp_min(torch.finfo(torch.float32).tiny), torch.zeros_like(z))
    centred = z - z.mean(0, keepdim=True)
    k = U.shape[0]
    C = centred.T @ centred / batch
    step = torch.linalg.solve(C + torch.eye(k, device=C.device, dtype=C.dtype), centred.T).T
    step = step * usable
    if not bool(torch.isfinite(step).all()):
        return None
    return match_activation_norm(step, s, band)


def _tf_hook(run, grid, d_model):
    def hook(module, args, kwargs, output):
        live = LIVE.get()
        if live is None or live.run is not run or SCOPE.get() is not run:
            return None
        options = kwargs.get("transformer_options") or (args[5] if len(args) > 5 else {})
        if output.shape[-1] != d_model:
            raise RuntimeError("ReDiverse directions do not match the loaded model width.")
        start, end = _image_rows(options, output.shape[1], grid)
        branches = live.branches
        rows = output.shape[0]
        if rows != run.batch * len(branches):
            raise RuntimeError("ReDiverse could not verify the conditioning batch layout.")
        if run.offsets is None:
            hidden = None
            if run.config.tf_mode == "cohort_logdet" and 0 in branches:
                offset = branches.index(0) * run.batch
                hidden = output[offset:offset + run.batch, start:end]
            _tf_offsets(run, output.device, hidden)
        offsets = run.offsets.to(output.dtype)
        result = output.clone()
        view = result[:, start:end]
        for index in range(len(branches)):
            view[index * run.batch:(index + 1) * run.batch] += offsets
        run.tf_applications += 1
        run.tf_rows += rows
        if run.forwards:
            run.forwards[-1]["tf"] += 1
        if run.config.diagnostics and len(run.realized) < 8:
            # Realized displacement after native-dtype rounding, per row, next to the intended norm.
            delta = (view.float() - output[:, start:end].float()).flatten(1)
            base = output[:, start:end].float().flatten(1).norm(dim=1).clamp_min(1e-12)
            norms = delta.norm(dim=1)
            run.realized.append({"block": run.config.directions.block, "rows": rows,
                                 "branches": list(branches),
                                 "realized_offset_norms": [float(v) for v in norms.cpu()],
                                 "relative_offset_norms": [float(v) for v in (norms / base).cpu()]})
        return result
    return hook


@dataclass
class _Live:
    run: ReDiverseRun
    branches: tuple


@contextlib.contextmanager
def evaluation_scope(run, dit, x, options):
    """Install ReDiverse hooks for one admitted evaluation and verify every forward it executes.

    Execution evidence comes from the transformer itself: a pre-hook on the first block
    opens a record for each actual forward (live or auxiliary) and a hook on the last
    block closes it. Every closed forward must have executed every adapted linear
    through its module call, and every closed live forward must have received the TF
    offset exactly once; otherwise sampling stops. A scope in which no forward ran was
    replaced by a forecast. Hooks act only inside the execution context that owns the
    scope, and one transformer accepts one scope at a time.
    """
    rd = run.rediverse
    if SCOPE.get() is not None or rd.schedule.closed:
        yield
        return
    sigma = _sigma(options)
    if sigma is None:
        raise RuntimeError("ReDiverse could not read one uniform sigma for this evaluation.")
    if not rd.schedule.admit(sigma):
        rd.release()
        yield
        return
    branches = tuple(options.get("cond_or_uncond", ()))
    steer = rd.steer
    if steer and (not branches or x.shape[0] != rd.batch * len(branches)):
        raise RuntimeError("ReDiverse steering could not verify the conditioning batch layout of this evaluation "
                           "(for example a batch split under memory pressure).")
    with _ACTIVE_LOCK:
        if id(dit) in _ACTIVE:
            raise RuntimeError("Another ReDiverse evaluation is active on this transformer. Overlapping execution on "
                               "shared model modules is unsupported.")
        _ACTIVE[id(dit)] = rd
    handles = []
    forced = []
    expected = set()
    layers = []
    live_token = scope_token = None
    rd.forwards = []
    closed = []
    try:
        if rd.config.adapts:
            for block, modules in rd.config.adapter.modules.items():
                for name, (down, up, scale) in modules.items():
                    key = (block, name, x.device)
                    if key not in rd.device_weights:
                        rd.device_weights[key] = (down.to(x.device), up.to(x.device))
                    layer = resolve_module(dit.blocks[block], name)
                    layers.append((block, name, layer))
                    expected.add(id(layer))
                    handles.append(layer.register_forward_hook(
                        _lora_hook(rd, *rd.device_weights[key], scale * rd.config.strength)))
                    # As ComfyUI's bypass adapters do: fused quantized paths must call the module.
                    forced.append((layer, layer.__dict__.get("comfy_force_forward", _MISSING)))
                    layer.comfy_force_forward = True
        if steer:
            if not rd.conditioning_checked:
                rd.conditioning_checked = True
                archived = rd.config.directions.metadata["conditioning_sha256"]
                if rd.conditioning is None or tensor_fingerprint(rd.conditioning) != archived:
                    raise RuntimeError("The ReDiverse direction archive was calibrated for different positive conditioning. "
                                       "Calibrate directions for this prompt.")
            grid = tuple(math.ceil(size / dit.patch) for size in x.shape[-2:])
            if grid != rd.config.directions.grid:
                raise RuntimeError(f"The ReDiverse direction archive was calibrated for a {rd.config.directions.grid} token grid; "
                                   f"this latent has {grid}. Directions are never resized.")
            block = rd.config.directions.block
            if not 0 <= block < len(dit.blocks):
                raise RuntimeError("The ReDiverse direction block is outside the loaded model.")
            handles.append(dit.blocks[block].register_forward_hook(
                _tf_hook(rd, grid, rd.config.directions.d_model), with_kwargs=True))
        # Marks the live solver evaluation; auxiliary forwards clear it (isolated_reference).
        live_token = LIVE.set(_Live(rd, branches))

        def verify(record, upto=None):
            """Every adapted linear in blocks <= upto (all blocks when None) ran; live TF applied once."""
            missing = [f"{b}.{n}" for b, n, layer in layers
                       if (upto is None or b <= upto) and id(layer) not in record["fired"]]
            if missing:
                raise RuntimeError("ReDiverse adapter modules were not executed through their module call ("
                                   + ", ".join(missing[:4]) + f"{', ...' if len(missing) > 4 else ''}); "
                                   "the native Krea 2 forward path differs from the supported one.")
            tf_reached = steer and (upto is None or rd.config.directions.block <= upto)
            if tf_reached and record["live"] and record["tf"] != 1:
                raise RuntimeError("The ReDiverse steering hook did not apply exactly once in a live forward.")

        rd.verify_prefix = verify

        def opened(module, args):
            if SCOPE.get() is rd:
                live = LIVE.get()
                rd.forwards.append({"live": live is not None and live.run is rd, "fired": set(), "tf": 0})

        def finished(module, args, output):
            if SCOPE.get() is rd and rd.forwards:
                record = rd.forwards.pop()
                verify(record)
                closed.append(record["live"])

        # Registered after the adapter and steering hooks, so it closes after them.
        handles.append(dit.blocks[0].register_forward_pre_hook(opened))
        handles.append(dit.blocks[-1].register_forward_hook(finished))
        scope_token = SCOPE.set(rd)
        yield
        live = sum(closed)
        rd.live_forwards += live
        rd.auxiliary_forwards += len(closed) - live
        if live == 0:
            # No actual live transformer forward ran. Guarded Spectrum keeps admitted evaluations
            # actual, so with a requested intervention this is a broken invariant, not a forecast.
            if rd.config.adapts or steer:
                raise RuntimeError("An admitted ReDiverse evaluation ran no live transformer forward, so the requested "
                                   "intervention was not applied. Use guarded Spectrum with the ModeExplorer overlay.")
            rd.forecast_consumed += 1
            rd.reasons.add("admitted_evaluation_not_executed")
        else:
            rd.adapted_calls += 1
            rd.adapted_rows += x.shape[0]
    finally:
        for handle in handles:
            handle.remove()
        for layer, previous in forced:
            if previous is _MISSING:
                layer.__dict__.pop("comfy_force_forward", None)
            else:
                layer.comfy_force_forward = previous
        if scope_token is not None:
            SCOPE.reset(scope_token)
        if live_token is not None:
            LIVE.reset(live_token)
        rd.forwards = []
        rd.verify_prefix = None
        with _ACTIVE_LOCK:
            _ACTIVE.pop(id(dit), None)


@contextlib.contextmanager
def auxiliary():
    """Auxiliary forwards keep the adapted model but never receive live-sample offsets."""
    token = LIVE.set(None)
    try:
        yield
    finally:
        LIVE.reset(token)


# ------------------------------------------------------------------ research capture --
class _Stop(Exception):
    """Raised after the deepest captured block: the rest of the forward is unnecessary."""


class CaptureWrapper:
    """Diffusion-model wrapper that captures block-output image rows on selected calls.

    ``plan(call_index) -> blocks`` names the blocks captured on each 1-based diffusion
    call; ``sink(call, block, rows)`` receives [B, tokens, d] image rows; when
    ``stop_after`` is set, the forward is aborted after the deepest block of that call.
    """

    def __init__(self, plan, sink, stop_after=None):
        self.plan = plan
        self.sink = sink
        self.stop_after = stop_after
        self.calls = 0
        self.inputs = hashlib.sha256()

    def __call__(self, executor, x, timesteps, context, attention_mask=None, ref_latents=None,
                 transformer_options=None, **kwargs):
        options = transformer_options or {}
        if AUXILIARY.get() or 0 not in options.get("cond_or_uncond", (0,)):
            # ModeExplorer auxiliary forwards (references, response probes) and separately batched
            # negative branches are never captured, counted, fingerprinted or stopped.
            return executor(x, timesteps, context, attention_mask, ref_latents, options, **kwargs)
        self.calls += 1
        call = self.calls
        blocks = sorted(self.plan(call))
        if not blocks:
            return executor(x, timesteps, context, attention_mask, ref_latents, options, **kwargs)
        if ref_latents:
            raise RuntimeError("ReDiverse capture does not support reference-image conditioning.")
        dit = executor.class_obj
        grid = tuple(math.ceil(size / dit.patch) for size in x.shape[-2:])
        stop = self.stop_after is not None and call >= self.stop_after
        branches = tuple(options.get("cond_or_uncond", (0,)))
        live = x.shape[0] // len(branches)
        offset = branches.index(0) * live
        # Values as executed, in the model's input dtype, then widened: equal digests mean equal states.
        rows = x[offset:offset + live].detach().to("cpu", torch.float32).contiguous()
        header = json.dumps([list(rows.shape[1:]), float(timesteps.reshape(-1)[0])]).encode()
        for row in rows:  # per row, so the digest does not depend on batch chunking
            self.inputs.update(header)
            self.inputs.update(row.numpy().tobytes())
        seen = set()

        def make(block):
            def hook(module, args, kwargs, output):
                trans = kwargs.get("transformer_options") or {}
                start, end = _image_rows(trans, output.shape[1], grid)
                self.sink(call, block, output[:, start:end].detach(), options)
                seen.add(block)
                if stop and block == blocks[-1]:
                    # The forward ends here: verify the adapter on the executed block prefix.
                    owner = SCOPE.get()
                    if owner is not None and owner.forwards and owner.verify_prefix is not None:
                        owner.verify_prefix(owner.forwards[-1], upto=block)
                    raise _Stop
            return hook

        handles = [dit.blocks[block].register_forward_hook(make(block), with_kwargs=True) for block in blocks]
        try:
            result = executor(x, timesteps, context, attention_mask, ref_latents, options, **kwargs)
        except _Stop:
            raise
        finally:
            for handle in handles:
                handle.remove()
        if seen != set(blocks):
            raise RuntimeError("ReDiverse capture hooks did not fire; the executing model differs from the patched one.")
        return result


def conditioning_fingerprint(positive):
    """Fingerprint of the single positive text conditioning a direction archive belongs to."""
    if len(positive) != 1 or not torch.is_tensor(positive[0][0]):
        raise ValueError("ReDiverse calibration needs exactly one positive text conditioning.")
    flags = {"area", "mask", "control", "hooks", "gligen", "reference_latents", "start_percent", "end_percent"}
    if flags.intersection(positive[0][1]):
        raise ValueError("ReDiverse calibration does not support area, mask, control, hook or reference conditioning.")
    return tensor_fingerprint(positive[0][0])


def _guider(model, positive, negative, cfg):
    import comfy.samplers

    guider = comfy.samplers.CFGGuider(model)
    if negative is None:
        guider.inner_set_conds({"positive": positive})
    else:
        guider.set_conds(positive, negative)
    guider.set_cfg(cfg)
    return guider


def _noise_batch(seeds, shape):
    """Reference packed draws in the latent layout of ``shape`` ([C, H, W] or [C, 1, H, W])."""
    rows = []
    for seed in seeds:
        noise = packed_noise(seed, shape[-2], shape[-1], channels=shape[0])
        rows.append(noise.reshape(shape))
    return torch.stack(rows)


def _latent_shape(latent, model):
    """Per-sample model latent shape, after ComfyUI's native empty-latent channel fix."""
    import comfy.sample

    samples = comfy.sample.fix_empty_latent_channels(model, latent["samples"], latent.get("downscale_ratio_spacial"),
                                                     latent.get("downscale_ratio_temporal"))
    if samples.ndim == 5 and samples.shape[2] == 1:
        return tuple(samples.shape[1:])
    if samples.ndim == 4:
        return tuple(samples.shape[1:])
    raise ValueError("ReDiverse calibration needs a single-frame image latent.")


def _verify_research_model(model):
    from .nodes import ModeExplorer
    from .runtime import has_spectrum

    ModeExplorer.verify_model(model, 0)
    if has_spectrum(model.model_options, model.wrappers):
        raise ValueError("Run ReDiverse calibration and probes on a MODEL without Spectrum.")


def _attached_settings(model):
    from comfy.patcher_extension import WrappersMP

    from .runtime import KEY

    samplers = model.get_wrappers(WrappersMP.SAMPLER_SAMPLE, KEY)
    return dict(vars(samplers[0].settings)) if samplers else None


def _attached_rediverse(model):
    from comfy.patcher_extension import WrappersMP

    from .runtime import KEY

    samplers = model.get_wrappers(WrappersMP.SAMPLER_SAMPLE, KEY)
    config = getattr(samplers[0], "rediverse", None) if samplers else None
    return config.describe() if config is not None else None


def _workspace_device(choice):
    import comfy.model_management

    return comfy.model_management.get_torch_device() if choice == "auto" else torch.device(choice)


GRAM_CHUNK = 1 << 20


def research_workspace_bytes(n, features, k=0, staged_blocks=1, reference=False):
    """Peak research allocations (staging and analysis temporaries), excluding model weights and activations.

    staging: staged_blocks * n * features bf16; Gram/projection chunk: two float64 n x chunk
    copies; column mean: features float32; PCA reconstruction: k x features float32 plus
    its bf16 copy; a reference basis on the workspace device: k x features bf16.
    """
    chunk = min(GRAM_CHUNK, features)
    total = staged_blocks * n * features * 2 + 2 * n * chunk * 8 + features * 4
    total += k * features * (4 + 2)
    if reference:
        total += k * features * 2
    return total


def _check_budget(need, budget_mib, what):
    if need > budget_mib * (1 << 20):
        raise ValueError(f"{what} needs about {need / (1 << 20):.0f} MiB of research workspace (staging and analysis "
                         f"temporaries, excluding model weights and activations); the limit is {budget_mib} MiB. "
                         "Reduce samples, blocks or resolution, or raise the limit if the device has room.")


def _unique_path(directory, stem, suffix):
    """Never replace an existing archive: repeated executions get -r2, -r3, ..."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, stem + suffix)
    index = 2
    while os.path.exists(path):
        path = os.path.join(directory, f"{stem}-r{index}{suffix}")
        index += 1
    return path


def _canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _noise_digest(seeds, shape):
    digest = hashlib.sha256(json.dumps([list(shape), seeds]).encode())
    for seed in seeds:
        digest.update(packed_noise(seed, shape[-2], shape[-1], channels=shape[0]).numpy().tobytes())
    return digest.hexdigest()


def _validate_schedule(schedule, forward):
    if (len(schedule) < forward + 1 or not all(math.isfinite(v) for v in schedule)
            or abs(schedule[0] - 1.0) > 1e-5 or not all(0.0 <= v <= 1.0 + 1e-5 for v in schedule)
            or any(a <= b for a, b in zip(schedule, schedule[1:]))):
        raise ValueError("Connected teacher sigmas must be finite, start at 1, stay in [0, 1], decrease strictly "
                         "and cover the requested teacher forward.")


def _model_options_summary(model):
    options = model.model_options
    trans = options.get("transformer_options", {})
    return {
        "wrappers": sorted(f"{kind}:{key}" for kind, by_key in model.wrappers.items() for key, items in by_key.items() if items),
        "transformer_patches": sorted(trans.get("patches", {})),
        "attention_override": getattr(trans.get("optimized_attention_override"), "__qualname__", None),
        "model_function_wrapper": options.get("model_function_wrapper") is not None,
    }


def calibrate(teacher, positive, latent, *, negative=None, guidance=1.0, steps=TEACHER_STEPS, forward=TEACHER_FORWARD,
              block=ANCHOR_BLOCK, rollouts=ROLLOUTS, directions=DIRECTIONS, seed=0, batch=5, sigmas=None,
              label="", device="auto", workspace_mib=8192, store_mean=True):
    """Estimate ReDiverse-TF directions from teacher rollouts (paper Sec. 4.2).

    Rollout j (0-based) starts from the packed reference draw seeded seed * 1000 + 100 + j.
    Forwards 1..forward-1 are full Euler steps on the teacher schedule; forward ``forward``
    is captured at the block output and stopped there. The reference is conditional-only;
    guided calibration needs an explicit negative conditioning. Returns (path, Directions,
    timings); the archive stores rollout and PCA seconds, the log line and the returned
    timings add archive write/validation and total wall time.
    """
    import comfy.model_management
    import comfy.samplers
    from comfy.patcher_extension import WrappersMP

    from .rediverse import ARCHIVE_SCHEMA, archive_root, load_directions

    total_start = time.perf_counter()
    _verify_research_model(teacher)
    if teacher.get_wrappers(WrappersMP.SAMPLER_SAMPLE, "mode_explorer_krea2"):
        raise ValueError("Calibrate on the plain teacher MODEL, without ModeExplorer or ReDiverse nodes.")
    if not math.isfinite(guidance) or guidance < 1:
        raise ValueError("Calibration CFG must be finite and at least 1.")
    if guidance != 1 and negative is None:
        raise ValueError("Guided calibration (CFG above 1) needs a negative conditioning.")
    if guidance == 1:
        negative = None  # CFG 1 evaluates the positive branch only.
    dit = teacher.get_model_object("diffusion_model")
    if not 0 <= block < len(dit.blocks):
        raise ValueError(f"Block {block} is outside the teacher's {len(dit.blocks)} blocks.")
    if not 1 <= forward <= steps or rollouts < 2 or not 1 <= directions <= rollouts:
        raise ValueError("Calibration needs 1 <= forward <= steps and 1 <= directions <= rollouts.")
    shape = _latent_shape(latent, teacher)
    grid = (math.ceil(shape[-2] / dit.patch), math.ceil(shape[-1] / dit.patch))
    tokens = grid[0] * grid[1]
    width = dit.blocks[block].attn.wo.out_features
    if sigmas is None:
        schedule = reference_sigmas(steps, tokens)
        source = "reference_flow_match_euler"
    else:
        schedule = [float(v) for v in sigmas]
        source = "connected_sigmas"
        _validate_schedule(schedule, forward)
    features = tokens * width
    _check_budget(research_workspace_bytes(rollouts, features, directions), workspace_mib, "This calibration")
    run_sigmas = torch.tensor(schedule[:forward + 1], dtype=torch.float32)
    workspace = _workspace_device(device)
    X = torch.empty(rollouts, features, dtype=torch.bfloat16, device=workspace)
    seeds = [seed * 1000 + ROLLOUT_SAMPLE_BASE + j for j in range(rollouts)]
    state = {"filled": 0}

    def sink(call, captured_block, rows, options):
        branches = tuple(options.get("cond_or_uncond", ()))
        live = rows.shape[0] // len(branches)
        offset = branches.index(0) * live
        X[state["filled"]:state["filled"] + live] = rows[offset:offset + live].reshape(live, -1).to(workspace, torch.bfloat16)
        state["filled"] += live

    clone = teacher.clone()
    wrapper = CaptureWrapper(lambda call: (block,) if call == forward else (), sink, stop_after=forward)
    clone.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, CAPTURE_KEY, wrapper)
    guider = _guider(clone, positive, negative, guidance)
    fingerprint = conditioning_fingerprint(positive)
    negative_fingerprint = conditioning_fingerprint(negative) if negative is not None else None
    sampler = comfy.samplers.ksampler("euler")
    rollout_start = time.perf_counter()
    for start in range(0, rollouts, batch):
        comfy.model_management.throw_exception_if_processing_interrupted()
        chunk = seeds[start:start + batch]
        noise = _noise_batch(chunk, shape)
        before = state["filled"]
        wrapper.calls = 0
        try:
            guider.sample(noise, torch.zeros_like(noise), sampler, run_sigmas, disable_pbar=True, seed=chunk[0])
            raise RuntimeError("The calibration rollout finished without reaching the requested teacher forward.")
        except _Stop:
            pass
        if state["filled"] - before != len(chunk):
            raise RuntimeError("Calibration captured an unexpected number of rollouts.")
    rollout_seconds = time.perf_counter() - rollout_start
    pca_start = time.perf_counter()
    G = covariance.centred_gram(X, GRAM_CHUNK)
    values, vectors = covariance.eigenpairs(G)
    U, s, valid = covariance.principal_basis(X, vectors, values, directions, GRAM_CHUNK)
    if valid < 1:
        raise RuntimeError("Teacher rollouts produced no variance at the requested block.")
    mean = covariance.column_mean(X, GRAM_CHUNK).to(torch.bfloat16).cpu() if store_mean else None
    stats = covariance.spectrum_stats(values, rollouts)
    U = U.cpu()
    del X
    pca_seconds = time.perf_counter() - pca_start
    model_hash = model_fingerprint(dit)
    sampling = teacher.get_model_object("model_sampling")
    metadata = {
        "schema": ARCHIVE_SCHEMA, "implementation": IMPLEMENTATION, "upstream_revision": UPSTREAM_REVISION,
        "coordinate": "block_output_image_rows", "block": block, "token_grid": list(grid), "d_model": width,
        "latent_shape": list(shape), "teacher_model_sha256": model_hash,
        "teacher_model_fingerprint": "sampled: raw values of fixed slices of five named tensors; not a full checkpoint hash",
        "teacher_patches": patch_fingerprint(teacher),
        "teacher_patch_fingerprint": "sampled: keys, strengths and the first 256 values of each patch tensor",
        "teacher_model_options": _model_options_summary(teacher), "teacher_blocks": len(dit.blocks),
        "teacher_noise_scale": float(getattr(sampling, "noise_scale", 1.0)),
        "conditioning_sha256": fingerprint, "negative_conditioning_sha256": negative_fingerprint,
        "sigma_in": schedule[forward - 1], "sigma_out": schedule[forward], "teacher_forward": forward,
        "teacher_steps": len(schedule) - 1, "schedule_source": source, "teacher_sigmas": schedule,
        "executed_sigmas": schedule[:forward + 1], "solver": "euler",
        "guidance": {"cfg": guidance, "branches": "conditional_only" if negative is None else "guided"},
        "seeds": seeds, "seed_rule": "seed * 1000 + 100 + j, packed CPU randn", "rollouts": rollouts, "batch": batch,
        "initial_noise_sha256": _noise_digest(seeds, shape),
        "capture_input_sha256": wrapper.inputs.hexdigest(),
        "normalization": "eigenvalues of the unnormalized centred Gram; covariance eigenvalue = value / rollouts; "
                         "s_k = sqrt(value_k / rollouts)",
        "staging_dtype": "bfloat16", "basis_dtype": "bfloat16", "directions": len(s), "valid_directions": valid,
        "pc1_share": stats["pc1_share"], "participation_ratio": stats["pr"], "total_variance": stats["tv"],
        "label": label, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    volatile = {"created", "label", "pc1_share", "participation_ratio", "total_variance", "valid_directions",
                "capture_input_sha256", "timings"}
    identity = _canonical_hash({key: value for key, value in metadata.items() if key not in volatile})[:8]
    metadata["identity_sha256"] = identity
    stem = f"{label + '-' if label else ''}{(fingerprint or 'nocond')[:12]}-{model_hash[:8]}-b{block}-f{forward}-n{rollouts}-{identity}"
    write_start = time.perf_counter()
    path = _unique_path(os.path.join(archive_root(), "directions"), stem, ".safetensors")
    metadata["timings"] = {"rollout_seconds": rollout_seconds, "pca_seconds": pca_seconds}
    save_directions(path, U, s, values.float().cpu(), metadata, mean)
    timings = dict(metadata["timings"])
    loaded = load_directions(path)
    timings.update(write_and_validate_seconds=time.perf_counter() - write_start,
                   total_seconds=time.perf_counter() - total_start)
    LOGGER.info("ModeExplorer ReDiverse calibration: %s", json.dumps({
        "file": os.path.basename(path), "valid_directions": valid, "pc1_share": stats["pc1_share"],
        **{key: round(value, 3) for key, value in timings.items()}}))
    return path, loaded, timings


PROBE_CATEGORIES = ("exact_matched_noisy_state", "matched_noise_state_representation_differs",
                    "matched_initial_noise_evolved_trajectories", "same_conditioning_independent_samples",
                    "cross_conditioning_exploratory")


def comparison_category(a, b):
    """Qualify a comparison between two captured cohorts from their recorded evidence.

    Each side provides conditioning_sha256, initial_noise_sha256, capture_input_sha256 (the
    executed model inputs at the captured forward), sigma and noise_scale; equal sigma alone
    never certifies a matched state.
    """
    if a.get("conditioning_sha256") is None or a.get("conditioning_sha256") != b.get("conditioning_sha256"):
        return "cross_conditioning_exploratory"
    if a.get("capture_input_sha256") and a.get("capture_input_sha256") == b.get("capture_input_sha256"):
        return "exact_matched_noisy_state"
    if a.get("initial_noise_sha256") and a.get("initial_noise_sha256") == b.get("initial_noise_sha256"):
        if (a.get("forward", 1) == 1 and b.get("forward", 1) == 1 and a.get("sigma") == b.get("sigma")
                and a.get("noise_scale") == b.get("noise_scale")):
            return "matched_noise_state_representation_differs"
        return "matched_initial_noise_evolved_trajectories"
    return "same_conditioning_independent_samples"


def _archive_side(archive):
    meta = archive.metadata
    return {"conditioning_sha256": meta.get("conditioning_sha256"), "initial_noise_sha256": meta.get("initial_noise_sha256"),
            "capture_input_sha256": meta.get("capture_input_sha256"), "sigma": meta.get("sigma_in"),
            "noise_scale": meta.get("teacher_noise_scale"), "forward": meta.get("teacher_forward")}


def probe(model, positive, latent, *, blocks=(ANCHOR_BLOCK,), samples=200, seed=0, sigma=1.0, batch=8,
          reference=None, device="auto", workspace_mib=16384, label=""):
    """Activation covariance of block-output image rows at one frozen first-step state.

    Every sample starts from the packed reference draw 1_000_000 + seed * 100003 + i at
    ``sigma`` and runs one conditional forward. Blocks are processed in groups that fit
    ``workspace_mib`` (staging and analysis temporaries); each group repeats the forwards
    and stops after its deepest block. With a reference direction archive at the same
    coordinate and geometry, the cohort is also compared in the reference subspace, and
    the comparison is qualified from recorded input evidence.
    """
    import comfy.model_management
    import comfy.samplers
    from comfy.patcher_extension import WrappersMP

    from .rediverse import archive_root

    _verify_research_model(model)
    dit = model.get_model_object("diffusion_model")
    blocks = sorted(set(int(b) for b in blocks))
    if not blocks or blocks[0] < 0 or blocks[-1] >= len(dit.blocks):
        raise ValueError("Probe blocks are outside the loaded model.")
    if not 0 < sigma <= 1:
        raise ValueError("Probe sigma must be in (0, 1].")
    attached = _attached_rediverse(model)
    if attached is not None and sigma != 1.0:
        raise ValueError("A probe through ReDiverse must start at sigma 1; other sigmas bypass the adapter.")
    shape = _latent_shape(latent, model)
    grid = (math.ceil(shape[-2] / dit.patch), math.ceil(shape[-1] / dit.patch))
    width = dit.blocks[0].attn.wo.out_features
    features = grid[0] * grid[1] * width
    k = reference.U.shape[0] if reference is not None else 0
    analysis = research_workspace_bytes(samples, features, 0, 0, reference is not None) + k * features * 4
    per_block = samples * features * 2
    _check_budget(per_block + analysis, workspace_mib, "Probing one block")
    group_size = max(1, (workspace_mib * (1 << 20) - analysis) // per_block)
    groups = [blocks[i:i + group_size] for i in range(0, len(blocks), group_size)]
    workspace = _workspace_device(device)
    seeds = [1_000_000 + seed * 100_003 + i for i in range(samples)]
    run_sigmas = torch.tensor([sigma, 0.0], dtype=torch.float32)
    sampling = model.get_model_object("model_sampling")
    report = {"schema": 2, "implementation": IMPLEMENTATION, "coordinate": "block_output_image_rows",
              "state": "first_forward_from_noise", "sigma": sigma, "samples": samples, "seed": seed,
              "seed_rule": "1000000 + seed * 100003 + i, packed CPU randn", "batch": batch,
              "noise_scale": float(getattr(sampling, "noise_scale", 1.0)),
              "initial_noise_sha256": _noise_digest(seeds, shape), "latent_shape": list(shape),
              "token_grid": list(grid), "d_model": width, "model_sha256": model_fingerprint(dit),
              "model_fingerprint": "sampled: raw values of fixed slices of five named tensors; not a full checkpoint hash",
              "patches": patch_fingerprint(model), "model_options": _model_options_summary(model),
              "rediverse": attached, "modeexplorer": _attached_settings(model),
              "conditioning_sha256": conditioning_fingerprint(positive), "requested_blocks": blocks,
              "reference_archive": reference.describe() if reference is not None else None,
              "workspace_device": workspace.type, "staging_dtype": "bfloat16",
              "blocks": {}, "label": label, "groups": [list(g) for g in groups]}
    started = time.perf_counter()
    sampler = comfy.samplers.ksampler("euler")
    digests = set()
    for group in groups:
        staged = {b: torch.empty(samples, features, dtype=torch.bfloat16, device=workspace) for b in group}
        filled = {b: 0 for b in group}

        def sink(call, block, rows, options, staged=staged, filled=filled):
            branches = tuple(options.get("cond_or_uncond", ()))
            live = rows.shape[0] // len(branches)
            offset = branches.index(0) * live
            staged[block][filled[block]:filled[block] + live] = rows[offset:offset + live].reshape(live, -1).to(workspace, torch.bfloat16)
            filled[block] += live

        clone = model.clone()
        wrapper = CaptureWrapper(lambda call, group=group: tuple(group) if call == 1 else (), sink, stop_after=1)
        clone.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, CAPTURE_KEY, wrapper)
        guider = _guider(clone, positive, None, 1.0)
        for start in range(0, samples, batch):
            comfy.model_management.throw_exception_if_processing_interrupted()
            chunk = seeds[start:start + batch]
            noise = _noise_batch(chunk, shape)
            wrapper.calls = 0
            try:
                guider.sample(noise, torch.zeros_like(noise), sampler, run_sigmas, disable_pbar=True, seed=chunk[0])
            except _Stop:
                pass
        digests.add(wrapper.inputs.hexdigest())
        for block in group:
            if filled[block] != samples:
                raise RuntimeError("The probe captured an unexpected number of samples.")
            X = staged.pop(block)
            G = covariance.centred_gram(X, GRAM_CHUNK)
            values, vectors = covariance.eigenpairs(G)
            entry = covariance.spectrum_stats(values, samples)
            if reference is not None and reference.block == block:
                if reference.grid != grid or reference.d_model != width:
                    raise ValueError("The reference directions use another geometry.")
                if reference.metadata.get("coordinate") != report["coordinate"]:
                    raise ValueError("The reference directions use another coordinate.")
                U = reference.U.to(workspace)
                P = covariance.project(X, U, chunk=GRAM_CHUNK)
                entry["reference"] = covariance.subspace_comparison(reference.s, P, values, vectors)
                entry["reference"]["archive"] = reference.describe()
                if reference.mean is not None:
                    displacement = covariance.column_mean(X, GRAM_CHUNK) - reference.mean.to(workspace, torch.float32)
                    along = torch.stack([U[i].float() @ displacement for i in range(U.shape[0])])
                    entry["reference"]["mean_displacement_rms"] = float(displacement.square().mean().sqrt())
                    entry["reference"]["mean_displacement_along_directions"] = [float(v) for v in along.cpu()]
                    entry["reference"]["mean_displacement_relative_to_cohort_rms"] = float(
                        displacement.norm() / max(math.sqrt(entry["tv"]), 1e-30))
                del U, P
            report["blocks"][f"block{block:02d}"] = entry
            del X
    if len(digests) != 1:
        raise RuntimeError("Repeated probe forwards received different inputs across block groups.")
    report["capture_input_sha256"] = digests.pop()
    if reference is not None:
        category = comparison_category(_archive_side(reference), {**report, "forward": 1})
        for entry in report["blocks"].values():
            if "reference" in entry:
                entry["reference"]["comparison"] = category
        report["reference_comparison"] = category
    report["seconds"] = time.perf_counter() - started
    identity = _canonical_hash({key: report[key] for key in (
        "implementation", "coordinate", "state", "sigma", "samples", "seed", "batch", "noise_scale",
        "initial_noise_sha256", "latent_shape", "token_grid", "d_model", "model_sha256", "patches", "model_options",
        "rediverse", "modeexplorer", "conditioning_sha256", "requested_blocks", "reference_archive",
        "workspace_device")})[:8]
    report["identity_sha256"] = identity
    stem = (f"{label + '-' if label else ''}{report['model_sha256'][:8]}-{report['conditioning_sha256'][:12]}"
            f"-n{samples}-s{sigma:g}-{identity}")
    path = _unique_path(os.path.join(archive_root(), "probes"), stem, ".json")
    with open(path + ".tmp", "w") as handle:
        json.dump(report, handle, indent=1)
    os.replace(path + ".tmp", path)
    return path, report


def compare_probes(teacher, student):
    """Per-block student/teacher ratios and the upstream collapsed-block rule (App. A).

    ``comparison`` qualifies the pair from recorded input evidence (see comparison_category);
    ``matched_inputs`` is true only for an exact matched noisy state. Ratios of unmatched
    cohorts compare populations, not states.
    """
    rows = {}
    for name in sorted(set(teacher["blocks"]) & set(student["blocks"])):
        t, s = teacher["blocks"][name], student["blocks"][name]
        ratio = lambda key: (s[key] / t[key]) if t.get(key) and s.get(key) is not None else None
        rows[name] = {"pr_ratio": ratio("pr"), "tv_ratio": ratio("tv"), "conc_ratio": ratio("conc"),
                      "entropy_rank_ratio": ratio("entropy_rank")}
    names = list(rows)
    steps = [(names[i], rows[names[i]]["conc_ratio"] - rows[names[i - 1]]["conc_ratio"]) for i in range(1, len(names))
             if rows[names[i]]["conc_ratio"] is not None and rows[names[i - 1]]["conc_ratio"] is not None]
    geometry = all(teacher.get(key) == student.get(key) and teacher.get(key) is not None
                   for key in ("coordinate", "token_grid", "d_model", "samples"))
    category = comparison_category(teacher, student) if geometry else "incompatible_geometry"
    return {"blocks": rows, "largest_concentration_step": max(steps, key=lambda v: v[1])[0] if steps else None,
            "comparison": category, "matched_inputs": category == "exact_matched_noisy_state",
            "caveat": "Ratios use N-sample spectra; participation ratios are capped at N - 1."}
