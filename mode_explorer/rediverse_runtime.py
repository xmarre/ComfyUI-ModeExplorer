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
import json
import logging
import math
import os
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
TF_MODES = ("random", "cohort_logdet")
IMPLEMENTATION = "modeexplorer-rediverse-1"
# Set on the live solver evaluation; auxiliary forwards clear it.
LIVE = ContextVar("modeexplorer_rediverse_live", default=None)
SCOPE = ContextVar("modeexplorer_rediverse_scope", default=None)


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
        return self.schedule != "first_evaluation" or self.evaluations != 1 or self.tf_mode != "random"

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
    forecast_consumed: int = 0
    conditioning_checked: bool = False
    device_weights: dict = field(default_factory=dict)
    offsets: object = None
    coefficient_values: object = None
    realized: list = field(default_factory=list)
    steer: bool = True

    def pending(self, sigma):
        return (self.config.adapts or self.steer) and self.schedule.pending(sigma)

    def release(self):
        self.device_weights.clear()
        self.offsets = None

    def summary(self):
        out = {
            "config": self.config.describe(), "seed": self.seed, "admitted_sigmas": self.schedule.seen,
            "adapted_evaluations": self.adapted_calls, "adapted_rows": self.adapted_rows,
            "adapter_module_calls": self.module_calls, "tf_applications": self.tf_applications,
            "tf_rows": self.tf_rows, "forecast_consumed": self.forecast_consumed, "reasons": sorted(self.reasons),
        }
        if self.coefficient_values is not None:
            out["coefficients"] = [[round(float(v), 5) for v in row] for row in self.coefficient_values]
        if self.realized:
            out["tf_realized"] = self.realized
        return out


def start_run(config, seed, batch, conditioning, sigmas):
    boundary = float(sigmas[1]) if len(sigmas) > 1 else None
    run = ReDiverseRun(config, seed, batch, conditioning,
                       FirstEvaluations(config.schedule, config.evaluations, boundary))
    run.steer = config.steers
    if config.steers and not isinstance(seed, int):
        run.reasons.add("tf_missing_seed")
        run.steer = False
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
        run.module_calls += 1
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


def _tf_offsets(run, device, live_rows, hidden=None, positive=None):
    """Per-sample block offsets [batch, tokens, d_model] in the basis dtype."""
    cfg, archive = run.config, run.config.directions
    key = ("basis", device)
    if key not in run.device_weights:
        run.device_weights[key] = (archive.U[:cfg.count].to(device), archive.s[:cfg.count].to(device))
    U, s = run.device_weights[key]
    if cfg.tf_mode == "cohort_logdet" and hidden is not None and positive is not None and run.batch > 1:
        alpha = cohort_coefficients(hidden[positive], U, s, cfg.band)
        if alpha is None:
            run.reasons.add("tf_cohort_degenerate_random_fallback")
    else:
        alpha = None
        if cfg.tf_mode == "cohort_logdet":
            run.reasons.add("tf_cohort_requires_batch_random_fallback")
    if alpha is None:
        alpha = coefficients(run.seed, run.batch, cfg.count, cfg.band, cfg.trial)
    run.coefficient_values = alpha.cpu()
    weights = (alpha.to(device) * s).to(U.dtype)
    run.offsets = (weights @ U).reshape(run.batch, *archive.grid, archive.d_model).flatten(1, 2)
    return run.offsets


def cohort_coefficients(hidden, U, s, band):
    """Experimental: one projected log-det ascent step for an equally conditioned cohort.

    hidden [B, tokens, d] is the student's block output, U [k, D] the teacher basis and
    s [k] its spreads. In teacher-whitened coordinates z_b = (U h_b) / s, the cohort
    covariance C has gradient of log det(C + I) proportional to (C + I)^-1 (z_b - mean z).
    The identity is the teacher's own whitened covariance, so directions in which the
    cohort already matches or exceeds the teacher are damped. Each step is rescaled to
    the root-mean-square norm of a uniform U[-band, band] draw, band * sqrt(k / 3), and
    clipped per coordinate to the band. Returns None for a degenerate cohort.
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
    norms = step.norm(dim=1, keepdim=True)
    if not bool(torch.isfinite(step).all()) or float(norms.min()) <= 1e-12:
        return None
    radius = band * math.sqrt(int(usable.sum()) / 3.0)
    return (step / norms * radius).clamp(-band, band).cpu()


def _tf_hook(run, grid, d_model):
    def hook(module, args, kwargs, output):
        live = LIVE.get()
        if live is None or live.run is not run:
            return None
        options = kwargs.get("transformer_options") or (args[5] if len(args) > 5 else {})
        if output.shape[-1] != d_model:
            raise RuntimeError("ReDiverse directions do not match the loaded model width.")
        start, end = _image_rows(options, output.shape[1], grid)
        branches = live.branches
        rows = output.shape[0]
        if rows != run.batch * len(branches):
            raise RuntimeError("ReDiverse could not verify the conditioning batch layout.")
        hidden = None
        positive = None
        if run.offsets is None:
            if run.config.tf_mode == "cohort_logdet" and 0 in branches:
                offset = branches.index(0) * run.batch
                positive = slice(offset, offset + run.batch)
                hidden = output[:, start:end]
            _tf_offsets(run, output.device, rows, hidden, positive)
        offsets = run.offsets.to(output.dtype)
        result = output.clone()
        view = result[:, start:end]
        for index in range(len(branches)):
            view[index * run.batch:(index + 1) * run.batch] += offsets
        run.tf_applications += 1
        run.tf_rows += rows
        if run.config.diagnostics and len(run.realized) < 8:
            delta = (view.float() - output[:, start:end].float())
            base = output[:, start:end].float().square().mean().sqrt().clamp_min(1e-12)
            rms = delta.square().mean().sqrt()
            run.realized.append({"block": run.config.directions.block, "offset_rms": float(rms),
                                 "relative_rms": float(rms / base), "rows": rows})
        return result
    return hook


@dataclass
class _Live:
    run: ReDiverseRun
    branches: tuple


@contextlib.contextmanager
def evaluation_scope(run, dit, x, options):
    """Install ReDiverse hooks for one admitted live evaluation."""
    rd = run.rediverse
    if SCOPE.get() is not None or rd.schedule.closed:
        yield
        return
    sigma = _sigma(options)
    if sigma is None:
        rd.reasons.add("unverified_sigma_dispatch")
        rd.schedule.closed = True
        yield
        return
    if not rd.schedule.admit(sigma):
        rd.release()
        yield
        return
    handles = []
    live_token = scope_token = None
    branches = tuple(options.get("cond_or_uncond", ()))
    try:
        if rd.config.adapts:
            for block, modules in rd.config.adapter.modules.items():
                for name, (down, up, scale) in modules.items():
                    key = (block, name, x.device)
                    if key not in rd.device_weights:
                        rd.device_weights[key] = (down.to(x.device), up.to(x.device))
                    weights = rd.device_weights[key]
                    layer = resolve_module(dit.blocks[block], name)
                    handles.append(layer.register_forward_hook(
                        _lora_hook(rd, *weights, scale * rd.config.strength)))
        steer = rd.steer and branches and x.shape[0] == rd.batch * len(branches)
        if rd.steer and not steer:
            rd.reasons.add("tf_unverified_condition_dispatch")
        if steer:
            if not rd.conditioning_checked:
                rd.conditioning_checked = True
                expected = rd.config.directions.metadata["conditioning_sha256"]
                if rd.conditioning is None or tensor_fingerprint(rd.conditioning) != expected:
                    raise RuntimeError("The ReDiverse direction archive was calibrated for different positive conditioning. "
                                       "Calibrate directions for this prompt.")
            patch = dit.patch
            grid = tuple(math.ceil(size / patch) for size in x.shape[-2:])
            if grid != rd.config.directions.grid:
                raise RuntimeError(f"The ReDiverse direction archive was calibrated for a {rd.config.directions.grid} token grid; "
                                   f"this latent has {grid}. Directions are never resized.")
            block = rd.config.directions.block
            if not 0 <= block < len(dit.blocks):
                raise RuntimeError("The ReDiverse direction block is outside the loaded model.")
            handles.append(dit.blocks[block].register_forward_hook(
                _tf_hook(rd, grid, rd.config.directions.d_model), with_kwargs=True))
            live_token = LIVE.set(_Live(rd, branches))
        scope_token = SCOPE.set(rd)
        before = rd.module_calls, rd.tf_applications
        yield
        rd.adapted_calls += 1
        rd.adapted_rows += x.shape[0]
        if (rd.config.adapts and rd.module_calls == before[0]) or (steer and rd.tf_applications == before[1]):
            rd.forecast_consumed += 1
            rd.adapted_calls -= 1
            rd.reasons.add("admitted_evaluation_not_executed")
    finally:
        for handle in handles:
            handle.remove()
        if scope_token is not None:
            SCOPE.reset(scope_token)
        if live_token is not None:
            LIVE.reset(live_token)


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

    def __call__(self, executor, x, timesteps, context, attention_mask=None, ref_latents=None,
                 transformer_options=None, **kwargs):
        options = transformer_options or {}
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
        seen = set()

        def make(block):
            def hook(module, args, kwargs, output):
                trans = kwargs.get("transformer_options") or {}
                start, end = _image_rows(trans, output.shape[1], grid)
                self.sink(call, block, output[:, start:end].detach(), options)
                seen.add(block)
                if stop and block == blocks[-1]:
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


def _latent_shape(latent):
    samples = latent["samples"]
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


def _staging(n, features, device, budget_mib):
    need = n * features * 2
    if need > budget_mib * (1 << 20):
        raise ValueError(f"Staging {n} x {features} bf16 activations needs {need / (1 << 20):.0f} MiB; "
                         f"the workspace limit is {budget_mib} MiB.")
    return torch.empty(n, features, dtype=torch.bfloat16, device=device)


def _workspace_device(choice):
    import comfy.model_management

    return comfy.model_management.get_torch_device() if choice == "auto" else torch.device(choice)


def calibrate(teacher, positive, latent, *, negative=None, guidance=1.0, steps=TEACHER_STEPS, forward=TEACHER_FORWARD,
              block=ANCHOR_BLOCK, rollouts=ROLLOUTS, directions=DIRECTIONS, seed=0, batch=5, sigmas=None,
              label="", device="auto", workspace_mib=8192, store_mean=True):
    """Estimate ReDiverse-TF directions from conditional teacher rollouts (paper Sec. 4.2).

    Rollout j (0-based) starts from the packed reference draw seeded seed * 1000 + 100 + j.
    Forwards 1..forward-1 are full Euler steps on the teacher schedule; forward ``forward``
    is captured at the block output and stopped there. Returns (path, Directions).
    """
    import comfy.model_management
    import comfy.samplers
    from comfy.patcher_extension import WrappersMP

    from .rediverse import ARCHIVE_SCHEMA, archive_root, load_directions

    _verify_research_model(teacher)
    dit = teacher.get_model_object("diffusion_model")
    if not 0 <= block < len(dit.blocks):
        raise ValueError(f"Block {block} is outside the teacher's {len(dit.blocks)} blocks.")
    if not 1 <= forward <= steps or rollouts < 2 or not 1 <= directions <= rollouts:
        raise ValueError("Calibration needs 1 <= forward <= steps and 1 <= directions <= rollouts.")
    shape = _latent_shape(latent)
    grid = (math.ceil(shape[-2] / dit.patch), math.ceil(shape[-1] / dit.patch))
    tokens = grid[0] * grid[1]
    width = dit.blocks[block].attn.wo.out_features
    if sigmas is None:
        schedule = reference_sigmas(steps, tokens)
        source = "reference_flow_match_euler"
    else:
        schedule = [float(v) for v in sigmas]
        source = "connected_sigmas"
        if len(schedule) < forward + 1 or abs(schedule[0] - 1.0) > 1e-5:
            raise ValueError("Connected teacher sigmas must start at 1 and cover the requested forward.")
    run_sigmas = torch.tensor(schedule[:forward + 1], dtype=torch.float32)
    workspace = _workspace_device(device)
    X = _staging(rollouts, tokens * width, workspace, workspace_mib)
    seeds = [seed * 1000 + ROLLOUT_SAMPLE_BASE + j for j in range(rollouts)]
    state = {"filled": 0, "conditioning": None}

    def sink(call, captured_block, rows, options):
        count = rows.shape[0]
        branches = tuple(options.get("cond_or_uncond", ()))
        offset = branches.index(0) * (count // len(branches))
        live = count // len(branches)
        X[state["filled"]:state["filled"] + live] = rows[offset:offset + live].reshape(live, -1).to(workspace, torch.bfloat16)
        state["filled"] += live

    clone = teacher.clone()
    wrapper = CaptureWrapper(lambda call: (block,) if call == forward else (), sink, stop_after=forward)
    clone.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, CAPTURE_KEY, wrapper)
    guider = _guider(clone, positive, negative, guidance)
    fingerprint = tensor_fingerprint(guider.original_conds["positive"][0]["cross_attn"]) \
        if "cross_attn" in guider.original_conds["positive"][0] else None
    started = time.perf_counter()
    sampler = comfy.samplers.ksampler("euler")
    for start in range(0, rollouts, batch):
        comfy.model_management.throw_exception_if_processing_interrupted()
        chunk = seeds[start:start + batch]
        noise = _noise_batch(chunk, shape)
        empty = torch.zeros_like(noise)
        before = state["filled"]
        wrapper.calls = 0
        try:
            guider.sample(noise, empty, sampler, run_sigmas, disable_pbar=True, seed=chunk[0])
            raise RuntimeError("The calibration rollout finished without reaching the requested teacher forward.")
        except _Stop:
            pass
        if state["filled"] - before != len(chunk):
            raise RuntimeError("Calibration captured an unexpected number of rollouts; use conditional-only CFG 1 or a matching negative.")
    elapsed = time.perf_counter() - started
    G = covariance.centred_gram(X)
    values, vectors = covariance.eigenpairs(G)
    U, s, valid = covariance.principal_basis(X, vectors, values, directions)
    if valid < 1:
        raise RuntimeError("Teacher rollouts produced no variance at the requested block.")
    mean = X.float().mean(0).to(torch.bfloat16).cpu() if store_mean else None
    stats = covariance.spectrum_stats(values, rollouts)
    model_hash = model_fingerprint(dit)
    metadata = {
        "schema": ARCHIVE_SCHEMA, "implementation": IMPLEMENTATION, "upstream_revision": UPSTREAM_REVISION,
        "coordinate": "block_output_image_rows", "block": block, "token_grid": list(grid), "d_model": width,
        "latent_shape": list(shape), "teacher_model_sha256": model_hash, "teacher_patches": patch_fingerprint(teacher),
        "teacher_blocks": len(dit.blocks), "conditioning_sha256": fingerprint,
        "sigma_in": schedule[forward - 1], "sigma_out": schedule[forward], "teacher_forward": forward,
        "teacher_steps": len(schedule) - 1, "schedule_source": source,
        "guidance": {"cfg": guidance, "branches": "conditional_only" if negative is None or guidance == 1 else "guided"},
        "seeds": seeds, "seed_rule": "seed * 1000 + 100 + j, packed CPU randn", "rollouts": rollouts,
        "normalization": "eigenvalues of the unnormalized centred Gram; covariance eigenvalue = value / rollouts; "
                         "s_k = sqrt(value_k / rollouts)",
        "basis_dtype": "bfloat16", "directions": len(s), "valid_directions": valid,
        "pc1_share": stats["pc1_share"], "participation_ratio": stats["pr"], "total_variance": stats["tv"],
        "label": label, "seconds": elapsed, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    name = f"{label + '-' if label else ''}{(fingerprint or 'nocond')[:12]}-{model_hash[:8]}-b{block}-f{forward}-n{rollouts}.safetensors"
    path = save_directions(os.path.join(archive_root(), "directions", name), U.cpu(), s, values.float().cpu(), metadata, mean)
    del X
    LOGGER.info("ModeExplorer ReDiverse calibration: %s", json.dumps({"file": name, "seconds": round(elapsed, 3),
                                                                       "valid_directions": valid, "pc1_share": stats["pc1_share"]}))
    return path, load_directions(path)


def probe(model, positive, latent, *, blocks=(ANCHOR_BLOCK,), samples=200, seed=0, sigma=1.0, batch=8,
          reference=None, device="auto", workspace_mib=16384, label=""):
    """Activation covariance of block-output image rows at one frozen first-step state.

    Every sample starts from the packed reference draw 1_000_000 + seed * 100003 + i at
    ``sigma`` and runs one conditional forward. Blocks are processed in groups whose
    staged bf16 activations fit ``workspace_mib``; each group repeats the forwards and
    stops after its deepest block. With a reference direction archive at the same
    coordinate and geometry, the cohort is also compared in the reference subspace.
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
    shape = _latent_shape(latent)
    grid = (math.ceil(shape[-2] / dit.patch), math.ceil(shape[-1] / dit.patch))
    width = dit.blocks[0].attn.wo.out_features
    features = grid[0] * grid[1] * width
    per_block = samples * features * 2
    budget = workspace_mib * (1 << 20)
    if per_block > budget:
        raise ValueError(f"One block needs {per_block / (1 << 20):.0f} MiB of staged activations; raise the workspace limit "
                         "or reduce samples or resolution.")
    group_size = max(1, budget // per_block)
    groups = [blocks[i:i + group_size] for i in range(0, len(blocks), group_size)]
    workspace = _workspace_device(device)
    seeds = [1_000_000 + seed * 100_003 + i for i in range(samples)]
    if not 0 < sigma <= 1:
        raise ValueError("Probe sigma must be in (0, 1].")
    run_sigmas = torch.tensor([sigma, 0.0], dtype=torch.float32)
    report = {"schema": 1, "implementation": IMPLEMENTATION, "coordinate": "block_output_image_rows",
              "comparison": "matched_frozen_state", "sigma": sigma, "samples": samples, "seed_rule": "1000000 + seed * 100003 + i",
              "token_grid": list(grid), "d_model": width, "model_sha256": model_fingerprint(dit),
              "patches": patch_fingerprint(model), "blocks": {}, "label": label, "groups": [list(g) for g in groups]}
    started = time.perf_counter()
    sampler = comfy.samplers.ksampler("euler")
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
        if "fingerprint" not in report:
            first = guider.original_conds["positive"][0]
            report["conditioning_sha256"] = tensor_fingerprint(first["cross_attn"]) if "cross_attn" in first else None
            report["fingerprint"] = True
        for start in range(0, samples, batch):
            comfy.model_management.throw_exception_if_processing_interrupted()
            chunk = seeds[start:start + batch]
            noise = _noise_batch(chunk, shape)
            wrapper.calls = 0
            try:
                guider.sample(noise, torch.zeros_like(noise), sampler, run_sigmas, disable_pbar=True, seed=chunk[0])
            except _Stop:
                pass
        for block in group:
            if filled[block] != samples:
                raise RuntimeError("The probe captured an unexpected number of samples.")
            X = staged.pop(block)
            G = covariance.centred_gram(X)
            values, vectors = covariance.eigenpairs(G)
            entry = covariance.spectrum_stats(values, samples)
            if reference is not None and reference.block == block:
                if reference.grid != grid or reference.d_model != width:
                    raise ValueError("The reference directions use another geometry.")
                U = reference.U.to(workspace)
                P = covariance.project(X, U)
                entry["reference"] = covariance.subspace_comparison(reference.s, P, values, vectors)
                entry["reference"]["archive"] = reference.describe()
                entry["reference"]["coordinate_match"] = (
                    "matched_frozen_state" if abs(float(reference.metadata["sigma_in"]) - sigma) < 1e-6
                    else "native_trajectory_states")
                if reference.mean is not None:
                    displacement = X.float().mean(0) - reference.mean.to(workspace, torch.float32)
                    entry["reference"]["mean_displacement_rms"] = float(displacement.square().mean().sqrt())
                    entry["reference"]["mean_displacement_along_directions"] = [
                        float(v) for v in (U.float() @ displacement).cpu()]
                    entry["reference"]["mean_displacement_relative_to_cohort_rms"] = float(
                        displacement.norm() / max(math.sqrt(entry["tv"]), 1e-30))
            report["blocks"][f"block{block:02d}"] = entry
            del X
    report.pop("fingerprint", None)
    report["seconds"] = time.perf_counter() - started
    os.makedirs(os.path.join(archive_root(), "probes"), exist_ok=True)
    name = f"{label + '-' if label else ''}{report['model_sha256'][:8]}-{(report.get('conditioning_sha256') or 'nocond')[:12]}-n{samples}-s{sigma:g}.json"
    path = os.path.join(archive_root(), "probes", name)
    with open(path + ".tmp", "w") as handle:
        json.dump(report, handle, indent=1)
    os.replace(path + ".tmp", path)
    return path, report


def compare_probes(teacher, student):
    """Per-block student/teacher ratios and the upstream collapsed-block rule (App. A)."""
    rows = {}
    for name in sorted(set(teacher["blocks"]) & set(student["blocks"])):
        t, s = teacher["blocks"][name], student["blocks"][name]
        ratio = lambda key: (s[key] / t[key]) if t.get(key) and s.get(key) is not None else None
        rows[name] = {"pr_ratio": ratio("pr"), "tv_ratio": ratio("tv"), "conc_ratio": ratio("conc"),
                      "entropy_rank_ratio": ratio("entropy_rank")}
    names = list(rows)
    steps = [(names[i], rows[names[i]]["conc_ratio"] - rows[names[i - 1]]["conc_ratio"]) for i in range(1, len(names))
             if rows[names[i]]["conc_ratio"] is not None and rows[names[i - 1]]["conc_ratio"] is not None]
    matched = (teacher.get("sigma") == student.get("sigma") and teacher.get("samples") == student.get("samples")
               and teacher.get("conditioning_sha256") == student.get("conditioning_sha256")
               and teacher.get("token_grid") == student.get("token_grid"))
    return {"blocks": rows, "largest_concentration_step": max(steps, key=lambda v: v[1])[0] if steps else None,
            "matched_inputs": matched,
            "caveat": "Ratios use N-sample spectra; participation ratios are capped at N - 1."}
