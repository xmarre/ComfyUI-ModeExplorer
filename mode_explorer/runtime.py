"""Clone-local ComfyUI integration. All tensor state belongs to a sampler invocation."""

import json
import logging
import math
import sys
import time
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from functools import partial
from types import FunctionType
from dataclasses import dataclass, field

import torch

from .objectives import (
    cell_areas,
    cell_size,
    direction_control,
    feature_shift,
    initial_direction,
    noise_candidate,
    objective,
    pool,
    reference_noise,
    seed_key,
    select_candidate,
    token_pool,
    unit_rows,
)

LOGGER = logging.getLogger("ModeExplorer")
KEY = "mode_explorer_krea2"
STAGE = "spectrum_krea2_stage"
SOL_KEY = "sol_krea2_sparse_attention_v1"
RUN = f"{KEY}.run"
FRAME = f"{KEY}.frame"
BLOCK = 1  # second single-stream block, zero-indexed
MAX_BATCH = 8
MAX_POINTS = 12
MAX_INITIALIZATION_DIT = 12
MAX_INITIALIZATION_SECONDS = 45
MAX_INITIALIZATION_NOISE_BYTES = 32 * 1024**2
MAX_REFERENCE_STATE_BYTES = 64 * 1024**2
FEATURES = ("native", "feature_cosine", "feature_volume")
OPERATIONS = ("auto", "sequential", "batch")
REFERENCE_SOURCES = ("bank", "per_seed")
REFERENCE_SELECTIONS = ("all", "closest")
REFERENCE_CONTINUATIONS = ("first_estimate", "rolling_euler")
FEATURE_UPDATES = ("bounded", "verified")
APPLICATION_ORDINALS = {"first_only": 1, "second_only": 2}
FEATURE_APPLICATIONS = ("all", *APPLICATION_ORDINALS)
FEATURE_DIRECTIONS = ("repel", "reverse", "channel_sign")
FEATURE_GRIDS = (2, 4, 6, 8)
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
    feature_block: int = BLOCK
    feature_grid: int = 4
    cosine_temperature: float = 0.5
    volume_bandwidth: float = 0.55
    volume_median: bool = False
    reference_source: str = "bank"
    reference_bank_size: int = 8
    reference_trajectories: int = 1
    reference_selection: str = "all"
    reference_continuation: str = "first_estimate"
    feature_update: str = "bounded"
    sham: bool = False
    feature_direction: str = "repel"
    direction_trial: int = 0
    trajectory_label: str = ""
    trajectory_roi: str = ""
    feature_application: str = "all"
    ablation_route: str = ""

    def __post_init__(self):
        if self.feature_application not in FEATURE_APPLICATIONS:
            raise ValueError("Unknown ModeExplorer feature application schedule.")
        if self.ablation_route:
            if (not 2 <= len(self.ablation_route) <= 32 or not self.ablation_route.startswith("AA")
                    or set(self.ablation_route) - {"A", "F"} or not self.trajectory_label
                    or self.feature == "native" or self.strength <= 0 or self.initialization or self.full_spectrum
                    or self.reference_budget != 2 or self.reference_trajectories != 1 or self.operation == "batch"
                    or self.sham and self.feature_application != "all"):
                raise ValueError("Temporal ablation requires two reference evaluations, a guarded route and a trajectory label.")
        if self.trajectory_label:
            from .trajectory import LABEL

            if not LABEL.fullmatch(self.trajectory_label):
                raise ValueError("Use 1–64 letters, digits, hyphens or underscores for the trajectory label.")
        if self.trajectory_roi:
            from .trajectory import parse_roi

            if not self.trajectory_label:
                raise ValueError("A trajectory ROI requires an export label.")
            parse_roi(self.trajectory_roi)
        if self.feature not in FEATURES or self.operation not in OPERATIONS:
            raise ValueError("ModeExplorer received an unknown feature or batch mode")
        for key, low, high in (
            ("strength", 0, 1), ("fidelity", 0, 1), ("min_sigma", 0.5, 1),
            ("similarity", 0, 0.99), ("cosine_temperature", 0.1, 2),
            ("volume_bandwidth", 0.25, 1.25),
        ):
            if not low <= getattr(self, key) <= high:
                raise ValueError(f"ModeExplorer {key} must be in [{low}, {high}]")
        if type(self.reference_budget) is not int or not 0 <= self.reference_budget <= 4:
            raise ValueError("ModeExplorer reference_budget must be an integer in [0, 4]")
        if type(self.initialization_trials) is not int or not 1 <= self.initialization_trials <= 3:
            raise ValueError("ModeExplorer initialization_trials must be an integer in [1, 3]")
        if type(self.feature_block) is not int or self.feature_block < 0:
            raise ValueError("ModeExplorer feature_block must be a nonnegative integer.")
        for key, low, high in (
            ("reference_bank_size", 2, 256),
            ("reference_trajectories", 1, 3),
        ):
            value = getattr(self, key)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"ModeExplorer {key} must be an integer in [{low}, {high}]")
        for key, choices in (
            ("feature_grid", FEATURE_GRIDS), ("reference_source", REFERENCE_SOURCES),
            ("reference_selection", REFERENCE_SELECTIONS), ("reference_continuation", REFERENCE_CONTINUATIONS),
            ("feature_update", FEATURE_UPDATES), ("feature_direction", FEATURE_DIRECTIONS),
        ):
            if getattr(self, key) not in choices:
                raise ValueError(f"ModeExplorer received an unknown {key}")
        if self.feature_direction != "repel" and self.feature_update != "bounded":
            raise ValueError("ModeExplorer direction controls require the bounded feature update")
        if type(self.direction_trial) is not int or not 0 <= self.direction_trial <= 31:
            raise ValueError("ModeExplorer direction_trial must be an integer in [0, 31]")
        if self.reference_source == "bank" and self.reference_trajectories > self.reference_bank_size:
            raise ValueError("ModeExplorer reference bank must fit all trajectories")
        for key in ("full_spectrum", "volume_median", "sham"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"ModeExplorer {key} must be a boolean")


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
    sol: object = None
    shadows: dict = field(default_factory=dict)
    stage: str = "standalone"
    decisions: list = field(default_factory=list)
    points: list[dict] = field(default_factory=list)
    reasons: set[str] = field(default_factory=set)
    response: object = None
    guidance: object = None
    trajectory: object = None


@dataclass
class Shadow:
    noise: torch.Tensor
    first_sigma: float
    sigma: float
    clean: torch.Tensor | None = None
    latent: torch.Tensor | None = None
    velocity: torch.Tensor | None = None
    last_sampled_sigma: float | None = None


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
    edit_metrics: dict | None = None
    seen: bool = False
    prepare: object = None
    attempted: bool = False
    reference_slots: list = field(default_factory=list)
    replay: object = None
    replay_attention: object = None
    taps: dict | None = None
    tap_from: int = 0
    tap_grid: int = 4
    sol_evaluations: int | None = None
    anchor_blocks: tuple[int, ...] = ()
    mechanistic_capture: object = None
    realized_rms: torch.Tensor | None = None
    trajectory_proposal: torch.Tensor | None = None
    trajectory_roi: torch.Tensor | None = None
    proposal_relative_rms: torch.Tensor | None = None
    realization_error_rms: torch.Tensor | None = None
    objective_after: torch.Tensor | None = None
    reference_history: list = field(default_factory=list)


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
    matches = getattr(adapter, "matches", None) if getattr(adapter, "MODE_EXPLORER_DEFERRED_PROBES", False) is True else None
    return owner, runtime.CURRENT, adapter.CAPTURE, matches


def is_kj_loader_sage(override):
    """Recognize the actual KJ diffusion loader's Sage output dispatcher."""
    if not isinstance(override, FunctionType):
        return False
    module = sys.modules.get(override.__module__)
    if (module is None or not module.__name__.endswith(".nodes.model_optimization_nodes")
            or override.__globals__ is not vars(module)):
        return False
    loader = getattr(module, "DiffusionModelLoaderKJ", None)
    load = getattr(loader, "patch_and_load", None)
    factory = getattr(module, "get_sage_func", None)
    if (not isinstance(load, FunctionType) or not isinstance(factory, FunctionType)
            or not any(code is override.__code__ for code in load.__code__.co_consts)
            or override.__name__ != "attention_override_sage"
            or override.__code__.co_freevars != ("new_attention",)):
        return False
    try:
        attention = getattr(override.__closure__[0].cell_contents, "__wrapped__", None)
    except ValueError:  # empty closure cell
        return False
    return (isinstance(attention, FunctionType) and attention.__globals__ is vars(module)
            and any(code is attention.__code__ for code in factory.__code__.co_consts))


def is_core_attention_backend(override):
    """Recognize the built-in selector's dense PyTorch or Kitchen override."""
    if not isinstance(override, FunctionType):
        return False

    from comfy.ldm.modules import attention
    from comfy.model_patcher import ModelPatcher

    setter = getattr(ModelPatcher, "set_model_optimized_attention", None)
    if (not isinstance(setter, FunctionType) or override.__globals__ is not setter.__globals__
            or not any(code is override.__code__ for code in setter.__code__.co_consts)
            or override.__code__.co_freevars != ("optimized_attention",)):
        return False
    try:
        backend = override.__closure__[0].cell_contents
    except ValueError:  # empty closure cell
        return False
    if backend is not attention.attention_pytorch:
        kitchen = getattr(attention, "attention_comfy_kitchen_int8", None)
        if kitchen is None or backend is not kitchen:
            return False
    return getattr(override, "container_function", None) is getattr(backend, "container_function", None)


def verified_sol(options, registered):
    trans = options.get("transformer_options", {})
    override = trans.get("optimized_attention_override")
    marker = trans.get(SOL_KEY)
    tree = registered if registered is not None else wrappers(options)
    outer = tree.get("outer_sample", {}).get(SOL_KEY, ())
    diffusion = tree.get("diffusion_model", {}).get(SOL_KEY, ())
    if marker is None and not outer and not diffusion:
        if override is not None and not (is_core_attention_backend(override) or is_kj_loader_sage(override)):
            raise RuntimeError(
                "ModeExplorer found an unrecognized optimized_attention_override. "
                "Use the core PyTorch or Kitchen backend, a supported KJ diffusion loader, or verified Sol-Krea2."
            )
        return None
    module = sys.modules.get("sol_h3.krea2")
    owner = getattr(override, "_sol_krea2_owner", None)
    if (module is None or owner is None or len(outer) != 1 or len(diffusion) != 1
            or type(outer[0]) is not getattr(module, "_SamplingScope", None)
            or type(diffusion[0]) is not getattr(module, "_DiffusionScope", None)
            or outer[0].owner is not owner or diffusion[0].owner is not owner
            or getattr(override, "__code__", None) not in module._override.__code__.co_consts
            or not isinstance(getattr(module, "_ACTIVE", None), ContextVar)
            or not isinstance(getattr(module, "_EXECUTION", None), ContextVar)
            or type(outer[0].settings) is not module.Settings):
        raise RuntimeError("ModeExplorer could not verify Sol-Krea2 callback ownership.")
    settings = outer[0].settings
    if marker != {"api": 1, **vars(settings)}:
        raise RuntimeError("ModeExplorer could not verify Sol-Krea2 settings.")
    ordered = list(tree.get("diffusion_model", {}))
    if KEY in ordered and ordered.index(KEY) > ordered.index(SOL_KEY):
        raise RuntimeError("Apply ModeExplorer before Sol-Krea2 so auxiliary passes can use an isolated attention scope.")
    return module, owner


@contextmanager
def isolated_reference(run, *, sol_evaluations=None):
    """Do not advance Spectrum or Sol state while evaluating auxiliary inputs."""
    if run.spectrum is None:
        current = capture = None
    else:
        owner, current, capture, _ = run.spectrum
        session = current.get()
        if session is not None and session.binding is not owner:
            raise RuntimeError("ModeExplorer encountered a foreign Spectrum session.")
    if run.sol is None:
        sol = None
    else:
        sol, owner = run.sol
        state = sol._ACTIVE.get()
        if state is None or state.owner is not owner:
            raise RuntimeError("ModeExplorer lost the verified Sol-Krea2 sampling scope.")
    token_session = current.set(None) if current is not None else None
    token_capture = capture.set(None) if capture is not None else None
    token_sol = None
    try:
        if sol is not None:
            # Use the same dense-warmup position and isolated mutable counters.
            # Keep cached device code without sharing per-invocation statistics.
            evaluations = state.evaluations if sol_evaluations is None else sol_evaluations
            aux = sol._Run(owner, state.settings, evaluations=evaluations,
                           kernel=state.kernel, kernel_failure=state.kernel_failure,
                           kernel_device=state.kernel_device, sparse_verified=set(state.sparse_verified))
            token_sol = sol._ACTIVE.set(aux)
        yield
    finally:
        if token_sol is not None:
            sol._ACTIVE.reset(token_sol)
        if capture is not None:
            capture.reset(token_capture)
            current.reset(token_session)


def spectrum_forecast_count(run):
    if run.spectrum is None:
        return None
    owner, current, _, _ = run.spectrum
    session = current.get()
    return session.counts["forecast"] if session is not None and session.binding is owner else None


def spectrum_can_prepare(run, options):
    if run.spectrum is None:
        return False
    owner, current, _, matches = run.spectrum
    session = current.get()
    if not callable(matches) or session is None or session.binding is not owner:
        return False
    policy = session.patch_policy
    mode = getattr(policy, "mode_explorer", None)
    # Mirror the adapter's native dispatch gate, including live options.
    return mode is not None and mode.prepare_native is prepare_native_options and matches(policy, options)


def verify_patch_graph(options, settings, registered=None):
    spectrum = None
    sol = verified_sol(options, registered)
    for tree in (wrappers(options), registered or {}):
        sampler = tree.get("sampler_sample", {}).get(KEY, ())
        diffusion = tree.get("diffusion_model", {}).get(KEY, ())
        if sampler or diffusion:
            if (len(sampler) != 1 or type(sampler[0]) is not SamplerWrapper
                    or len(diffusion) != 1 or diffusion[0] is not diffusion_wrapper):
                raise RuntimeError("ModeExplorer could not verify its sampler and diffusion callbacks.")
        verified = verified_spectrum(tree)
        if verified is not None:
            if spectrum is not None and verified[0] is not spectrum[0]:
                raise RuntimeError("ModeExplorer found inconsistent Spectrum bindings.")
            spectrum = verified
        for name in ("sampler_sample", "predict_noise", "calc_cond_batch", "apply_model", "diffusion_model"):
            for key, value in tree.get(name, {}).items():
                if value and key != KEY and not (sol is not None and key == SOL_KEY and name == "diffusion_model") and not (key == "spectrum_krea2"
                    and name in ("sampler_sample", "predict_noise", "diffusion_model")):
                    raise RuntimeError(f"ModeExplorer needs the native {name} path except for verified Spectrum.")
    tr = options.get("transformer_options", {})
    if options.get("model_function_wrapper") or options.get("sampler_calc_cond_batch_function"):
        raise RuntimeError("ModeExplorer needs the native model and conditioning dispatch.")
    if tr.get("patches_replace") or tr.get("patches", {}).get("post_input"):
        raise RuntimeError("ModeExplorer needs the native Krea 2 image-token layout.")
    if settings.initialization and not settings.sham and any(
        options.get(key) for key in ("sampler_cfg_function", "sampler_pre_cfg_function", "sampler_post_cfg_function")
    ):
        raise RuntimeError("ModeExplorer initialization requires native CFG callbacks.")
    if tr.get("patches") and any(
        fn is not attention_feature for functions in tr["patches"].values() for fn in functions
    ):
        raise RuntimeError("ModeExplorer needs the native attention path. Use a separate MODEL branch for other transformer patches.")
    return spectrum, sol


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


def pooled_edit_metrics(before, after, pairs, proposal, repulsion, live, areas=None):
    """Measure the actual pooled update after native-dtype addition."""
    dims = (1, 2, 3)
    difference = after[:live] - before[:live]
    rms = difference.square().mean(dim=dims).sqrt()
    base_rms = before[:live].square().mean(dim=dims).sqrt()
    a, b = proposal[:live].flatten(1), repulsion[:live].flatten(1)
    alignment = (a * b).sum(dim=1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(torch.finfo(a.dtype).tiny)
    alignment = alignment.clamp(-1, 1)
    if areas is None:
        centered = after - after.mean(dim=(-1, -2), keepdim=True)
    else:
        centered = after - (after * areas).sum(dim=(-1, -2), keepdim=True) / areas.sum()
    unit, _, valid = unit_rows(centered.flatten(1))
    cosine = (unit @ unit.T).clamp(-1, 1)
    nearest = cosine.masked_fill(~(pairs & valid & valid.T), -1).amax(dim=-1)
    return {
        "applied_pooled_shift_rms": rms,
        "applied_pooled_relative_rms": rms / base_rms.clamp_min(1e-7),
        "proposal_direction_cosine": alignment,
        "nearest_cosine_after": nearest[:live],
    }


def attention_feature(output, options):
    frame = options.get(FRAME)
    if frame is None or options.get("block_type") != "single":
        return output
    block = options.get("block_index")
    if frame.mechanistic_capture is not None:
        return frame.mechanistic_capture.attention(output, options, frame)
    if frame.anchor_blocks:
        if block in frame.anchor_blocks:
            if block in frame.taps:
                raise RuntimeError("ModeExplorer captured the same reference block twice.")
            height, width = frame.grid
            start, end = token_slice(options, output.shape[1], height * width)
            frame.taps[block] = token_pool(output[:, start:end], height, width, frame.tap_grid)
        return output
    if frame.taps is not None and block > frame.tap_from:
        height, width = frame.grid
        start, end = token_slice(options, output.shape[1], height * width)
        frame.taps[block] = token_pool(output[frame.offset:frame.offset + frame.batch, start:end],
                                       height, width, frame.tap_grid)
    if frame.phase == "response_capture" or block != frame.run.settings.feature_block:
        return output
    if frame.seen:
        raise RuntimeError("ModeExplorer received the same Krea 2 attention callback twice in one forward.")
    frame.seen = True
    if frame.phase == "response_pass":
        return output
    height, width = frame.grid
    start, end = token_slice(options, output.shape[1], height * width)
    numbers = frame.offset, frame.offset + frame.batch
    image = output[numbers[0]:numbers[1], start:end]
    settings = frame.run.settings
    current = token_pool(image, height, width, settings.feature_grid)
    if frame.phase == "anchor":
        frame.pooled = current
        return output

    trajectory = frame.run.trajectory
    observing = trajectory is not None and trajectory.failure is None
    if observing and trajectory.roi is not None and frame.batch == 1:
        y0, y1, x0, x1 = trajectory.roi
        if y1 <= height and x1 <= width:
            frame.trajectory_roi = image[0].unflatten(0, (height, width))[y0:y1, x0:x1].clone()
    if observing or (settings.diagnostics and len(frame.run.points) < MAX_POINTS):
        frame.pooled = current
    pairs = frame.pairs
    if frame.reference is not None:
        reference = frame.reference
        if settings.reference_selection == "closest" and len(reference) > 1:
            joined = torch.cat((current, reference), dim=0)
            unit, _, valid = unit_rows((joined - joined.mean(dim=(-1, -2), keepdim=True)).flatten(1))
            sim = (unit[1:] @ unit[0]).masked_fill(~valid[1:, 0], -torch.inf)
            reference = reference.index_select(0, sim.argmax().reshape(1))
        current = torch.cat((current, reference), dim=0)
        pairs = ~torch.eye(len(current), device=output.device, dtype=torch.bool)
    if pairs is None:
        return output
    areas = None
    if settings.feature_update == "verified":
        areas = cell_areas(height, width, *current.shape[-2:], settings.feature_grid, output.device)
    frame.run.feature_attempts += 1
    # Research qualifies fixed actual-evaluation ordinals. Production counts
    # eligible proposals, so forecasts and negative-only calls cannot spend one.
    ordinal = frame.run.native_dit_calls if settings.ablation_route else frame.run.feature_attempts
    apply_edit = not settings.sham and (
        settings.feature_application == "all" or ordinal == APPLICATION_ORDINALS[settings.feature_application]
    )
    frame.attempted = apply_edit
    shift, redundancy, loss = feature_shift(
        current, pairs, settings.feature, settings.strength, settings.fidelity,
        settings.adaptive, settings.similarity, temperature=settings.cosine_temperature,
        bandwidth=settings.volume_bandwidth, median_bandwidth=settings.volume_median,
        live=frame.batch, areas=areas,
    )
    repulsion = shift
    if settings.feature_direction != "repel":
        shift = direction_control(shift, settings.feature_direction, frame.run.seed, settings.direction_trial)
    if observing:
        frame.trajectory_proposal = shift[:frame.batch]
        if settings.ablation_route:
            base_rms = current[:frame.batch].square().mean(dim=(1, 2, 3)).sqrt()
            frame.proposal_relative_rms = shift[:frame.batch].square().mean(dim=(1, 2, 3)).sqrt() / base_rms.clamp_min(1e-12)
    trace_edit = observing or (settings.diagnostics and len(frame.run.points) < MAX_POINTS)
    if trace_edit:
        frame.metrics = (
            redundancy[:frame.batch], loss,
            shift[:frame.batch].square().mean(dim=(1, 2, 3)).sqrt(),
        )
    if frame.replay is not None:
        from .response_runtime import feature_delta

        cfg, stats = frame.replay
        if cfg.feature_direction == "sham":
            stats["realized_edit_l2"] = stats["realized_edit_rms"] = 0.0
            return output
        if cfg.feature_direction in ("reverse", "channel_sign"):
            shift = direction_control(shift, cfg.feature_direction, frame.run.seed, cfg.trial)
        result = output.clone()
        view = result[numbers[0]:numbers[1], start:end].unflatten(1, (height, width))
        sy, sx = cell_size(height, width, settings.feature_grid)
        feature_delta(shift, current, output, view, sy, sx, frame)
        return result
    if not apply_edit:
        if trace_edit:
            frame.edit_metrics = pooled_edit_metrics(current, current, pairs, shift, repulsion, frame.batch, areas)
            if settings.ablation_route:
                frame.objective_after = loss.reshape(-1)
        return output
    result = output.clone()
    view = result[numbers[0]:numbers[1], start:end].unflatten(1, (height, width))
    sy, sx = cell_size(height, width, settings.feature_grid)
    # Expand a single coarse cell row over width, with no full-size delta tensor.
    for iy in range(shift.shape[-2]):
        delta = shift[:frame.batch, :, iy].to(output.dtype).repeat_interleave(sx, dim=-1)
        target = view[:, iy * sy:(iy + 1) * sy]
        target.add_(delta[:, :, :width].transpose(1, 2)[:, None])
    if observing:
        # Bounded strips avoid a full activation-sized FP32 difference.
        energy = view.new_zeros((frame.batch,), dtype=torch.float32)
        after = result[numbers[0]:numbers[1], start:end]
        error_energy = torch.zeros_like(energy) if settings.ablation_route else None
        for row_index, (edited_row, native_row) in enumerate(zip(after.split(width, dim=1), image.split(width, dim=1))):
            delta = edited_row.to(dtype=torch.float32, copy=True).sub_(native_row)
            energy.add_((delta.square() if error_energy is not None else delta.square_()).sum(dim=(1, 2)))
            if error_energy is not None:
                if row_index % sy == 0:
                    intended = shift[:frame.batch, :, row_index // sy].repeat_interleave(sx, dim=-1)
                error_energy.add_(delta.sub_(intended[:, :, :width].transpose(1, 2)).square_().sum(dim=(1, 2)))
        frame.realized_rms = (energy / image[0].numel()).sqrt()
        if error_energy is not None:
            frame.realization_error_rms = (error_energy / image[0].numel()).sqrt()
    if trace_edit:
        edited = token_pool(result[numbers[0]:numbers[1], start:end], height, width, settings.feature_grid)
        after = torch.cat((edited, current[frame.batch:]), dim=0) if frame.reference is not None else edited
        frame.edit_metrics = pooled_edit_metrics(current, after, pairs, shift, repulsion, frame.batch, areas)
        if settings.ablation_route:
            mean = (after.mean(dim=(-1, -2), keepdim=True) if areas is None
                    else (after * areas).sum(dim=(-1, -2), keepdim=True) / areas.sum())
            # Temporal ablations have one live row and one reference. The
            # pair uses the same fallback objective parameters as feature_shift.
            _, _, after_loss = objective((after - mean).flatten(1), pairs, settings.feature, .55, .5, gradient=False)
            frame.objective_after = after_loss.reshape(-1)
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
        "kind": "trajectory", "sigma": frame.sigma, "block": frame.run.settings.feature_block,
        "grid": list(frame.pooled.shape[-2:]),
        "attention": summary(frame.pooled - frame.pooled.mean(dim=(-1, -2), keepdim=True)),
        "conditional_x0": summary(pool(spatial_image(x0))),
    }
    if frame.metrics is not None:
        cosine, loss, rms = frame.metrics
        point.update(nearest_cosine=as_list(cosine), objective=as_list(loss), shift_rms=as_list(rms),
                     edit_applied=frame.attempted, reference_slots=frame.reference_slots)
    if frame.edit_metrics is not None:
        point.update({key: as_list(value) for key, value in frame.edit_metrics.items()})
    frame.run.points.append(point)


def reference_slot(settings, seed, index):
    if settings.reference_source == "bank":
        return 2 + (seed_key(seed, "single-reference-assignment") + index) % settings.reference_bank_size
    return seed_key(seed, f"feature-trajectory-{index}")


def shadow_forward(executor, run, frame, x, timestep, context, mask, refs, options, kwargs, index,
                   *, sol_evaluations=None, capture_blocks=(), retain_state=True):
    from comfy.model_management import throw_exception_if_processing_interrupted

    throw_exception_if_processing_interrupted()
    row = slice(frame.offset, frame.offset + 1)
    settings = run.settings
    state = run.shadows.get(index)
    rolling = settings.reference_continuation == "rolling_euler"
    slot = reference_slot(settings, run.seed, index)
    frame.reference_slots.append(slot)
    if settings.ablation_route:
        from .response_runtime import _tensor_hash

        frame.reference_history.append({
            "slot": slot, "sigma": frame.sigma, "continuation": settings.reference_continuation,
            "first_sigma": frame.sigma if state is None else state.first_sigma,
            "previous_sigma": None if state is None else state.last_sampled_sigma,
            "previous_clean_sha256": _tensor_hash(state.clean) if state is not None else None,
        })
    if state is None:
        noise = reference_noise(x[row], slot).float() * run.noise_scale
        state = Shadow(noise, frame.sigma, frame.sigma)
        run.shadows[index] = state
    if rolling and state.latent is not None:
        shadow = state.latent + (frame.sigma - state.sigma) * state.velocity
    else:
        initial_sigma = run.start if rolling else frame.sigma
        shadow = initial_sigma * state.noise
        if state.clean is not None:
            shadow = shadow + (1 - frame.sigma / state.first_sigma) * state.clean
    shadow = shadow.to(x.dtype)
    anchor = Frame(frame.grid, 1, 0, "anchor", run, frame.sigma,
                   anchor_blocks=capture_blocks, taps={} if capture_blocks else None,
                   tap_grid=settings.feature_grid)
    probe_options = {
        **options, FRAME: anchor, "cond_or_uncond": [0],
        "uuids": options.get("uuids", [None] * x.shape[0])[frame.offset:frame.offset + 1],
    }

    def row_of(tensor):
        return tensor[row] if tensor is not None and tensor.shape[0] == x.shape[0] else tensor

    run.reference_calls += 1
    if settings.ablation_route:
        LOGGER.info("ModeExplorer temporal reference: actual evaluation %d, sigma %.10g, slot %d (%d/2)",
                    run.native_dit_calls - 1, frame.sigma, slot, run.reference_calls)
    with isolated_reference(run, sol_evaluations=sol_evaluations):
        velocity = executor(
            shadow, timestep[row], context[row], row_of(mask),
            [row_of(ref) for ref in refs] if refs is not None else None, probe_options, **kwargs,
        )
    observed = anchor.taps.keys() == set(capture_blocks) if capture_blocks else anchor.seen
    if not observed:
        raise RuntimeError("ModeExplorer did not reach the expected Krea 2 attention hook during the reference forward.")
    if retain_state:
        if rolling:
            state.latent, state.velocity, state.sigma = shadow.float(), velocity.float(), frame.sigma
        elif state.clean is None:
            state.clean = shadow.float() - frame.sigma * velocity.float()
    if settings.ablation_route:
        state.last_sampled_sigma = frame.sigma
    return anchor.taps if capture_blocks else anchor.pooled


def guidance_pending(model_options, sigma):
    run = model_options.get("transformer_options", {}).get(RUN)
    if type(run) is not Run or run.phase != "sample":
        return False
    if run.guidance is not None:
        state = run.guidance
        return state.config.strength(state.ordinal, sigma) > 0
    return (run.feature_active and sigma_eligible(sigma, run.settings.min_sigma)
            and (run.batch > 1 or run.reference_calls < run.settings.reference_budget))


def native_feature_attempted(options):
    frame = options.get(FRAME)
    return type(frame) is Frame and frame.attempted


def prepare_native_options(options):
    """Run pending reference probes immediately before a qualified actual DiT."""
    frame = options.get(FRAME)
    if type(frame) is Frame and frame.prepare is not None:
        prepare, frame.prepare = frame.prepare, None
        prepare()


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
    prior_refs = run.reference_calls

    def forward(args):
        result = executor(*args, **kwargs)
        after = spectrum_forecast_count(run)
        forecast = before_forecast is not None and after is not None and after > before_forecast
        if forecast:
            run.spectrum_forecasts += after - before_forecast
            run.native_dit_calls -= 1
        if run.trajectory is not None:
            try:
                run.trajectory.observe(run, args, result, forecast, run.reference_calls - prior_refs,
                                       kwargs.get("ref_latents_method", executor.class_obj.default_ref_method))
            except (MemoryError, OSError, RuntimeError, ValueError) as exc:
                run.trajectory.failure = "capture_failed"
                LOGGER.warning("ModeExplorer trajectory capture stopped: %s", exc)
        if run.settings.diagnostics and len(run.decisions) < 256:
            run.decisions.append({
                "route": "forecast" if forecast else "actual",
                "branches": list(options.get("cond_or_uncond", ())),
                "reference_calls": run.reference_calls - prior_refs,
            })
        return result

    if run.guidance is not None:
        from .projection_guidance import dispatch

        return dispatch(forward, arguments, executor.class_obj, run)

    response_active = run.response is not None and run.response.eligibility is None
    if not (run.feature_active or run.settings.diagnostics or run.trajectory is not None or response_active) or spatial_image(x) is None:
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
    observe = run.trajectory is not None and run.trajectory.failure is None
    may_steer = run.feature_active and (run.batch > 1 or run.reference_calls < run.settings.reference_budget)
    if not (trace or observe or may_steer or response_active):
        return forward(arguments)
    sigmas = options.get("sigmas")
    if sigmas is None or sigmas.shape != (run.batch,):
        run.reasons.add("unverified_sigma_dispatch")
        return forward(arguments)
    # All native solver rows share the same scalar. Verify custom schedules
    # defensively so guidance never couples distinct timesteps.
    if run.batch == 1:
        low = high = float(sigmas[0].float())
    else:
        low, high = as_list(torch.stack((sigmas.amin(), sigmas.amax())))
    if not math.isfinite(low) or high != low:
        run.reasons.add("nonuniform_sigma")
        return forward(arguments)
    active = may_steer and sigma_eligible(low, run.settings.min_sigma)
    if not (active or trace or observe or response_active):
        return forward(arguments)
    offset = branches.index(0) * run.batch
    patch = executor.class_obj.patch
    grid = tuple(math.ceil(size / patch) for size in x.shape[-2:])
    frame = Frame(grid, run.batch, offset, "sample", run, low)
    if response_active and run.sol is not None:
        sol, owner = run.sol
        sol_state = sol._ACTIVE.get()
        if sol_state is not None and sol_state.owner is owner:
            frame.sol_evaluations = sol_state.evaluations
    if grid == (1, 1):
        active = False
        run.reasons.add("degenerate_feature_grid")
    if active and run.batch == 1:
        def prepare():
            remaining = run.settings.reference_budget - run.reference_calls
            indices = [index for index in range(min(run.settings.reference_trajectories, remaining))
                       if index not in run.shadows or low <= run.shadows[index].sigma]
            retain_state = (len(indices) < remaining or run.settings.diagnostics
                            or run.trajectory is not None or run.response is not None)
            references = [
                shadow_forward(executor, run, frame, x, timesteps, context, attention_mask,
                               ref_latents, options, kwargs, index, retain_state=retain_state)
                for index in indices
            ]
            if not retain_state:
                run.shadows.clear()
            if references:
                frame.reference = references[0] if len(references) == 1 else torch.cat(references, dim=0)
        frame.prepare = prepare
        if not spectrum_can_prepare(run, options):
            prepare_native_options({FRAME: frame})
    elif active and run.batch > 1:
        pairs = equal_condition_pairs(run.batch, offset, context, attention_mask, ref_latents)
        # A batch without an equally conditioned pair has no proposal to compute
        # and must not spend a scheduled application.
        frame.pairs = pairs if bool(pairs.any()) else None
    if frame.pairs is None and frame.reference is None and frame.prepare is None and not (trace or observe):
        if not response_active:
            return forward(arguments)
        frame.phase = "response_pass"
    try:
        result = forward((*arguments[:-1], {**options, FRAME: frame}))
        if frame.prepare is not None and frame.seen:
            run.reasons.add("reference_prepare_unavailable")
    finally:
        # Forecasts and interrupted forwards leave a pending closure. Release
        # its latent, context and executor immediately instead of waiting for GC.
        frame.prepare = None
    if not frame.seen:
        if before_forecast is not None and spectrum_forecast_count(run) > before_forecast:
            return result
        raise RuntimeError("ModeExplorer did not reach the expected Krea 2 attention hook.")
    if trace:
        record(frame, x, result)
    if response_active:
        from .response_runtime import run_response

        return run_response(executor, run, frame, arguments, kwargs, result)
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
        with isolated_reference(run):
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
        if not valid.all():
            run.reasons.add("initialization_invalid_clean_baseline")
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
            prior_guider = run.initialization_guider_calls
            try:
                candidate_clean = spatial_image(estimate(candidate))
            except InitializationLimit as exc:
                if run.settings.diagnostics:
                    attempts.append({
                        "rms_budget": as_list(radius), "volume_gain": None, "relative_clean_shift": None,
                        "status": [str(exc) if attempted else "inactive" for attempted in changed.tolist()],
                        "guider_calls": run.initialization_guider_calls - prior_guider,
                        "dit_calls": run.initialization_dit_calls - before,
                    })
                raise
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
    def __init__(self, settings, response=None, guidance=None):
        self.settings = settings
        self.response = response
        self.guidance = guidance

    def __call__(self, executor, model_wrap, sigmas, extra_args, callback, noise,
                 latent_image=None, denoise_mask=None, disable_pbar=False):
        from comfy.model_sampling import CONST, ModelSamplingFlux
        from comfy.samplers import CFGGuider, KSAMPLER

        def sample(args=extra_args, initial=noise, sample_callback=callback):
            return executor(model_wrap, sigmas, args, sample_callback, initial, latent_image, denoise_mask, disable_pbar)

        settings = self.settings
        model_options = extra_args["model_options"]
        if self.guidance is not None:
            from .projection_guidance import sample_guidance

            return sample_guidance(self.guidance, settings, executor, model_wrap, sigmas, extra_args, noise, sample)
        if model_options.get(STAGE) == "highres" or getattr(model_wrap, "model_options", {}).get(STAGE) == "highres":
            if settings.ablation_route:
                raise RuntimeError("Temporal ablation must use the planner-only output.")
            return sample()
        spectrum, sol = verify_patch_graph(model_options, settings)
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
            if settings.ablation_route:
                raise RuntimeError("Temporal ablation requires an ER-SDE schedule.")
            return sample()
        seed = extra_args.get("seed")
        start = float(sigmas[0])
        batch = noise.shape[0]
        active = settings.strength > 0 and (settings.feature != "native" or settings.initialization)
        reasons = set()
        if active or self.response is not None:
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
        initialize = active and not reasons and settings.initialization and not settings.sham and batch_allowed
        if active and not reasons and not batch_allowed:
            reasons.add("batch_mode_mismatch")
            LOGGER.info("ModeExplorer skipped batch_mode_mismatch (seed=%s)", seed)
        reference_state_tensors = 3 if settings.reference_continuation == "rolling_euler" else 2
        reference_bytes = (4 * noise.numel() * reference_state_tensors
                           * min(settings.reference_trajectories, settings.reference_budget))
        if feature_active and batch == 1 and reference_bytes > MAX_REFERENCE_STATE_BYTES:
            feature_active = False
            reasons.add("reference_latent_limit")
        if feature_active and batch == 1 and settings.reference_budget == 0:
            LOGGER.info("ModeExplorer sequential feature guidance has zero reference budget (seed=%s)", seed)
            feature_active = False
            reasons.add("zero_reference_budget")
        from comfy.k_diffusion.sampling import sample_er_sde

        capture_solver = (bool(settings.trajectory_label) and executor.class_obj.sampler_function is sample_er_sde
                          and float(sigmas[-1]) == 0 and getattr(model_wrap, "cfg", None) == 1.0
                          and denoise_mask is None
                          and not any(model_options.get(key) for key in
                                      ("disable_cfg1_optimization", "sampler_pre_cfg_function",
                                       "sampler_post_cfg_function", "sampler_cfg_function")))
        if settings.ablation_route and (not capture_solver or not feature_active or batch != 1
                                        or spectrum is None or self.response is not None
                                        or model_options.get(STAGE) != "planner"
                                        or len(settings.ablation_route) != len(sigmas) - 1):
            raise RuntimeError("Temporal ablation requires CFG 1, native ER-SDE ending at zero, verified guarded Spectrum and two eligible reference calls.")
        if not (feature_active or initialize or settings.diagnostics or settings.trajectory_label or self.response is not None):
            return sample()

        scale = getattr(sampling, "noise_scale", 1.0)
        if not math.isfinite(scale) or scale <= 0:
            raise RuntimeError("ModeExplorer requires a finite positive flow noise scale.")
        run = Run(settings, int(seed or 0), start, scale, batch, tuple(noise.shape), feature_active,
                  spectrum=spectrum, sol=sol, reasons=reasons, stage=model_options.get(STAGE, "standalone"))
        sample_callback = callback
        if settings.trajectory_label:
            from .trajectory import Trajectory

            run.trajectory = Trajectory(run, executor.class_obj, sigmas, noise, model_wrap.inner_model,
                                        getattr(model_wrap, "cfg", None))

            def sample_callback(step, denoised, noisy, total):
                if capture_solver:
                    try:
                        run.trajectory.observe_solver(step, denoised, noisy)
                    except (MemoryError, OSError, RuntimeError, ValueError) as exc:
                        # Archive-only work. An explicit ablation still fails
                        # its qualification through the recorded failure.
                        run.trajectory.failure = run.trajectory.failure or "solver_capture_failed"
                        LOGGER.warning("ModeExplorer solver capture stopped: %s", exc)
                run.trajectory.step = int(step) + 1
                if callback is not None:
                    return callback(step, denoised, noisy, total)
        force_native = nullcontext()
        response = self.response
        if response is not None:
            from .response_runtime import ResponseRun, spectrum_all_actual

            eligibility = None
            if reasons - {"batch_mode_mismatch", "reference_latent_limit", "zero_reference_budget"}:
                eligibility = "sampler_or_conditioning_unsupported"
            elif batch != 1:
                eligibility = "single_image_required"
            elif (not isinstance(sampling, ModelSamplingFlux)
                  or getattr(type(sampling), "timestep") is not ModelSamplingFlux.timestep):
                eligibility = "unverified_flow_timestep"
            elif response.steers and response.selection == "external" and response.scorer is None:
                eligibility = "score_unavailable"
            elif response.steers and (getattr(model_wrap, "cfg", None) != 1.0 or any(
                model_options.get(key) for key in ("sampler_cfg_function", "sampler_pre_cfg_function", "sampler_post_cfg_function")
            )):
                eligibility = "response_steering_requires_native_cfg_one"
            elif response.steers and (feature_active or initialize):
                eligibility = "use_native_model_for_response_steering"
            elif response.steers and settings.full_spectrum and spectrum is not None:
                eligibility = "active_response_requires_all_actual"
            if eligibility is None and (response.all_actual or response.steers) and spectrum is not None:
                force_native = spectrum_all_actual(spectrum)
                if force_native is None:
                    eligibility = "missing_spectrum_all_actual_contract"
                    force_native = nullcontext()
            is_mechanistic = response.mechanistic is not None and eligibility is None
            run.response = ResponseRun(
                response, model_wrap.inner_model.process_latent_out, eligibility, cfg_scale=getattr(model_wrap, "cfg", None),
                base_model=model_wrap.inner_model if is_mechanistic else None,
                positive_metadata_mask=conds["positive"][0].get("attention_mask") if is_mechanistic else None,
            )
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
            with force_native:
                result = sample({**extra_args, "model_options": options}, initial, sample_callback)
            if capture_solver and run.trajectory is not None:
                try:
                    run.trajectory.verify_terminal(result)
                except (MemoryError, OSError, RuntimeError, ValueError) as exc:
                    # The sampled latent is valid. Only the archive loses its
                    # terminal certificate; an explicit ablation rejects it below.
                    run.trajectory.solver_terminal = {"status": "terminal_capture_failed"}
                    run.trajectory.failure = run.trajectory.failure or "terminal_capture_failed"
                    LOGGER.warning("ModeExplorer terminal verification stopped: %s", exc)
            if settings.ablation_route:
                run.trajectory.verify_ablation(run)
            completed = True
            return result
        finally:
            run.shadows.clear()
            trajectory = None
            if run.trajectory is not None:
                try:
                    trajectory = run.trajectory.save(completed, run)
                except (MemoryError, OSError, RuntimeError, ValueError) as exc:
                    trajectory = {"status": "failed", "reason": str(exc)}
                    if settings.ablation_route and completed:
                        run.trajectory.ablation.update(status="unqualified", reason="trajectory_write_failed")
            if settings.diagnostics or run.response is not None or trajectory is not None:
                payload = {
                    "seed": run.seed, "completed": completed, "batch": batch, "stage": run.stage,
                    "sham": settings.sham, "feature_direction": settings.feature_direction,
                    "direction_trial": settings.direction_trial,
                    "seconds": time.perf_counter() - begin, "reasons": sorted(run.reasons),
                    "native_dit_calls": run.native_dit_calls, "feature_reference_dit_calls": run.reference_calls,
                    "initialization_dit_calls": run.initialization_dit_calls,
                    "initialization_guider_calls": run.initialization_guider_calls,
                    "feature_attempts": run.feature_attempts, "spectrum_forecast_calls": run.spectrum_forecasts,
                    "spectrum_policy": ("full" if settings.full_spectrum else "guarded")
                        if spectrum is not None else "none",
                    "points": run.points, "decisions": run.decisions,
                }
                if trajectory is not None:
                    payload["trajectory"] = trajectory
                if settings.ablation_route:
                    payload["ablation"] = run.trajectory.ablation
                if run.response is not None:
                    rep = run.response
                    payload["response"] = {
                        "schema": 1, "auxiliary_dit_calls": rep.forwards, "auxiliary_dit_rows": rep.rows,
                        "auxiliary_guider_calls": 0,
                        "eligible_actual_positive_evaluations": rep.evaluation + 1,
                        "status": ("completed" if any(p["status"] == "completed" for p in rep.points)
                                   else "rejected" if any(p["status"] == "rejected" for p in rep.points) else "skipped"),
                        "reason": rep.eligibility or ("no_matching_actual_evaluation" if not rep.points else None),
                        "points": rep.points,
                    }
                LOGGER.info("MODEEXPLORER_TRACE %s", json.dumps(finite_json(payload), separators=(",", ":"), allow_nan=False))
            if completed and settings.ablation_route and trajectory["status"] != "completed":
                raise RuntimeError("ModeExplorer temporal ablation could not export its qualified trajectory.")
