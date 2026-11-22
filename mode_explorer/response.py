"""Bounded empirical response measurements in native Krea 2 flow coordinates."""

from copy import deepcopy
from dataclasses import dataclass
import math
from numbers import Real

import torch
import torch.nn.functional as F

from .objectives import cell_size, lift, reference_noise, seed_key

BASES = ("gaussian", "coarse", "bilinear", "dct", "reference")
MODES = ("latent_probe", "feature_replay", "joint_probe", "latent_steer", "feature_steer")
POLICIES = ("response_rms", "minimum_change", "response_alignment", "external", "export_only")
LOCALIZATIONS = ("whole", "only_raw", "remove_raw", "only_matched", "remove_matched")
REPLAY_DIRECTIONS = ("repel", "reverse", "channel_sign", "sham", "tangent", "spatial_random")
MAX_FLOW_SIGMA = 1.00001
CALIBRATION_OFF_AXIS_LIMIT = 0.10
MAX_STATES = 2
MAX_WORKSPACE_MIB = 1024
MAX_EXPORT_BYTES = 64 * 1024**2


@dataclass(frozen=True)
class ResponseSettings:
    mode: str = "latent_probe"
    basis: str = "bilinear"
    probe_grid: int = 4
    directions: int = 2
    trial: int = 0
    displacement: float = 0.02
    norm: str = "rms"
    calibration: str = "repeat_and_half"
    min_sigma: float = 0.94
    max_sigma: float = 0.98
    evaluation: int = -1
    states: int = 1
    auxiliary_budget: int = 12
    memory_mb: int = 128
    clean_limit: float = 0.06
    candidate_gain: float = 0.5
    candidate_target_rms: float = 0.0
    selection: str = "minimum_change"
    response_direction: str = "singular"
    all_actual: bool = False
    feature_block: int = 5
    feature_grid: int = 4
    feature_strength: float = 0.025
    feature_references: int = 1
    feature_pair: bool = True
    feature_direction: str = "repel"
    localization: str = "whole"
    head: int = -1
    cell: int = -1
    export: bool = False
    checkpoint_label: str = ""
    frozen_state: str = ""
    downstream: bool = False
    scorer: object = None
    input_limit: float = 0.25
    calibration_max_change: float = 0.25
    latent_cell: int = -1
    feature_sites: str = ""
    feature_projection: str = "native"
    require_conditioning: bool = False
    require_fidelity: bool = False
    reference_min_sigma: float = -1.0
    reference_start_sigma: float = -1.0
    probe_precision: str = "auto"
    mechanistic: object = None

    def __post_init__(self):
        for key, choices in (
            ("mode", MODES), ("basis", BASES), ("norm", ("rms", "l2")),
            ("calibration", ("off", "repeat_and_half")),
            ("probe_precision", ("auto", "fixed")), ("selection", POLICIES),
            ("response_direction", ("singular", "symmetric")),
            ("feature_projection", ("native", "response_subspace")),
            ("feature_direction", REPLAY_DIRECTIONS), ("localization", LOCALIZATIONS),
        ):
            if getattr(self, key) not in choices:
                raise ValueError(f"Unknown ModeExplorer response {key}.")
        for key, low, high in (
            ("probe_grid", 1, 16), ("directions", 1, 4), ("trial", 0, 65535),
            ("evaluation", -1, 127), ("states", 1, 3 if self.mechanistic is not None else MAX_STATES), ("auxiliary_budget", 0, 48),
            ("memory_mb", 16, MAX_WORKSPACE_MIB), ("feature_block", 0, 63), ("feature_grid", 1, 16),
            ("latent_cell", -1, 255), ("head", -1, 255), ("cell", -1, 255), ("feature_references", 1, 3),
        ):
            value = getattr(self, key)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"ModeExplorer response {key} must be an integer in [{low}, {high}].")
        for key, low, high in (
            ("input_limit", 1e-5, 0.25), ("calibration_max_change", 0.01, 2.0),
            ("displacement", 1e-5, 0.25), ("clean_limit", 1e-5, 0.25),
            ("candidate_gain", 0.01, 1.0), ("candidate_target_rms", 0.0, 0.25), ("feature_strength", 0.0, 1.0),
            ("min_sigma", 0.0, 1.0), ("max_sigma", 0.0, 1.0),
        ):
            if not low <= getattr(self, key) <= high:
                raise ValueError(f"ModeExplorer response {key} must be in [{low}, {high}].")
        for key, high in (("reference_min_sigma", 1.0), ("reference_start_sigma", MAX_FLOW_SIGMA)):
            value = getattr(self, key)
            if value != -1 and not 0 <= value <= high:
                raise ValueError(f"ModeExplorer {key} must be -1 or in [0, {high}].")
        if self.candidate_target_rms > self.clean_limit:
            raise ValueError("The candidate target RMS exceeds the clean trust limit.")
        if self.frozen_state and self.steers:
            raise ValueError("Frozen-state replay requires a diagnostic or export-only mode.")
        if self.selection == "response_alignment" and self.mode != "feature_steer":
            raise ValueError("Output-response alignment is available for feature steering.")
        if (self.feature_sites or self.feature_projection != "native") and self.mode != "feature_steer":
            raise ValueError("Feature site selection and projection require feature steering.")
        if (self.require_conditioning or self.require_fidelity) and self.selection != "external":
            raise ValueError("Conditioning and fidelity gates require external selection.")
        if self.feature_sites:
            self.sites
        if self.feature_grid not in (2, 4, 6, 8):
            raise ValueError("Use a feature grid of 2, 4, 6, or 8.")
        if self.min_sigma > self.max_sigma:
            raise ValueError("ModeExplorer response sigma bounds are reversed.")
        if self.localization != "whole" and self.head == self.cell == -1 and not self.feature_sites:
            raise ValueError("Select an output head or spatial cell to localize.")
        if self.selection == "external" and self.scorer is not None and not callable(self.scorer):
            raise TypeError("The ModeExplorer response scorer must be callable.")

    @property
    def sites(self):
        if not self.feature_sites:
            return ((self.feature_block, self.head),)
        try:
            sites = tuple(tuple(int(n) for n in pair.split(":")) for pair in self.feature_sites.split(","))
            if (len(sites) > 4 or len(set(sites)) != len(sites)
                    or any(not 0 <= block <= 63 or not 0 <= head <= 255 for block, head in sites)):
                raise ValueError
        except ValueError as exc:
            raise ValueError("Enter up to four unique feature sites as block:head pairs.") from exc
        return sites

    @property
    def steers(self):
        return self.mode in ("latent_steer", "feature_steer") and self.selection != "export_only"


def rms(value):
    return float(torch.linalg.vector_norm(value) / math.sqrt(value.numel()))


def flow_clean(x, velocity, sigma):
    """ComfyUI CONST, with a native-coordinate input and FP32 output."""
    return x.float() - sigma * velocity.float()


def effective_tilt_scale(sigma, noise_scale):
    """For isotropic corruption with noise standard deviation sigma * noise_scale."""
    if not 0 < sigma < 1 or not math.isfinite(noise_scale) or noise_scale <= 0:
        return None
    std = sigma * noise_scale
    variance = std * std
    if not math.isfinite(variance) or variance <= 0:
        return None
    scale = (1 - sigma) / variance
    return scale if math.isfinite(scale) else None


def velocity_for_clean(x, clean, sigma):
    if not 1e-5 <= sigma <= MAX_FLOW_SIGMA:
        raise ValueError("Cannot express a clean correction at this flow sigma.")
    return (x.float() - clean.float()) / sigma


def direction_field(x, basis, grid, seed, trial, index):
    """Create a field on the physical latent grid before normalization."""
    generator = torch.Generator(device="cpu").manual_seed(seed_key(seed, f"response-{trial}-{index}"))
    shape = x.shape
    height, width = shape[-2:]
    rows, cols = min(grid, height), min(grid, width)

    def normal(size):
        return torch.randn(size, generator=generator, dtype=torch.float32, device="cpu")

    if basis == "gaussian":
        return normal(shape).to(x.device)
    if basis == "reference":
        slot = seed_key(seed, f"latent-response-reference-{trial}-{index}")
        return reference_noise(x.float(), slot) - x.float()
    cells = normal((shape[0], shape[1], rows, cols))
    if basis == "coarse":
        field = lift(cells, height, width, grid=grid)
    elif basis == "bilinear":
        field = F.interpolate(cells, size=(height, width), mode="bilinear", align_corners=False)
    else:
        y = torch.arange(height, dtype=torch.float32) + 0.5
        u = torch.arange(rows, dtype=torch.float32)
        h = torch.arange(width, dtype=torch.float32) + 0.5
        v = torch.arange(cols, dtype=torch.float32)
        dy = torch.cos(math.pi / height * y[:, None] * u[None, :])
        dx = torch.cos(math.pi / width * h[:, None] * v[None, :])
        field = dy @ cells @ dx.T
    return field.reshape(shape).to(x.device)


def latent_cell_bounds(shape, grid, cell):
    height, width = shape[-2:]
    sy, sx = cell_size(height, width, grid)
    rows, cols = math.ceil(height / sy), math.ceil(width / sx)
    if not 0 <= cell < rows * cols:
        raise ValueError("latent_cell_out_of_range")
    y, x = divmod(cell, cols)
    return y * sy, min((y + 1) * sy, height), x * sx, min((x + 1) * sx, width)


def localize_direction(value, bounds):
    if bounds is None:
        return value
    y0, y1, x0, x1 = bounds
    field = torch.zeros_like(value)
    field[..., y0:y1, x0:x1] = value[..., y0:y1, x0:x1]
    return field


def orthonormalize(vectors):
    """Twice-reorthogonalized small basis with relative rank rejection."""
    basis = []
    for value in vectors:
        candidate = value.reshape(-1).float().clone()
        original = torch.linalg.vector_norm(candidate)
        if not torch.isfinite(original) or original <= 0:
            continue
        for _ in range(2):
            for q in basis:
                candidate.sub_(torch.dot(candidate, q) * q)
        length = torch.linalg.vector_norm(candidate)
        if torch.isfinite(length) and length > 1e-5 * original:
            basis.append(candidate / length)
    return basis


def shifted_pair(x, unit, displacement, norm, trust_rms):
    requested = displacement if norm == "l2" else displacement * math.sqrt(x.numel())
    base = x.float()
    pos = (base + requested * unit.reshape_as(x)).to(x.dtype)
    neg = (base - requested * unit.reshape_as(x)).to(x.dtype)
    dp, dn = (pos.float() - base).flatten(), (base - neg.float()).flatten()
    p_len, n_len = torch.linalg.vector_norm(dp), torch.linalg.vector_norm(dn)
    a, b = torch.dot(dp, unit), torch.dot(dn, unit)
    p, n, a, b = map(float, (p_len, n_len, a, b))
    off_axis = max(math.sqrt(max(0.0, p*p - a*a)) / max(p, 1e-30),
                   math.sqrt(max(0.0, n*n - b*b)) / max(n, 1e-30))
    metrics = {
        "requested_l2": requested, "requested_rms": requested / math.sqrt(x.numel()),
        "positive_l2": p, "negative_l2": n, "asymmetry": abs(p - n) / max(p, n, 1e-30),
        "positive_projected_l2": a, "negative_projected_l2": b, "off_axis_fraction": off_axis,
        "input_dtype": str(x.dtype),
    }
    if not all(math.isfinite(v) for v in (p, n, a, b)):
        return None, metrics, "nonfinite_input"
    if min(a, b) <= requested * 0.1:
        return None, metrics, "unresolved_native_displacement"
    if off_axis > 0.25:
        return None, metrics, "excessive_rounding_off_axis"
    if max(p, n) / math.sqrt(x.numel()) > trust_rms:
        return None, metrics, "input_trust_bound"
    return (pos, neg, a, b), metrics, None



def calibrated_pairs(x, unit, settings, amplitude):
    args = (settings.norm, settings.input_limit)
    full = shifted_pair(x, unit, amplitude, *args)
    half = shifted_pair(x, unit, amplitude * 0.5, *args) if settings.calibration != "off" else None
    return full, half


def common_probe_displacement(x, directions, settings):
    """Choose a single scale for the largest usable subset of this basis."""
    requested = settings.displacement
    if (settings.calibration == "off" or settings.probe_precision == "fixed"
            or x.dtype not in (torch.bfloat16, torch.float16)):
        return requested, None

    cap = settings.input_limit * (math.sqrt(x.numel()) if settings.norm == "l2" else 1)
    best_amplitude, best_indices = requested, ()
    amplitudes = (requested, *(min(requested * scale, cap) for scale in (1.25, 1.5, 2.0, 3.0, 4.0)))
    for amplitude in dict.fromkeys(amplitudes):
        if amplitude < requested:
            continue
        usable = []
        for i, unit in enumerate(directions):
            full, half = calibrated_pairs(x, unit, settings, amplitude)
            if (full[2] is None and half[2] is None
                    and max(full[1]["off_axis_fraction"], half[1]["off_axis_fraction"]) <= CALIBRATION_OFF_AXIS_LIMIT):
                usable.append(i)
        if len(usable) > len(best_indices):
            best_amplitude, best_indices = amplitude, tuple(usable)
        if len(usable) == len(directions):
            break
    return best_amplitude, best_indices


def differences(plus, minus, base, a, b):
    """Three-point differences with unequal realized steps on each side."""
    upper, lower = (plus - base) / a, (base - minus) / b
    return (b * upper + a * lower) / (a + b), 2 * (upper - lower) / (a + b)


def projected_operator(basis, responses):
    matrix = torch.stack([torch.stack([torch.dot(b, response) for response in responses])
                          for b in basis]).to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(matrix).all():
        raise ValueError("nonfinite_projected_response")
    symmetric = (matrix + matrix.T) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    _, singular, right = torch.linalg.svd(matrix, full_matrices=False)
    total = singular.sum()
    probability = singular[singular > 0] / total if total > 0 else singular[:0]
    rank = float(torch.exp(-torch.sum(probability * probability.log()))) if len(probability) else 0.0
    spectral_mass = eigenvalues.abs().sum()
    return {
        "matrix": matrix.tolist(),
        "singular_values": singular.tolist(),
        "symmetric_eigenvalues": eigenvalues.tolist(),
        "symmetry_error": float(torch.linalg.matrix_norm(matrix - matrix.T)
                                / torch.linalg.matrix_norm(matrix).clamp_min(1e-15)),
        "negative_spectral_mass": float((-eigenvalues.clamp_max(0)).sum() / spectral_mass.clamp_min(1e-15)),
        "quadratic_forms": matrix.diag().tolist(),
        "amplification": [float(torch.linalg.vector_norm(r)) for r in responses],
        "effective_rank": rank,
    }, right[0], eigenvectors[:, -1]


def output_projection(delta, responses, *, return_projection=False):
    flat = delta.float().flatten()
    basis = orthonormalize(responses)
    weights = [torch.dot(flat, q) for q in basis]
    coefficients = [float(weight) for weight in weights]
    energy = float(torch.dot(flat, flat))
    projected = min(energy, math.fsum(c*c for c in coefficients))
    report = {
        "output_basis_coefficients": coefficients,
        "cosines": [float(torch.dot(flat, r) / (
            torch.linalg.vector_norm(flat) * torch.linalg.vector_norm(r)).clamp_min(1e-15)) for r in responses],
        "explained_energy_fraction": projected / energy if energy else None,
        "residual_energy": energy - projected,
    }
    if return_projection:
        projected_delta = torch.zeros_like(flat)
        for q, weight in zip(basis, weights):
            projected_delta.add_(q * weight)
        return report, projected_delta.reshape_as(delta)
    return report


def clean_candidate(base, delta, settings, info):
    gain = settings.candidate_gain
    if settings.candidate_target_rms:
        available = rms(delta)
        if math.isfinite(available) and available * gain >= settings.candidate_target_rms:
            gain = settings.candidate_target_rms / available
        else:
            info["reason"] = "target_exceeds_available_response"
    info["gain"] = gain
    return (base + gain * delta, info)


def select_response(candidates, baseline, settings, noise_floor, provenance, process_latent_out=None):
    """Candidate order is stable. External scores are finite, larger-is-better."""
    chosen = None
    best = -math.inf
    categories = tuple(name for name, enabled in (
        ("conditioning", settings.require_conditioning), ("fidelity", settings.require_fidelity)
    ) if enabled)
    baseline_score = None
    if settings.selection == "external" and settings.scorer is not None:
        baseline_score = get_score(settings.scorer, baseline, provenance, process_latent_out)
    for clean, info in candidates:
        if "reason" in info:
            continue
        change = rms(clean - baseline)
        info["clean_displacement_rms"] = change
        if not math.isfinite(change) or change > settings.clean_limit:
            info["reason"] = "clean_trust_bound"
        elif settings.candidate_target_rms and abs(change - settings.candidate_target_rms) > 0.1 * settings.candidate_target_rms:
            info["reason"] = "realized_response_misses_target"
        elif change <= max(noise_floor * 2.0, 1e-8):
            info["reason"] = "below_repeat_floor"
        elif settings.selection == "export_only":
            info["reason"] = "offline_review"
        elif settings.selection == "external" and baseline_score is None:
            info["reason"] = "score_unavailable"
        elif settings.selection == "external" and any(name not in baseline_score for name in categories):
            info["reason"] = "baseline_acceptance_metric_unavailable"
        else:
            if settings.selection == "external":
                scores = get_score(settings.scorer, clean, provenance, process_latent_out)
                info["external_scores"] = scores
                info["external_score"] = scores["quality"] if scores is not None else None
                acceptance = {
                    name: ("unavailable" if scores is None or name not in scores
                           else "regressed" if scores[name] < baseline_score[name] else "passed")
                    for name in categories
                }
                if acceptance:
                    info["acceptance"] = acceptance
                if scores is None:
                    info["reason"] = "invalid_external_score"
                    continue
                if any(value != "passed" for value in acceptance.values()):
                    info["reason"] = "external_acceptance_gate"
                    continue
                score = scores["quality"]
                if score <= baseline_score["quality"]:
                    info["reason"] = "no_valid_score_improvement"
                    continue
            elif settings.selection == "response_alignment":
                score = info.get("response_alignment")
                if score is None or not math.isfinite(score):
                    info["reason"] = "response_alignment_unavailable"
                    continue
            else:
                score = -change if settings.selection == "minimum_change" else change
            if score > best:
                chosen, best = (clean, info), score
            info["reason"] = "lower_rank"
    if chosen is not None:
        chosen[1]["reason"] = "selected"
    return chosen, baseline_score


def get_score(scorer, clean, provenance, process_latent_out):
    try:
        # Give the scorer the same coordinates as VAE Decode and local exports.
        clean = clean.detach().clone()
        if process_latent_out is not None:
            clean = process_latent_out(clean)
        value = scorer(clean, deepcopy(provenance))
    except (ValueError, TypeError, RuntimeError, OSError):
        return None
    if isinstance(value, Real):
        value = {"quality": value}
    if not isinstance(value, dict):
        return None
    scores = {}
    for key in ("quality", "conditioning", "fidelity"):
        number = value.get(key)
        if isinstance(number, Real) and not isinstance(number, bool):
            try:
                number = float(number)
            except (OverflowError, TypeError, ValueError):
                continue
            if math.isfinite(number):
                scores[key] = number
    return scores if "quality" in scores else None
