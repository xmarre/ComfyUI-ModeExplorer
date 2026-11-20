"""Inference-only objectives on small, spatially pooled feature tensors."""

import hashlib
import math

import torch
import torch.nn.functional as F

EPS = 1e-7
RIDGE = 1e-3


def seed_key(seed: int, stream: str) -> int:
    data = (seed % (1 << 64)).to_bytes(8, "little") + stream.encode("ascii")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "little")


def reference_noise(like: torch.Tensor, slot: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed_key(slot, "reference"))
    sample = torch.randn((1, *like.shape[1:]), generator=generator, device="cpu", dtype=torch.float32)
    return sample.to(device=like.device, dtype=like.dtype).expand_as(like)


def cell_size(height: int, width: int, grid: int = 4) -> tuple[int, int]:
    return math.ceil(height / grid), math.ceil(width / grid)


def pool(images: torch.Tensor, grid: int = 4) -> torch.Tensor:
    extent = cell_size(*images.shape[-2:], grid)
    return F.avg_pool2d(images, extent, stride=extent, ceil_mode=True).float()


def token_pool(tokens: torch.Tensor, height: int, width: int, grid: int = 4) -> torch.Tensor:
    return pool(tokens.transpose(1, 2).unflatten(-1, (height, width)), grid)


def cell_areas(height: int, width: int, rows: int, cols: int, grid: int, device):
    sy, sx = cell_size(height, width, grid)
    ny = (height - sy * torch.arange(rows, device=device)).clamp(max=sy)
    nx = (width - sx * torch.arange(cols, device=device)).clamp(max=sx)
    return ny[:, None] * nx[None, :]


def lift(cells: torch.Tensor, height: int, width: int, *, adjoint: bool = False, grid: int = 4) -> torch.Tensor:
    """Broadcast cell offsets, or divide by the actual cell area for the pooling adjoint."""
    sy, sx = cell_size(height, width, grid)
    if adjoint:
        cells = cells / cell_areas(height, width, *cells.shape[-2:], grid, cells.device)
    return cells.repeat_interleave(sy, dim=-2).repeat_interleave(sx, dim=-1)[..., :height, :width]


def unit_rows(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    valid = torch.isfinite(features).all(dim=-1, keepdim=True)
    clean = torch.where(valid, features, 0)
    norm = torch.linalg.vector_norm(clean, dim=-1, keepdim=True)
    valid = valid & torch.isfinite(norm) & (norm > EPS)
    norm = torch.where(valid, norm, 1)
    return torch.where(valid, clean / norm, 0), norm, valid


def objective(
    features: torch.Tensor,
    eligible: torch.Tensor,
    style: str,
    bandwidth: float | torch.Tensor = 0.55,
    temperature: float | torch.Tensor = 0.5,
    *, gradient: bool = True,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """Return descent direction, strongest eligible cosine, and objective per cohort.

    Features are [..., N, D]. The off-diagonal mask must be symmetric and split
    equivalence classes, so the masked RBF matrix stays positive semidefinite.
    """
    with torch.autocast(device_type=features.device.type, enabled=False):
        unit, norm, valid = unit_rows(features.float() if features.dtype != torch.float64 else features)
        eligible = eligible & valid & valid.transpose(-1, -2)
        cosine = (unit @ unit.transpose(-1, -2)).clamp(-1, 1)
        nearest = cosine.masked_fill(~eligible, -1).amax(dim=-1, keepdim=True)
        if style == "feature_cosine":
            logits = (cosine / temperature).masked_fill(~eligible, -torch.inf)
            count = eligible.sum(dim=(-1, -2)).clamp_min(1)
            loss = logits.flatten(-2).logsumexp(dim=-1) - count.log()
            loss = torch.where(eligible.any(dim=-1).any(dim=-1), loss, 0)
            if not gradient:
                return None, nearest, loss
            weights = logits.flatten(-2).softmax(dim=-1).unflatten(-1, cosine.shape[-2:]).nan_to_num()
            grad_unit = 2 * ((weights / temperature) @ unit)
        elif style == "feature_volume":
            bw2 = bandwidth ** 2
            identity = torch.eye(unit.shape[-2], dtype=unit.dtype, device=unit.device)
            kernel = ((cosine - 1) / bw2).exp().masked_fill(~eligible, 0)
            matrix = kernel + (1 + RIDGE) * identity
            factor, info = torch.linalg.cholesky_ex(matrix, check_errors=False)
            # A block-diagonal RBF Gram plus ridge is positive definite; guard
            # against unexpected backend/numerical errors without a host sync.
            ok = (info == 0)[..., None, None]
            factor = torch.where(ok, factor, identity)
            loss = -2 * factor.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
            if not gradient:
                return None, nearest, torch.where(ok[..., 0, 0], loss, torch.inf)
            inverse_factor = torch.linalg.solve_triangular(factor, identity.expand_as(factor), upper=False)
            inv = inverse_factor.transpose(-1, -2) @ inverse_factor
            weights = inv * kernel / bw2
            # Inverse Gram coefficients can be large near a duplicate. Form
            # pair differences before weighting so exact duplicates stay
            # stationary and avoid subtracting two large FP32 reductions.
            grad_unit = torch.zeros_like(unit)
            for j in range(unit.shape[-2]):
                grad_unit.addcmul_(unit - unit[..., j:j + 1, :], weights[..., j:j + 1])
            grad_unit *= 2
            grad_unit = torch.where(ok, grad_unit, 0)
        else:
            raise ValueError(f"Unknown objective {style}")
        tangent = grad_unit - (grad_unit * unit).sum(dim=-1, keepdim=True) * unit
        return -tangent / norm, nearest, loss


def feature_shift(
    features: torch.Tensor,
    eligible: torch.Tensor,
    style: str,
    gain: float,
    fidelity: float,
    adaptive: bool,
    threshold: float,
    *, temperature: float = 0.5, bandwidth: float = 0.55,
    median_bandwidth: bool = False, live: int | None = None,
    areas: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    def center(values):
        if areas is None:
            return values - values.mean(dim=(-1, -2), keepdim=True)
        return values - (values * areas).sum(dim=(-1, -2), keepdim=True) / areas.sum()

    centered = center(features).flatten(1)
    # The gradient for one competitor has a fixed direction. Keep its
    # established magnitude, including the existing saturation function.
    if temperature != 0.5 or bandwidth != 0.55 or median_bandwidth:
        unit, _, valid = unit_rows(centered)
        peers = (eligible & valid & valid.T).sum(dim=-1) >= 2
        collective = peers[:, None] & peers[None, :]
        if style == "feature_cosine":
            temperature = torch.where(collective, temperature, 0.5)
        else:
            if median_bandwidth:
                distances = (2 - 2 * (unit @ unit.T).clamp(-1, 1)).clamp_min(0).sqrt()
                distances = distances.masked_fill(~(eligible & collective & valid & valid.T), torch.nan)
                bandwidth = distances.nanmedian().nan_to_num(nan=0.55).clamp(0.25, 1.25)
            bandwidth = torch.where(collective, bandwidth, 0.55)
    direction, nearest, loss = objective(centered, eligible, style, bandwidth, temperature)
    _, size, _ = unit_rows(centered)
    relative = (direction * size).reshape_as(features)
    # Pullback through centering. A common per-sample scale preserves the
    # coarse-grid channel means while respecting the tightest local bound.
    relative = center(relative)
    magnitude = torch.linalg.vector_norm(relative.flatten(1), dim=-1, keepdim=True)
    direction = relative / magnitude.clamp_min(EPS)[..., None, None]
    cell_norm = torch.linalg.vector_norm(direction, dim=1, keepdim=True)
    anchor_norm = torch.linalg.vector_norm(features.nan_to_num(), dim=1, keepdim=True)
    anchor_norm = torch.where(torch.isfinite(anchor_norm), anchor_norm, 0)
    rate = gain * (1 - 0.85 * fidelity) * magnitude / (magnitude + 0.15)
    if adaptive:
        rate = rate * ((nearest - threshold) / (1.0 - threshold)).clamp(0, 1)
    limit = (anchor_norm / cell_norm.clamp_min(EPS)).amin(dim=(-1, -2), keepdim=True)
    shift = direction * limit * rate[..., None, None]
    if areas is not None:
        # Verify a small, area-weighted feature edit against the actual pooled
        # objective. Moving reference rows would certify a different edit.
        if live is not None:
            shift[live:] = 0
        dims = (-1, -2, -3)
        anchor_energy = (features.nan_to_num().square() * areas).sum(dim=dims)
        shift_energy = (shift.square() * areas).sum(dim=dims)
        multiplier = (0.08 * (anchor_energy / shift_energy.clamp_min(EPS)).sqrt()).clamp(max=1)
        shift = shift * multiplier[:, None, None, None]
        result = torch.zeros_like(shift)
        accepted = torch.zeros_like(loss, dtype=torch.bool)
        for scale in (1.0, 0.5, 0.25):
            proposed = shift * scale
            _, _, actual_loss = objective(center(features + proposed).flatten(1), eligible, style, bandwidth, temperature, gradient=False)
            passes = torch.isfinite(actual_loss) & (actual_loss < loss - EPS)
            result = torch.where(passes & ~accepted, proposed, result)
            accepted = accepted | passes
        shift = result
    return shift, nearest.squeeze(-1), loss


def noise_candidate(noise: torch.Tensor, direction: torch.Tensor, rms_budget: float | torch.Tensor) -> torch.Tensor:
    """Rotate each noise tensor toward an ascent direction on its original radius."""
    flat = noise.float().flatten(1)
    drift = direction.float().flatten(1)
    radius = torch.linalg.vector_norm(flat, dim=1, keepdim=True)
    unit = flat / radius.clamp_min(EPS)
    tangent = drift - (drift * unit).sum(dim=1, keepdim=True) * unit
    magnitude = torch.linalg.vector_norm(tangent, dim=1, keepdim=True)
    tangent = tangent / magnitude.clamp_min(EPS)
    chord = rms_budget * math.sqrt(flat.shape[1])
    angle = 2 * (chord / (2 * radius.clamp_min(EPS))).clamp(max=0.5).asin()
    moved = radius * (unit * angle.cos() + tangent * angle.sin())
    moved = torch.where((magnitude > EPS) & (radius > EPS) & torch.isfinite(moved), moved, flat)
    return moved.reshape_as(noise).to(noise.dtype)


def initial_direction(clean: torch.Tensor, refs: tuple[torch.Tensor, torch.Tensor]):
    members = torch.stack([pool(t) for t in (clean, *refs)], dim=1)
    unit, _, _ = unit_rows(members.flatten(2))
    bandwidth = (unit[:, 1] - unit[:, 2]).norm(dim=-1).clamp(0.25, 1.25)[:, None, None]
    pairs = ~torch.eye(3, dtype=torch.bool, device=clean.device)
    direction, _, loss = objective(members.flatten(2), pairs, "feature_volume", bandwidth)
    lifted = lift(direction[:, 0].reshape_as(members[:, 0]), *clean.shape[-2:], adjoint=True)
    return lifted, -loss, bandwidth


def select_candidate(
    original: torch.Tensor,
    candidate: torch.Tensor,
    clean: torch.Tensor,
    candidate_clean: torch.Tensor,
    refs: tuple[torch.Tensor, torch.Tensor],
    initial_volume: torch.Tensor,
    bandwidth: torch.Tensor,
    fidelity: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    members = torch.stack([pool(t).flatten(1) for t in (candidate_clean, *refs)], dim=1)
    pairs = ~torch.eye(3, dtype=torch.bool, device=clean.device)
    _, _, loss = objective(members, pairs, "feature_volume", bandwidth, gradient=False)
    dims = tuple(range(1, clean.ndim))
    displacement = (candidate_clean - clean).square().mean(dim=dims).sqrt()
    scale = clean.square().mean(dim=dims).sqrt().clamp_min(EPS)
    relative = displacement / scale
    gain = -loss - initial_volume
    # Validate against a fresh real model evaluation, with separate volume
    # improvement and clean-prediction trust-region thresholds.
    valid = torch.isfinite(members).all(dim=(1, 2)) & torch.isfinite(relative)
    accepted = valid & (gain > 1e-4) & (relative <= 0.32 - 0.27 * fidelity)
    return torch.where(accepted[:, None, None, None], candidate, original), accepted, gain, relative
