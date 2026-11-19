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


def cell_size(height: int, width: int) -> tuple[int, int]:
    return math.ceil(height / 4), math.ceil(width / 4)


def pool(images: torch.Tensor) -> torch.Tensor:
    extent = cell_size(*images.shape[-2:])
    return F.avg_pool2d(images, extent, stride=extent, ceil_mode=True).float()


def token_pool(tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return pool(tokens.transpose(1, 2).unflatten(-1, (height, width)))


def lift(cells: torch.Tensor, height: int, width: int, *, adjoint: bool = False) -> torch.Tensor:
    """Broadcast cell offsets, or divide by the actual cell area for the pooling adjoint."""
    sy, sx = cell_size(height, width)
    if adjoint:
        ny = (height - sy * torch.arange(cells.shape[-2], device=cells.device)).clamp(max=sy)
        nx = (width - sx * torch.arange(cells.shape[-1], device=cells.device)).clamp(max=sx)
        cells = cells / (ny[:, None] * nx[None, :])
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            logits = (cosine / 0.5).masked_fill(~eligible, -torch.inf)
            weights = logits.flatten(-2).softmax(dim=-1).unflatten(-1, cosine.shape[-2:]).nan_to_num()
            grad_unit = (2 / 0.5) * (weights @ unit)
            count = eligible.sum(dim=(-1, -2)).clamp_min(1)
            loss = logits.flatten(-2).logsumexp(dim=-1) - count.log()
            loss = torch.where(eligible.any(dim=-1).any(dim=-1), loss, 0)
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
            inverse_factor = torch.linalg.solve_triangular(factor, identity.expand_as(factor), upper=False)
            inv = inverse_factor.transpose(-1, -2) @ inverse_factor
            weights = inv * kernel
            grad_unit = (2 / bw2) * (weights.sum(dim=-1, keepdim=True) * unit - weights @ unit)
            grad_unit = torch.where(ok, grad_unit, 0)
            loss = -2 * factor.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    direction, nearest, loss = objective(features.flatten(1), eligible, style)
    # Restore the scale of dL/d(unit feature). Saturate the update to avoid
    # amplifying near-zero gradients.
    values = features.flatten(1)
    _, size, _ = unit_rows(values)
    relative = direction * size
    magnitude = torch.linalg.vector_norm(relative, dim=-1, keepdim=True)
    direction = (relative / magnitude.clamp_min(EPS)).reshape_as(features)
    cell_norm = torch.linalg.vector_norm(direction, dim=1, keepdim=True)
    anchor_norm = torch.linalg.vector_norm(features.nan_to_num(), dim=1, keepdim=True)
    anchor_norm = torch.where(torch.isfinite(anchor_norm), anchor_norm, 0)
    rate = gain * (1 - 0.85 * fidelity) * magnitude / (magnitude + 0.15)
    if adaptive:
        rate = rate * ((nearest - threshold) / (1.0 - threshold)).clamp(0, 1)
    # Relative norm of each coarse-cell offset is bounded against its local anchor.
    correction = direction * anchor_norm / cell_norm.clamp_min(1.0)
    return correction * rate[..., None, None], nearest.squeeze(-1), loss


def noise_candidate(noise: torch.Tensor, direction: torch.Tensor, rms_budget: float) -> torch.Tensor:
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
    moved = torch.where((magnitude > EPS) & (radius > EPS), moved, flat)
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
    _, _, loss = objective(members, pairs, "feature_volume", bandwidth)
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
