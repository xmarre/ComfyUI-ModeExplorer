"""Full-resolution spatial and spectral measurements of native model tensors."""

from dataclasses import dataclass
from functools import cached_property
import math

import torch

SITES = (
    "attention_pre", "attention_post", "gate_logits", "sigmoid_gate",
    "projection_input", "projection_output", "attention_contribution",
    "after_attention", "mlp_input", "mlp_output", "mlp_contribution", "after_mlp",
    "final_projection", "velocity", "clean",
)
UNAVAILABLE = {"sigmoid_gate", "attention_contribution", "mlp_contribution"}
MODES = ("signed", "conditioning_fixed", "conditioning_recomputed", "joint_fixed", "joint_recomputed")
COMPONENTS = ("observe", "complete", "even", "odd", "interaction", "control")
FILTERS = ("all", "low", "mid", "high")


@dataclass(frozen=True)
class MechanisticSettings:
    mode: str = "signed"
    blocks: str = "5"
    sites: str = "velocity,clean"
    chunk: int = 8
    capture_mib: int = 2048
    low_cut: float = 0.125
    high_cut: float = 0.25
    transition: float = 0.0
    component: str = "observe"
    band: str = "all"
    cell: int = -1
    symmetry_limit: float = 0.1
    amplitude_check: bool = False
    alternate: object = None
    checkpoint_provenance: str = ""
    match_cross_dtype: bool = False

    @cached_property
    def block_ids(self):
        result = []
        for field in self.blocks.split(","):
            endpoints = [int(n) for n in field.split("-")]
            if len(endpoints) not in (1, 2) or not 0 <= endpoints[0] <= endpoints[-1] < 128:
                raise ValueError("Enter block numbers or increasing ranges between 0 and 127.")
            result.extend(range(endpoints[0], endpoints[-1] + 1))
        return tuple(dict.fromkeys(result))

    @cached_property
    def site_names(self):
        return tuple(dict.fromkeys(name.strip() for name in self.sites.split(",") if name.strip()))

    @property
    def conditioning(self):
        return self.mode != "signed"

    @property
    def signed(self):
        return self.mode == "signed" or self.mode.startswith("joint")

    def __post_init__(self):
        if self.mode not in MODES or self.component not in COMPONENTS or self.band not in FILTERS:
            raise ValueError("Unknown mechanistic mode, component or frequency band.")
        if set(self.site_names) - set(SITES) or not self.block_ids:
            raise ValueError("Unknown measurement site or empty block selection.")
        if not 1 <= self.chunk <= 64 or not 16 <= self.capture_mib <= 8192:
            raise ValueError("Chunk and capture budget are outside the supported range.")
        if not 0 < self.low_cut < self.high_cut < math.sqrt(0.5) or not 0 <= self.transition <= 0.25:
            raise ValueError("Frequency cutoffs must increase within the spatial Nyquist grid.")
        if not 0 <= self.symmetry_limit <= 1 or not -1 <= self.cell <= 255:
            raise ValueError("Symmetry tolerance or cell is outside its supported range.")
        if self.conditioning and self.alternate is None:
            raise ValueError("Connect alternate conditioning for the conditioning experiment.")
        if self.component in ("even", "odd") and not self.signed:
            raise ValueError("The odd and even components require a signed experiment.")
        if self.component == "interaction" and not self.conditioning:
            raise ValueError("The interaction component requires alternate conditioning.")
        if self.amplitude_check and not self.signed:
            raise ValueError("The amplitude check requires opposite-signed arms.")


def radius(shape, device):
    h, w = shape[-2:]
    fy = torch.fft.fftfreq(h, device=device)
    fx = torch.fft.rfftfreq(w, device=device)
    return (fy[:, None].square() + fx[None, :].square()).sqrt()


def band_weights(shape, low, high, device):
    radii = radius(shape, device)
    dc = radii == 0
    labels = (dc, (~dc) & (radii <= low), (radii > low) & (radii <= high), radii > high)
    w = shape[-1]
    weights = torch.full((w // 2 + 1,), 2.0, device=device)
    weights[0] = 1
    if w % 2 == 0:
        weights[-1] = 1
    return torch.stack(labels).flatten(1).to(weights.dtype) * weights.repeat(shape[-2])


def combination(terms, channel):
    result = terms[0][1][channel].float() * terms[0][0]
    for scale, tensor in terms[1:]:
        result.add_(tensor[channel].float(), alpha=scale)
    return result


def paired_measurement(baseline, edited, chunk=8, low=0.125, high=0.25):
    """Measure two channel-first fields, optionally specified as linear combinations."""
    baseline = ((1.0, baseline),) if torch.is_tensor(baseline) else baseline
    edited = ((1.0, edited),) if torch.is_tensor(edited) else edited
    fields = [tensor for _, tensor in (*baseline, *edited)]
    shape = fields[0].shape
    if len(shape) != 3 or any(t.shape != shape or t.device != fields[0].device for t in fields):
        raise ValueError("Incompatible channel-first spatial tensors.")
    device = fields[0].device
    weights = band_weights(shape, low, high, device).double().T
    powers = torch.zeros((4, 4), dtype=torch.float64, device=device)
    totals = torch.zeros(4, dtype=torch.float64, device=device)
    spatial = torch.zeros(shape[-2:], dtype=torch.float64, device=device)
    for i in range(0, shape[0], chunk):
        channels = slice(i, i + chunk)
        base = combination(baseline, channels)
        arm = combination(edited, channels)
        delta = arm - base
        spatial += delta.square().sum(dim=0)
        totals += torch.stack((base.square().sum(dtype=torch.float64), arm.square().sum(dtype=torch.float64),
                               delta.square().sum(dtype=torch.float64), (base * arm).sum(dtype=torch.float64)))
        a = torch.fft.rfft2(base, norm="ortho")
        d = torch.fft.rfft2(delta, norm="ortho")
        b = a + d
        per_bin = torch.stack((a.abs().square(), b.abs().square(), d.abs().square(),
                               2 * (a.conj() * d).real)).sum(dim=1).flatten(1)
        powers += per_bin.double() @ weights
    if not torch.isfinite(totals).all() or not torch.isfinite(powers).all():
        raise ValueError("Nonfinite spatial or spectral measurement.")
    base, arm, delta, dot = totals.tolist()
    n = math.prod(shape)
    bn, an, dn = math.sqrt(base), math.sqrt(arm), math.sqrt(delta)
    cosine = max(-1., min(1., dot / (bn * an))) if bn * an else None
    ordered = spatial.flatten()
    top = {str(p): float(ordered.topk(max(1, math.ceil(ordered.numel() * p / 100))).values.sum()) / delta
           if delta else None for p in (1, 10, 25)}
    # Sum disjoint spatial tiles so the map is comparable to total displacement
    # energy and never mixes channels, overlapping windows, or normalized activations.
    h, w = shape[-2:]
    ty, tx = min(4, h), min(4, w)
    tiles = [[float(spatial[i*h//ty:(i+1)*h//ty, j*w//tx:(j+1)*w//tx].sum())
              for j in range(tx)] for i in range(ty)]
    result = {}
    for name, (b, a, d, cross) in zip(("dc", "low", "mid", "high"), powers.T.tolist()):
        result[name] = {
            "baseline": b, "edited": a, "displacement": d, "cross_term": cross, "net_change": d + cross,
            "baseline_fraction": b / base if base else None, "edited_fraction": a / arm if arm else None,
            "displacement_fraction": d / delta if delta else None,
            "relative_net_change": (d + cross) / b if b else None,
        }
    return {
        "shape_chw": list(shape), "baseline_dtype": str(baseline[0][1].dtype),
        "edited_dtype": str(edited[0][1].dtype), "baseline_l2": bn, "edited_l2": an,
        "displacement_l2": dn, "baseline_rms": bn / math.sqrt(n), "edited_rms": an / math.sqrt(n),
        "displacement_rms": dn / math.sqrt(n), "relative_displacement": dn / bn if bn else None,
        "cosine": cosine, "normalized_displacement": math.sqrt(max(0., 2 - 2 * cosine)) if cosine is not None else None,
        "top_spatial_percent_energy_fraction": top, "spatial_tile_energy": tiles, "bands": result,
    }


def filter_field(field, band, low, high, transition=0.0):
    if band == "all":
        return field
    r = radius(field.shape, field.device)

    def low_pass(cut):
        if transition == 0:
            return (r <= cut).float()
        t = ((r - cut + transition / 2) / transition).clamp(0, 1)
        return (1 + torch.cos(torch.pi * t)) * 0.5

    first, second = low_pass(low), low_pass(high)
    mask = {"low": first, "mid": second - first, "high": 1 - second}[band]
    freq = torch.fft.rfft2(field.float(), norm="ortho")
    return torch.fft.irfft2(freq * mask, s=field.shape[-2:], norm="ortho")


def signed_terms(base, plus, minus):
    return ((0.5, plus), (-0.5, minus)), ((0.5, plus), (0.5, minus), (-1.0, base))


def conditioning_terms(d00, d10, d01, d11):
    return ((1.0, d10), (-1.0, d00)), ((1.0, d11), (-1.0, d01))
