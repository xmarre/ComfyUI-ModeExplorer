"""Full-resolution spatial and spectral measurements of native model tensors."""

from dataclasses import dataclass
from functools import cached_property
import math

import torch

SITES = (
    "attention_pre", "attention_post", "gate_logits", "sigmoid_gate",
    "projection_input", "projection_output", "attention_contribution",
    "after_attention", "mlp_input", "mlp_output", "mlp_contribution", "after_mlp",
    "final_projection", "final_projection_unpatched", "velocity", "clean",
)
UNAVAILABLE = {"sigmoid_gate", "attention_contribution", "mlp_contribution"}
MODES = ("signed", "conditioning_fixed", "conditioning_recomputed", "joint_fixed", "joint_recomputed")
COMPONENTS = ("observe", "complete", "even", "odd", "interaction", "control")
FILTERS = ("all", "low", "mid", "high", "sub_patch")


@dataclass(frozen=True)
class MechanisticSettings:
    mode: str = "signed"
    blocks: str = "5"
    sites: str = "velocity,clean"
    chunk: int = 8
    capture_mib: int = 2048
    low_cut: float = 0.0625
    high_cut: float = 0.125
    transition: float = 0.0
    component: str = "observe"
    band: str = "all"
    cell: int = -1
    symmetry_limit: float = 0.1
    amplitude_check: bool = False
    alternate: object = None
    checkpoint_provenance: str = ""
    match_cross_dtype: bool = False
    analysis_device: str = "cuda"
    capture_vram_mib: int = 8192
    cuda_chunk: int = 32

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

    @property
    def core_arms(self):
        return 2 + int(self.signed) + 2 * int(self.conditioning)

    def __post_init__(self):
        if self.mode not in MODES or self.component not in COMPONENTS or self.band not in FILTERS:
            raise ValueError("Unknown mechanistic mode, component or frequency band.")
        if set(self.site_names) - set(SITES) or not self.block_ids:
            raise ValueError("Unknown measurement site or empty block selection.")
        if not 1 <= self.chunk <= 64 or not 16 <= self.capture_mib <= 8192:
            raise ValueError("Chunk and capture budget are outside the supported range.")
        if self.analysis_device not in ("cuda", "cpu"):
            raise ValueError("Analysis device must be cuda or cpu.")
        if not 16 <= self.capture_vram_mib <= 32768 or not 1 <= self.cuda_chunk <= 128:
            raise ValueError("CUDA channel chunk and capture budget are outside the supported range.")
        if not 0 < self.low_cut < self.high_cut < math.sqrt(0.5) / 2 or not 0 <= self.transition <= 0.25:
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


def frequency_grid(shape, device, sample_stride=1, patch_size=None):
    h, w = shape[-2:]
    fy = torch.fft.fftfreq(h, d=sample_stride, device=device)[:, None]
    fx = torch.fft.rfftfreq(w, d=sample_stride, device=device)[None, :]
    common = (fy.abs() <= .5 / patch_size) & (fx <= .5 / patch_size) if patch_size else None
    return (fy.square() + fx.square()).sqrt(), common


def band_weights(shape, low, high, device, sample_stride=1, patch_size=None):
    radii, common = frequency_grid(shape, device, sample_stride, patch_size)
    dc = radii == 0
    marks = {"dc": dc, "low": (~dc) & (radii <= low), "mid": (radii > low) & (radii <= high),
             "high": radii > high}
    if common is not None:
        marks = {name: mask & common for name, mask in marks.items()}
        marks["sub_patch"] = ~common
    w = shape[-1]
    weights = torch.full((w // 2 + 1,), 2.0, device=device)
    weights[0] = 1
    if w % 2 == 0:
        weights[-1] = 1
    return tuple(marks), torch.stack(tuple(marks.values())).flatten(1).to(weights.dtype) * weights.repeat(shape[-2])


def combination(terms, fields, zero):
    """Combine only a channel batch. Preserve cached input and Fourier fields."""
    if not terms:
        return zero
    scale, name = terms[0]
    if len(terms) == 1 and scale == 1:
        return fields[name]
    result = fields[name] * scale
    for scale, name in terms[1:]:
        result.add_(fields[name], alpha=scale)
    return result


def measure_comparisons(basis, comparisons, chunk=8, low=.125, high=.25, *, sample_stride=1, patch_size=None):
    """Measure (baseline, displacement) in a shared basis, with one FFT per basis field and channel."""
    source = [value for terms in basis.values() for _, value in terms]
    shape, device = source[0].shape, source[0].device
    if len(shape) != 3 or any(value.shape != shape or value.device != device for value in source):
        raise ValueError("Incompatible channel-first spatial tensors.")

    names, weights = band_weights(shape, low, high, device, sample_stride, patch_size)
    counts = weights.sum(dim=1)
    n = math.prod(shape)
    h, w = shape[-2:]
    power = torch.zeros((len(comparisons), 4, len(names)), dtype=torch.float64, device=device)
    totals = torch.zeros((len(comparisons), 4), dtype=torch.float64, device=device)
    spatial = torch.zeros((len(comparisons), h, w), device=device)
    for i in range(0, shape[0], chunk):
        ch = slice(i, i + chunk)
        # Use one cached FP32 cast per distinct source tensor in the batch.
        values = {id(value): value[ch].float() for value in {id(t): t for t in source}.values()}
        fields = {name: combination([(scale, id(value)) for scale, value in terms], values, None)
                  for name, terms in basis.items()}
        packed = torch.stack(tuple(fields.values()))
        # Center channels before FFT and restore DC analytically. A large
        # offset otherwise leaks through FP32 round-off into small AC responses.
        mean = packed.mean(dim=(-1, -2), keepdim=True)
        packed.sub_(mean)
        transformed = torch.fft.rfft2(packed, norm="ortho")
        transformed[..., 0, 0] = mean[..., 0, 0] * math.sqrt(h * w)
        spectra = dict(zip(fields, transformed.unbind()))
        zero = torch.zeros_like(packed[0])
        zero_fft = torch.zeros_like(transformed[0])
        for idx, (base, change) in enumerate(comparisons.values()):
            a = combination(base, fields, zero)
            d = combination(change, fields, zero)
            b = a + d
            energy = d.square()
            spatial[idx].add_(energy.sum(dim=0))
            totals[idx].add_(torch.stack((a.square().sum(), b.square().sum(), energy.sum(), (a * b).sum())))

            af = combination(base, spectra, zero_fft)
            df = combination(change, spectra, zero_fft)
            bf = af + df
            per_bin = torch.stack(((af.real.square() + af.imag.square()).sum(0),
                                   (bf.real.square() + bf.imag.square()).sum(0),
                                   (df.real.square() + df.imag.square()).sum(0),
                                   2 * (af.real * df.real + af.imag * df.imag).sum(0))).flatten(1)
            # Explicit FP32 products and reductions avoid TF32 matmul settings on CUDA.
            # Accumulate only the small chunk summaries in FP64.
            power[idx].add_((per_bin[:, None] * weights).sum(dim=-1))

    ordered = spatial.flatten(1)
    top = [ordered.topk(max(1, math.ceil(h * w * p / 100)), dim=1).values.sum(dim=1)
           for p in (1, 10, 25)]
    ty, tx = min(4, h), min(4, w)
    tiles = [spatial[:, i*h//ty:(i+1)*h//ty, j*w//tx:(j+1)*w//tx].sum(dim=(1, 2))
             for i in range(ty) for j in range(tx)]
    summary = torch.cat((totals, power.transpose(1, 2).flatten(1), torch.stack(top + tiles, dim=1)), dim=1)
    # Move only compact results to the host, once per site.
    host = torch.cat((counts.expand(len(comparisons), -1), summary), dim=1).cpu()
    if not torch.isfinite(host).all():
        raise ValueError("Nonfinite spatial or spectral measurement.")
    dtype = str(source[0].dtype)
    return {name: _measurement(row, names, shape, dtype, n, ty, tx)
            for name, row in zip(comparisons, host.tolist())}


def _measurement(row, names, shape, dtype, n, ty, tx):
    counts = row[:len(names)]
    base, arm, delta, dot = row[len(names):len(names)+4]
    offset = len(names) + 4
    bn, an, dn = math.sqrt(base), math.sqrt(arm), math.sqrt(delta)
    cosine = max(-1., min(1., dot / (bn * an))) if bn * an else None
    bands = {}
    for i, name in enumerate(names):
        b, a, d, cross = row[offset + 4*i:offset + 4*i + 4]
        bands[name] = {
            "frequency_bins": int(counts[i]), "baseline": b, "edited": a, "displacement": d,
            "cross_term": cross, "net_change": d + cross,
            "baseline_fraction": b / base if base else None, "edited_fraction": a / arm if arm else None,
            "displacement_fraction": d / delta if delta else None,
            "relative_net_change": (d + cross) / b if b else None,
        }
    offset += 4 * len(names)
    top = {str(p): value / delta if delta else None for p, value in zip((1, 10, 25), row[offset:offset+3])}
    tiles = row[offset+3:]
    return {
        "shape_chw": list(shape), "baseline_dtype": dtype, "edited_dtype": dtype,
        "baseline_l2": bn, "edited_l2": an, "displacement_l2": dn,
        "baseline_rms": bn / math.sqrt(n), "edited_rms": an / math.sqrt(n),
        "displacement_rms": dn / math.sqrt(n), "relative_displacement": dn / bn if bn else None,
        "cosine": cosine, "normalized_displacement": math.sqrt(max(0., 2 - 2 * cosine)) if cosine is not None else None,
        "top_spatial_percent_energy_fraction": top,
        "spatial_tile_energy": [tiles[i*tx:(i+1)*tx] for i in range(ty)], "bands": bands,
    }


def paired_measurement(baseline, edited, chunk=8, low=.125, high=.25, *, sample_stride=1, patch_size=None):
    """Measure two channel-first fields, optionally specified as linear combinations."""
    baseline = ((1., baseline),) if torch.is_tensor(baseline) else baseline
    edited = ((1., edited),) if torch.is_tensor(edited) else edited
    basis = {"base": baseline, "delta": (*edited, *((-scale, value) for scale, value in baseline))}
    result = measure_comparisons(basis, {"pair": (((1., "base"),), ((1., "delta"),))},
                                 chunk, low, high, sample_stride=sample_stride, patch_size=patch_size)["pair"]
    result["edited_dtype"] = str(edited[0][1].dtype)
    return result


def filter_field(field, band, low, high, transition=0.0, *, patch_size=1):
    if band == "all":
        return field
    r, common = frequency_grid(field.shape, field.device, patch_size=patch_size)

    def low_pass(cut):
        if transition == 0:
            return (r <= cut).float()
        t = ((r - cut + transition / 2) / transition).clamp(0, 1)
        return (1 + torch.cos(torch.pi * t)) * 0.5

    first, second = low_pass(low), low_pass(high)
    mask = ~common if band == "sub_patch" else {"low": first, "mid": second - first, "high": 1 - second}[band] * common
    freq = torch.fft.rfft2(field.float(), norm="ortho")
    return torch.fft.irfft2(freq * mask, s=field.shape[-2:], norm="ortho")


def signed_terms(base, plus, minus):
    return ((0.5, plus), (-0.5, minus)), ((0.5, plus), (0.5, minus), (-1.0, base))


def conditioning_terms(d00, d10, d01, d11):
    return ((1.0, d10), (-1.0, d00)), ((1.0, d11), (-1.0, d01))
