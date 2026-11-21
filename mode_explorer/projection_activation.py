"""Packed Kitchen INT8 qualification and ordered activation counterfactuals."""

from dataclasses import dataclass
import math

import torch

from .mechanism import measure_comparisons


@dataclass
class ActivationArm:
    rotated: torch.Tensor
    codes: torch.Tensor
    scales: torch.Tensor


def _chw(value, grid):
    return value.T.unflatten(-1, grid)


class ActivationStudy:
    """Emulate the audited eager sequence, then remove stages one at a time."""

    def __init__(self, reference, spec, interrupt):
        from comfy_kitchen.backends.eager.quantization import mm_int8, quantize_int8_rowwise
        from comfy_kitchen.tensor.int8_utils import _rotate_activation

        self.reference = reference
        self.quantize = quantize_int8_rowwise
        self.mm_int8 = mm_int8
        self.rotate_activation = _rotate_activation
        self.token_chunk = spec.projection_token_chunk
        self.interrupt = interrupt
        self.codes_t = reference.storage.T.contiguous()
        self.weight_scales = reference.scale.float()

    def _epilogue(self, raw, scales, dtype):
        # Kitchen casts the scaled accumulation before adding the bias.
        output = (raw.float() * (scales * self.weight_scales)).to(dtype)
        if self.reference.bias is not None:
            output = output + self.reference.bias.to(dtype)
        return output

    def qualify(self, inputs, expected, bounds):
        rotated = inputs
        rotation = self.reference.rotation
        if rotation is not None:
            rotated = self.rotate_activation(inputs, rotation.to(inputs.dtype), rotation.shape[0])
        rotated = rotated.flatten(0, -2)
        codes, scales = self.quantize(rotated)
        native = expected.flatten(0, -2)
        mismatch = torch.zeros((), dtype=torch.int64, device=inputs.device)
        error_sq = torch.zeros((), dtype=torch.float64, device=inputs.device)
        for start in range(0, native.shape[0], self.token_chunk):
            self.interrupt()
            rows = slice(start, start + self.token_chunk)
            prediction = self._epilogue(self.mm_int8(codes[rows], self.codes_t), scales[rows], native.dtype)
            observed = native[rows]
            # Compare element bytes, including the sign of zero.
            different = prediction.contiguous().view(torch.uint8) != observed.contiguous().view(torch.uint8)
            mismatch.add_(different.unflatten(-1, (observed.shape[-1], observed.element_size())).any(-1).sum())
            error_sq.add_((prediction.double() - observed.double()).square().sum())
        count, error = int(mismatch), float(error_sq.sqrt())
        telemetry = {"packed_bit_exact": count == 0 and math.isfinite(error), "packed_mismatched_values": count,
                     "packed_l2": error, "packed_rms": error / math.sqrt(native.numel())}
        if not telemetry["packed_bit_exact"]:
            return telemetry, None
        lo, hi = bounds
        return telemetry, ActivationArm(rotated[lo:hi].clone(), codes[lo:hi].clone(), scales[lo:hi].clone())

    def without_quantization(self, source, arm, bounds, dtype):
        """Keep row scales and the eager epilogue while removing input coding."""
        lo, hi = bounds
        source = source[0, lo:hi]
        rows, count = source.shape[0], self.reference.weight.shape[0]
        rounded = torch.empty((rows, count), dtype=dtype, device=source.device)
        rotation = self.reference.rotation
        ideal = torch.empty_like(rounded) if rotation is not None else rounded
        for start in range(0, rows, self.token_chunk):
            self.interrupt()
            span = slice(start, start + self.token_chunk)
            scales = arm.scales[span]
            continuous = arm.rotated[span].double()
            # Continuous row-normalized codes discard activation division
            # rounding, integer rounding and clipping while retaining the
            # same FP32 row-scale/weight-scale product and output epilogue.
            rounded[span] = self._epilogue((continuous / scales.double()) @ self.reference.weight.T, scales, dtype)
            if rotation is not None:
                group = rotation.shape[0]
                high_precision = (source[span].double().unflatten(-1, (-1, group)) @ rotation).flatten(-2)
                ideal[span] = self._epilogue((high_precision / scales.double()) @ self.reference.weight.T, scales, dtype)
        return rounded, ideal


def transitions(base, plus, minus, token_chunk, interrupt):
    """Count code and scale changes over matching image-token rows."""
    keys = ("codes_plus", "codes_minus", "codes_either", "codes_both", "rows_plus", "rows_minus",
            "scales_plus", "scales_minus", "scales_either", "scales_both",
            "codes_plus_fixed_scale", "codes_minus_fixed_scale")
    result = torch.zeros(len(keys), dtype=torch.int64, device=base.codes.device)
    for start in range(0, base.codes.shape[0], token_chunk):
        interrupt()
        span = slice(start, start + token_chunk)
        p = plus.codes[span] != base.codes[span]
        m = minus.codes[span] != base.codes[span]
        ps = plus.scales[span] != base.scales[span]
        ms = minus.scales[span] != base.scales[span]
        values = (p, m, p | m, p & m, p.any(-1), m.any(-1), ps, ms, ps | ms, ps & ms, p & ~ps, m & ~ms)
        result.add_(torch.stack([value.sum() for value in values]))
    counts = dict(zip(keys, result.tolist()))
    counts.update(total_codes=base.codes.numel(), total_rows=base.codes.shape[0])
    counts["codes_one_sign_only"] = counts["codes_either"] - counts["codes_both"]
    counts["scales_one_sign_only"] = counts["scales_either"] - counts["scales_both"]
    return counts


def stage_metrics(native, uncoded, unrounded, ideal, grid, spec, patch, chunk):
    fields = {name: ((1., _chw(value, grid)),) for name, value in (
        ("native", native), ("no_activation_quantization", uncoded),
        ("no_rotation_rounding", unrounded), ("ideal_affine", ideal),
    )}
    n, q, r, a = ((1., name) for name in fields)
    # Report the ordered counterfactuals alongside the differences. Signed
    # fields add, but the squared energies of their differences can overlap.
    comparisons = {name: ((), ((1., name),)) for name in fields}
    comparisons.update(activation_quantization=((q,), (n, (-1., q[1]))),
                       rotation_rounding=((r,), (q, (-1., r[1]))),
                       eager_epilogue=((a,), (r, (-1., a[1]))))
    return measure_comparisons(fields, comparisons, chunk, spec.low_cut, spec.high_cut,
                               sample_stride=patch, patch_size=patch)
