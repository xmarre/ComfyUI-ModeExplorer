"""Packed Kitchen INT8 qualification and ordered activation counterfactuals."""

from dataclasses import dataclass
import math

import torch

from .mechanism import measure_comparisons


class ActivationUnavailable(RuntimeError):
    pass


EAGER = "fp32_row_scale_times_weight_scale_then_product_output_cast_then_bias_add"
CUDA = "fp32_sequential_scales_bias_add_then_output_cast"
CUDA_FMA = "fp32_activation_scale_fused_weight_scale_and_bias_then_output_cast"


class CudaActivationPath:
    """Adapter for the audited Kitchen CUDA INT8 GEMM route."""

    def __init__(self, version):
        if version not in ("0.2.40", "0.2.41"):
            raise ActivationUnavailable("unaudited_kitchen_cuda_int8_version")
        from comfy_kitchen.backends import cuda

        self.cuda = cuda

    def quantize(self, source, rotation, rotate):
        cuda = self.cuda
        rows = source.flatten(0, -2).contiguous()
        m, k = rows.shape
        if m < 3 or (128 * 128) * k > torch.iinfo(torch.int32).max:
            raise ActivationUnavailable("unsupported_kitchen_cuda_int8_gemm_geometry")
        group = rotation.shape[0] if rotation is not None else 0
        # The secondary fused quantizer's predicate is a subset of this one
        # in both audited releases. It cannot be reached by this linear call.
        if (group == 256 and k % group == 0 and group <= k <= cuda._CONVROT_FUSED_MAX_K
                and cuda._convrot_fused_shared_memory_fits(rows, k, group)):
            codes, scales = cuda.quantize_int8_rowwise_convrot64(rows, group)
            return "cuda_convrot64_fused", None, codes, scales

        rotated = rotate(source, rotation).flatten(0, -2)
        codes, scales = cuda.quantize_int8_rowwise(rotated)
        return "cuda_split_convrot" if group else "cuda_rowwise", rotated, codes, scales

    def gemm_candidate(self, source, out_dim):
        cuda = self.cuda
        k = source.shape[-1]
        m = source.numel() // k
        device = source.get_device()
        if (cuda._prefer_turing_fused_int8(m, out_dim, k)
                and cuda._cuda_device_should_use_turing_kernels(device)
                and hasattr(cuda._C, "cutlass_turing_int8_dequant")):
            return "cutlass_turing_with_cublas_fallback"
        if (not cuda._DISABLE_CUTLASS_INT8 and cuda._cuda_device_supports_cutlass_int8_dequant(source)
                and not cuda._prefer_cublas_int8_fallback(m, out_dim, k, device)):
            return "cutlass_with_cublas_fallback"
        return "cublas_separate_dequant"

    def continuous_rotation(self, source, group):
        # These releases use the same four FP32 butterfly stages in this
        # helper and in the fused quantizer. FP32 input retains FP32 output.
        try:
            return self.cuda.rotate_int8_convrot_weight(source.float().contiguous(), group)
        except torch.OutOfMemoryError:
            raise
        except (AttributeError, RuntimeError) as exc:
            raise ActivationUnavailable("kitchen_fp32_convrot_unavailable") from exc


@dataclass
class ActivationArm:
    rotated: torch.Tensor | None
    codes: torch.Tensor
    scales: torch.Tensor


def _chw(value, grid):
    return value.T.unflatten(-1, grid)


class ActivationStudy:
    """Replay the selected activation quantizer and qualify a common epilogue."""

    def __init__(self, reference, spec, interrupt, backend="eager"):
        from comfy_kitchen.backends.eager.quantization import mm_int8, quantize_int8_rowwise
        from comfy_kitchen.tensor.int8_utils import _rotate_activation

        self.reference = reference
        self.cuda_path = CudaActivationPath(reference.contract["comfy_kitchen_version"]) if backend == "cuda" else None
        self.epilogues = ((CUDA, CUDA_FMA, EAGER) if reference.bias is not None else (CUDA, EAGER)) if self.cuda_path else (EAGER,)
        self.quantize = quantize_int8_rowwise
        self.mm_int8 = mm_int8
        self.rotate_activation = _rotate_activation
        self.token_chunk = spec.projection_token_chunk
        self.interrupt = interrupt
        self.codes_t = reference.storage.T.contiguous()
        self.weight_scales = reference.scale.float()

    def _epilogue(self, raw, scales, dtype, model=EAGER):
        bias = self.reference.bias
        if model == EAGER:
            output = (raw.float() * (scales * self.weight_scales)).to(dtype)
            return output if bias is None else output + bias.to(dtype)
        scaled = raw.float() * scales
        if model == CUDA_FMA:
            return torch.addcmul(bias.to(dtype).float(), scaled, self.weight_scales).to(dtype)
        output = scaled * self.weight_scales
        return (output if bias is None else output + bias.to(dtype).float()).to(dtype)

    def _rotate(self, inputs, rotation):
        return inputs if rotation is None else self.rotate_activation(inputs, rotation.to(inputs.dtype), rotation.shape[0])

    def replay(self, inputs, expected, bounds, epilogues):
        rotation = self.reference.rotation
        try:
            if self.cuda_path:
                route, rotated, codes, scales = self.cuda_path.quantize(inputs, rotation, self._rotate)
                gemm = self.cuda_path.gemm_candidate(inputs, expected.shape[-1])
            else:
                rotated = self._rotate(inputs, rotation).flatten(0, -2)
                codes, scales = self.quantize(rotated)
                route, gemm = "eager_rowwise", "eager_int32_matmul"
        except (ActivationUnavailable, torch.OutOfMemoryError):
            raise
        except (AttributeError, RuntimeError) as exc:
            raise ActivationUnavailable("kitchen_activation_quantizer_unavailable") from exc
        native = expected.flatten(0, -2)
        mismatches = torch.zeros(len(epilogues), dtype=torch.int64, device=inputs.device)
        errors_sq = torch.zeros(len(epilogues), dtype=torch.float64, device=inputs.device)
        for start in range(0, native.shape[0], self.token_chunk):
            self.interrupt()
            rows = slice(start, start + self.token_chunk)
            try:
                raw = self.mm_int8(codes[rows], self.codes_t)
            except torch.OutOfMemoryError:
                raise
            except RuntimeError as exc:
                reason = "cuda_int8_accumulation_unavailable" if self.cuda_path else "eager_int8_accumulation_unavailable"
                raise ActivationUnavailable(reason) from exc
            observed = native[rows]
            for index, model in enumerate(epilogues):
                prediction = self._epilogue(raw, scales[rows], native.dtype, model)
                # Compare element bytes, including the sign of zero.
                different = prediction.contiguous().view(torch.uint8) != observed.contiguous().view(torch.uint8)
                mismatches[index].add_(different.unflatten(-1, (observed.shape[-1], observed.element_size())).any(-1).sum())
                errors_sq[index].add_((prediction.double() - observed.double()).square().sum())
        telemetry = {
            model: {"packed_bit_exact": count == 0 and math.isfinite(error), "packed_mismatched_values": count,
                    "packed_l2": error, "packed_rms": error / math.sqrt(native.numel())}
            for model, count, error in zip(epilogues, mismatches.tolist(), errors_sq.sqrt().tolist())
        }
        lo, hi = bounds
        arm = ActivationArm(rotated[lo:hi].clone() if rotated is not None else None, codes[lo:hi].clone(), scales[lo:hi].clone())
        return {"quantizer": route, "gemm_route_candidate": gemm}, telemetry, arm

    def without_quantization(self, source, arm, bounds, dtype, epilogue):
        """Keep row scales and the qualified epilogue while removing input coding."""
        lo, hi = bounds
        source = source[0, lo:hi]
        rows, count = source.shape[0], self.reference.weight.shape[0]
        rounded = torch.empty((rows, count), dtype=dtype, device=source.device)
        rotation = self.reference.rotation
        unrounded = torch.empty_like(rounded) if rotation is not None else rounded
        for start in range(0, rows, self.token_chunk):
            self.interrupt()
            span = slice(start, start + self.token_chunk)
            scales = arm.scales[span]
            continuous = (self.cuda_path.continuous_rotation(source[span], rotation.shape[0])
                          if arm.rotated is None else arm.rotated[span]).double()
            # Continuous row-normalized codes discard activation division
            # rounding, integer rounding and clipping while retaining the
            # same row scale and output epilogue.
            rounded[span] = self._epilogue((continuous / scales.double()) @ self.reference.weight.T, scales, dtype, epilogue)
            if rotation is not None:
                group = rotation.shape[0]
                high_precision = (source[span].double().unflatten(-1, (-1, group)) @ rotation).flatten(-2)
                unrounded[span] = self._epilogue((high_precision / scales.double()) @ self.reference.weight.T, scales, dtype, epilogue)
        return rounded, unrounded


def component_bounds(checks, components, fraction):
    """Conservatively propagate the packed per-arm errors into each response."""
    result = {}
    for name, (magnitude, terms) in components.items():
        error = sum(abs(scale) * checks[arm]["packed_l2"] for scale, arm in terms)
        result[name] = {"native_l2": magnitude, "model_l2_bound": error,
                        "relative_model_bound": error / magnitude if magnitude else None,
                        "qualified": math.isfinite(error) and math.isfinite(magnitude) and error <= fraction * magnitude}
    return result


def select_epilogue(models, components, fraction, allow_bounded):
    candidates = {}
    for model, checks in models.items():
        exact = all(check["packed_bit_exact"] for check in checks.values())
        bounds = component_bounds(checks, components, fraction)
        eligible = exact or (allow_bounded and all(bound["qualified"] for bound in bounds.values()))
        candidates[model] = {
            "qualification": ("bit_exact" if exact else "component_bounded") if eligible else "unqualified",
            "components": bounds,
        }
    qualified = [model for model in models if candidates[model]["qualification"] != "unqualified"]
    selected = min(qualified, default=None, key=lambda model: (
        candidates[model]["qualification"] != "bit_exact",
        max(bound["relative_model_bound"] or 0. for bound in candidates[model]["components"].values()),
    ))
    return selected, candidates


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


def stage_metrics(native, uncoded, unrounded, ideal, grid, spec, patch, chunk, epilogue):
    fields = {name: ((1., _chw(value, grid)),) for name, value in (
        ("native", native), ("no_activation_quantization", uncoded),
        ("no_rotation_rounding", unrounded), ("ideal_affine", ideal),
    )}
    n, q, r, a = ((1., name) for name in fields)
    # Report the ordered counterfactuals alongside the differences. Signed
    # fields add, but the squared energies of their differences can overlap.
    comparisons = {name: ((), ((1., name),)) for name in fields}
    comparisons.update(activation_quantization=((q,), (n, (-1., q[1]))),
                       rotation_rounding=((r,), (q, (-1., r[1]))))
    comparisons["eager_epilogue" if epilogue == EAGER else "epilogue"] = ((a,), (r, (-1., a[1])))
    return measure_comparisons(fields, comparisons, chunk, spec.low_cut, spec.high_cut,
                               sample_stride=patch, patch_size=patch)
