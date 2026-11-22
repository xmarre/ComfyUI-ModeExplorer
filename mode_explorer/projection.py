"""Qualified packed projection replay and local numerical decomposition."""

from dataclasses import dataclass
from importlib.metadata import version
import math

import torch
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode

from .mechanism import measure_comparisons
from .projection_activation import ActivationStudy, ActivationUnavailable, EAGER, select_epilogue, stage_metrics, transitions


class ProjectionUnavailable(ValueError):
    pass


def bitwise_equal(left, right):
    return torch.equal(left.view(torch.uint8), right.view(torch.uint8))


def geometry(tensor):
    return {"shape": list(tensor.shape), "stride": list(tensor.stride()),
            "offset": tensor.storage_offset(), "alignment_mod_16": tensor.data_ptr() % 16,
            "dtype": str(tensor.dtype), "device": str(tensor.device), "layout": str(tensor.layout)}


@dataclass
class ProjectionCapture:
    bounds: tuple
    inputs: torch.Tensor
    input_geometry: dict
    output: torch.Tensor = None
    output_geometry: dict = None


class LinearOperand(TorchFunctionMode):
    """Observe the effective operands of the actual F.linear call in local replay."""

    def __init__(self, accept):
        self.accept = accept
        self.calls = 0
        self.error = None
        self.output = None

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is F.linear:
            self.calls += 1
            if self.calls == 1:
                try:
                    source = args[0] if args else kwargs["input"]
                    weight = args[1] if len(args) > 1 else kwargs["weight"]
                    bias = args[2] if len(args) > 2 else kwargs.get("bias")
                    self.accept(source, weight, bias)
                except ProjectionUnavailable as exc:
                    self.error = str(exc)
            self.output = func(*args, **kwargs)
            return self.output
        return func(*args, **kwargs)


class EffectiveProjection:
    """Same resident/replayed operands, evaluated as an ideal affine map in FP64."""

    def __init__(self, source, weight, bias, analysis_device):
        from comfy.quant_ops import QuantizedTensor, TensorWiseINT8Layout

        self.rotation = None
        self.scale = None
        self.storage = None
        self.bias = bias.detach().to(device=analysis_device, dtype=torch.float64, copy=True) if bias is not None else None
        self.contract = {
            "observed_call": "torch.nn.functional.linear", "linear_input_dtype": str(source.dtype),
            "effective_weight_dtype": str(weight.dtype), "bias_dtype": str(bias.dtype) if bias is not None else None,
            "backend": "unobserved", "accumulation_dtype": "unobserved", "activation_codes": "unobserved",
            "activation_scales": "unobserved",
        }
        if isinstance(weight, QuantizedTensor):
            if weight._layout_cls != "TensorWiseINT8Layout" or not isinstance(weight.params, TensorWiseINT8Layout.Params):
                raise ProjectionUnavailable("unsupported_effective_quantized_layout")
            params = weight.params
            data, scale = TensorWiseINT8Layout.get_plain_tensors(weight)
            if (not params.is_weight or params.transposed or data.dtype != torch.int8 or data.ndim != 2
                    or scale.numel() not in (1, data.shape[0])):
                raise ProjectionUnavailable("unsupported_int8_operand_geometry")
            self.storage = data.detach().to(device=analysis_device, copy=True)
            self.weight = self.storage.to(torch.float64)
            self.scale = scale.detach().to(device=analysis_device, dtype=torch.float64, copy=True).reshape(-1)
            size = params.convrot_groupsize if params.convrot else 0
            if size:
                from comfy_kitchen.tensor.int8_utils import _build_hadamard
                if data.shape[-1] % size or size not in (4, 16, 64, 256, 1024):
                    raise ProjectionUnavailable("unsupported_convrot_group_size")
                self.rotation = _build_hadamard(size, analysis_device, torch.float64)
            self.contract.update(weight_representation="stored_int8_codes", storage_dtype=str(data.dtype),
                                 comfy_kitchen_version=version("comfy-kitchen"), weight_scale_count=scale.numel(),
                                 weight_scale_dtype=str(scale.dtype), convrot_group=size,
                                 input_quantization="dynamic_per_flattened_row_in_audited_kitchen_int8_path",
                                 affine_reference="stored_int8_with_ideal_rotation_and_weight_scales")
            # Registry selection describes the public custom-op dispatch for
            # these operands. It cannot identify the eventual fused/split kernel.
            from comfy_kitchen.exceptions import NoCapableBackendError
            from comfy_kitchen.registry import registry
            try:
                self.contract["backend_dispatch_candidate"] = registry.get_capable_backend("int8_linear", {
                    "x": source, "weight": data, "weight_scale": scale, "bias": bias, "out_dtype": source.dtype,
                    "convrot": params.convrot, "convrot_groupsize": params.convrot_groupsize,
                    "input_act": None, "input_act_weight": None, "input_act_eps": 0.,
                    "residual": None, "residual_scale": None,
                })
            except NoCapableBackendError:
                self.contract["backend_dispatch_candidate"] = "unobserved"
        else:
            if type(weight) not in (torch.Tensor, torch.nn.Parameter) or weight.dtype not in (
                torch.float16, torch.bfloat16, torch.float32, torch.float64
            ):
                raise ProjectionUnavailable("unsupported_effective_dense_weight")
            self.weight = weight.detach().to(device=analysis_device, dtype=torch.float64, copy=True)
            self.contract.update(weight_representation="effective_dense_cast_operand", convrot_group=0,
                                 input_quantization="none_at_observed_linear",
                                 affine_reference="effective_dense_weight")
        self.equal = torch.ones((), device=analysis_device, dtype=torch.bool)

    def check(self, source, weight, bias):
        from comfy.quant_ops import QuantizedTensor, TensorWiseINT8Layout

        if (bias is None) != (self.bias is None) or str(source.dtype) != self.contract["linear_input_dtype"]:
            raise ProjectionUnavailable("effective_operand_changed")
        device = self.weight.device
        if bias is not None:
            self.equal.logical_and_((self.bias == bias.to(device=device, dtype=torch.float64)).all())
        if self.storage is None:
            if isinstance(weight, QuantizedTensor) or self.weight.shape != weight.shape or str(weight.dtype) != self.contract["effective_weight_dtype"]:
                raise ProjectionUnavailable("effective_operand_changed")
            self.equal.logical_and_((self.weight == weight.to(device=device, dtype=torch.float64)).all())
        else:
            if not isinstance(weight, QuantizedTensor) or weight._layout_cls != "TensorWiseINT8Layout":
                raise ProjectionUnavailable("effective_operand_changed")
            p = weight.params
            data, scale = TensorWiseINT8Layout.get_plain_tensors(weight)
            if (not p.is_weight or p.transposed or (p.convrot_groupsize if p.convrot else 0) != self.contract["convrot_group"]
                    or data.shape != self.storage.shape or scale.numel() != self.scale.numel()
                    or str(scale.dtype) != self.contract["weight_scale_dtype"]):
                raise ProjectionUnavailable("effective_operand_changed")
            self.equal.logical_and_((self.storage == data.to(device)).all())
            self.equal.logical_and_((self.scale == scale.to(device=device, dtype=torch.float64).reshape(-1)).all())

    def apply(self, x, *, split=False, bias=False):
        if self.rotation is not None:
            group = self.rotation.shape[0]
            x = (x.unflatten(-1, (-1, group)) @ self.rotation).flatten(-2)
        if split:
            cut = x.shape[-1] // 2
            y = x[..., :cut] @ self.weight[:, :cut].T + x[..., cut:] @ self.weight[:, cut:].T
        else:
            y = x @ self.weight.T
        if self.scale is not None:
            y.mul_(self.scale)
        if bias and self.bias is not None:
            y.add_(self.bias)
        return y


def _chw(value, grid):
    return value.T.unflatten(-1, grid)


def _response_sizes(captures, terms, chunk):
    first = captures[terms[0][1]]
    count = first.bounds[1] - first.bounds[0]
    norm = torch.zeros((), dtype=torch.float64, device=first.output.device)
    for start in range(0, count, chunk):
        stop = min(start + chunk, count)
        delta = sum(scale * captures[name].output[0, captures[name].bounds[0]+start:captures[name].bounds[0]+stop].double()
                    for scale, name in terms)
        norm.add_(delta.square().sum())
    return math.sqrt(float(norm))


def _component_fields(terms, captures, reference, token_chunk, cast_check, interrupt, staged=None):
    device = reference.weight.device
    first = captures[terms[0][1]]
    in_dim, out_dim = first.inputs.shape[-1], first.output.shape[-1]
    count = first.bounds[1] - first.bounds[0]
    input_delta = torch.empty((count, in_dim), dtype=torch.float32, device=device)
    linear = torch.empty((count, out_dim), dtype=torch.float32, device=device)
    residual = torch.empty_like(linear)
    executed = torch.empty_like(linear)
    cast = torch.empty_like(linear) if cast_check else None
    no_quant, no_rotation = (torch.empty_like(linear), torch.empty_like(linear)) if staged is not None else (None, None)
    floor = torch.zeros((), dtype=torch.float64, device=device)

    for t in range(0, count, token_chunk):
        interrupt()
        span = slice(t, min(t + token_chunk, count))

        def gather(name, field):
            capture = captures[name]
            return getattr(capture, field)[0, capture.bounds[0] + span.start:capture.bounds[0] + span.stop].to(torch.float64)

        delta = 0
        output_delta = 0
        cast_delta = 0
        for scale, name in terms:
            activation = gather(name, "inputs")
            delta = delta + scale * activation
            output_delta = output_delta + scale * gather(name, "output")
            if cast_check:
                # Single output cast of the affine map. This is a controlled
                # counterfactual, including when native bias is fused elsewhere.
                y = reference.apply(activation, bias=True)
                cast_delta = cast_delta + scale * (y.to(first.output.dtype).double() - y)
        expected = reference.apply(delta)
        floor.add_((expected - reference.apply(delta, split=True)).square().sum())
        input_delta[span] = delta
        linear[span] = expected
        executed[span] = output_delta
        residual[span] = output_delta - expected
        if cast is not None:
            cast[span] = cast_delta
        if staged is not None:
            no_quant[span] = sum(scale * staged[name][0][span].double() for scale, name in terms)
            no_rotation[span] = sum(scale * staged[name][1][span].double() for scale, name in terms)
    return input_delta, linear, residual, executed, cast, math.sqrt(float(floor)), no_quant, no_rotation


def _replay_error(result, cap, grid, spec, patch, chunk):
    lo, hi = cap.bounds
    if bitwise_equal(result, cap.output):
        return {"packed_bit_exact": True, "packed_l2": 0., "packed_rms": 0.,
                "displacement_l2": 0., "displacement_rms": 0., "relative_output_error": 0.}
    # The fixup case is rare. Keep one error field, and only for image tokens.
    difference = _chw(result[0, lo:hi].float() - cap.output[0, lo:hi].float(), grid)
    metrics = measure_comparisons({"error": ((1., difference),)}, {"replay": ((), ((1., "error"),))},
                                  chunk, spec.low_cut, spec.high_cut, sample_stride=patch, patch_size=patch)["replay"]
    packed_l2 = float(torch.linalg.vector_norm(result.float() - cap.output.float()))
    native_l2 = float(torch.linalg.vector_norm(cap.output.float()))
    metrics.update(packed_bit_exact=False, packed_l2=packed_l2, packed_rms=packed_l2 / math.sqrt(result.numel()),
                   relative_output_error=packed_l2 / native_l2 if native_l2 else None)
    return metrics


def _activation_stages(reference, captures, components, floors, spec, report, timing, interrupt):
    info = report["activation_counterfactual"] = {"status": "unqualified", "packed_emulation": {},
                                                 "reference_matmul_passes": 0}
    if reference.storage is None:
        info["reason"] = "requires_observed_tensorwise_int8_weight"
        return None
    if any(cap.inputs.dtype not in (torch.float16, torch.bfloat16, torch.float32) or
           cap.output.dtype != cap.inputs.dtype for cap in captures.values()):
        info["reason"] = "unsupported_activation_or_output_dtype"
        return None
    backend = reference.contract["backend_dispatch_candidate"]
    info["backend_dispatch_candidate"] = backend
    if backend not in ("cuda", "eager"):
        info["reason"] = "unsupported_activation_backend"
        return None
    try:
        study = ActivationStudy(reference, spec, interrupt, backend)
    except ImportError:
        info["reason"] = "kitchen_int8_contract_unavailable"
        return None
    except ActivationUnavailable as exc:
        info["reason"] = str(exc)
        return None

    arms = {}
    checks = {model: {} for model in study.epilogues}
    info["packed_routes"] = {}
    device = reference.weight.device
    with timing.measure("projection_activation_emulation", device):
        for name, cap in captures.items():
            try:
                route, stats, arms[name] = study.replay(cap.inputs, cap.output, cap.bounds, study.epilogues)
            except ActivationUnavailable as exc:
                info["reason"] = str(exc)
                return None
            info["packed_routes"][name] = route
            for model, check in stats.items():
                checks[model][name] = check
        responses = {name: (floors[name]["native_l2"], terms) for name, terms in components.items()}
        epilogue, candidates = select_epilogue(checks, responses, spec.projection_replay_fraction, backend == "cuda")
        info["epilogue_candidates"] = candidates
        if epilogue is None:
            if backend == "eager":
                info["packed_emulation"] = checks[EAGER]
                info["reason"] = "packed_eager_emulation_mismatch"
            else:
                info["rejected_packed_emulations"] = checks
                info["reason"] = "cuda_epilogue_exceeds_component_floor"
            return None
        selected = candidates[epilogue]
        info.update(packed_emulation=checks[epilogue], epilogue_model=epilogue,
                    qualification=selected["qualification"], component_floors=selected["components"])
        info["quantization_observation"] = (
            "replayed_kitchen_cuda_quantizer_codes_and_row_scales" if backend == "cuda" else
            "eager_emulated_codes_and_row_scales_after_packed_bitwise_match"
        )
        if spec.projection_study == "activation_factorial" and selected["qualification"] != "bit_exact":
            info["reason"] = "factorial_requires_bit_exact_packed_emulation"
            return None
        if spec.projection_study == "activation_factorial" and backend != "cuda" and device.type == "cuda":
            info["reason"] = "factorial_requires_kitchen_cuda_quantizer"
            return None
        qualify_pairs = {}
        if "D0minus" in arms:
            qualify_pairs["full"] = ("D01", "D0minus")
        if "D0half_minus" in arms:
            qualify_pairs["half"] = ("D0half_plus", "D0half_minus")
        info["quantization_transitions"] = {
            label: transitions(arms["D00"], arms[plus], arms[minus], study.token_chunk, interrupt)
            for label, (plus, minus) in qualify_pairs.items()
        }

    staged = {}
    try:
        with timing.measure("projection_activation_reference", device):
            for name, cap in captures.items():
                arm = arms[name] if name == "D00" and spec.projection_symmetric_check else arms.pop(name)
                include_unrounded = spec.projection_study != "activation_factorial"
                staged[name] = study.without_quantization(cap.inputs, arm, cap.bounds, cap.output.dtype, epilogue,
                                                           include_unrounded=include_unrounded)
                info["reference_matmul_passes"] += 1 + int(include_unrounded and reference.rotation is not None)
    except ActivationUnavailable as exc:
        info["reason"] = str(exc)
        return None
    info.update(status="qualified", components={})
    return study, staged, arms.get("D00"), epilogue


def _symmetric_activation(study, epilogue, source, native, bounds, name, info, timing):
    with timing.measure("projection_activation_emulation", source.device):
        try:
            route, checks, arm = study.replay(source, native, bounds, (epilogue,))
        except ActivationUnavailable as exc:
            info["reason"] = str(exc)
            return None
    info["packed_routes"][name] = route
    info["packed_emulation"][name] = checks[epilogue]
    return arm


def _resolved_activation(metrics, model_floor):
    metrics["model_l2_bound"] = model_floor["model_l2_bound"]
    metrics["activation_quantization_resolved"] = (
        metrics["activation_quantization"]["displacement_l2"] > 4 * model_floor["model_l2_bound"]
    )


def _symmetric_probe(captures, evaluate, spec, grid, patch, chunk, report, timing, interrupt, activation):
    base, positive, negative = (captures[key] for key in ("D00", "D01", "D0minus"))
    lo, hi = base.bounds
    device = base.inputs.device
    x = base.inputs.clone(memory_format=torch.preserve_format)
    sums = torch.zeros(2, dtype=torch.float64, device=device)
    kept = torch.zeros((), dtype=torch.int64, device=device)
    with timing.measure("projection_reference", device):
        for start in range(lo, hi, spec.projection_token_chunk):
            interrupt()
            stop = min(start + spec.projection_token_chunk, hi)
            x0 = base.inputs[:, start:stop].double()
            p = positive.inputs[:, start:stop].double()
            m = negative.inputs[:, start:stop].double()
            target = (p - m) * .5
            plus = (x0 + target).to(x.dtype)
            minus = (2 * x0 - plus.double()).to(x.dtype)
            exact = (plus.double() + minus.double() == 2 * x0) & plus.isfinite() & minus.isfinite()
            edit = torch.where(exact, plus.double() - x0, 0.)
            kept.add_(edit.count_nonzero())
            sums.add_(torch.stack((target.square().sum(), edit.square().sum())))
            x[:, start:stop] = x0 + edit

    total, realized = sums.sqrt().tolist()
    info = report["symmetric_check"] = {"status": "unqualified", "input_l2": realized,
                                       "target_input_l2": total, "realized_to_target_l2": realized / total if total else None,
                                       "nonzero_values": int(kept), "exactly_antisymmetric": True}
    if realized == 0:
        info["reason"] = "no_representable_symmetric_values"
        return
    result = evaluate(x, base)
    if activation is not None:
        study, stages, base_arm, epilogue = activation
        stages_info = info["activation_counterfactual"] = {
            "status": "unqualified", "packed_emulation": {}, "packed_routes": {}, "epilogue_model": epilogue,
        }
        plus_arm = _symmetric_activation(study, epilogue, x, result, base.bounds, "plus", stages_info, timing)
        if plus_arm is not None:
            try:
                with timing.measure("projection_activation_reference", device):
                    plus_stages = study.without_quantization(x, plus_arm, base.bounds, result.dtype, epilogue)
            except ActivationUnavailable as exc:
                stages_info["reason"] = str(exc)
                plus_arm = None
    plus = result[0, lo:hi].clone()
    del result
    with timing.measure("projection_reference", device):
        for start in range(lo, hi, spec.projection_token_chunk):
            stop = min(start + spec.projection_token_chunk, hi)
            x[:, start:stop] = 2 * base.inputs[:, start:stop].double() - x[:, start:stop].double()
    result = evaluate(x, base)
    minus = result[0, lo:hi]
    if activation is not None and plus_arm is not None:
        minus_arm = _symmetric_activation(study, epilogue, x, result, base.bounds, "minus", stages_info, timing)
        if minus_arm is not None:
            # The positive image slice is retained by the native probe after
            # its full packed output has been qualified and released.
            native_pair = {"D00": base, "plus": ProjectionCapture((0, hi - lo), None, {}, plus.unsqueeze(0)),
                           "minus": ProjectionCapture(base.bounds, None, {}, result)}
            pair_checks = dict(stages_info["packed_emulation"], D00=report["activation_counterfactual"]["packed_emulation"]["D00"])
            terms = {"odd": ((.5, "plus"), (-.5, "minus")),
                     "even": ((.5, "plus"), (.5, "minus"), (-1., "D00"))}
            sizes = {name: (_response_sizes(native_pair, ts, spec.projection_token_chunk), ts)
                     for name, ts in terms.items()}
            selected, candidates = select_epilogue({epilogue: pair_checks}, sizes, spec.projection_replay_fraction,
                                                   study.cuda_path is not None)
            stages_info["component_floors"] = candidates[epilogue]["components"]
            stages_info["qualification"] = candidates[epilogue]["qualification"]
            if selected is None:
                stages_info["reason"] = ("cuda_epilogue_exceeds_component_floor" if study.cuda_path else
                                         "packed_eager_emulation_mismatch")
                minus_arm = None
        if plus_arm is not None and minus_arm is not None:
            try:
                with timing.measure("projection_activation_reference", device):
                    minus_stages = study.without_quantization(x, minus_arm, base.bounds, result.dtype, epilogue)
            except ActivationUnavailable as exc:
                stages_info["reason"] = str(exc)
                minus_arm = None
        if minus_arm is not None:
            with timing.measure("projection_activation_emulation", device):
                stages_info["quantization_transitions"] = transitions(base_arm, plus_arm, minus_arm, study.token_chunk, interrupt)
            del plus_arm, minus_arm
            c0 = stages["D00"]
            # The ideal map is exactly antisymmetric by construction.
            def even(p, m, b):
                delta = torch.empty_like(b, dtype=torch.float32)
                for start in range(0, b.shape[0], study.token_chunk):
                    interrupt()
                    span = slice(start, start + study.token_chunk)
                    delta[span] = .5 * p[span].double() + .5 * m[span].double() - b[span].double()
                return delta

            with timing.measure("projection_activation_reference", device):
                n = even(plus, minus, base.output[0, lo:hi])
                q, r = (even(p, m, b) for p, m, b in zip(plus_stages, minus_stages, c0))
            with timing.measure("projection_spectral", device):
                stages_info["even"] = stage_metrics(n, q, r, torch.zeros_like(n), grid, spec, patch, chunk, epilogue)
            _resolved_activation(stages_info["even"], stages_info["component_floors"]["even"])
            stages_info.update(status="qualified",
                               reference_matmul_passes=2 * (1 + int(study.reference.rotation is not None)))
    with timing.measure("projection_spectral", device):
        y0 = base.output[0, lo:hi].float()
        pos, neg = plus.float() - y0, minus.float() - y0
        basis = {"plus": ((1., _chw(pos, grid)),), "minus": ((1., _chw(neg, grid)),)}
        metrics = measure_comparisons(basis, {"odd": ((), ((.5, "plus"), (-.5, "minus"))),
                                             "even": ((), ((.5, "plus"), (.5, "minus")))},
                                      chunk, spec.low_cut, spec.high_cut, sample_stride=patch, patch_size=patch)
    info.update(status="measured", output=metrics)


def study_projection(module, captures, grid, spec, patch, chunk, report, timing, seed):
    """Return an eligible downstream displacement, or leave the baseline alone."""
    components = {"positive": ((1., "D01"), (-1., "D00"))}
    if "D0minus" in captures:
        components.update(odd=((.5, "D01"), (-.5, "D0minus")),
                          even=((.5, "D01"), (.5, "D0minus"), (-1., "D00")))
    if "D0half_plus" in captures:
        components.update(half_odd=((.5, "D0half_plus"), (-.5, "D0half_minus")),
                          half_even=((.5, "D0half_plus"), (.5, "D0half_minus"), (-1., "D00")))
    if "D11" in captures:
        components["conditioning_interaction"] = ((1., "D11"), (-1., "D10"), (-1., "D01"), (1., "D00"))
    report.update(status="unqualified", local_projection_calls=0, reference_matmul_passes=0,
                  geometry={key: {"input": cap.input_geometry, "output": cap.output_geometry,
                                  "generated_token_bounds": list(cap.bounds)} for key, cap in captures.items()})
    if any(capture.output is None for capture in captures.values()):
        raise ProjectionUnavailable("packed_projection_capture_incomplete")
    # Additional module hooks may change values or state outside the weight
    # contract. The temporary ModeExplorer hooks have already been removed.
    from torch.nn.modules import module as hooks
    from comfy.model_management import throw_exception_if_processing_interrupted as interrupt

    if (module.training or module._forward_hooks or module._forward_pre_hooks
            or hooks._global_forward_hooks or hooks._global_forward_pre_hooks):
        raise ProjectionUnavailable("projection_training_or_external_module_hooks")
    first = captures["D00"]
    device = first.inputs.device
    report["operator"] = f"{type(module).__module__}.{type(module).__qualname__}"
    report["comfy_quant_format"] = getattr(module, "quant_format", None)
    report["runtime_weight_functions"] = len(getattr(module, "weight_function", ()))
    report["runtime_bias_functions"] = len(getattr(module, "bias_function", ()))
    report["lazy_weight_patch"] = bool(getattr(module, "weight_lowvram_function", None))
    report["input_presmoothing"] = getattr(module, "pre_quant_scale", None) is not None
    reference = None
    replay = report["native_replay"] = {}
    reference_errors = set()

    def evaluate(inputs, cap):
        interrupt()
        if geometry(inputs) != cap.input_geometry or inputs.stride(-1) != 1 or cap.output.stride(-1) != 1:
            raise ProjectionUnavailable("packed_input_snapshot_geometry_mismatch")

        def accept(source, weight, bias):
            nonlocal reference
            if report["input_presmoothing"] or source is not inputs:
                raise ProjectionUnavailable("linear_operand_input_transformed_or_unverified")
            if reference is None:
                reference = EffectiveProjection(source, weight, bias, device)
            else:
                reference.check(source, weight, bias)

        with timing.measure("projection_native_replay", device):
            observer = LinearOperand(accept)
            report["local_projection_calls"] += 1
            with observer:
                result = module(inputs)
            if not torch.is_tensor(result) or geometry(result) != cap.output_geometry:
                raise ProjectionUnavailable("native_projection_replay_geometry_mismatch")
            if observer.error is not None or observer.calls != 1 or observer.output is not result:
                reference_errors.add(observer.error or "single_direct_linear_call_not_observed")
            return result

    for name, cap in captures.items():
        result = evaluate(cap.inputs, cap)
        with timing.measure("projection_spectral", device):
            replay[name] = _replay_error(result, cap, grid, spec, patch, chunk)
        del result
    floors = {}
    with timing.measure("projection_reference", device):
        for name, terms in components.items():
            magnitude = _response_sizes(captures, terms, spec.projection_token_chunk)
            # Use the full packed error to conservatively bound the contribution
            # that could have reached the image slice.
            error = sum(abs(scale) * replay[key]["packed_l2"] for scale, key in terms)
            floors[name] = {"native_l2": magnitude, "replay_l2_bound": error,
                            "relative_replay_bound": error / magnitude if magnitude else None,
                            "qualified": error <= spec.projection_replay_fraction * magnitude}
        for name, stats in replay.items():
            change = (_response_sizes(captures, ((1., name), (-1., "D00")), spec.projection_token_chunk)
                      if name != "D00" else 0.)
            stats["relative_intervention_error"] = stats["displacement_l2"] / change if change else None
    report.update(replay_floors=floors,
                  native_replay_status="qualified" if all(v["qualified"] for v in floors.values()) else "unqualified")
    if report["native_replay_status"] != "qualified":
        raise ProjectionUnavailable("native_replay_exceeds_component_floor")
    if reference_errors or reference is None or not bool(reference.equal):
        raise ProjectionUnavailable(",".join(sorted(reference_errors)) or "effective_operands_not_constant")
    # Replay evidence is stronger than the observed Python module or registry name.
    report["effective_operator"] = reference.contract
    report["owned_effective_weight_bytes"] = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (reference.weight, reference.storage, reference.scale, reference.bias, reference.rotation)
        if tensor is not None
    )
    report["reference_dtype"] = "torch.float64"
    report["reference_equivalence"] = "same_observed_operands_across_packed_native_replays"
    report["components"] = {}
    activation = (_activation_stages(reference, captures, components, floors, spec, report, timing, interrupt)
                  if spec.projection_activation_check else None)
    if spec.projection_study == "activation_factorial":
        if activation is None:
            raise ProjectionUnavailable(report["activation_counterfactual"]["reason"])
        if not all(check["packed_bit_exact"] for check in replay.values()):
            raise ProjectionUnavailable("factorial_requires_bit_exact_packed_native_replay")
        report["status"] = "qualified"
        return {name: values[0] for name, values in activation[1].items()}
    chosen = None
    for name, terms in components.items():
        with timing.measure("projection_reference", device):
            i, linear, residual, native, cast, arithmetic, no_quant, no_rotation = _component_fields(
                terms, captures, reference, spec.projection_token_chunk, spec.projection_cast_check, interrupt,
                activation[1] if activation is not None else None,
            )
        report["reference_matmul_passes"] += 3 + len(terms) * int(spec.projection_cast_check)
        with timing.measure("projection_spectral", device):
            lin = ((1., "linear"),)
            res = ((1., "residual"),)
            ins = ((1., "input"),)
            basis = {"linear": ((1., _chw(linear, grid)),), "residual": ((1., _chw(residual, grid)),),
                     "input": ((1., _chw(i, grid)),)}
            comparisons = {"input": ((), ins), "linear": ((), lin), "executed": ((), lin + res),
                           "residual": ((), res), "accounting": (lin, res)}
            if cast is not None:
                c = ((1., "cast"),)
                basis["cast"] = ((1., _chw(cast, grid)),)
                comparisons.update(output_cast_only=((), c), residual_minus_output_cast=((), res + ((-1., "cast"),)))
            metrics = measure_comparisons(basis, comparisons, chunk, spec.low_cut, spec.high_cut,
                                          sample_stride=patch, patch_size=patch)
        amp = metrics["input"]["displacement_l2"]
        linear_l2 = metrics["linear"]["displacement_l2"]
        residual_l2 = metrics["residual"]["displacement_l2"]
        native_l2 = metrics["executed"]["displacement_l2"]
        replay_floor = floors[name]["replay_l2_bound"]
        # Include FP32 storage error for the direct component and the
        # empirical split/unsplit FP64 reference repeat.
        arithmetic += torch.finfo(torch.float32).eps * (linear_l2 + native_l2 + residual_l2)
        floor = max(arithmetic, replay_floor)
        metrics.update(native_amplification=native_l2 / amp if amp else None, linear_amplification=linear_l2 / amp if amp else None,
                       reference_arithmetic_l2=arithmetic, native_replay_l2_bound=replay_floor,
                       residual_resolved=residual_l2 > 4 * floor, significant_residual_threshold_l2=4 * floor,
                       residual_to_executed_l2=residual_l2 / native_l2 if native_l2 else None)
        report["components"][name] = metrics
        if activation is not None:
            with timing.measure("projection_spectral", device):
                stages_info = report["activation_counterfactual"]
                # Keep `metrics` as the affine record: downstream selection
                # below reads its residual qualification.
                activation_metrics = stages_info["components"][name] = stage_metrics(
                    native, no_quant, no_rotation, linear, grid, spec, patch, chunk, activation[3],
                )
                _resolved_activation(activation_metrics, stages_info["component_floors"][name])
        if spec.projection_effect == name and spec.projection_study != "observe":
            if spec.projection_study == "control" and metrics["residual_resolved"]:
                generator = torch.Generator(device=device).manual_seed(seed)
                control = torch.randn(residual.shape, generator=generator, device=device)
                chosen = control.mul_(residual_l2 / torch.linalg.vector_norm(control))
                report["control_target_l2"] = residual_l2
            elif spec.projection_study in ("linear", "executed") or metrics["residual_resolved"]:
                chosen = {"linear": linear, "residual": residual, "executed": native}[spec.projection_study]
        del i, linear, residual, native, cast, basis, no_quant, no_rotation
    if {"half_odd", "half_even"} <= report["components"].keys():
        ratios = report["half_over_full"] = {}
        for kind in ("odd", "even"):
            full = report["components"][kind]
            half = report["components"][f"half_{kind}"]
            ratios[kind] = {field: half[field]["displacement_l2"] / full[field]["displacement_l2"]
                            if full[field]["displacement_l2"] else None
                            for field in ("input", "linear", "executed", "residual")}
    if spec.projection_symmetric_check:
        _symmetric_probe(captures, evaluate, spec, grid, patch, chunk, report, timing, interrupt, activation)
        if reference_errors or not bool(reference.equal):
            report["symmetric_check"].update(status="unqualified", reason="effective_operands_not_constant")
            report["symmetric_check"].pop("output", None)
            report["symmetric_check"].pop("activation_counterfactual", None)
    report["status"] = "qualified"
    return chosen
