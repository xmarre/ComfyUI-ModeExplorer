"""Bounded, per-layer qualification for a frozen downstream INT8 intervention."""

from contextlib import contextmanager
from contextvars import ContextVar

import torch

from .projection import EffectiveProjection, LinearOperand, ProjectionUnavailable, bitwise_equal, geometry
from .projection_activation import ActivationStudy, ActivationUnavailable
from .runtime import token_slice

_ACTIVE = ContextVar("modeexplorer_continuous_layers", default=None)
_ATTENTION = ("wq", "wk", "wv", "gate", "wo")
_MLP = ("gate", "up", "down")


def layer_plan(dit, injection, selection):
    """Inventory the native post-edit token stream, excluding timestep-only linears."""
    from comfy.quant_ops import QuantizedTensor

    inventory = {}
    for index, block in enumerate(dit.blocks):
        if index < injection:
            continue
        for name in (_ATTENTION if index > injection else ("wo",)):
            inventory[f"b{index}.attn.{name}"] = (block, getattr(block.attn, name))
        for name in _MLP:
            inventory[f"b{index}.mlp.{name}"] = (block, getattr(block.mlp, name))
    inventory["last.linear"] = (dit.blocks[-1], dit.last.linear)

    def format_of(module):
        if isinstance(module.weight, QuantizedTensor):
            return getattr(module.weight, "_layout_cls", "unknown_quantized_layout")
        return getattr(module, "quant_format", None) or "dense"

    formats = {name: format_of(module) for name, (_, module) in inventory.items()}
    supported = {"int8_tensorwise", "TensorWiseINT8Layout"}
    if selection == "downstream":
        requested = [name for name in inventory if formats[name] in supported]
    else:
        requested = [item.strip() for item in selection.split(",")]
        if len(requested) != len(set(requested)) or any(name not in inventory for name in requested):
            raise ProjectionUnavailable("unknown_duplicate_or_pre_injection_projection_layer")
        if any(formats[name] not in supported for name in requested):
            raise ProjectionUnavailable("requested_projection_layer_is_not_tensorwise_int8")
    if not requested:
        raise ProjectionUnavailable("no_downstream_tensorwise_int8_layers")
    targets = {name: target for name, target in inventory.items() if name in requested}
    if len({id(module) for _, module in targets.values()}) != len(targets):
        raise ProjectionUnavailable("shared_downstream_projection_module")
    return targets, formats


def workspace_bytes(targets, packed_rows, token_chunk):
    """Peak workspace for one native replay, activation emulation and FP64 substitution."""
    reserve = max(
        16 * module.in_features * module.out_features
        + 32 * packed_rows * (module.in_features + module.out_features)
        + 64 * token_chunk * (module.in_features + module.out_features)
        for _, module in targets.values()
    )
    return reserve + 16 * 1024**2


class ContinuousLayers:
    """Qualify each actual C/D operand, then substitute only its generated rows."""

    def __init__(self, targets, token_count, packed_limit, spec, timing, report):
        self.targets = targets
        self.token_count = token_count
        self.packed_limit = packed_limit
        self.spec = spec
        self.timing = timing
        self.report = report
        self.contracts = {}

    @contextmanager
    def hooks(self, arm):
        from torch.nn.modules import module as hooks

        # Native forward hooks could change operands or the returned tensor outside our contract.
        if hooks._global_forward_hooks or hooks._global_forward_pre_hooks:
            raise ProjectionUnavailable("projection_global_hooks")
        if any(module.training or module._forward_hooks or module._forward_pre_hooks
               for _, module in self.targets.values()):
            raise ProjectionUnavailable("projection_training_or_external_module_hooks")

        bounds, seen = {}, {}
        self.report["layers"][arm] = seen
        handles = []
        token = _ACTIVE.set(seen)

        def record_bounds(block, args, kwargs):
            if _ACTIVE.get() is seen:
                bounds[block] = token_slice(kwargs["transformer_options"], args[0].shape[1], self.token_count)

        def hook(name, block):
            def replace(module, args, output):
                if _ACTIVE.get() is not seen:
                    return
                if name in seen or block not in bounds or len(args) != 1:
                    raise ProjectionUnavailable(f"{name}:projection_duplicate_or_missing_bounds")
                out, detail = self.replace(name, module, args[0], output, bounds[block])
                seen[name] = detail
                return out
            return replace

        try:
            for block in dict.fromkeys(block for block, _ in self.targets.values()):
                handles.append(block.register_forward_pre_hook(record_bounds, with_kwargs=True))
            for name, (block, module) in self.targets.items():
                handles.append(module.register_forward_hook(hook(name, block)))
            yield
            if seen.keys() != self.targets.keys():
                raise ProjectionUnavailable("some_requested_projection_layers_were_not_observed")
        finally:
            for handle in reversed(handles):
                handle.remove()
            _ACTIVE.reset(token)

    def replace(self, name, module, source, native, bounds):
        from comfy.model_management import throw_exception_if_processing_interrupted as interrupt
        from .response_runtime import _tensor_hash

        lo, hi = bounds
        if (not torch.is_tensor(source) or not torch.is_tensor(native) or source.ndim != 3 or native.ndim != 3
                or source.shape[0] != 1 or source.shape[:2] != native.shape[:2]
                or not 0 <= lo < hi <= source.shape[1] <= self.packed_limit
                or source.dtype not in (torch.float16, torch.bfloat16, torch.float32) or native.dtype != source.dtype
                or source.stride(-1) != 1 or native.stride(-1) != 1):
            raise ProjectionUnavailable(f"{name}:unsupported_packed_activation_geometry")
        reference = None

        def accept(actual, weight, bias):
            nonlocal reference
            if actual is not source or getattr(module, "pre_quant_scale", None) is not None:
                raise ProjectionUnavailable("linear_operand_input_transformed_or_unverified")
            reference = EffectiveProjection(actual, weight, bias, source.device)

        try:
            with self.timing.measure("projection_native_replay", source.device):
                observer = LinearOperand(accept)
                self.report["local_projection_calls"] += 1
                # Call the same forward directly to avoid re-entering this hook.
                with observer:
                    replayed = module.forward(source)
                if observer.error or observer.calls != 1 or observer.output is not replayed or reference is None:
                    raise ProjectionUnavailable(observer.error or "single_direct_linear_call_not_observed")
                if geometry(replayed) != geometry(native) or not bitwise_equal(replayed, native):
                    raise ProjectionUnavailable("packed_native_replay_not_bit_exact")
                del replayed
            if reference.storage is None:
                raise ProjectionUnavailable("requires_observed_tensorwise_int8_weight")
            backend = reference.contract["backend_dispatch_candidate"]
            if backend != ("cuda" if source.is_cuda else "eager"):
                raise ProjectionUnavailable("requires_audited_native_activation_quantizer")
            with self.timing.measure("factorial_operand_verification", source.device):
                weights = [_tensor_hash(t) for t in (reference.storage, reference.scale, reference.bias)]
            with self.timing.measure("projection_activation_emulation", source.device):
                study = ActivationStudy(reference, self.spec, interrupt, backend)
                route, checks, arm = study.replay(source, native, bounds, study.epilogues)
                prior = self.contracts.get(name)
                exact = [key for key in study.epilogues if checks[key]["packed_bit_exact"]]
                epilogue = (exact[0] if exact else None) if prior is None else prior["epilogue"]
                if epilogue not in exact:
                    raise ProjectionUnavailable("no_common_bit_exact_activation_epilogue")
                contract = {"epilogue": epilogue, "route": route, "weights": weights,
                            "effective_operator": reference.contract, "input": geometry(source),
                            "output": geometry(native), "generated_token_bounds": list(bounds)}
                if prior is not None and prior != contract:
                    raise ProjectionUnavailable("effective_operator_or_packed_geometry_changed_between_arms")
                self.contracts[name] = contract
            with self.timing.measure("projection_activation_reference", source.device):
                values, _ = study.without_quantization(source, arm, bounds, native.dtype, epilogue, include_unrounded=False)
                if not bool(torch.isfinite(values).all()):
                    raise ProjectionUnavailable("nonfinite_continuous_output")
                result = native.clone()
                result[0, lo:hi] = values
            return result, {**contract, "packed_native_replay_bit_exact": True,
                            "activation_emulation": checks[epilogue], "continuous_matmul_dtype": "torch.float64"}
        except (ActivationUnavailable, ProjectionUnavailable, ImportError) as exc:
            raise ProjectionUnavailable(f"{name}:{exc}") from exc
