import pytest
import torch

pytest.importorskip("comfy")
from comfy.cli_args import args

args.cpu = True
from comfy import ops
from comfy.quant_ops import QuantizedTensor

from mode_explorer.mechanism import MechanisticSettings
from mode_explorer.mechanism_runtime import ResearchTiming
from mode_explorer.projection import ProjectionCapture, ProjectionUnavailable, geometry, study_projection


def record(module, values, bounds=(1, 7)):
    result = {}
    for name, x in values.items():
        y = module(x)
        result[name] = ProjectionCapture(bounds, x.clone(), geometry(x), y.clone(), geometry(y))
    return result


def run_study(module, captures, **options):
    spec = MechanisticSettings(projection_study="observe", **options)
    report = {}
    study_projection(module, captures, (2, 3), spec, 2, 4, report, ResearchTiming(), 9)
    assert report["status"] == "qualified"
    return report


@pytest.mark.parametrize("even", [True, False])
def test_direct_component_handles_large_baseline_and_amplified_asymmetry(even):
    with torch.inference_mode():
        layer = torch.nn.Linear(2, 2, dtype=torch.float32).eval()
        layer.weight.copy_(torch.diag(torch.tensor([1., 128.])))
        layer.bias.fill_(64)
        x0 = torch.full((1, 8, 2), 16.)
        odd = torch.tensor([.5, 0.])
        asym = torch.tensor([0., .015625 if even else 0.])
        report = run_study(layer, record(layer, {"D00": x0, "D01": x0 + odd + asym, "D0minus": x0 - odd + asym}))
    expected = 2. * (6.**.5) if even else 0.
    entry = report["components"]["even"]
    assert entry["executed"]["displacement_l2"] == pytest.approx(expected)
    assert entry["linear"]["displacement_l2"] == pytest.approx(expected)
    assert entry["residual"]["displacement_l2"] == 0
    assert not entry["residual_resolved"]
    assert report["effective_operator"]["bias_dtype"] == "torch.float32"


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable"))])
def test_actual_kitchen_int8_convrot_observation_and_symmetric_probe(device):
    import comfy_kitchen as kitchen

    with torch.inference_mode(), kitchen.use_backend("eager"):
        layer = ops.mixed_precision_ops(compute_dtype=torch.float16).Linear(8, 8, device=device).eval()
        gen = torch.Generator().manual_seed(119)
        weight = torch.randn((8, 8), generator=gen).to(device=device, dtype=torch.float16)
        layer.weight = torch.nn.Parameter(QuantizedTensor.from_float(
            weight, "TensorWiseINT8Layout", convrot=True, convrot_groupsize=4, per_channel=True,
        ), requires_grad=False)
        layer.quant_format = "int8_tensorwise"
        layer.layout_type = "TensorWiseINT8Layout"
        layer.bias.fill_(.25)
        x0 = torch.randn((1, 8, 8), generator=gen).to(device=device, dtype=torch.float16)
        delta = torch.randn((1, 8, 8), generator=gen).to(device=device, dtype=torch.float16) * .005
        snapshots = record(layer, {"D00": x0, "D01": x0 + delta, "D0minus": x0 - delta})
        weight_before = layer.weight
        report = run_study(layer, snapshots, projection_symmetric_check=True, projection_cast_check=True,
                           projection_activation_check=True)
        assert layer.weight is weight_before
    contract = report["effective_operator"]
    assert contract["weight_representation"] == "stored_int8_codes"
    assert contract["convrot_group"] == 4 and contract["weight_scale_count"] == 8
    assert contract["backend_dispatch_candidate"] == "eager"
    assert report["local_projection_calls"] == 5
    activation = report["activation_counterfactual"]
    assert activation["status"] == "qualified", activation
    assert all(item["packed_bit_exact"] for item in activation["packed_emulation"].values())
    assert activation["quantization_transitions"]["full"]["codes_one_sign_only"] > 0
    assert activation["components"]["even"]["native"]["displacement_l2"] > 0
    symmetric = report["symmetric_check"]
    assert symmetric["status"] == "measured" and symmetric["exactly_antisymmetric"]
    assert symmetric["output"]["even"]["displacement_l2"] > 0
    sym = symmetric["activation_counterfactual"]
    assert sym["status"] == "qualified"
    assert sym["even"]["ideal_affine"]["displacement_l2"] == 0
    assert sym["even"]["no_activation_quantization"]["displacement_l2"] < sym["even"]["native"]["displacement_l2"]
    even = report["components"]["even"]
    assert even["residual_resolved"]
    assert even["residual"]["displacement_l2"] > even["linear"]["displacement_l2"]
    accounting = even["accounting"]["bands"]
    executed = even["executed"]["bands"]
    for name, band in accounting.items():
        total = band["baseline"] + band["displacement"] + band["cross_term"]
        assert total == pytest.approx(executed[name]["displacement"], abs=2e-8, rel=2e-5)


def test_replay_rejects_mismatch_outside_image_slice():
    with torch.inference_mode():
        layer = torch.nn.Linear(4, 4).eval()
        x0 = torch.zeros((1, 8, 4))
        snaps = record(layer, {"D00": x0, "D01": x0+.01, "D0minus": x0-.01})
        # Non-image token only. The generated-image replay error is still zero.
        snaps["D01"].output[0, 0].fill_(20.)
        report = {}
        with pytest.raises(ProjectionUnavailable, match="native_replay_exceeds_component_floor"):
            study_projection(layer, snaps, (2, 3), MechanisticSettings(projection_study="observe"),
                             2, 4, report, ResearchTiming(), 11)
    assert report["native_replay"]["D01"]["displacement_l2"] == 0
    assert report["native_replay"]["D01"]["packed_l2"] > 1
    assert "components" not in report


def test_unsupported_linear_input_transform_preserves_native_replay():
    class Coupled(torch.nn.Linear):
        def forward(self, x):
            return super().forward(x / x.abs().amax().clamp_min(.01))

    with torch.inference_mode():
        layer = Coupled(4, 4).eval()
        x0 = torch.ones((1, 8, 4))
        x0[:, 0] = 32.
        values = {"D00": x0, "D01": x0 + .1, "D0minus": x0 - .1}
        snaps = record(layer, values)
        assert not torch.equal(layer(x0)[:, 1:7], layer(x0[:, 1:7]))
        report = {}
        with pytest.raises(ProjectionUnavailable, match="linear_operand_input_transformed_or_unverified"):
            study_projection(layer, snaps, (2, 3), MechanisticSettings(projection_study="observe"),
                             2, 4, report, ResearchTiming(), 11)
    assert report["native_replay_status"] == "qualified"
    assert "components" not in report



def test_reference_uses_executed_runtime_weight_and_bias_functions():
    with torch.inference_mode():
        layer = ops.manual_cast.Linear(2, 2, dtype=torch.float32).eval()
        layer.weight.copy_(torch.eye(2))
        layer.bias.zero_()
        layer.weight_function = [lambda weight: weight * 3]
        layer.bias_function = [lambda bias: bias + 14]
        x0 = torch.full((1, 8, 2), 4.)
        d = torch.tensor([.25, .125])
        report = run_study(layer, record(layer, {"D00": x0, "D01": x0 + d, "D0minus": x0 - d}))
    measures = report["components"]["odd"]
    assert report["runtime_weight_functions"] == report["runtime_bias_functions"] == 1
    assert measures["linear_amplification"] == pytest.approx(3.)
    assert measures["residual"]["displacement_l2"] == 0
    assert torch.equal(layer.weight, torch.eye(2))
    assert torch.count_nonzero(layer.bias) == 0



def test_emulation_epilogue_casts_before_adding_bias():
    from types import SimpleNamespace
    from mode_explorer.projection_activation import ActivationStudy, CUDA

    ref = SimpleNamespace(storage=torch.ones((1, 1), dtype=torch.int8),
                          scale=torch.tensor([1.]), bias=torch.tensor([2.**-11], dtype=torch.float64))
    spec = MechanisticSettings(projection_study="observe")
    study = ActivationStudy(ref, spec, lambda: None)
    scaled_product = torch.tensor([[1. + 2.**-11]])
    output = study._epilogue(scaled_product, torch.ones(1, 1), torch.float16)
    assert output.item() == 1.
    fused = study._epilogue(scaled_product, torch.ones(1, 1), torch.float16, CUDA)
    assert fused.item() == 1. + 2.**-10


@pytest.mark.parametrize("group, fits, expected", [(256, True, "cuda_convrot64_fused"),
                                                 (256, False, "cuda_split_convrot"),
                                                 (64, True, "cuda_split_convrot"),
                                                 (0, True, "cuda_rowwise")])
def test_cuda_quantizer_dispatch_matches_audited_convrot_predicate(group, fits, expected):
    from types import SimpleNamespace
    from mode_explorer.projection_activation import CudaActivationPath

    source = torch.zeros(1, 4, 6144, dtype=torch.float16)
    rotation = torch.empty(group, group) if group else None
    path = CudaActivationPath.__new__(CudaActivationPath)

    def quantize(rows, *args):
        assert rows.shape == (4, 6144) and rows.is_contiguous()
        assert args == ((256,) if expected == "cuda_convrot64_fused" else ())
        return rows.to(torch.int8), torch.ones(4, 1)

    path.cuda = SimpleNamespace(
        _CONVROT_FUSED_MAX_K=16384, _convrot_fused_shared_memory_fits=lambda *args: fits,
        quantize_int8_rowwise_convrot64=quantize, quantize_int8_rowwise=quantize,
        rotate_int8_convrot_weight=lambda value, size: value + size,
    )
    route, rotated, _, _ = path.quantize(source, rotation, lambda value, _: value + 1)
    assert route == expected
    if route == "cuda_convrot64_fused":
        assert rotated is None
        continuous = path.continuous_rotation(source[0], group)
        assert continuous.dtype == torch.float32 and continuous[0, 0].item() == group
    else:
        assert rotated[0, 0].item() == 1


def test_cuda_epilogue_uses_one_model_and_the_smallest_component_bound():
    from mode_explorer.projection_activation import CUDA, EAGER, select_epilogue

    signed = {
        "odd": (20., ((.5, "plus"), (-.5, "minus"))),
        "even": (1., ((.5, "plus"), (.5, "minus"), (-1., "base"))),
    }

    def checks(*errors):
        return {arm: {"packed_bit_exact": error == 0., "packed_l2": error}
                for arm, error in zip(("base", "plus", "minus"), errors)}

    models = {CUDA: checks(.02, .02, .02), EAGER: checks(0., .15, 0.)}
    selected, candidates = select_epilogue(models, signed, .05, allow_bounded=True)
    assert selected == CUDA and candidates[CUDA]["qualification"] == "component_bounded"
    assert candidates[CUDA]["components"]["even"]["model_l2_bound"] == pytest.approx(.04)
    assert candidates[EAGER]["components"]["odd"]["qualified"]
    assert not candidates[EAGER]["components"]["even"]["qualified"]
    assert select_epilogue(models, signed, .05, allow_bounded=False)[0] is None
    signed["even"] = (0., signed["even"][1])
    assert select_epilogue(models, signed, .05, allow_bounded=True)[0] is None
    models[EAGER] = checks(0., 0., 0.)
    assert select_epilogue(models, signed, .05, allow_bounded=True)[0] == EAGER


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_native_cuda_fused_convrot_activation_replay():
    import comfy_kitchen as kitchen
    from importlib.metadata import version
    from comfy_kitchen.registry import registry

    backend = registry.list_backends().get("cuda", {})
    if version("comfy-kitchen") not in ("0.2.40", "0.2.41") or not backend.get("available") or backend.get("disabled"):
        pytest.skip("audited Kitchen CUDA extension is unavailable")
    with torch.inference_mode(), kitchen.use_backend("cuda"):
        device = "cuda"
        generator = torch.Generator().manual_seed(112)
        weight = torch.randn((256, 6144), generator=generator).to(device=device, dtype=torch.float16)
        layer = ops.mixed_precision_ops(compute_dtype=torch.float16).Linear(6144, 256, device=device).eval()
        layer.weight = torch.nn.Parameter(QuantizedTensor.from_float(
            weight, "TensorWiseINT8Layout", convrot=True, convrot_groupsize=256, per_channel=True,
        ), requires_grad=False)
        layer.quant_format = "int8_tensorwise"
        layer.layout_type = "TensorWiseINT8Layout"
        layer.bias.copy_(torch.randn(256, generator=generator).to(device) * .125)
        x = torch.randn((1, 8, 6144), generator=generator).to(device=device, dtype=torch.float16)
        edit = torch.randn((1, 8, 6144), generator=generator).to(device=device, dtype=torch.float16) * .025
        report = run_study(layer, record(layer, {"D00": x, "D01": x + edit, "D0minus": x - edit}),
                           projection_activation_check=True)
    activation = report["activation_counterfactual"]
    assert activation["backend_dispatch_candidate"] == "cuda"
    assert activation["status"] == "qualified", activation
    assert all(route["quantizer"] == "cuda_convrot64_fused" for route in activation["packed_routes"].values())
    assert activation["quantization_transitions"]["full"]["codes_one_sign_only"] > 0
    for component in ("even", "odd"):
        assert activation["component_floors"][component]["qualified"]
        assert "activation_quantization" in activation["components"][component]


@pytest.mark.parametrize("reject", ["mismatch", "backend"])
def test_unqualified_activation_emulation_keeps_only_affine_report(monkeypatch, reject):
    import comfy_kitchen as kitchen
    from mode_explorer.projection_activation import ActivationStudy

    epilogue = ActivationStudy._epilogue

    def offset(self, *args, **kwargs):
        result = epilogue(self, *args, **kwargs)
        result[0, 0] += 1  # Outside the generated-image slice.
        return result

    if reject == "mismatch":
        monkeypatch.setattr(ActivationStudy, "_epilogue", offset)
    else:
        from comfy_kitchen.backends.eager import quantization

        def unavailable(*args):
            raise RuntimeError("int8 matrix multiply unsupported")

        monkeypatch.setattr(quantization, "mm_int8", unavailable)
    with torch.inference_mode(), kitchen.use_backend("eager"):
        layer = ops.mixed_precision_ops(compute_dtype=torch.float16).Linear(8, 8, device="cpu").eval()
        layer.weight = torch.nn.Parameter(QuantizedTensor.from_float(
            torch.eye(8, dtype=torch.float16), "TensorWiseINT8Layout", convrot=True,
            convrot_groupsize=4, per_channel=True), requires_grad=False)
        layer.quant_format = "int8_tensorwise"
        layer.layout_type = "TensorWiseINT8Layout"
        layer.bias.zero_()
        x = torch.ones(1, 8, 8, dtype=torch.float16)
        replay = record(layer, {"D00": x, "D01": x+.0625, "D0minus": x-.0625})
        report = run_study(layer, replay, projection_activation_check=True)
    assert report["native_replay_status"] == "qualified"
    activation = report["activation_counterfactual"]
    assert activation["status"] == "unqualified"
    assert "components" not in activation
    if reject == "mismatch":
        assert activation["reason"] == "packed_eager_emulation_mismatch"
        assert all(check["packed_mismatched_values"] == 1 for check in activation["packed_emulation"].values())
    else:
        assert activation["reason"] == "eager_int8_accumulation_unavailable"


def _int8_convrot_snapshots(seed=119):
    layer = ops.mixed_precision_ops(compute_dtype=torch.float16).Linear(8, 8, device="cpu").eval()
    gen = torch.Generator().manual_seed(seed)
    weight = torch.randn((8, 8), generator=gen).to(dtype=torch.float16)
    layer.weight = torch.nn.Parameter(QuantizedTensor.from_float(
        weight, "TensorWiseINT8Layout", convrot=True, convrot_groupsize=4, per_channel=True,
    ), requires_grad=False)
    layer.quant_format = "int8_tensorwise"
    layer.layout_type = "TensorWiseINT8Layout"
    layer.bias.fill_(.25)
    x0 = torch.randn((1, 8, 8), generator=gen).to(dtype=torch.float16)
    delta = torch.randn((1, 8, 8), generator=gen).to(dtype=torch.float16) * .005
    return layer, record(layer, {"D00": x0, "D01": x0 + delta, "D0minus": x0 - delta})


def _downstream_study(layer, snapshots, study):
    spec = MechanisticSettings(projection_study=study, component="complete", projection_effect="even",
                               projection_activation_check=True)
    report = {}
    chosen = study_projection(layer, snapshots, (2, 3), spec, 2, 4, report, ResearchTiming(), 9)
    assert report["status"] == "qualified"
    assert report["activation_counterfactual"]["status"] == "qualified"
    return report, chosen


@pytest.mark.parametrize("study", ["residual", "control"])
def test_activation_counterfactual_keeps_residual_gate_for_downstream_isolation(study):
    import comfy_kitchen as kitchen

    with torch.inference_mode(), kitchen.use_backend("eager"):
        layer, snapshots = _int8_convrot_snapshots()
        report, chosen = _downstream_study(layer, snapshots, study)
    affine = report["components"]["even"]
    activation = report["activation_counterfactual"]["components"]["even"]
    # The affine record keeps its gate; the activation record keeps its own.
    assert affine["residual_resolved"] is True
    assert "residual_resolved" not in activation
    assert "activation_quantization_resolved" in activation
    assert chosen is not None and chosen.shape == (6, 8)
    target = affine["residual"]["displacement_l2"]
    assert float(torch.linalg.vector_norm(chosen.double())) == pytest.approx(target, rel=1e-5)
    if study == "control":
        assert report["control_target_l2"] == pytest.approx(target)


@pytest.mark.parametrize("study", ["residual", "control"])
def test_unresolved_residual_rejects_downstream_with_qualified_activation(monkeypatch, study):
    import comfy_kitchen as kitchen
    from mode_explorer import projection

    fields = projection._component_fields

    def without_residual(*args, **kwargs):
        # Make the executed even response equal its affine prediction.
        values = list(fields(*args, **kwargs))
        values[2].zero_()
        values[3].copy_(values[1])
        return tuple(values)

    monkeypatch.setattr(projection, "_component_fields", without_residual)
    with torch.inference_mode(), kitchen.use_backend("eager"):
        layer, snapshots = _int8_convrot_snapshots()
        report, chosen = _downstream_study(layer, snapshots, study)
    assert report["components"]["even"]["residual_resolved"] is False
    assert chosen is None
    assert "control_target_l2" not in report
