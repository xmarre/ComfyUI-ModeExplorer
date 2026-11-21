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
    from mode_explorer.projection_activation import ActivationStudy

    ref = SimpleNamespace(storage=torch.ones((1, 1), dtype=torch.int8),
                          scale=torch.tensor([1.]), bias=torch.tensor([2.**-11], dtype=torch.float64))
    spec = MechanisticSettings(projection_study="observe")
    study = ActivationStudy(ref, spec, lambda: None)
    scaled_product = torch.tensor([[1. + 2.**-11]])
    output = study._epilogue(scaled_product, torch.ones(1, 1), torch.float16)
    assert output.item() == 1.
    assert (scaled_product + ref.bias).half().item() == 1. + 2.**-10


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
