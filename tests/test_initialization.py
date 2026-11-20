import math
from types import SimpleNamespace

import pytest
import torch

from comfy.cli_args import args
from comfy.model_sampling import CONST

args.cpu = True
from mode_explorer import runtime
from mode_explorer.objectives import reference_noise


@pytest.mark.parametrize("variant", ["overshoot", "flat", "nonfinite", "limit", "cancel", "memory", "time"])
def test_bounded_initialization_search(monkeypatch, variant):
    template = torch.zeros(1, 16, 1, 12, 12)
    original = reference_noise(template, 11).clone()
    anchor = original.reshape_as(template)
    other = [reference_noise(template, i).clone() for i in range(2)]
    clean_references = [anchor + 0.28 * reference_noise(template, i + 5) for i in range(2)]
    run = runtime.Run(runtime.Settings(diagnostics=True), 42, 1.0, 1.0, 1, tuple(template.shape), False)
    sigmas = torch.tensor([1.0, 0.0])
    from comfy import model_management
    monkeypatch.setattr(model_management, "throw_exception_if_processing_interrupted", lambda: None)

    class Guider:
        inner_model = SimpleNamespace(model_sampling=CONST())

        def __call__(self, x, sigma, **kwargs):
            runtime.check_initialization_budget(run)
            run.initialization_dit_calls += 1
            assert math.isclose(float(sigma[0]), 0.99, abs_tol=1e-6)
            if variant == "cancel" and run.initialization_dit_calls == 4:
                raise model_management.InterruptProcessingException()
            for i, ref in enumerate(other):
                if torch.equal(x, ref):
                    return clean_references[i]
            if variant == "flat":
                return anchor
            if variant == "nonfinite" and not torch.equal(x, anchor):
                return x * torch.nan
            return anchor + 20 * (x - anchor)

    if variant == "limit":
        monkeypatch.setattr(runtime, "MAX_INITIALIZATION_DIT", 4)
    if variant == "memory":
        monkeypatch.setattr(runtime, "MAX_INITIALIZATION_NOISE_BYTES", 1)
    if variant == "time":
        monkeypatch.setattr(runtime, "MAX_INITIALIZATION_SECONDS", 0)

    def optimize():
        return runtime.optimize_noise(Guider(), sigmas, {}, original, template, run, torch.tensor(0.99))

    if variant == "cancel":
        with pytest.raises(model_management.InterruptProcessingException):
            optimize()
        assert run.phase == "sample"
        return
    result = optimize()
    assert run.phase == "sample"
    if variant in ("memory", "time"):
        assert run.initialization_dit_calls == 0 and torch.equal(result, original)
        assert "initialization_latent_limit" in run.reasons or "initialization_time_limit" in run.reasons
        return
    attempts = run.points[-1]["attempts"]
    if variant == "overshoot":
        assert len(attempts) == 2
        assert attempts[0]["status"] == ["outside_trust_region"]
        assert attempts[1]["status"] == ["accepted"]
        assert attempts[1]["rms_budget"][0] < attempts[0]["rms_budget"][0] * 0.5
        assert attempts[1]["relative_clean_shift"][0] <= run.points[-1]["relative_clean_limit"]
        assert attempts[1]["volume_gain"][0] > 0
        assert not torch.equal(result, original)
        torch.testing.assert_close(result.norm(), original.norm())
    else:
        assert torch.equal(result, original)
        expected = {"flat": "insufficient_volume_gain", "nonfinite": "nonfinite", "limit": "initialization_dit_limit"}
        assert attempts[-1]["status"] == [expected[variant]]
    assert run.initialization_guider_calls <= 6
    assert run.initialization_dit_calls <= 6


def test_concurrent_probe_scope_restores_outer_state():
    from concurrent.futures import ThreadPoolExecutor
    from contextvars import ContextVar
    from threading import Barrier

    owner = object()
    session = ContextVar("test session", default=None)
    capture = ContextVar("test capture", default=None)
    sync = Barrier(2)

    def execute(seed):
        outer = SimpleNamespace(binding=owner)
        marker = object()
        session_token = session.set(outer)
        capture_token = capture.set(marker)
        try:
            run = runtime.Run(runtime.Settings(), seed, 1.0, 1.0, 1, (1, 16, 2, 2), True,
                              spectrum=(owner, session, capture, True))
            with pytest.raises(ArithmeticError):
                with runtime.isolated_reference(run):
                    sync.wait(timeout=5)
                    assert session.get() is None and capture.get() is None
                    raise ArithmeticError
            assert session.get() is outer and capture.get() is marker
        finally:
            session.reset(session_token)
            capture.reset(capture_token)

    with ThreadPoolExecutor(2) as pool:
        list(pool.map(execute, (14, 91)))
    assert session.get() is None and capture.get() is None
