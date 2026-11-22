"""Opt-in planner capture. Decoding and comparison happen after sampling."""

import json
import math
import re
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch

from .response import flow_clean
from .response_runtime import _tensor_hash

MAX_BYTES = 512 * 1024**2
MAX_STATES = 32
LABEL = re.compile(r"[A-Za-z0-9_-]{1,64}")
IDENTIFIER = re.compile(r"(trajectory-[a-z0-9_]{8})/(\d{3})")


def _root():
    import folder_paths

    return (Path(folder_paths.get_output_directory()) / "modeexplorer_trajectory").resolve()


def _option(value):
    cls = type(value)
    if (cls.__name__ == "Krea2PatchCoupledERSDENoise" and cls.__module__.endswith(".slrd_runtime")):
        return {"type": f"{cls.__module__}.{cls.__name__}",
                **{name: _option(getattr(value, name)) for name in
                   ("low_shape", "stage_shapes", "stage_index", "seed", "start_step", "sigmas", "device", "dtype")}}
    if isinstance(value, (tuple, list)):
        return [_option(item) for item in value]
    if isinstance(value, (torch.device, torch.dtype)):
        return str(value)

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return f"{type(value).__module__}.{type(value).__qualname__}"


class Trajectory:
    def __init__(self, run, sampler, sigmas, noise, base, cfg):
        from .runtime import as_list

        self.convert = base.process_latent_out
        self.states = []
        self.routes = []
        self.step = 0
        self.used = 0
        self.failure = None
        self.solver = sampler.sampler_function
        self.packed_conditioning = None
        self.noise_sampler = sampler.extra_options.get("noise_sampler")
        self.info = {
            "schema": 1, "label": run.settings.trajectory_label, "seed": run.seed,
            "stage": run.stage, "settings": asdict(run.settings), "cfg": cfg,
            "solver": f"{self.solver.__module__}.{self.solver.__name__}",
            "solver_options": {key: _option(value) for key, value in sampler.extra_options.items()},
            "schedule": as_list(sigmas[:MAX_STATES + 1]), "schedule_length": len(sigmas),
            "initial_noise_sha256": _tensor_hash(noise), "noise_scale": run.noise_scale,
            "device": str(noise.device), "torch": str(torch.__version__),
            "resume_supported": False, "auxiliary_dit_calls": 0,
        }

    def observe(self, run, args, velocity, forecast, reference_calls, reference_method):
        from .runtime import FRAME, as_list, spatial_image, summary

        x, timestep, context, mask, refs, options = args
        branches = list(options.get("cond_or_uncond", ()))
        if len(self.routes) >= 4 * MAX_STATES:
            self.failure = self.failure or "route_limit"
            return
        route = {
            "opportunity": len(self.routes), "solver_step": self.step, "branches": branches,
            "route": "forecast" if forecast else "actual", "reference_calls": reference_calls,
        }
        self.routes.append(route)
        sigmas = options.get("sigmas")
        if sigmas is not None:
            route["sigmas"] = as_list(sigmas)
        if forecast or self.failure:
            return
        if branches.count(0) != 1 or run.batch != 1 or spatial_image(x) is None or x.shape[0] != len(branches):
            route["capture"] = "unsupported_packed_geometry"
            return
        if sigmas is None or sigmas.numel() != 1 or velocity.shape != x.shape:
            route["capture"] = "unsupported_flow_coordinates"
            return
        sigma = route["sigmas"][0]
        if not math.isfinite(sigma) or not 0 < sigma <= 1.00001:
            route["capture"] = "unsupported_sigma"
            return
        offset = branches.index(0)
        if len(self.states) == MAX_STATES or self.used + 12 * x[offset].numel() > MAX_BYTES:
            self.failure = "trajectory_budget"
            return
        # Record the first packed conditioning, which is sufficient to check
        # conditioning identity for the fixed single-branch discovery graphs.
        if self.packed_conditioning is None:
            from .response_runtime import _packed_condition_value_hashes

            self.packed_conditioning = _packed_condition_value_hashes(context, mask, refs)
        row = slice(offset, offset + 1)
        with torch.no_grad():
            clean = self.convert(flow_clean(x[row], velocity[row], sigma))
            tensors = {
                "frozen_native_input": x[row].detach().to(device="cpu", dtype=torch.float32, copy=True).contiguous(),
                "conditional_velocity": velocity[row].detach().to(device="cpu", dtype=torch.float32, copy=True).contiguous(),
                "baseline_clean": clean.detach().to(device="cpu", dtype=torch.float32, copy=True).contiguous(),
            }
        frame = options.get(FRAME)
        state = {
            **route, "ordinal": len(self.states), "sigma": sigma,
            "timestep": float(timestep.flatten()[offset]), "input_dtype": str(x.dtype),
            "conditioning_sha256": _tensor_hash(context), "attention_mask_sha256": _tensor_hash(mask),
            "reference_latent_sha256": [_tensor_hash(ref) for ref in refs or ()],
            "reference_method": reference_method, "native_input_sha256": _tensor_hash(x[row]),
            "velocity_dtype": str(velocity.dtype), "device": str(x.device),
            "er_sde_noise_cursor": (self.noise_sampler.cursor if
                type(self.noise_sampler).__name__ == "Krea2PatchCoupledERSDENoise"
                and type(self.noise_sampler).__module__.endswith(".slrd_runtime") else None),
            "feature_applied": bool(frame and frame.attempted and frame.metrics is not None),
        }
        if frame is not None:
            state["reference_slots"] = list(frame.reference_slots)
            if frame.pooled is not None:
                state["attention_cell_groups"] = summary(frame.pooled)
                state["attention_grid"] = list(frame.pooled.shape[-2:])
            if frame.metrics is not None:
                state.update(zip(("nearest_cosine", "objective", "proposed_shift_rms"), map(as_list, frame.metrics)))
                state["realized_shift_rms"] = as_list(frame.realized_rms) if frame.realized_rms is not None else [0.0]
            if frame.edit_metrics:
                state["pooled_edit"] = {key: as_list(value) for key, value in frame.edit_metrics.items()}
        self.used += sum(value.numel() * value.element_size() for value in tensors.values())
        self.states.append((state, tensors))

    def save(self, completed, run):
        from safetensors.torch import save_file
        from .runtime import finite_json

        root = _root()
        root.mkdir(parents=True, exist_ok=True)
        folder = Path(tempfile.mkdtemp(prefix="trajectory-", dir=root))
        states = []
        reference_start = next((state["sigma"] for state, _ in self.states if state.get("reference_slots")), None)
        try:
            for state, tensors in self.states:
                filename = f'{state["ordinal"]:03}'
                save_file(tensors, folder / f"{filename}.safetensors")
                states.append({**state, "export_id": f"{folder.name}/{filename}",
                               "feature_reference_initial_sigma": reference_start})
            manifest = {
                **self.info, "status": ("completed" if completed and not self.failure and states
                                        else "partial" if states else "skipped"),
                "reason": self.failure or ("interrupted" if not completed else "no_actual_states" if not states else None),
                "native_dit_calls": run.native_dit_calls, "spectrum_forecast_calls": run.spectrum_forecasts,
                "live_reference_dit_calls": run.reference_calls, "packed_conditioning": self.packed_conditioning,
                "tensor_bytes": self.used, "states": states, "routes": self.routes,
                "initialization_dit_calls": run.initialization_dit_calls,
                "feature_reasons": sorted(run.reasons),
            }
            (folder / "manifest.json").write_text(json.dumps(finite_json(manifest), allow_nan=False, indent=2))
        except Exception:
            import shutil

            shutil.rmtree(folder, ignore_errors=True)
            raise
        return {"id": folder.name, "status": manifest["status"], "reason": manifest["reason"],
                "captured_actual_states": len(states), "tensor_bytes": self.used}


def read_trajectory(export_id):
    match = IDENTIFIER.fullmatch(export_id)
    if match is None or int(match[2]) >= MAX_STATES:
        raise ValueError("invalid_trajectory_export_id")
    root = _root()
    folder = (root / match[1]).resolve()
    manifest = (folder / "manifest.json").resolve()
    data = (folder / f"{match[2]}.safetensors").resolve()
    if folder.parent != root or manifest.parent != folder or data.parent != folder:
        raise ValueError("invalid_trajectory_export_path")
    if manifest.stat().st_size > 512 * 1024 or data.stat().st_size > MAX_BYTES + 65536:
        raise ValueError("trajectory_export_size_limit")
    info = json.loads(manifest.read_text())
    index = int(match[2])
    if info.get("schema") != 1 or index >= len(info.get("states", ())) or info["states"][index]["export_id"] != export_id:
        raise ValueError("unknown_trajectory_state")
    return {**info["states"][index], "seed": info["seed"], "cfg_scale": info["cfg"],
            "noise_scale": info["noise_scale"], "origin": "trajectory"}, data
