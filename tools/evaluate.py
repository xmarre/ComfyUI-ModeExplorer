#!/usr/bin/env python3
"""Run matched sequential Krea 2 image generations against a running ComfyUI server."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import subprocess
import threading
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

METHODS = {
    "native": ("native", False),
    "cosine": ("feature_cosine", False),
    "volume": ("feature_volume", False),
    "initialization": ("native", True),
    "combined": ("feature_cosine", True),
}
PROFILE_CONTROLS = {
    "feature_block", "feature_grid", "cosine_temperature", "volume_bandwidth", "volume_median",
    "reference_source", "reference_bank_size", "reference_trajectories", "reference_selection",
    "reference_continuation", "feature_update", "sham", "initialization_trials",
    "feature_direction", "direction_trial",
    "reference_budget", "strength", "fidelity", "min_sigma", "adaptive", "similarity", "diagnostics", "mode",
}
RESPONSE_CONTROLS = {
    "mode", "basis", "probe_grid", "directions", "trial", "displacement", "norm", "calibration",
    "min_sigma", "max_sigma", "evaluation", "states", "auxiliary_budget", "memory_mb", "clean_limit",
    "candidate_gain", "candidate_target_rms", "selection", "response_direction", "all_actual", "feature_block", "feature_grid",
    "feature_strength", "feature_references", "feature_pair", "feature_direction", "localization", "head", "cell", "export", "checkpoint_label",
    "frozen_state", "downstream", "input_limit", "calibration_max_change", "latent_cell",
    "feature_sites", "feature_projection", "require_conditioning", "require_fidelity",
    "reference_min_sigma", "reference_start_sigma", "probe_precision",
}
REDIVERSE_CONTROLS = {
    "adapter", "strength", "band", "schedule", "adapted_sigmas", "direction_count", "coefficient_trial",
    "steering", "diagnostics", "calibrate",
}
CALIBRATION_CONTROLS = {
    "teacher_model", "block", "teacher_forward", "teacher_steps", "rollouts", "directions", "seed", "batch",
    "workspace", "workspace_mib",
}
SPECTRUM_CONTROLS = {
    "preset", "degree", "ridge", "warmup", "tail", "window", "growth", "max_history",
    "storage", "budget_mb", "debug", "bootstrap", "extrapolation_guard_enabled",
    "max_extrapolation_span_ratio", "sa_pece_forecast_policy", "extrapolation_guard_scope",
}
IDENTITY = (
    "waiting for a tram in a city at dawn", "choosing apples at a market in a wool coat",
    "working at a sewing machine", "leaving a rehearsal with a bicycle",
    "sitting under trees during a summer picnic",
)
SEEDS = [206627 + i * 104729 for i in range(16)]


def load_profiles(path):
    profiles = json.loads(path.read_text())
    if not isinstance(profiles, dict) or len(profiles) < 2:
        raise ValueError("Profiles must contain at least two named variants.")
    for label, cfg in profiles.items():
        if not re.fullmatch(r"[A-Za-z0-9_-]+", label) or not isinstance(cfg, dict):
            raise ValueError(f"Invalid profile {label!r}.")
        rediverse = cfg.get("rediverse", {})
        if (set(cfg) - {"method", "settings", "spectrum", "spectrum_settings", "response", "model", "steps", "cfg",
                        "rediverse", "schedule"}
                or not isinstance(rediverse, dict) or set(rediverse) - REDIVERSE_CONTROLS
                or not isinstance(rediverse.get("calibrate", {}), dict)
                or set(rediverse.get("calibrate", {})) - CALIBRATION_CONTROLS
                or ("calibrate" in rediverse and not rediverse["calibrate"].get("teacher_model"))
                or (rediverse and not (rediverse.get("adapter") or "calibrate" in rediverse))
                or cfg.get("schedule", "native") not in {"native", "reference_turbo", "reference_raw"}
                or cfg.get("method") not in METHODS
                or cfg.get("spectrum", "off") not in {"off", "guarded", "full", "all_actual"}
                or not isinstance(cfg.get("settings", {}), dict)
                or not isinstance(cfg.get("spectrum_settings", {}), dict)
                or set(cfg.get("settings", {})) - PROFILE_CONTROLS
                or set(cfg.get("spectrum_settings", {})) - SPECTRUM_CONTROLS
                or not isinstance(cfg.get("response", {}), dict)
                or set(cfg.get("response", {})) - RESPONSE_CONTROLS
                or ("model" in cfg and (not isinstance(cfg["model"], str) or not cfg["model"]))
                or ("steps" in cfg and (type(cfg["steps"]) is not int or not 1 <= cfg["steps"] <= 10000))
                or ("cfg" in cfg and (type(cfg["cfg"]) not in (int, float) or not 0 <= cfg["cfg"] <= 100))):
            raise ValueError(f"Invalid options for profile {label!r}.")
    return profiles


def graph(args, method, prompt, seed, *, name):
    profile = getattr(args, "variants", {}).get(method, {"method": method})
    mode = profile["method"]
    feature, initialization = METHODS[mode]
    spectrum = profile.get("spectrum", "off")
    def resolve(value):
        """Profiles name the teacher checkpoint and ReDiverse adapter as $teacher and $adapter."""
        for token, option in (("$teacher", "teacher_model"), ("$adapter", "rediverse_adapter")):
            if value == token:
                if not getattr(args, option, None):
                    raise ValueError(f"This profile needs --{option.replace('_', '-')}.")
                return getattr(args, option)
        return value

    checkpoint = resolve(profile.get("model", args.model))
    steps, cfg = profile.get("steps", args.steps), profile.get("cfg", args.cfg)
    def node(kind, **inputs):
        return {"class_type": kind, "inputs": inputs}

    g = {
        "1": node("UNETLoader", unet_name=checkpoint, weight_dtype=args.weight_dtype),
        "2": node("CLIPLoader", clip_name=args.encoder, type="krea2", device="default"),
        "3": node("VAELoader", vae_name=args.vae),
        "4": node("CLIPTextEncode", clip=["2", 0], text=prompt),
        "5": node("CLIPTextEncode", clip=["2", 0], text=args.negative),
        "6": node("EmptyLatentImage", width=args.width, height=args.height, batch_size=1),
        "8": node("KSampler", model=["7", 0], seed=seed, steps=steps, cfg=cfg,
                  sampler_name=args.sampler, scheduler=args.scheduler, positive=["4", 0], negative=["5", 0],
                  latent_image=["6", 0], denoise=1.0),
        "9": node("VAEDecode", samples=["8", 0], vae=["3", 0]),
        "10": node("SaveImage", images=["9", 0], filename_prefix=f"modeexplorer_eval/{name}"),
    }
    model_input = ["1", 0]
    for key, lora_name, strength in (
        ("11", args.model_lora, args.model_lora_strength), ("12", args.identity_lora, args.lora_strength),
    ):
        if lora_name:
            g[key] = node("LoraLoaderModelOnly", model=model_input, lora_name=lora_name, strength_model=strength)
            model_input = [key, 0]
    rediverse = profile.get("rediverse")
    if rediverse:
        controls = {"enabled": True, "adapter": "none", "strength": 1.0, "band": 2.0, "diagnostics": bool(args.diagnostics)}
        controls.update({key: resolve(value) for key, value in rediverse.items() if key != "calibrate"})
        if "calibrate" in rediverse:
            calibration = {"block": 26, "teacher_forward": 5, "teacher_steps": 52, "rollouts": 25, "directions": 8,
                           "seed": 0, "batch": 5, "label": "", "cfg": 1.0, "workspace": "auto", "workspace_mib": 8192}
            calibration.update(rediverse["calibrate"])
            g["19"] = node("UNETLoader", unet_name=resolve(calibration.pop("teacher_model")), weight_dtype=args.weight_dtype)
            teacher = ["19", 0]
            if args.model_lora:
                # The student's weight LoRA is applied to the teacher too, as in the calibration guide.
                g["23"] = node("LoraLoaderModelOnly", model=teacher, lora_name=args.model_lora,
                               strength_model=args.model_lora_strength)
                teacher = ["23", 0]
            g["20"] = node("ModeExplorerReDiverseCalibrateKrea2", teacher=teacher, positive=["4", 0],
                           latent=["6", 0], **calibration)
            controls["directions"] = ["20", 0]
        g["18"] = node("ModeExplorerReDiverseKrea2", model=model_input, **controls)
        model_input = ["18", 0]
    if mode != "native" or args.diagnostics:
        g["7"] = node(
            "ModeExplorerKrea2", model=model_input, enabled=True, feature=feature, initialization=initialization,
            strength=args.strength, fidelity=args.fidelity, min_sigma=args.min_sigma, adaptive=not args.fixed,
            similarity=args.similarity, operation="sequential", reference_budget=args.reference_budget,
            diagnostics=args.diagnostics, full_spectrum=spectrum == "full",
        )
        g["7"]["inputs"].update(profile.get("settings", {}))
        model_input = ["7", 0]
    if "response" in profile:
        controls = {
            "mode": "latent_probe", "basis": "bilinear", "probe_grid": 4, "directions": 2, "trial": 0,
            "displacement": 0.02, "norm": "rms", "calibration": "repeat_and_half",
            "min_sigma": 0.82, "max_sigma": 0.99999, "evaluation": -1, "states": 1,
            "auxiliary_budget": 12, "memory_mb": 128, "checkpoint_label": checkpoint,
        }
        controls.update(profile["response"])
        g["16"] = node("ModeExplorerResponseKrea2", model=model_input, enabled=True, **controls)
        model_input = ["16", 0]
    if spectrum != "off":
        settings = {"enabled": True, "preset": getattr(args, "spectrum_preset", "turbo"),
                    "degree": 1, "ridge": 0.01, "warmup": 2, "tail": 1,
                    "window": 1, "growth": 0.0, "max_history": 8, "storage": "vram",
                    "budget_mb": 4096, "debug": bool(args.diagnostics)}
        settings.update(profile.get("spectrum_settings", {}))
        if spectrum == "all_actual":
            settings.update(window=0, growth=0.0, bootstrap=False, sa_pece_forecast_policy="off")
        g["13"] = node("SpectrumApplyKrea2", model=model_input, **settings)
        model_input = ["13", 0]
    sampler = g["8"]["inputs"]
    sampler["model"] = model_input
    schedule = profile.get("schedule", "native")
    if schedule != "native":
        if getattr(args, "planner_megapixels", None) is not None:
            raise ValueError("Reference schedules use SamplerCustom and cannot drive the Scale-Locked planner.")
        g["21"] = node("KSamplerSelect", sampler_name=args.sampler)
        g["22"] = node("ModeExplorerReferenceSigmasKrea2", latent=["6", 0], model=schedule.removeprefix("reference_"),
                       steps=steps)
        g["8"] = node("SamplerCustom", model=model_input, add_noise=True, noise_seed=seed, cfg=cfg,
                      positive=["4", 0], negative=["5", 0], sampler=["21", 0], sigmas=["22", 0],
                      latent_image=["6", 0])
    if getattr(args, "planner_megapixels", None) is not None:
        g["8"]["class_type"] = "Krea2TurboFlowAlignedRegenerate"
        del sampler["denoise"]
        sampler.update(planner_megapixels=args.planner_megapixels, coarse_lock_strength=0.0,
                       mid_lock_strength=0.0, composition_lock_sigma=args.composition_lock_sigma,
                       handoff_diagnostics="log" if args.diagnostics else "off")
        g["14"] = node("VAEDecode", samples=["8", 1], vae=["3", 0])
        g["15"] = node("SaveImage", images=["14", 0], filename_prefix=f"modeexplorer_eval/{name}_planner")
    return g


def sampler_config(workflow):
    """Effective sampling configuration of a graph from ``graph``, for KSampler and SamplerCustom."""
    node = workflow["8"]
    inputs = node["inputs"]
    if node["class_type"] == "SamplerCustom":
        schedule = workflow[inputs["sigmas"][0]]["inputs"]
        return {"steps": schedule["steps"], "cfg": inputs["cfg"],
                "sampler": workflow[inputs["sampler"][0]]["inputs"]["sampler_name"],
                "scheduler": f"reference_{schedule['model']}"}
    if node["class_type"] in ("KSampler", "Krea2TurboFlowAlignedRegenerate"):
        return {"steps": inputs["steps"], "cfg": inputs["cfg"], "sampler": inputs["sampler_name"],
                "scheduler": inputs["scheduler"]}
    raise ValueError(f"Unknown sampler node {node['class_type']!r}.")


def request(base, endpoint, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = Request(base.rstrip("/") + endpoint, data=data, headers={"Content-Type": "application/json"} if data else {})
    with urlopen(req, timeout=30) as response:
        return response.read()


class VramSampler:
    """Approximate total device VRAM high-water mark during an isolated job."""

    def __init__(self, device):
        self.device = device
        self.peak_mib = None
        self.stop = threading.Event()
        self.thread = None

    def __enter__(self):
        if self.device is not None:
            self.thread = threading.Thread(target=self.poll, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        if self.thread is not None:
            self.thread.join()

    def poll(self):
        while not self.stop.is_set():
            try:
                signal = subprocess.check_output(
                    ["nvidia-smi", "-i", self.device, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    text=True, timeout=3, stderr=subprocess.DEVNULL,
                )
                value = int(signal.strip())
                self.peak_mib = max(self.peak_mib or 0, value)
            except (OSError, subprocess.SubprocessError, ValueError):
                return
            self.stop.wait(0.2)


def read_observations(path, offset, seed):
    trace = None
    spectrum = []
    calibration = []
    if path is None:
        return trace, spectrum, calibration
    with path.open("rb") as logfile:
        if offset <= path.stat().st_size:
            logfile.seek(offset)
        for line in logfile:
            for marker in (b"MODEEXPLORER_TRACE ", b"Spectrum Krea 2 summary: ", b"ModeExplorer ReDiverse calibration: "):
                _, found, payload = line.partition(marker)
                if not found:
                    continue
                try:
                    observed = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if marker.startswith(b"Spectrum"):
                    spectrum.append(observed)
                elif marker.startswith(b"ModeExplorer ReDiverse"):
                    calibration.append(observed)
                elif observed["seed"] == seed and observed["completed"]:
                    trace = observed
    return trace, spectrum, calibration


def render(args, workflow, seed, destination):
    offset = args.server_log.stat().st_size if args.server_log else 0
    start = time.perf_counter()
    with VramSampler(args.gpu) as vram:
        response = json.loads(request(args.server, "/prompt", {"prompt": workflow}))
        prompt_id = response["prompt_id"]
        while time.perf_counter() - start < args.timeout:
            history = json.loads(request(args.server, f"/history/{prompt_id}"))
            if prompt_id in history:
                history = history[prompt_id]
                break
            time.sleep(0.2)
        else:
            raise TimeoutError(f"Timed out waiting for ComfyUI prompt {prompt_id}")
    finish = time.perf_counter()
    status = history["status"]
    if status.get("status_str") != "success":
        raise RuntimeError(f"ComfyUI prompt {prompt_id} failed: {status}")
    def download(slot, path):
        metadata, = history["outputs"][slot]["images"]
        content = request(args.server, "/view?" + urlencode(
            {key: metadata[key] for key in ("filename", "subfolder", "type")},
        ))
        path.write_bytes(content)

    download("10", destination)
    if "15" in workflow:
        download("15", destination.with_name(destination.stem + "_planner.png"))
    trace, spectrum, calibration = read_observations(args.server_log, offset, seed)
    stamps = {kind: data.get("timestamp") for kind, data in status.get("messages", [])}
    start_ms, end_ms = stamps.get("execution_start"), stamps.get("execution_success")
    return {
        "job_id": prompt_id,
        "wall_seconds": finish - start,
        "execution_seconds": (end_ms - start_ms) / 1000 if start_ms is not None and end_ms is not None else None,
        "sampled_peak_device_mib": vram.peak_mib,
        "trace": trace, "spectrum": spectrum,
        # Teacher calibration executed in this job (ComfyUI caches it for later seeds of a prompt).
        "rediverse_calibration": calibration,
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--server", default="http://127.0.0.1:8188")
    p.add_argument("--server-log", type=Path, help="Readable ComfyUI INFO log from a dedicated server.")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", required=True, help="ComfyUI diffusion model filename.")
    p.add_argument("--encoder", default="qwen3vl_4b_fp8_scaled.safetensors")
    p.add_argument("--vae", default="qwen_image_vae.safetensors")
    p.add_argument("--weight-dtype", default="default")
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--cfg", type=float, default=1.0)
    p.add_argument("--sampler", default="euler")
    p.add_argument("--scheduler", default="simple")
    p.add_argument("--negative", default="")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--methods", nargs="+", choices=sorted(METHODS))
    p.add_argument("--profiles", type=Path, help="Named parameter variants for a matched panel.")
    p.add_argument("--spectrum-preset", choices=("raw", "turbo"), default="turbo")
    p.add_argument("--planner-megapixels", type=float, help="Run the real Scale-Locked node and export both planner and final images.")
    p.add_argument("--composition-lock-sigma", type=float, default=0.60)
    p.add_argument("--seed-count", type=int, choices=range(2, 17), default=16)
    p.add_argument("--panel", type=Path, help="Local JSON with explicit case categories/prompts and optional seeds.")
    p.add_argument("--panel-purpose", choices=("pilot", "held_out"), default="held_out")
    p.add_argument("--category", action="append", help="Restrict the prompt panel to this category.")
    p.add_argument("--repeats", type=int, choices=(1, 2, 3), default=1,
                   help="Repeat each same-seed job. Use a dedicated server with --cache-none and a readable --server-log.")
    p.add_argument("--model-lora", help="Optional model LoRA for a separate external baseline.")
    p.add_argument("--model-lora-strength", type=float, default=1.0)
    p.add_argument("--teacher-model", help="Krea 2 Raw diffusion model filename for $teacher in ReDiverse profiles.")
    p.add_argument("--rediverse-adapter", help="ReDiverse-FT adapter filename in the loras folder for $adapter.")
    p.add_argument("--identity-lora")
    p.add_argument("--identity-trigger")
    p.add_argument("--lora-strength", type=float, default=1.0)
    p.add_argument("--strength", type=float, default=0.45)
    p.add_argument("--fidelity", type=float, default=0.55)
    p.add_argument("--min-sigma", type=float, default=0.82)
    p.add_argument("--similarity", type=float, default=0.70)
    p.add_argument("--reference-budget", type=int, default=2)
    p.add_argument("--fixed", action="store_true", help="Disable adaptive cosine gating.")
    p.add_argument("--diagnostics", action="store_true")
    p.add_argument("--gpu", help="Optional nvidia-smi device index or GPU UUID to monitor.")
    p.add_argument("--smoke", action="store_true", help="Two prompts by two seeds.")
    p.add_argument("--timeout", type=int, default=1800)
    args = p.parse_args()
    if args.identity_lora and not args.identity_trigger:
        p.error("--identity-lora requires --identity-trigger")
    if args.server_log is not None:
        args.diagnostics = True
    if args.profiles is not None and args.methods is not None:
        p.error("--profiles and --methods are alternatives.")
    try:
        args.variants = load_profiles(args.profiles) if args.profiles else {}
    except (OSError, ValueError) as exc:
        p.error(str(exc))
    args.methods = args.methods or list(args.variants or METHODS)
    if args.repeats > 1 and args.server_log is None:
        p.error("Repeated jobs require --server-log to verify fresh sampling.")
    if args.planner_megapixels is not None and not 0.25 <= args.planner_megapixels < args.width * args.height / 1e6:
        p.error("The Scale-Locked planner must be at least 0.25 MP and smaller than the requested output.")
    if len(set(args.methods)) != len(args.methods):
        p.error("Each method may be selected once.")
    if not all(size >= 16 and size % 16 == 0 for size in (args.height, args.width)):
        p.error("Width and height must be positive multiples of 16.")
    return args


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "runs.jsonl").open("x", encoding="utf-8") as records:
        gallery = args.out / "images"
        gallery.mkdir()
        seeds = SEEDS[:2 if args.smoke else args.seed_count]
        if args.panel:
            panel = json.loads(args.panel.read_text())
            if not isinstance(panel, dict) or set(panel) - {"prompts", "seeds"}:
                raise ValueError("A local panel must contain prompts and may include seeds.")
            prompts = panel.get("prompts")
            if not isinstance(prompts, list) or not prompts or any(
                not isinstance(row, list) or len(row) != 2 or not all(isinstance(item, str) and item for item in row)
                for row in prompts
            ):
                raise ValueError("Local panel prompts must be [category, text] pairs.")
            seeds = panel.get("seeds", seeds)
            if (not isinstance(seeds, list) or not seeds
                    or any(type(seed) is not int or not 0 <= seed < 2**64 for seed in seeds)
                    or len(seeds) != len(set(seeds))):
                raise ValueError("Local panel seeds must be distinct unsigned 64-bit integers.")
            if args.smoke:
                seeds = seeds[:2]
        else:
            prompts = json.loads(Path(__file__).with_name("prompts.json").read_text())
        if args.identity_lora:
            prompts += [["identity", f"Photograph of {args.identity_trigger} {setting}."] for setting in IDENTITY]
        if args.category:
            prompts = [item for item in prompts if item[0] in args.category]
        if args.smoke and len(prompts) > 2:
            prompts = [prompts[0], prompts[-1]]
        if not prompts:
            raise ValueError("The selected categories produced an empty prompt panel.")
        manifest = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        manifest["seeds"] = seeds
        manifest["prompts"] = prompts
        (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        for method in args.methods:
            warmup = graph(args, method, "A paper cup on a windowsill.", 814917, name=f"warmup_{method}")
            render(args, warmup, 814917, args.out / f"warmup_{method}.png")
        for i, (category, prompt) in enumerate(prompts):
            for seed in manifest["seeds"]:
                for method in random.Random(f"{i}:{seed}").sample(args.methods, len(args.methods)):
                    tag = hashlib.sha256(f"{i}:{seed}:{method}".encode()).hexdigest()[:16]
                    relative = Path("images") / f"{tag}.png"
                    workflow = graph(args, method, prompt, seed, name=f"run_{tag}")
                    result = render(args, workflow, seed, args.out / relative)
                    result.update(
                        id=tag, method=method, prompt_id=i, category=category, seed=seed,
                        image=relative.as_posix(),
                        checkpoint=workflow["1"]["inputs"]["unet_name"],
                        weight_dtype=workflow["1"]["inputs"]["weight_dtype"],
                        **sampler_config(workflow),
                    )
                    if args.planner_megapixels is not None:
                        result["planner_image"] = relative.with_name(relative.stem + "_planner.png").as_posix()
                    result["repeats"] = []
                    for retry in range(1, args.repeats):
                        repeat_tag = f"run_{tag}_repeat_{retry}"
                        repeat_path = args.out / "images" / f"{repeat_tag}.png"
                        run = render(args, graph(args, method, prompt, seed, name=repeat_tag), seed, repeat_path)
                        result["repeats"].append({
                            "image": repeat_path.relative_to(args.out).as_posix(),
                            "sampled": bool(run["trace"] or run["spectrum"]),
                        })
                    records.write(json.dumps(result, allow_nan=False) + "\n")
                    records.flush()
                    print(f"prompt={i} seed={seed} method={method} seconds={result['wall_seconds']:.2f}", flush=True)


if __name__ == "__main__":
    main()
