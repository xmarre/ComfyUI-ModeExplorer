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
    "reference_budget", "strength", "fidelity", "min_sigma", "adaptive", "similarity", "diagnostics",
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
        if (set(cfg) - {"method", "settings", "spectrum", "spectrum_settings"}
                or cfg.get("method") not in METHODS
                or cfg.get("spectrum", "off") not in {"off", "guarded", "full", "all_actual"}
                or not isinstance(cfg.get("settings", {}), dict)
                or not isinstance(cfg.get("spectrum_settings", {}), dict)
                or set(cfg.get("settings", {})) - PROFILE_CONTROLS
                or set(cfg.get("spectrum_settings", {})) - SPECTRUM_CONTROLS):
            raise ValueError(f"Invalid options for profile {label!r}.")
    return profiles


def graph(args, method, prompt, seed, *, name):
    profile = getattr(args, "variants", {}).get(method, {"method": method})
    mode = profile["method"]
    feature, initialization = METHODS[mode]
    spectrum = profile.get("spectrum", "off")
    def node(kind, **inputs):
        return {"class_type": kind, "inputs": inputs}

    g = {
        "1": node("UNETLoader", unet_name=args.model, weight_dtype=args.weight_dtype),
        "2": node("CLIPLoader", clip_name=args.encoder, type="krea2", device="default"),
        "3": node("VAELoader", vae_name=args.vae),
        "4": node("CLIPTextEncode", clip=["2", 0], text=prompt),
        "5": node("CLIPTextEncode", clip=["2", 0], text=args.negative),
        "6": node("EmptyLatentImage", width=args.width, height=args.height, batch_size=1),
        "8": node("KSampler", model=["7", 0], seed=seed, steps=args.steps, cfg=args.cfg,
                  sampler_name=args.sampler, scheduler=args.scheduler, positive=["4", 0], negative=["5", 0],
                  latent_image=["6", 0], denoise=1.0),
        "9": node("VAEDecode", samples=["8", 0], vae=["3", 0]),
        "10": node("SaveImage", images=["9", 0], filename_prefix=f"modeexplorer_eval/{name}"),
    }
    model_input = ["1", 0]
    for key, name, strength in (
        ("11", args.model_lora, args.model_lora_strength), ("12", args.identity_lora, args.lora_strength),
    ):
        if name:
            g[key] = node("LoraLoaderModelOnly", model=model_input, lora_name=name, strength_model=strength)
            model_input = [key, 0]
    if mode != "native" or args.diagnostics:
        g["7"] = node(
            "ModeExplorerKrea2", model=model_input, enabled=True, feature=feature, initialization=initialization,
            strength=args.strength, fidelity=args.fidelity, min_sigma=args.min_sigma, adaptive=not args.fixed,
            similarity=args.similarity, operation="sequential", reference_budget=args.reference_budget,
            diagnostics=args.diagnostics, full_spectrum=spectrum == "full",
        )
        g["7"]["inputs"].update(profile.get("settings", {}))
        model_input = ["7", 0]
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
    if getattr(args, "planner_megapixels", None) is not None:
        g["8"]["class_type"] = "Krea2TurboFlowAlignedRegenerate"
        del sampler["denoise"]
        sampler.update(planner_megapixels=args.planner_megapixels, coarse_lock_strength=0.0,
                       mid_lock_strength=0.0, composition_lock_sigma=args.composition_lock_sigma,
                       handoff_diagnostics="log" if args.diagnostics else "off")
        g["14"] = node("VAEDecode", samples=["8", 1], vae=["3", 0])
        g["15"] = node("SaveImage", images=["14", 0], filename_prefix=f"modeexplorer_eval/{name}_planner")
    return g


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
    if path is None:
        return trace, spectrum
    with path.open("rb") as logfile:
        if offset <= path.stat().st_size:
            logfile.seek(offset)
        for line in logfile:
            for marker in (b"MODEEXPLORER_TRACE ", b"Spectrum Krea 2 summary: "):
                _, found, payload = line.partition(marker)
                if not found:
                    continue
                try:
                    observed = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if marker.startswith(b"Spectrum"):
                    spectrum.append(observed)
                elif observed["seed"] == seed and observed["completed"]:
                    trace = observed
    return trace, spectrum


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
    trace, spectrum = read_observations(args.server_log, offset, seed)
    stamps = {kind: data.get("timestamp") for kind, data in status.get("messages", [])}
    start_ms, end_ms = stamps.get("execution_start"), stamps.get("execution_success")
    return {
        "job_id": prompt_id,
        "wall_seconds": finish - start,
        "execution_seconds": (end_ms - start_ms) / 1000 if start_ms is not None and end_ms is not None else None,
        "sampled_peak_device_mib": vram.peak_mib,
        "trace": trace, "spectrum": spectrum,
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
    p.add_argument("--category", action="append", help="Restrict the prompt panel to this category.")
    p.add_argument("--repeats", type=int, choices=(1, 2, 3), default=1,
                   help="Repeat each same-seed job. Use a dedicated server with --cache-none and a readable --server-log.")
    p.add_argument("--model-lora", help="Optional model LoRA for a separate external baseline.")
    p.add_argument("--model-lora-strength", type=float, default=1.0)
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
        manifest["seeds"] = SEEDS[:2 if args.smoke else args.seed_count]
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
