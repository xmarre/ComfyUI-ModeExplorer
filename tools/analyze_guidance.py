"""Compare clean predictions from matched frozen Krea 2 evaluations."""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from mode_explorer.mechanism import measure_comparisons


def load(path, field):
    directory = Path(path)
    manifest = directory / "manifest.json"
    data = directory / "predictions.safetensors"
    if manifest.stat().st_size > 16 * 1024**2 or data.stat().st_size > 64 * 1024**2 + 65536:
        raise ValueError("Export exceeds the response size limit.")
    point = json.loads(manifest.read_text())
    if point.get("schema") != 1:
        raise ValueError("Unsupported response manifest.")
    tensors = load_file(data, device="cpu")
    clean = tensors[field]
    if clean.ndim == 5 and clean.shape[2] == 1:
        clean = clean.squeeze(2)
    if clean.ndim != 4 or clean.shape[0] != 1 or not torch.isfinite(clean).all():
        raise ValueError("Expected one finite image clean prediction.")
    return point, clean[0]


def matched(left, right):
    for field in ("sigma", "stage", "native_input_fp32_sha256", "native_condition_values"):
        if field not in left or left[field] is None or left[field] != right.get(field):
            raise ValueError(f"Frozen states differ or lack a verified {field}.")


def l2(values):
    return float(torch.linalg.vector_norm(values.flatten()))


def compare(guided_path, baseline_path, controls):
    guided, prediction = load(guided_path, "guided_clean")
    baseline, native = load(baseline_path, "baseline_clean")
    matched(guided, baseline)
    if prediction.shape != native.shape:
        raise ValueError("Clean shapes do not match.")
    delta = prediction - native
    relative = l2(delta) / delta.numel()**.5
    intervention = guided["rows"][0]["realized_rms"]
    fields = {"delta": ((1., delta),)}
    comparisons = {"guided": ((), ((1., "delta"),))}
    metrics = measure_comparisons(fields, comparisons, 16, .0625, .125, sample_stride=1, patch_size=2)["guided"]
    result = {"sigma": guided["sigma"], "clean_space": "comfy_vae_latent",
              "guidance_rms": intervention, "clean_displacement_rms": relative,
              "clean_to_projection_rms": relative / intervention if intervention else None,
              "clean_measurement": metrics, "direction_alignment": {}}
    for label, directory, field in controls:
        metadata, candidate = load(directory, field)
        matched(baseline, metadata)
        if candidate.shape != native.shape:
            raise ValueError(f"Unexpected clean shape for {label}.")
        other = candidate - native
        denom = l2(delta) * l2(other)
        result["direction_alignment"][label] = float(torch.sum(delta.double() * other.double())) / denom if denom else None
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("guided_path", help="Projection Guidance clean export directory")
    parser.add_argument("baseline_path", help="Matched Response or Mechanistic export directory with baseline_clean")
    parser.add_argument("--control", action="append", nargs=3, default=[], metavar=("LABEL", "DIR", "CLEAN_FIELD"))
    args = parser.parse_args()
    print(json.dumps(compare(args.guided_path, args.baseline_path, args.control), indent=2, allow_nan=False))
