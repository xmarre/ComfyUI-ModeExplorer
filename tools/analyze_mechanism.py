"""Read versioned ModeExplorer Mechanistic exports without loading ComfyUI."""

import argparse
import json
import math
from pathlib import Path


MAX_MANIFEST = 16 * 1024**2
MAX_TENSORS = 64 * 1024**2 + 65536


def load(path):
    path = Path(path)
    path = path / "manifest.json" if path.is_dir() else path
    if path.stat().st_size > MAX_MANIFEST:
        raise ValueError("Manifest exceeds the mechanistic size limit.")
    point = json.loads(path.read_text())
    study = point.get("mechanistic", {})
    if point.get("schema") != 1 or study.get("schema") not in (1, 2):
        raise ValueError(f"{path} is not a supported mechanistic export.")
    return path, point, study


def profile(study, comparison):
    result = {}
    for key, site in study["sites"].items():
        value = site.get("comparisons", {}).get(comparison)
        result[key] = {
            "status": "completed" if value else site["status"] if site["status"] != "completed" else "comparison_missing",
            "reason": site.get("reason"),
            "coordinate": site["coordinate"],
            "latent_pixels_per_sample": site.get("latent_pixels_per_sample"),
            "even_energy_fraction": site.get("even_energy_fraction"),
            "even_energy_fraction_by_band": site.get("even_energy_fraction_by_band"),
            "measurement": value,
        }
    return result


def compare(left, right):
    point0, study0 = left
    point1, study1 = right
    fields = (
        "sigma", "timestep", "noise_scale", "probe_seed", "feature_reference_initial_sigma",
        "feature_reference_initial_sol_evaluations", "sol_evaluations", "feature_settings",
        "reference_method",
    )
    pairs = {name: (point0.get(name), point1.get(name)) for name in fields}
    pairs["input_values"] = tuple(p.get("native_input_fp32_sha256", p.get("native_input_sha256")) for p in (point0, point1))
    pairs["references"] = tuple(
        p.get("native_condition_values", {}).get("references", p.get("reference_latent_sha256")) for p in (point0, point1)
    )

    def condition_values(study):
        return {key: (data.get("context_values_sha256", data["sha256"]),
                      data.get("mask_values_sha256", data["mask_sha256"]), data["metadata_mask_sha256"], data["tokens"])
                for key, data in study["conditioning"].items()}

    pairs["conditioning"] = condition_values(study0), condition_values(study1)
    for name in ("schema", "mode", "injection_block", "frequency_cycles_per_grid_cell",
                 "frequency_cycles_per_latent_pixel", "patch_nyquist_per_axis", "amplitude_check"):
        pairs[name] = (study0.get(name), study1.get(name))
    mismatches = [name for name, (a, b) in pairs.items() if a != b]
    sites0, sites1 = study0["sites"], study1["sites"]
    geometry = {}
    for key in sorted(sites0.keys() | sites1.keys()):
        if key not in sites0 or key not in sites1:
            geometry[key] = "site_missing"
            continue
        a, b = sites0[key], sites1[key]
        if a["status"] != "completed" or b["status"] != "completed":
            geometry[key] = "measurement_unavailable"
            continue
        shape0 = a["comparisons"]["positive"]["shape_chw"]
        shape1 = b["comparisons"]["positive"]["shape_chw"]
        if shape0 != shape1 or a["coordinate"] != b["coordinate"]:
            geometry[key] = "site_geometry_mismatch"
    return {"status": "matched_inputs" if not mismatches else "incompatible_inputs", "mismatched": mismatches,
            "site_differences": geometry,
            "execution": [study.get("kernel_contract") for study in (study0, study1)],
            "checkpoint_provenance": [study.get("checkpoint_provenance") for study in (study0, study1)]}


def component_pair(left, left_name, right, right_name):
    import torch
    from safetensors import safe_open

    if any(not name.endswith("_native_delta") for name in (left_name, right_name)):
        raise ValueError("Spatial comparisons require native-coordinate component keys.")
    fields = []
    for manifest, key in ((left, left_name), (right, right_name)):
        data = manifest.with_name("predictions.safetensors")
        if not data.is_file():
            return {"status": "tensor_file_missing", "key": key}
        if data.stat().st_size > MAX_TENSORS:
            raise ValueError("Tensor file exceeds the response size limit.")
        with safe_open(data, framework="pt", device="cpu") as package:
            if key not in package.keys():
                return {"status": "component_missing", "key": key}
            a = package.get_tensor(key)
        if a.ndim not in (4, 5) or a.shape[0] != 1 or (a.ndim == 5 and a.shape[2] != 1):
            return {"status": "component_not_spatial", "key": key}
        fields.append(a.reshape(-1, *a.shape[-2:]).double())
    a, b = fields
    if a.shape != b.shape:
        return {"status": "component_geometry_mismatch"}
    if not (a.isfinite().all() and b.isfinite().all()):
        return {"status": "nonfinite_component"}
    ea, eb = a.square().sum(0), b.square().sum(0)
    sa, sb = float(ea.sum()), float(eb.sum())
    if not sa or not sb:
        return {"status": "zero_energy", "left_energy": sa, "right_energy": sb}
    overlap = torch.minimum(ea / sa, eb / sb).sum()
    return {"status": "completed", "shape_chw": list(a.shape), "left_energy": sa, "right_energy": sb,
            "displacement_cosine": float((a * b).sum()) / math.sqrt(sa * sb),
            "spatial_energy_overlap": float(overlap)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exports", nargs="+", help="One export directory or manifest, or a matched pair.")
    parser.add_argument("--comparison", default="positive", help="Site comparison key, such as even or conditioning_interaction.")
    parser.add_argument("--components", nargs=2, metavar=("LEFT", "RIGHT"),
                        help="Compare two exported native component fields.")
    args = parser.parse_args()
    if len(args.exports) > 2:
        parser.error("Supply one export or a pair.")
    loaded = [load(path) for path in args.exports]
    result = {"comparison": args.comparison, "profiles": [{
        "path": str(path), "status": point["status"], "sigma": point["sigma"],
        "amplitude": study.get("amplitude"), "sites": profile(study, args.comparison),
        "frequency_cycles_per_latent_pixel": study.get("frequency_cycles_per_latent_pixel"),
        "frequency_cycles_per_grid_cell": study.get("frequency_cycles_per_grid_cell"),
        "patch_nyquist_per_axis": study.get("patch_nyquist_per_axis"),
    } for path, point, study in loaded]}
    if len(loaded) == 2:
        result["compatibility"] = compare(loaded[0][1:], loaded[1][1:])
    if args.components:
        if result.get("compatibility", {}).get("status") == "incompatible_inputs":
            result["components"] = {"status": "incompatible_inputs"}
        else:
            result["components"] = component_pair(loaded[0][0], args.components[0],
                                                  loaded[-1][0], args.components[1])
    print(json.dumps(result, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
