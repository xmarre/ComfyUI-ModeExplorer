"""Review matched trajectory exports using the production VAE only."""

import hashlib
import json
import logging
from pathlib import Path
from time import perf_counter

import torch

from .response import flow_clean
from .response_runtime import _tensor_hash
from .trajectory import IDENTIFIER, LABEL, _root, read_trajectory

LOGGER = logging.getLogger("ModeExplorer")
REPRODUCTION = ("native", "native_repeat", "sham", "active")
TEMPORAL = ("sham", "first_only", "second_only", "active")
CONTROL_KEYS = ("seed", "cfg", "stage", "solver", "solver_options", "schedule",
                "initial_noise_sha256", "packed_conditioning")
STATE_KEYS = ("conditioning_sha256", "attention_mask_sha256", "reference_latent_sha256",
              "backend", "reference_method", "timestep", "er_sde_noise_cursor", "reference_calls")


def _manifest(folder):
    path = folder / "manifest.json"
    if folder.resolve().parent != _root() or path.resolve().parent != folder.resolve() or path.stat().st_size > 512 * 1024:
        raise ValueError("Invalid trajectory manifest path or size.")
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or data.get("schema") != 1:
        raise ValueError("Unsupported trajectory manifest format.")
    return data


def _resolve_arm(arm, pin, baseline=None):
    if "id" in pin:
        name = pin["id"]
        if not IDENTIFIER.fullmatch(f"{name}/000"):
            raise ValueError(f"{arm}: invalid archived trajectory ID.")
        folder = _root() / name
        return folder, _manifest(folder)
    label = pin["label"]
    found = []
    for folder in _root().glob("trajectory-*"):
        if not IDENTIFIER.fullmatch(f"{folder.name}/000") or not (folder / "manifest.json").is_file():
            continue
        data = _manifest(folder)
        if data.get("label") == label and (data.get("ablation") or {}).get("status") == "qualified" and all(
                data.get(key) == baseline.get(key) for key in CONTROL_KEYS):
            found.append((folder, data))
    if not found:
        raise ValueError(f"{arm}: no qualified trajectory matching {label} and the archived controls. Queue the corresponding ablation workflow first.")
    # A repeated queue creates another export. Resolve by completion time and
    # record exactly which directory was selected in the review manifest.
    return max(found, key=lambda item: ((item[0] / "manifest.json").stat().st_mtime_ns, item[0].name))


def qualify_runs(runs, temporal):
    """Align by actual ER-SDE opportunity and exact sigma, never by state ordinal."""
    first = next(iter(runs.values()))
    if (first.get("stage") != "planner" or first.get("cfg") != 1.0
            or not first.get("solver", "").endswith(".sample_er_sde") or first["schedule"][-1] != 0):
        raise ValueError("Trajectory review requires a CFG-1 ER-SDE planner ending at zero.")
    for arm, data in runs.items():
        if data.get("status") != "completed" or any(data.get(key) != first.get(key) for key in CONTROL_KEYS):
            raise ValueError(f"{arm}: incomplete trajectory or different planner provenance.")
        expected = {"native": [], "native_repeat": [], "sham": [], "first_only": [0],
                    "second_only": [1], "active": [0, 1]}[arm]
        if [state["ordinal"] for state in data["states"] if state["feature_applied"]] != expected:
            raise ValueError(f"{arm}: applied intervention schedule does not match the review arm.")

    def routes(data):
        return [(r["solver_step"], r["route"], r["reference_calls"], r["sigmas"], r["branches"]) for r in data["routes"]]

    guarded = routes(runs["sham"])
    compared = TEMPORAL if temporal else ("sham", "active")
    if any(routes(runs[arm]) != guarded for arm in compared):
        raise ValueError("The feature arms have different Spectrum routes or reference opportunities.")
    if any(r[4] != [0] for r in guarded) or [r[2] for r in guarded] != [1, 1] + [0] * (len(guarded) - 2):
        raise ValueError("Expected exactly one positive reference on each of the first two opportunities.")

    states = {}
    for arm, run in runs.items():
        states[arm] = {}
        for state in run["states"]:
            key = (state["solver_step"], state["opportunity"], state["sigma"])
            if key in states[arm] or state["route"] != "actual" or state["branches"] != [0]:
                raise ValueError(f"{arm}: duplicate or nonactual captured state.")
            states[arm][key] = state
    common = sorted(set.intersection(*(set(records) for records in states.values())))
    if not common or common[-1][0] != len(first["schedule"]) - 2:
        raise ValueError("The terminal planner opportunity is not present as an actual state in every arm.")
    for key in common:
        matched = {arm: records[key] for arm, records in states.items()}
        if any(state["conditioning_sha256"] != matched["sham"]["conditioning_sha256"]
               for state in matched.values()):
            raise ValueError(f"Conditioning fingerprint differs at {key}.")
        if any(any(matched[arm].get(k) != matched["sham"].get(k) for k in STATE_KEYS)
               for arm in compared):
            raise ValueError(f"Routed feature-state provenance differs at {key}.")
        if any(matched[arm].get("reference_slots") != matched["sham"].get("reference_slots")
               for arm in compared):
            raise ValueError(f"Reference slots differ at {key}.")
    first_states = {arm: records[common[0]] for arm, records in states.items()}
    if len({s["native_input_sha256"] for s in first_states.values()}) != 1:
        raise ValueError("The initial actual noisy inputs differ.")
    if temporal:
        second = common[1]
        for a, b in (("sham", "second_only"), ("active", "first_only")):
            if states[a][second]["native_input_sha256"] != states[b][second]["native_input_sha256"]:
                raise ValueError(f"{a}/{b}: the second noisy state differs before their application schedules separate.")
        left, right = (runs[arm] for arm in ("first_only", "second_only"))
        if [s["reference_history"] for s in left["states"][:2]] != [s["reference_history"] for s in right["states"][:2]]:
            raise ValueError("The new arms have different reference continuation histories.")
        ignore = {"feature_application", "ablation_route", "trajectory_label", "sham"}
        recipes = [{key: value for key, value in runs[arm]["settings"].items() if key not in ignore} for arm in TEMPORAL]
        if any(recipe != recipes[0] for recipe in recipes[1:]):
            raise ValueError("Temporal feature settings differ beyond scheduled application.")
    return common, states


def checked_tensors(state):
    from comfy.latent_formats import Wan21
    from safetensors.torch import load_file

    _, path = read_trajectory(state["export_id"])
    tensors = load_file(path)
    computed = Wan21().process_out(flow_clean(tensors["frozen_native_input"], tensors["conditional_velocity"], state["sigma"]))
    if not torch.equal(computed, tensors["baseline_clean"]):
        raise ValueError(f"{state['export_id']}: archived clean tensor mismatches the Krea 2 latent conversion.")
    if "solver_clean" in tensors and _tensor_hash(tensors["solver_clean"]) != state.get("solver_clean_sha256"):
        raise ValueError(f"{state['export_id']}: solver clean fingerprint mismatch.")
    return tensors


def _pixel_hash(image):
    return hashlib.sha256(image.tobytes()).hexdigest()


def _compare_final(image, pin, output_dir):
    from PIL import Image
    import numpy as np

    digest = _pixel_hash(image)
    path = (output_dir / pin["output_png"]).resolve()
    if not path.is_relative_to(output_dir):
        raise ValueError("Final planner PNG must be inside ComfyUI output.")
    result = {"pixel_sha256": digest, "expected_pixel_sha256": pin["pixel_sha256"],
              "pixel_exact": digest == pin["pixel_sha256"]}
    if not path.is_file():
        return {**result, "status": "archived_png_not_in_output_digest_compared"}
    with Image.open(path) as handle:
        original = handle.convert("RGB")
    if original.size != image.size or _pixel_hash(original) != pin["pixel_sha256"]:
        return {**result, "status": "local_png_does_not_match_archive"}
    diff = np.asarray(original, dtype=np.float32) - np.asarray(image, dtype=np.float32)
    return {**result, "status": "compared_to_archived_planner_png", "pixel_rms_0_255": float(np.sqrt(np.mean(diff ** 2))),
            "pixel_max_abs_0_255": int(np.max(np.abs(diff)))}


def _sheet(images, caption):
    from PIL import Image, ImageDraw

    width, height = images[0][1].size
    sheet = Image.new("RGB", (width * len(images), height + 28), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (arm, image) in enumerate(images):
        draw.text((width * index + 5, 6), f"{arm}  {caption}", fill="black")
        sheet.paste(image, (width * index, 28))
    return sheet


class ModeExplorerTrajectoryReview:
    CATEGORY = "model_patches/ModeExplorer"
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "review"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"vae": ("VAE",), "review_plan": ("STRING", {"default": "", "multiline": True})}}

    @classmethod
    def IS_CHANGED(cls, vae, review_plan):
        # The requested local archives and output controls may change between queues.
        return float("nan")

    def review(self, vae, review_plan):
        import shutil
        import tempfile

        import folder_paths
        import numpy as np
        from PIL import Image
        from comfy.model_management import throw_exception_if_processing_interrupted
        from comfy.utils import ProgressBar
        from nodes import VAEDecode

        started = perf_counter()
        plan = json.loads(review_plan)
        case = plan["case"]
        temporal = plan["experiment"] == "temporal"
        expected = TEMPORAL if temporal else REPRODUCTION
        if not LABEL.fullmatch(case) or plan["experiment"] not in ("reproduction", "temporal") or set(plan["arms"]) != set(expected):
            raise ValueError("Invalid trajectory review plan.")
        output = Path(folder_paths.get_output_directory()).resolve()
        pins = plan["arms"]
        _, anchor = _resolve_arm("sham", pins["sham"])
        folders = {arm: _resolve_arm(arm, pins[arm], anchor) for arm in expected}
        runs = {arm: data for arm, (_, data) in folders.items()}
        common, states = qualify_runs(runs, temporal)
        # Coordinates must agree across an entire comparison. Legacy arms store
        # the model-input estimate, while new ER-SDE exports also store solver clean.
        exact = all("solver_clean_sha256" in states[arm][key] for arm in expected for key in common)
        tensor = "solver_clean" if exact else "baseline_clean"
        target = output / "modeexplorer_phase_iv"
        target.mkdir(parents=True, exist_ok=True)
        session = Path(tempfile.mkdtemp(prefix=f"{case}_{plan['experiment']}_", dir=target))
        report = {"schema": 1, "case": case, "experiment": plan["experiment"], "decode_tensor": tensor,
                  "coordinate": "exact_solver_vae_latent" if exact else "vae_latent_from_cast_dit_input",
                  "auxiliary_dit_calls": 0, "vae_decodes": len(common) * len(expected),
                  "arms": {arm: {"id": folder.name, "routes": data["routes"],
                                "forecast_steps": [r["solver_step"] for r in data["routes"] if r["route"] == "forecast"]}
                           for arm, (folder, data) in folders.items()},
                  "evaluations": [], "crop_boxes_xyxy": plan["crops"], "final_planner_comparison": {}}
        progress = ProgressBar(report["vae_decodes"])
        logger_args = (case, plan["experiment"], len(common), report["vae_decodes"])
        LOGGER.info("ModeExplorer trajectory decode: %s %s, %d common actual evaluations, %d VAE decodes, zero DiT calls", *logger_args)
        try:
            for key in common:
                step, opportunity, sigma = key
                LOGGER.info("ModeExplorer trajectory decode: %s step=%d sigma=%.17g", case, step, sigma)
                frame_dir = session / f"step_{step:02d}"
                frame_dir.mkdir()
                images, proposals, fingerprints = [], {}, {}
                for arm in expected:
                    throw_exception_if_processing_interrupted()
                    state = states[arm][key]
                    tensors = checked_tensors(state)
                    if "compact_feature_proposal" in tensors:
                        proposals[arm] = _tensor_hash(tensors["compact_feature_proposal"])
                    fingerprints[arm] = _tensor_hash(tensors[tensor])
                    pixels, = VAEDecode().decode(vae, {"samples": tensors[tensor]})
                    if pixels.ndim != 4 or pixels.shape[0] != 1 or pixels.shape[-1] != 3:
                        raise ValueError("Expected a single RGB planner image from VAE Decode.")
                    array = np.clip(255. * pixels[0].cpu().numpy(), 0, 255).astype(np.uint8)
                    image = Image.fromarray(array)
                    if list(image.size) != plan["resolution"]:
                        raise ValueError("VAE output does not match the archived planner resolution.")
                    image.save(frame_dir / f"{arm}.png")
                    images.append((arm, image))
                    if key == common[-1] and "final" in pins[arm]:
                        report["final_planner_comparison"][arm] = _compare_final(image, pins[arm]["final"], output)
                    progress.update(1)
                if key == common[0] and (set(proposals) != set(TEMPORAL if temporal else ("sham", "active"))
                                         or len(set(proposals.values())) != 1):
                    raise ValueError("The first compact feature proposals do not match across the feature arms.")
                if not temporal and fingerprints["native"] != fingerprints["native_repeat"]:
                    raise ValueError("The native repeat differs in captured clean coordinates.")
                caption = f"step {step}  sigma {sigma:.9f}"
                _sheet(images, caption).save(frame_dir / "full.png")
                for name, box in plan["crops"].items():
                    if not LABEL.fullmatch(name) or not (0 <= box[0] < box[2] <= plan["resolution"][0]
                                                         and 0 <= box[1] < box[3] <= plan["resolution"][1]):
                        raise ValueError("Invalid fixed crop bounds.")
                    _sheet([(arm, image.crop(box)) for arm, image in images], caption).save(frame_dir / f"{name}.png")
                report["evaluations"].append({"solver_step": step, "opportunity": opportunity, "sigma": sigma,
                                             "actual_ordinals": {arm: states[arm][key]["ordinal"] for arm in expected},
                                             "clean_sha256": fingerprints, "proposal_sha256": proposals})
            report["seconds"] = perf_counter() - started
            (session / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        except Exception:
            shutil.rmtree(session, ignore_errors=True)
            raise
        LOGGER.info("ModeExplorer trajectory decode complete: %s in %.2fs", session.name, report["seconds"])
        return ()
