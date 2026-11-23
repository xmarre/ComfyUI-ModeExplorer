"""Lossless evidence crops and bounded final-image preview panels."""

import hashlib
import json
from io import BytesIO

from .report_inputs import require
from .trajectory_review import _pixel_hash


def prompt_signature(prompt):
    copy = json.loads(json.dumps(prompt))
    for node in copy.values():
        # Frontend titles and registry annotations change on workflow import.
        # The executable class, input values and connections carry provenance.
        node.pop("_meta", None)
        inputs = node["inputs"]
        if node["class_type"] == "ModeExplorerKrea2":
            for opt in ("feature_block", "trajectory_label"):
                inputs.pop(opt, None)
        elif node["class_type"] == "SaveImage":
            inputs.pop("filename_prefix", None)
    return hashlib.sha256(json.dumps(copy, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def open_final(root, pin, spec, resolution):
    from PIL import Image

    # A prefix alone does not bind a PNG to its run. Historical images need the
    # original RGB fingerprint; new images need the executable prompt contract
    # plus the feature label and block, which the prompt hash excludes.
    require("pixel_sha256" in spec or ("prompt_sha256" in spec and spec.get("verify_feature_widget")),
            "missing_final_png_provenance_pin")
    prefix = (root / spec["prefix"]).resolve()
    require(prefix.is_relative_to(root), "unsafe_image_output_prefix")
    matches = {}
    seen = 0
    for path in sorted(prefix.parent.glob(prefix.name + "_*.png")):
        if not path.resolve().is_relative_to(root):
            continue
        seen += 1
        require(seen <= 64, "too_many_final_png_candidates")
        with Image.open(path) as handle:
            if list(handle.size) != resolution:
                continue
            if "prompt_sha256" in spec or spec.get("verify_feature_widget"):
                try:
                    prompt = json.loads(handle.info.get("prompt", "{}"))
                    signature = prompt_signature(prompt)
                    settings = [item["inputs"] for item in prompt.values()
                                if item["class_type"] == "ModeExplorerKrea2"]
                except (ValueError, TypeError, KeyError, AttributeError):
                    # Malformed metadata disqualifies only this candidate.
                    continue
            if "prompt_sha256" in spec and signature != spec["prompt_sha256"]:
                continue
            if spec.get("verify_feature_widget"):
                if len(settings) != 1 or settings[0].get("trajectory_label") != pin["label"]:
                    continue
                if settings[0].get("feature_block") != (-1 if pin["block"] == "middle" else pin["block"]):
                    continue
            pixels = handle.convert("RGB")
            digest = _pixel_hash(pixels)
            if "pixel_sha256" in spec and digest != spec["pixel_sha256"]:
                continue
            matches.setdefault(digest, (pixels, path))
            require(len(matches) == 1, "ambiguous_non_equivalent_final_images")
    require(matches, "unavailable_final_png_with_expected_provenance")
    digest, (pixels, path) = next(iter(matches.items()))
    return pixels, {"path": path.relative_to(root).as_posix(), "pixel_sha256": digest, "resolution": resolution}


def pixel_metrics(sham, edited):
    import numpy as np

    count, energy, changed, maximum = 0, 0., 0, 0
    w, h = sham.size
    for y in range(0, h, 128):
        box = (0, y, w, min(y + 128, h))
        delta = np.asarray(edited.crop(box), dtype=np.float32) - np.asarray(sham.crop(box), dtype=np.float32)
        energy += float(np.square(delta).sum(dtype=np.float64))
        count += delta.size
        changed += int(np.count_nonzero(delta))
        maximum = max(maximum, int(np.abs(delta).max()))
    return {"rms_0_255": (energy / count) ** .5, "max_abs_0_255": maximum, "changed_rgb_values": changed}


def encode(image, codec, **kwargs):
    if image.info:
        image = image.copy()
        image.info.clear()
    stream = BytesIO()
    image.save(stream, codec, **kwargs)
    return stream.getvalue()


def preview(image, width):
    from PIL import Image

    return image.resize((width, round(image.height / image.width * width)), Image.Resampling.LANCZOS)


def panel(images):
    from PIL import Image, ImageDraw

    width, height = images[0][1].size
    cols = min(4, len(images))
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (width * cols, (height + 24) * rows), "white")
    label = ImageDraw.Draw(sheet)
    for i, (name, image) in enumerate(images):
        x, y = i % cols * width, i // cols * (height + 24)
        label.text((x + 5, y + 5), name, fill="black")
        sheet.paste(image, (x, y + 24))
    return sheet


def blind_labels(plan):
    seed = plan.get("blind_seed")
    if seed is None:
        return None

    def shuffled(names, scope):
        def order(name):
            payload = json.dumps([seed, scope, name], separators=(",", ":"))
            return hashlib.sha256(payload.encode()).digest()

        return sorted(names, key=order)

    return {
        case: {"name": f"case_{i:02d}",
               "arms": {arm: f"arm_{j:02d}" for j, arm in enumerate(shuffled(spec["arms"], [case, "arms"]), 1)},
               "regions": {region: f"region_{j:02d}" for j, region in enumerate(shuffled(spec["crops"], [case, "regions"]), 1)}}
        for i, case in enumerate(shuffled(plan["cases"], "cases"), 1)
        for spec in (plan["cases"][case],)
    }


LOSSLESS_BUDGET_ERROR = ("Lossless evidence exceeds the report budget. "
                         "Narrow the report's fixed crop policy or increase its budget.")


def make_visuals(cases, plan, output, interrupt, presentation=None, limit=None):
    from PIL import features

    codec, extension = ("WEBP", "webp") if features.check("webp") else ("PNG", "png")
    lossless = {"lossless": True, "method": 4} if codec == "WEBP" else {}
    files, panels, retained = {}, {}, 0
    for case_name, spec in plan["cases"].items():
        report = cases[case_name]
        if report["comparison_status"] == "unavailable":
            continue
        images = {}
        for arm, pin in spec["arms"].items():
            if report["arms"][arm]["qualification"] != "qualified":
                continue
            interrupt()
            try:
                pixels, provenance = open_final(output, pin, pin["final"], spec["resolution"])
            except (OSError, ValueError, KeyError) as exc:
                report["arms"][arm]["final_image"] = {"status": "unavailable_or_unqualified", "reason": str(exc)}
                continue
            images[arm] = pixels
            report["arms"][arm]["final_image"] = {"status": "qualified", **provenance}
        entry = presentation[case_name] if presentation is not None else None
        names = entry["arms"] if entry else {arm: arm for arm in images}
        case_label = entry["name"] if entry else case_name
        ordered = [arm for arm in names if arm in images]
        if "sham" in images:
            for arm, pixels in images.items():
                report["arms"][arm]["final_image"]["pixel_delta_from_sham"] = pixel_metrics(images["sham"], pixels)
        # Preserve exact crop pixels. The JPEG previews alone are resized.
        for arm in ordered:
            interrupt()
            for region, box in spec.get("visual_crops", spec["crops"]).items():
                region_label = entry["regions"][region] if entry else region
                basename = f"visual/{case_label}/crops/{region_label}/{names[arm]}.{extension}"
                files[basename] = encode(images[arm].crop(box), codec, **lossless)
                # The ZIP stores compressed images without recompression, so the
                # retained crop bytes are a lower bound on the archive size.
                retained += len(files[basename])
                if limit is not None and retained > limit:
                    raise ValueError(LOSSLESS_BUDGET_ERROR)
        panels[case_label] = [(names[arm], preview(images[arm], 560)) for arm in ordered]
    return files, panels


def jpeg_evidence(panels, width, quality):
    files = {}
    for case, images in panels.items():
        if not images:
            continue
        reduced = [(name, preview(im, width) if im.width != width else im) for name, im in images]
        files[f"visual/{case}/final_panel.jpg"] = encode(panel(reduced), "JPEG", quality=quality, optimize=True)
        for name, image in reduced:
            files[f"visual/{case}/final/{name}.jpg"] = encode(image, "JPEG", quality=quality, optimize=True)
    return files
