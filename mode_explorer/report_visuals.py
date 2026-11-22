"""Lossless evidence crops and bounded final-image preview panels."""

import hashlib
import json
from io import BytesIO

from .report_inputs import require
from .trajectory_review import _pixel_hash


def prompt_signature(prompt):
    copy = json.loads(json.dumps(prompt))
    for node in copy.values():
        inputs = node["inputs"]
        if node["class_type"] == "ModeExplorerKrea2":
            for opt in ("feature_block", "trajectory_label"):
                inputs.pop(opt, None)
        elif node["class_type"] == "SaveImage":
            inputs.pop("filename_prefix", None)
    return hashlib.sha256(json.dumps(copy, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def open_final(root, pin, spec, resolution):
    from PIL import Image

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
            prompt = json.loads(handle.info.get("prompt", "{}"))
            if "prompt_sha256" in spec and prompt_signature(prompt) != spec["prompt_sha256"]:
                continue
            if spec.get("verify_feature_widget"):
                settings = [item["inputs"] for item in prompt.values() if item["class_type"] == "ModeExplorerKrea2"]
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


def make_visuals(cases, plan, output, interrupt):
    from PIL import features

    codec, extension = ("WEBP", "webp") if features.check("webp") else ("PNG", "png")
    lossless = {"lossless": True, "method": 4} if codec == "WEBP" else {}
    files, mapping, panels = {}, {}, {}
    for case_name, spec in plan["cases"].items():
        report = cases[case_name]
        if report["qualification"] != "qualified":
            continue
        images = {}
        for arm, pin in spec["arms"].items():
            interrupt()
            try:
                pixels, provenance = open_final(output, pin, pin["final"], spec["resolution"])
            except (OSError, ValueError, KeyError) as exc:
                report["arms"][arm]["final_image"] = {"status": "unavailable_or_unqualified", "reason": str(exc)}
                continue
            images[arm] = pixels
            report["arms"][arm]["final_image"] = {"status": "qualified", **provenance}
        ordered = list(images)
        if plan.get("blind_seed"):
            ordered.sort(key=lambda arm: hashlib.sha256(f"{plan['blind_seed']}:{case_name}:{arm}".encode()).digest())
        names = {arm: f"arm_{i+1:02d}" if plan.get("blind_seed") else arm for i, arm in enumerate(ordered)}
        mapping[case_name] = {name: arm for arm, name in names.items()}
        if "sham" in images:
            for arm, pixels in images.items():
                report["arms"][arm]["final_image"]["pixel_delta_from_sham"] = pixel_metrics(images["sham"], pixels)
        # Preserve exact crop pixels. The JPEG previews alone are resized.
        for arm in ordered:
            interrupt()
            for region, box in spec["crops"].items():
                basename = f"visual/{case_name}/crops/{region}/{names[arm]}.{extension}"
                files[basename] = encode(images[arm].crop(box), codec, **lossless)
        panels[case_name] = [(names[arm], preview(images[arm], 560)) for arm in ordered]
    if plan.get("blind_seed"):
        files["unblinding.json"] = json.dumps(mapping, indent=2).encode()
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
