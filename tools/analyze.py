#!/usr/bin/env python3
"""Analyze matched outputs and optionally make anonymized review sheets."""

import argparse
from collections import defaultdict
import csv
from itertools import combinations
import json
from pathlib import Path
import random
from statistics import fmean
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageOps

GROUPS = ("composition_group", "camera_group", "pose_group", "layout_group")
DEFECT_RATINGS = ("artifact", "anatomy_defect", "relation_defect")
RATINGS = (*GROUPS, "prompt_adherence", "quality", "artifact", "identity",
           "anatomy_defect", "relation_defect", "detail_quality")


def describe(path):
    with Image.open(path) as source:
        image = source.convert("L").resize((96, 96), Image.Resampling.LANCZOS)
    pixels = np.asarray(image, dtype=np.float32) / 255
    # Absolute gradient energy in spatially ordered horizontal/vertical bins.
    dx = np.abs(np.diff(pixels, axis=1, prepend=pixels[:, :1]))
    dy = np.abs(np.diff(pixels, axis=0, prepend=pixels[:1]))
    spatial = np.concatenate([d.reshape(6, 16, 6, 16).mean(axis=(1, 3)).flatten() for d in (dx, dy)])
    return spatial / max(float(np.linalg.norm(spatial)), 1e-8)


def cosine_distance(a, b):
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.clip(1 - np.dot(a, b) / denom, 0, 2)) if denom > 1e-8 else 0.0


def average(values):
    values = [v for v in values if v is not None]
    return fmean(values) if values else None


def bootstrap(paired, count=2000):
    if len(paired) < 2:
        return None
    rng = random.Random(26823)
    draws = sorted(fmean(rng.choices(paired, k=len(paired))) for _ in range(count))
    return [draws[int(0.025 * count)], draws[int(0.975 * count)]]


def load_ratings(path):
    if path is None:
        return {}
    with path.open(newline="", encoding="utf-8") as file:
        rows = {}
        for row in csv.DictReader(file):
            if row["id"] in rows:
                raise ValueError(f"Duplicate review ID {row['id']}")
            for key in RATINGS[len(GROUPS):]:
                if row.get(key):
                    value = float(row[key])
                    valid = value in (0, 1) if key in DEFECT_RATINGS else 1 <= value <= 5
                    if not valid:
                        raise ValueError(f"Review {row['id']} has invalid {key}")
                    row[key] = value
                else:
                    row[key] = None
            rows[row["id"]] = row
        return rows


def review_assets(root, manifest, runs, stage="final"):
    destination = root / ("review" if stage == "final" else "review_planner")
    destination.mkdir(exist_ok=True)
    ordered = sorted(runs, key=lambda row: (row["prompt_id"], row["id"]))
    with (destination / "scores.csv").open("x", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "prompt_id", *RATINGS])
        writer.writerows([row["id"], row["prompt_id"], *[""] * len(RATINGS)] for row in ordered)
    for prompt_id, group in partition(ordered, "prompt_id").items():
        label = manifest["prompts"][prompt_id][1]
        for offset in range(0, len(group), 20):
            page = group[offset:offset + 20]
            columns, cell_w, cell_h = 4, 260, 278
            height = ((len(page) + columns - 1) // columns) * cell_h + 58
            canvas = Image.new("RGB", (columns * cell_w, height), "#202428")
            draw = ImageDraw.Draw(canvas)
            draw.multiline_text((12, 12), "\n".join(textwrap.wrap(f"{prompt_id}. {label}", 130)), fill="white")
            for i, row in enumerate(page):
                left, top = i % columns * cell_w, i // columns * cell_h + 58
                with Image.open(root / row["image"]) as source:
                    thumb = ImageOps.contain(source.convert("RGB"), (cell_w - 12, cell_h - 30))
                canvas.paste(thumb, (left + (cell_w - thumb.width) // 2, top))
                draw.text((left + 9, top + cell_h - 23), row["id"], fill="white")
            canvas.save(destination / f"prompt_{prompt_id:02d}_{offset // 20 + 1:02d}.jpg", quality=91)


def partition(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    return groups


def perceptual_engine(device):
    import lpips
    import torch
    model = lpips.LPIPS(net="alex").eval().to(device)

    @torch.inference_mode()
    def measure(paths):
        tensors = []
        for path in paths:
            with Image.open(path) as src:
                rgb = src.convert("RGB")
                if max(rgb.size) != 256:
                    ratio = 256 / max(rgb.size)
                    rgb = rgb.resize(tuple(max(1, round(x * ratio)) for x in rgb.size), Image.Resampling.LANCZOS)
                values = np.array(rgb, dtype=np.float32)
            tensors.append(torch.from_numpy(values).permute(2, 0, 1) / 127.5 - 1)
        samples = torch.stack(tensors).to(device)
        pairs = list(combinations(range(len(paths)), 2))
        return average(
            value for offset in range(0, len(pairs), 12)
            for value in model(
                samples[[a for a, _ in pairs[offset:offset + 12]]],
                samples[[b for _, b in pairs[offset:offset + 12]]],
            ).flatten().cpu().tolist()
        )

    return measure


def trajectory(rows):
    by_metric = defaultdict(list)
    for row in rows:
        seen = set()
        for item in (row["trace"] or {}).get("points", []):
            if item["kind"] != "trajectory":
                continue
            sigma = round(item["sigma"], 5)
            if sigma in seen:
                continue
            seen.add(sigma)
            for key in ("attention", "conditional_x0"):
                values = np.asarray(item[key][0], dtype=np.float64)
                if np.isfinite(values).all():
                    by_metric[(sigma, key)].append(values)
    return {
        f"{key}@{sigma:g}": average(cosine_distance(a, b) for a, b in combinations(items, 2))
        for (sigma, key), items in by_metric.items() if len(items) >= 2
    }


def metrics(root, rows, ratings, perceptual, keep_traces):
    paths = [root / row["image"] for row in rows]
    features = [describe(path) for path in paths]
    result = {
        "layout_edge_distance": average(cosine_distance(a, b) for a, b in combinations(features, 2)),
        "wall_seconds": average(row["wall_seconds"] for row in rows),
        "execution_seconds": average(row["execution_seconds"] for row in rows),
        "sampled_peak_device_mib": average(row["sampled_peak_device_mib"] for row in rows),
        "total_dit_calls": average(sum(trace[k] for k in (
            "native_dit_calls", "feature_reference_dit_calls", "initialization_dit_calls",
        )) for row in rows if (trace := row["trace"])),
    }
    result["lpips"] = perceptual(paths) if perceptual is not None else None
    review = [ratings.get(row["id"], {}) for row in rows]
    for group in GROUPS:
        names = [score.get(group) for score in review]
        metric = group.removesuffix("_group")
        result[f"human_{metric}_distance"] = (
            average(float(a != b) for a, b in combinations(names, 2)) if all(names) else None
        )
        result[f"human_{metric}_unique_fraction"] = len(set(names)) / len(names) if all(names) else None
    initializations = [point for row in rows for point in (row["trace"] or {}).get("points", [])
                       if point["kind"] == "initialization"]
    result["initialization_acceptance"] = average(ok for run in initializations for ok in run["accepted"])
    result["initialization_proposals"] = average(len(run.get("attempts", [])) for run in initializations)
    def same_pixels(before, after):
        with Image.open(before) as a, Image.open(after) as b:
            return np.array_equal(np.asarray(a), np.asarray(b))
    result["repeat_pixel_match"] = average(
        same_pixels(root / row["image"], root / run["image"])
        for row in rows for run in row.get("repeats", []) if run["sampled"]
    )
    for route in ("actual", "forecast"):
        result[f"spectrum_{route}"] = average(
            sum(run[route] for run in row.get("spectrum", []))
            for row in rows if row.get("spectrum")
        )
    for key in RATINGS[len(GROUPS):]:
        values = [score.get(key) for score in review]
        result[key] = average(values) if all(value is not None for value in values) else None
    if keep_traces:
        result.update(trajectory(rows))
    return result


def forecast_route_signature(run):
    signatures = []
    for report in run.get("spectrum", []):
        decisions = report.get("denoiser_decisions")
        if not decisions:
            return None
        signatures.append((report["stage"], tuple(
            (d["step"], d["sigma"], d["route"], tuple(d["branches"]))
            for d in decisions if d["route"] != "forecast_rejected"
        )))
    return tuple(signatures) if signatures else None


def defect_transitions(rows, baseline_rows, ratings, key):
    reference = {row["seed"]: ratings.get(row["id"], {}).get(key) for row in baseline_rows}
    counts = dict(recovered=0, introduced=0, persistent=0, both_clear=0)
    for row in rows:
        before = reference.get(row["seed"])
        after = ratings.get(row["id"], {}).get(key)
        if before not in (0, 1) or after not in (0, 1):
            continue
        label = {(1, 0): "recovered", (0, 1): "introduced",
                 (1, 1): "persistent", (0, 0): "both_clear"}[before, after]
        counts[label] += 1
    return counts


def analyze(root, runs, baseline, ratings, perceptual, keep_traces):
    summaries = {}
    methods = partition(runs, "method")
    baseline_prompts = partition(methods[baseline], "prompt_id")
    baseline_cache = {}
    for method, group in methods.items():
        compared, n_seeds = {}, 0
        defects = {key: dict(recovered=0, introduced=0, persistent=0, both_clear=0) for key in DEFECT_RATINGS}
        for prompt_id, rows in partition(group, "prompt_id").items():
            native = baseline_prompts.get(prompt_id, [])
            matched = {r["seed"] for r in native} & {r["seed"] for r in rows}
            if len(matched) < 2:
                continue
            rows = [r for r in rows if r["seed"] in matched]
            native = [r for r in native if r["seed"] in matched]
            for key in DEFECT_RATINGS:
                for label, count in defect_transitions(rows, native, ratings, key).items():
                    defects[key][label] += count
            data = metrics(root, rows, ratings, perceptual, keep_traces)
            reference_routes = {row["seed"]: forecast_route_signature(row) for row in native}
            routes = [(forecast_route_signature(row), reference_routes[row["seed"]]) for row in rows]
            route_match = average(a == b for a, b in routes if a is not None and b is not None)
            compared.setdefault("matched_spectrum_route_fraction", []).append((route_match, 1.0))
            cache_key = (prompt_id, frozenset(matched))
            if method == baseline:
                baseline_cache[cache_key] = data
            if cache_key not in baseline_cache:
                baseline_cache[cache_key] = metrics(root, native, ratings, perceptual, keep_traces)
            ref = baseline_cache[cache_key]
            for key, value in data.items():
                compared.setdefault(key, []).append((value, ref.get(key)))
            n_seeds += len(matched)
        summaries[method] = {"matched_seed_pairs": n_seeds, "metrics": {}, "defect_transitions": defects}
        for key, records in compared.items():
            deltas = [a - b for a, b in records if a is not None and b is not None]
            summaries[method]["metrics"][key] = {
                "mean": average(a for a, _ in records),
                "mean_paired_delta": average(deltas),
                "paired_prompt_bootstrap_95": bootstrap(deltas),
                "matched_prompts": len(deltas),
            }
    return summaries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--baseline", default="native")
    p.add_argument("--review", action="store_true", help="Generate blinded contact sheets and a blank scoring CSV.")
    p.add_argument("--ratings", type=Path, help="A filled review CSV.")
    p.add_argument("--lpips-device", help="Optional LPIPS AlexNet calculation on the given torch device (cpu or cuda).")
    p.add_argument("--trajectory", action="store_true", help="Compare pooled features and conditional x0 by actual sigma.")
    p.add_argument("--stage", choices=("final", "planner"), default="final")
    args = p.parse_args()
    manifest = json.loads((args.output / "manifest.json").read_text())
    runs = [json.loads(line) for line in (args.output / "runs.jsonl").read_text().splitlines()]
    if args.stage == "planner":
        if not runs or not all("planner_image" in row for row in runs):
            p.error("Planner analysis requires a Scale-Locked panel with exported planner images.")
        runs = [{**row, "image": row["planner_image"], "repeats": []} for row in runs]
    if args.baseline not in {row["method"] for row in runs}:
        p.error("The baseline is absent from runs.jsonl.")
    if args.review:
        review_assets(args.output, manifest, runs, args.stage)
    ratings = load_ratings(args.ratings)
    engine = perceptual_engine(args.lpips_device) if args.lpips_device else None
    report = analyze(args.output, runs, args.baseline, ratings, engine, args.trajectory)
    filename = "summary.json" if args.stage == "final" else "summary_planner.json"
    (args.output / filename).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
