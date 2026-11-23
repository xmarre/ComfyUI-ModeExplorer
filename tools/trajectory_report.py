"""Compare exported planner trajectories and build a separate VAE decode workflow."""

import argparse
import json
from pathlib import Path


def load(path):
    path = Path(path)
    data = json.loads(path.read_text())
    if data.get("schema") != 1 or not data.get("states"):
        raise ValueError(f"{path} has no trajectory states")
    return path.parent, data


def aligned(first, second):
    def key(state):
        return state["solver_step"], state["opportunity"], state["sigma"]

    left = {key(state): state for state in first["states"]}
    right = {key(state): state for state in second["states"]}
    return [(left[k], right[k]) for k in sorted(left.keys() & right.keys())], sorted(left.keys() ^ right.keys())


def compare(left, right, regions=()):
    from safetensors import safe_open
    import torch
    from mode_explorer.mechanism import paired_measurement

    left_dir, first = load(left)
    right_dir, second = load(right)
    controls = ("seed", "cfg", "stage", "solver", "solver_options", "schedule", "initial_noise_sha256",
                "packed_conditioning")
    differences = [key for key in controls if first[key] != second[key]]
    pairs, missing = aligned(first, second)

    def values(directory, state):
        with safe_open(directory / f'{state["ordinal"]:03}.safetensors', framework="pt", device="cpu") as store:
            return {key: store.get_tensor(key) for key in ("frozen_native_input", "baseline_clean")}

    def region_metrics(field1, field2):
        measured = {}
        for name, y0, y1, x0, x1 in regions:
            h, w = field1.shape[-2:]
            if not (0 <= y0 < y1 <= h and 0 <= x0 < x1 <= w):
                raise ValueError(f"Region {name} lies outside the shared latent grid.")
            measured[name] = paired_measurement(field1[:, y0:y1, x0:x1], field2[:, y0:y1, x0:x1],
                                                low=.0625, high=.125, patch_size=2)
        return measured

    rows = []
    for a, b in pairs:
        v1, v2 = values(left_dir, a), values(right_dir, b)
        field1, field2 = (item["baseline_clean"][0].reshape(-1, *item["baseline_clean"].shape[-2:])
                          for item in (v1, v2))
        rows.append({
            "sigma": a["sigma"], "solver_step": a["solver_step"], "opportunity": a["opportunity"],
            "left": a["export_id"], "right": b["export_id"],
            "identical_noisy_input": torch.equal(v1["frozen_native_input"], v2["frozen_native_input"]),
            "identical_packed_condition": all(a[k] == b[k] for k in
                                              ("conditioning_sha256", "attention_mask_sha256", "reference_latent_sha256")),
            "clean": paired_measurement(field1, field2, low=.0625, high=.125, patch_size=2),
            "registered_latent_regions": region_metrics(field1, field2),
        })
    routing = [(a["solver_step"], a["route"]) for a in first["routes"]] == [
        (b["solver_step"], b["route"]) for b in second["routes"]]
    captures = {side: {"status": data.get("status"), "reason": data.get("reason")}
                for side, data in (("left", first), ("right", second))}
    # Interrupted or failed captures can agree on a shared prefix. Keep those
    # measurements, but never present them as a completed trajectory comparison.
    scope = ("completed_trajectories" if all(c["status"] == "completed" for c in captures.values())
             else "partial_capture_prefix")
    return {"comparison_scope": scope, "captures": captures,
            "input_provenance_differences": differences, "same_actual_forecast_route": routing,
            "unmatched_actual_keys": missing, "pairs": rows}


def decode_workflow(vae, items):
    """Return a LiteGraph workflow with a single shared VAE and no model/sampler."""
    def node(i, kind, xy, inputs, outputs, widgets):
        return {"id": i, "type": kind, "pos": xy, "size": [335, 120], "flags": {}, "order": i, "mode": 0,
                "inputs": inputs, "outputs": outputs, "properties": {"Node name for S&R": kind},
                "widgets_values": widgets}

    def port(name, kind, link):
        return {"name": name, "type": kind, "link": link}

    def output(name, kind, links, slot=0):
        return {"name": name, "type": kind, "links": links, "slot_index": slot}

    vae_links = []
    nodes = [node(1, "VAELoader", [0, 0], [], [output("VAE", "VAE", vae_links)], [vae])]
    links = []
    for index, (export, tensor, prefix) in enumerate(items):
        loader, decoder, saver = range(2 + 3 * index, 5 + 3 * index)
        latent, vae_link, image = range(1 + 3 * index, 4 + 3 * index)
        vae_links.append(vae_link)
        y = index * 165
        nodes.extend((
            node(loader, "ModeExplorerLoadResponseKrea2", [370, y], [],
                 [output("predicted_clean", "LATENT", [latent]), output("matched_sigma_schedule", "SIGMAS", [], 1)],
                 [export, tensor]),
            node(decoder, "VAEDecode", [750, y], [port("samples", "LATENT", latent), port("vae", "VAE", vae_link)],
                 [output("IMAGE", "IMAGE", [image])], []),
            node(saver, "SaveImage", [1130, y], [port("images", "IMAGE", image)], [], [prefix]),
        ))
        links.extend(((latent, loader, 0, decoder, 0, "LATENT"),
                      (vae_link, 1, 0, decoder, 1, "VAE"),
                      (image, decoder, 0, saver, 0, "IMAGE")))
    return {"last_node_id": len(nodes), "last_link_id": len(links), "nodes": nodes, "links": links,
            "groups": [], "config": {}, "extra": {}, "version": .4}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True, dest="command")
    cmp = commands.add_parser("compare", help="Align by exact sigma and solver opportunity, then measure latent bands.")
    cmp.add_argument("native", type=Path)
    cmp.add_argument("comparison", type=Path)
    cmp.add_argument("output", type=Path)
    cmp.add_argument("--roi", nargs=5, action="append", metavar=("NAME", "Y0", "Y1", "X0", "X1"),
                     help="Measure a visually registered shared ROI in latent-grid coordinates.")
    dec = commands.add_parser("decode", help="Build a model-free ComfyUI VAE decode workflow.")
    dec.add_argument("--vae", required=True)
    dec.add_argument("--output", type=Path, required=True)
    dec.add_argument("manifests", nargs="+", type=Path)
    args = parser.parse_args()
    if args.command == "compare":
        regions = [(name, *map(int, bounds)) for name, *bounds in args.roi or ()]
        payload = compare(args.native, args.comparison, regions)
    else:
        items = []
        for source in args.manifests:
            raw = json.loads(source.read_text())
            if "states" in raw:
                _, run = load(source)
                items.extend((s["export_id"], "baseline_clean",
                              f'modeexplorer_discovery/decoded/{run["label"]}/step_{s["solver_step"]:02d}')
                             for s in run["states"])
            else:
                factorial = raw["mechanistic"]["projection"].get("factorial")
                if not factorial or raw["mechanistic"]["projection"].get("status") != "qualified":
                    raise ValueError(f"{source} has no qualified factorial")
                items.extend((source.parent.name, prediction, f"modeexplorer_discovery/decoded/{source.parent.name}/{arm}")
                             for arm, prediction in zip("ABCD", ("baseline_clean", "D01_clean",
                                                                 "continuous_native_clean", "continuous_feature_clean")))
        payload = decode_workflow(args.vae, items)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
