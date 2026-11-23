"""Generate the predeclared planner PNG comparisons as ComfyUI workflows."""

import argparse
import copy
import hashlib
import json
import random
import secrets
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("native", "original_feature", "weak_gaussian", "higher_gaussian", "token_adaptive")


def attach_identity(workflow, guidance, lora):
    source = next(link for link in workflow["links"] if link[0] == guidance["inputs"][0]["link"])
    ident = workflow["last_node_id"] + 1
    link = workflow["last_link_id"] + 1
    source[3:5] = [ident, 0]
    guidance["inputs"][0]["link"] = link
    workflow["nodes"].append({
        "id": ident, "type": "LoraLoaderModelOnly", "pos": [425, 480], "size": [315, 110],
        "flags": {}, "order": 4, "mode": 0, "properties": {"Node name for S&R": "LoraLoaderModelOnly"},
        "inputs": [{"name": "model", "type": "MODEL", "link": source[0]}],
        "outputs": [{"name": "MODEL", "type": "MODEL", "links": [link], "slot_index": 0}],
        "widgets_values": [lora, 1.0],
    })
    workflow["links"].append([link, ident, 0, guidance["id"], 0, "MODEL"])
    workflow.update(last_node_id=ident, last_link_id=link)


def build(directory, split, arms, blind_key, identity=None):
    data = json.loads(Path(__file__).with_name("projection_guidance_prompts.json").read_text())
    variants = {}
    for arm in ("weak_gaussian", "higher_gaussian", "token_adaptive"):
        variants[arm] = json.loads((ROOT / "examples" / f"projection_guidance_{arm}.json").read_text())
    legacy = json.loads((ROOT / "examples" / "causal_directions.json").read_text())
    legacy_settings, = (node["widgets_values"] for node in legacy["nodes"]
                        if node["type"] == "ModeExplorerKrea2" and not node["widgets_values"][23]
                        and node["widgets_values"][-2] == "repel")
    directory.mkdir(parents=True, exist_ok=False)
    trial_seed = int.from_bytes(hashlib.sha256(f"{blind_key}:{split}".encode()).digest()[:8], "little")
    shuffle = random.Random(trial_seed)
    allocation = []
    # Separate generations save to separate folders, so a later allocation can
    # never relabel an earlier image. The ID carries no arm or key information.
    generation = secrets.token_hex(6)
    cases = data[split]
    if identity:
        trigger, lora = identity
        profile = data["optional_identity"]
        cases = cases + [{"id": "identity", "seed": profile[f"{split}_seed"],
                          "prompt": profile[f"{split}_prompt"].format(identity=trigger)}]
    for case in cases:
        order = list(arms)
        shuffle.shuffle(order)
        for index, arm in enumerate(order, 1):
            code = f"{case['id']}_{index:02d}"
            workflow = copy.deepcopy(variants.get(arm, variants["weak_gaussian"]))
            nodes = {n["type"]: n for n in workflow["nodes"]}
            nodes["CLIPTextEncode"]["widgets_values"][0] = case["prompt"]
            sampler = nodes["Krea2TurboFlowAlignedRegenerate"]
            sampler["widgets_values"][:2] = [case["seed"], "fixed"]
            nodes["SaveImage"]["widgets_values"][0] = f"modeexplorer/benchmark/{split}/{generation}/{code}"
            guidance = nodes["ModeExplorerProjectionGuidanceKrea2"]
            if arm == "native":
                guidance["widgets_values"][0] = False
            elif arm == "original_feature":
                guidance["type"] = "ModeExplorerKrea2"
                guidance["properties"] = {"Node name for S&R": "ModeExplorerKrea2"}
                guidance["title"] = "ModeExplorer Feature Guidance"
                guidance["widgets_values"] = copy.deepcopy(legacy_settings)
            if case["id"] == "identity":
                attach_identity(workflow, guidance, lora)
            (directory / f"{code}.json").write_text(json.dumps(workflow, indent=2) + "\n")
            allocation.append({"generation": generation, "code": code, "case": case["id"],
                               "seed": case["seed"], "arm": arm})
    (directory / "private_allocation.json").write_text(json.dumps(allocation, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="New directory for the workflows and private allocation key")
    parser.add_argument("--split", choices=("screening", "confirmation"), default="screening")
    parser.add_argument("--arms", choices=ARMS, nargs="+", default=ARMS)
    parser.add_argument("--blind-key", required=True, help="Use a private fixed string to reproduce masked arm labels")
    parser.add_argument("--identity", nargs=2, metavar=("TRIGGER", "LORA_FILENAME"), help="Include the optional identity case")
    args = parser.parse_args()
    if len(set(args.arms)) != len(args.arms):
        parser.error("each arm may be included once")
    build(args.directory, args.split, args.arms, args.blind_key, args.identity)
