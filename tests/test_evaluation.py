from argparse import Namespace
import csv
import io
import json

from pathlib import Path

from PIL import Image

from tools import analyze, evaluate


def test_graph_native_and_sequential_mode():
    args = Namespace(
        model="model", encoder="encoder", vae="vae", weight_dtype="default", width=1024, height=768,
        steps=8, cfg=1.0, sampler="euler", scheduler="simple", negative="", strength=0.45,
        fidelity=0.55, fixed=False, similarity=0.7, reference_budget=2, min_sigma=0.82, diagnostics=False,
        model_lora="sda", model_lora_strength=0.9, identity_lora="identity", lora_strength=0.8,
    )
    native = evaluate.graph(args, "native", "prompt", 42, name="native")
    guided = evaluate.graph(args, "combined", "prompt", 42, name="guided")
    assert "7" not in native
    assert native["8"]["inputs"]["model"] == guided["7"]["inputs"]["model"] == ["12", 0]
    assert guided["6"]["inputs"]["batch_size"] == 1
    assert guided["7"]["inputs"]["operation"] == "sequential"
    assert guided["7"]["inputs"]["initialization"]
    assert guided["8"]["inputs"]["seed"] == native["8"]["inputs"]["seed"] == 42

    args.variants = evaluate.load_profiles(Path(evaluate.__file__).with_name("spectrum_profiles.json"))
    args.planner_megapixels = 0.55
    args.composition_lock_sigma = 0.60
    args.diagnostics = True
    sham = evaluate.graph(args, "cosine_full_sham", "prompt", 42, name="test")
    assert sham["7"]["inputs"]["sham"] and sham["7"]["inputs"]["full_spectrum"]
    assert sham["13"]["inputs"]["model"] == ["7", 0]
    assert sham["8"]["inputs"]["model"] == ["13", 0]
    assert sham["8"]["class_type"] == "Krea2TurboFlowAlignedRegenerate"
    assert sham["14"]["inputs"]["samples"] == ["8", 1]
    controlled = evaluate.graph(args, "native", "prompt", 42, name="native")
    assert controlled["13"]["inputs"]["window"] == 0
    assert controlled["13"]["inputs"]["sa_pece_forecast_policy"] == "off"


def test_low_strength_profiles_match_guided_and_sham():
    args = Namespace(
        model="model", encoder="encoder", vae="vae", weight_dtype="default", width=1024, height=768,
        steps=13, cfg=1.0, sampler="euler", scheduler="simple", negative="", strength=0.45,
        fidelity=0.55, fixed=False, similarity=0.70, reference_budget=4, min_sigma=0.90, diagnostics=True,
        model_lora=None, model_lora_strength=1.0, identity_lora=None, lora_strength=1.0,
    )
    args.variants = evaluate.load_profiles(Path(evaluate.__file__).with_name("low_strength_profiles.json"))
    assert set(args.variants) == {"native", "low_strength", "low_strength_sham"}
    native, guided, sham = (
        evaluate.graph(args, arm, "prompt", 42, name="matched")
        for arm in ("native", "low_strength", "low_strength_sham")
    )
    active = guided["7"]["inputs"]
    assert active["feature"] == "feature_cosine" and not active["initialization"]
    assert active["strength"] == 0.03 and active["fidelity"] == 0.70
    assert active["min_sigma"] == 0.82 and not active["adaptive"]
    assert active["reference_budget"] == 2 and active["reference_trajectories"] == 1
    assert active["feature_block"] == 1 and active["feature_grid"] == "4"
    assert active["reference_bank_size"] == 8 and active["reference_source"] == "bank"
    assert not active["full_spectrum"] and not active["sham"]
    assert sham["7"]["inputs"] == {**active, "sham": True}
    assert native["7"]["inputs"]["feature"] == "native" and not native["7"]["inputs"]["initialization"]
    assert guided["13"] == sham["13"] == native["13"]
    assert guided["13"]["inputs"]["extrapolation_guard_scope"] == "planner_only"
    assert guided["8"] == sham["8"] == native["8"]
    assert {k: v for k, v in guided.items() if k != "7"} == {
        k: v for k, v in sham.items() if k != "7"
    }


def test_causal_profiles_change_only_direction():
    profiles = evaluate.load_profiles(Path(evaluate.__file__).with_name("causal_direction_profiles.json"))
    assert set(profiles) == {"native", "sham", "repel", "reverse", "channel_sign_0", "channel_sign_1", "channel_sign_2"}
    baseline = profiles["repel"]["settings"]
    assert baseline["strength"] == 0.025 and baseline["feature_block"] == 5
    assert profiles["sham"]["settings"] == {**baseline, "sham": True}
    assert profiles["reverse"]["settings"] == {**baseline, "feature_direction": "reverse"}
    for trial in range(3):
        assert profiles[f"channel_sign_{trial}"]["settings"] == {
            **baseline, "feature_direction": "channel_sign", "direction_trial": trial,
        }
    assert all(profile["spectrum"] == "all_actual" for profile in profiles.values())


def test_render_reads_native_api_history_image_and_trace(tmp_path, monkeypatch):
    log = tmp_path / "comfy.log"
    log.write_text("previous unrelated log\n")
    png = io.BytesIO()
    Image.new("RGB", (32, 32)).save(png, format="PNG")
    args = Namespace(server="http://example", server_log=log, timeout=5, gpu=None)
    job = "synthetic-job"

    def response(base, endpoint, payload=None):
        assert base == args.server
        if endpoint == "/prompt":
            assert payload == {"prompt": {"test": 5}}
            with log.open("a") as stream:
                stream.write("INFO MODEEXPLORER_TRACE {\"seed\": 42, \"completed\": true}\n")
            return json.dumps({"prompt_id": job})
        if endpoint == f"/history/{job}":
            return json.dumps({job: {
                "status": {"status_str": "success", "messages": [
                    ["execution_start", {"timestamp": 1000}], ["execution_success", {"timestamp": 1250}],
                ]},
                "outputs": {"10": {"images": [{"filename": "result.png", "subfolder": "a", "type": "output"}]}},
            }})
        assert endpoint == "/view?filename=result.png&subfolder=a&type=output"
        return png.getvalue()

    monkeypatch.setattr(evaluate, "request", response)
    destination = tmp_path / "job.png"
    record = evaluate.render(args, {"test": 5}, 42, destination)
    assert destination.read_bytes() == png.getvalue()
    assert record["job_id"] == job and record["execution_seconds"] == 0.25
    assert record["trace"] == {"seed": 42, "completed": True}
    assert record["sampled_peak_device_mib"] is None


def test_defect_transition_counts_exclude_missing_scores():
    native = [{"seed": i, "id": f"n{i}"} for i in range(5)]
    guided = [{"seed": i, "id": f"g{i}"} for i in range(5)]
    ratings = {f"n{i}": {"anatomy_defect": v} for i, v in enumerate([1, 0, 1, 0, 1])}
    ratings.update({f"g{i}": {"anatomy_defect": v} for i, v in enumerate([0, 1, 1, 0, None])})
    assert analyze.defect_transitions(guided, native, ratings, "anatomy_defect") == {
        "recovered": 1, "introduced": 1, "persistent": 1, "both_clear": 1,
    }


def test_paired_analysis_and_blinded_assets(tmp_path):
    rows = []
    ratings = {}
    for prompt in range(2):
        for method in ("native", "cosine"):
            for seed in range(2):
                identifier = f"{prompt}_{method}_{seed}"
                path = f"{identifier}.png"
                canvas = Image.new("RGB", (64, 64), "black")
                x = 8 if method == "native" or seed == 0 else 34
                canvas.paste("white", (x, 10, x + 10, 24))
                canvas.save(tmp_path / path)
                rows.append(dict(
                    method=method, prompt_id=prompt, id=identifier, seed=seed, image=path,
                    trace=None, wall_seconds=1, execution_seconds=0.8, sampled_peak_device_mib=None,
                ))
                ratings[identifier] = dict(composition_group=str(seed if method == "cosine" else 0), quality=5)
    manifest = {"prompts": [["open", "First prompt"], ["scene", "Second prompt"]]}
    analyze.review_assets(tmp_path, manifest, rows)
    with (tmp_path / "review" / "scores.csv").open() as file:
        blinded = list(csv.DictReader(file))
    assert len(blinded) == 8 and "method" not in blinded[0]
    assert (tmp_path / "review" / "prompt_00_01.jpg").is_file()
    summary = analyze.analyze(tmp_path, rows, "native", ratings, None, False)
    metrics = summary["cosine"]["metrics"]
    assert metrics["layout_edge_distance"]["mean_paired_delta"] > 0
    assert metrics["human_composition_distance"]["paired_prompt_bootstrap_95"] == [1.0, 1.0]
    assert metrics["quality"]["mean_paired_delta"] == 0
