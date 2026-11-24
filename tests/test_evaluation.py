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
    # LoRA names must not replace the requested output identifier.
    assert sham["10"]["inputs"]["filename_prefix"] == "modeexplorer_eval/test"
    assert sham["15"]["inputs"]["filename_prefix"] == "modeexplorer_eval/test_planner"
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


def test_paired_fingerprints_and_edit_measurements():
    def row(seed, value):
        return {"seed": seed, "trace": {"points": [{
            "kind": "trajectory", "sigma": 0.99, "attention": [[1, 1]],
            "conditional_x0": [[value, 0]], "nearest_cosine": [0.6], "nearest_cosine_after": [0.5],
            "applied_pooled_relative_rms": [0.01], "proposal_direction_cosine": [1.0],
        }]}}
    native = [row(1, 2), row(2, 4)]
    guided = [row(2, 6), row(1, 3)]
    assert analyze.paired_trajectory(guided, native)["paired_x0_fingerprint_relative_l2@0.99"] == 0.5
    summary = analyze.trajectory(guided)
    assert summary["applied_pooled_relative_rms@0.99"] == 0.01
    assert abs(summary["nearest_cosine_change@0.99"] + 0.1) < 1e-8
    guided[0]["trace"]["points"].append(guided[0]["trace"]["points"][0])
    guided[1]["trace"]["points"] = []
    assert not analyze.paired_trajectory(guided, native)


def test_defect_transition_counts_report_missing_separately():
    native = [{"seed": i, "id": f"n{i}"} for i in range(5)]
    guided = [{"seed": i, "id": f"g{i}"} for i in range(5)]
    ratings = {f"n{i}": {"anatomy_defect": v} for i, v in enumerate([1, 0, 1, 0, 1])}
    ratings.update({f"g{i}": {"anatomy_defect": v} for i, v in enumerate([0, 1, 1, 0, None])})
    assert analyze.defect_transitions(guided, native, ratings, "anatomy_defect") == {
        "recovered": 1, "introduced": 1, "persistent": 1, "both_clear": 1, "missing": 1,
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


def test_response_profile_and_offline_embeddings(tmp_path):
    from types import SimpleNamespace

    profiles = evaluate.load_profiles(Path(evaluate.__file__).with_name("response_profiles.json"))
    args = SimpleNamespace(model="krea2", encoder="enc", vae="vae", weight_dtype="default",
                           model_lora=None, identity_lora=None, model_lora_strength=1.0, lora_strength=1.0,
                           negative="", width=1024, height=768,
                           steps=8, cfg=1, sampler="euler", scheduler="simple", diagnostics=False,
                           variants=profiles)
    graph = evaluate.graph(args, "latent_steer", "prompt", 91, name="response")
    assert graph["16"]["class_type"] == "ModeExplorerResponseKrea2"
    assert graph["16"]["inputs"]["mode"] == "latent_steer"
    assert graph["8"]["inputs"]["model"] == ["16", 0]
    path = tmp_path / "embeddings.json"
    path.write_text('{"dino_v2":{"a":[1,0],"b":[0,1]}}')
    result = analyze.embedding_metrics([{"id": "a"}, {"id": "b"}], analyze.load_embeddings(path))
    assert result == {"dino_v2_cosine_distance": 1.0, "dino_v2_vendi": 2.0}


def test_rediverse_panels_build_matched_graphs():
    import pytest

    args = Namespace(
        model="turbo", encoder="encoder", vae="vae", weight_dtype="default", width=1024, height=1024,
        steps=8, cfg=1.0, sampler="euler", scheduler="simple", negative="", strength=0.45,
        fidelity=0.55, fixed=False, similarity=0.7, reference_budget=2, min_sigma=0.82, diagnostics=True,
        model_lora=None, model_lora_strength=1.0, identity_lora=None, lora_strength=1.0,
        teacher_model="raw", rediverse_adapter="krea-2-rediverse-comfy.safetensors",
    )
    root = Path(evaluate.__file__)
    args.variants = evaluate.load_profiles(root.with_name("rediverse_reference_profiles.json"))
    arms = {arm: evaluate.graph(args, arm, "prompt", 42, name=arm) for arm in args.variants}
    for arm, g in arms.items():
        sampler = g["8"]
        assert sampler["class_type"] == "SamplerCustom" and sampler["inputs"]["noise_seed"] == 42
        assert g["22"]["inputs"]["model"] == ("raw" if arm == "G_raw" else "turbo")
        assert g["21"]["inputs"]["sampler_name"] == "euler"
    assert arms["G_raw"]["1"]["inputs"]["unet_name"] == "raw" and arms["G_raw"]["22"]["inputs"]["steps"] == 52
    assert arms["G_raw"]["8"]["inputs"]["cfg"] == 4.5
    ft = arms["C_turbo_ft"]["18"]["inputs"]
    assert ft["adapter"] == "krea-2-rediverse-comfy.safetensors" and ft["strength"] == 1.0 and "directions" not in ft
    assert arms["C_turbo_ft"]["7"]["inputs"]["model"] == ["18", 0]  # ReDiverse precedes ModeExplorer
    combined = arms["D_turbo_ft_b1"]
    assert combined["7"]["inputs"]["mode"] == "Early Structural (B1)" and combined["7"]["inputs"]["model"] == ["18", 0]
    tf = arms["E_turbo_tf"]
    assert tf["18"]["inputs"]["adapter"] == "none" and tf["18"]["inputs"]["directions"] == ["20", 0]
    assert tf["20"]["inputs"]["teacher"] == ["19", 0] and tf["19"]["inputs"]["unet_name"] == "raw"
    assert tf["20"]["inputs"]["positive"] == ["4", 0] and tf["20"]["inputs"]["rollouts"] == 25
    assert arms["F_turbo_ft_tf"]["18"]["inputs"]["band"] == 4.0
    assert "18" not in arms["A_turbo"] and "18" not in arms["B_turbo_b1"]
    # Identity and model LoRAs reach the calibration teacher exactly as they reach the student.
    args.model_lora, args.identity_lora, args.lora_strength = "style", "person", 0.8
    identity = evaluate.graph(args, "F_turbo_ft_tf", "prompt", 42, name="identity")
    assert identity["23"]["inputs"] == {"model": ["19", 0], "lora_name": "style", "strength_model": 1.0}
    assert identity["24"]["inputs"] == {"model": ["23", 0], "lora_name": "person", "strength_model": 0.8}
    assert identity["20"]["inputs"]["teacher"] == ["24", 0] and identity["12"]["inputs"]["lora_name"] == "person"
    args.model_lora, args.identity_lora = None, None
    args.teacher_model = None
    with pytest.raises(ValueError, match="teacher-model"):
        evaluate.graph(args, "G_raw", "prompt", 42, name="raw")

    args.variants = evaluate.load_profiles(root.with_name("rediverse_production_profiles.json"))
    args.planner_megapixels, args.composition_lock_sigma, args.sampler = 0.55, 0.60, "er_sde"
    sham = evaluate.graph(args, "C_route_sham", "prompt", 7, name="sham")
    assert sham["18"]["inputs"]["strength"] == 0.0 and sham["13"]["inputs"]["model"] == ["7", 0]
    assert sham["8"]["class_type"] == "Krea2TurboFlowAlignedRegenerate"
    assert all(cfg.get("spectrum") == "guarded" for cfg in args.variants.values())
    args.variants = {"reference": {"method": "native", "schedule": "reference_turbo"}}
    with pytest.raises(ValueError, match="Scale-Locked planner"):
        evaluate.graph(args, "reference", "prompt", 1, name="reference")


def test_rediverse_panels_record_effective_sampler_configuration(tmp_path, monkeypatch):
    """The main recording path for SamplerCustom (reference) and planner graphs, with mocked rendering."""
    import sys

    rendered = []

    def fake_render(args, workflow, seed, destination):
        rendered.append(workflow)
        destination.write_bytes(b"png")
        return {"job_id": "x", "wall_seconds": 1.0, "execution_seconds": 1.0, "sampled_peak_device_mib": None,
                "trace": None, "spectrum": [], "rediverse_calibration": []}

    monkeypatch.setattr(evaluate, "render", fake_render)
    root = Path(evaluate.__file__)
    common = ["evaluate.py", "--model", "turbo", "--teacher-model", "raw", "--rediverse-adapter", "adapter", "--smoke"]
    monkeypatch.setattr(sys, "argv", common + ["--profiles", str(root.with_name("rediverse_reference_profiles.json")),
                                               "--out", str(tmp_path / "reference")])
    evaluate.main()
    records = [json.loads(line) for line in (tmp_path / "reference" / "runs.jsonl").read_text().splitlines()]
    assert len(records) == 2 * 2 * 7
    for record in records:
        raw = record["method"] == "G_raw"
        assert (record["steps"], record["cfg"]) == ((52, 4.5) if raw else (8, 1.0))
        assert record["scheduler"] == ("reference_raw" if raw else "reference_turbo") and record["sampler"] == "euler"
        assert record["checkpoint"] == ("raw" if raw else "turbo")
    monkeypatch.setattr(sys, "argv", common + ["--profiles", str(root.with_name("rediverse_production_profiles.json")),
                                               "--out", str(tmp_path / "production"), "--sampler", "er_sde",
                                               "--steps", "13", "--width", "2048", "--height", "2048",
                                               "--planner-megapixels", "1.0"])
    evaluate.main()
    records = [json.loads(line) for line in (tmp_path / "production" / "runs.jsonl").read_text().splitlines()]
    assert len(records) == 2 * 2 * 5
    assert all((r["steps"], r["sampler"], r["scheduler"]) == (13, "er_sde", "simple") for r in records)
    assert all(r["planner_image"].endswith("_planner.png") for r in records)
