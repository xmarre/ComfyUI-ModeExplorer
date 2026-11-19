from argparse import Namespace
import csv
import io
import json

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
