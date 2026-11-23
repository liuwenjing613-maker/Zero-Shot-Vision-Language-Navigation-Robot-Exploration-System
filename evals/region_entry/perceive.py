"""Run the existing scene perceptor on the exact selected MP3D frame."""
import argparse
import hashlib
import json
from pathlib import Path
from ollama_transport import OllamaPerceptor


def capture(folder, i, endpoint, model):
    rgb = folder / f"view-{i}.png"
    goal = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "results/mp3d-three-goals-dining/dining-three-goals.json"
        ).read_text()
    )[0]["goal"]
    result = OllamaPerceptor(endpoint, model, timeout=600).perceive_scene(
        str(rgb),
        goal,
        task_id="region-entry",
        observation_id=f"approach-{i}",
        view_id="front",
    )
    record = dict(
        rgb_sha256=hashlib.sha256(rgb.read_bytes()).hexdigest(), result=result
    )
    (folder / f"qwen-{i}.json").write_text(json.dumps(record, indent=2))
    print(i, result["visual_cues"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("folder", type=Path)
    p.add_argument("indices", type=int, nargs="+")
    p.add_argument("--endpoint", default="http://127.0.0.1:11434/v1")
    p.add_argument("--model", default="qwen3-vl:8b")
    args = p.parse_args()
    for i in args.indices:
        capture(args.folder, i, args.endpoint, args.model)
