"""First MP3D three-category check: two goals, one fixed RGB view."""
import argparse
import hashlib
import json
from pathlib import Path

from vln_perception import VLMClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    rgb = root / "results/mp3d-2azQ1b91cZZ-chair-challenges-v1/multiple.png"
    image_hash = hashlib.sha256(rgb.read_bytes()).hexdigest()
    client = VLMClient(args.base_url, "qwen3.8:27b", timeout=240)
    with args.output.open("x", encoding="utf-8") as out:
        for category in ("chair", "table"):
            goal = client.parse_goal(f"Find a {category}.")
            if goal["target_category"] != category or goal["target_attributes"] or goal["spatial_relations"]:
                raise ValueError(f"Unexpected parsed goal: {goal}")
            result = client.perceive_scene(rgb, goal, task_id="find_" + category,
                                           observation_id="mp3d_dining", view_id="multiple")
            print(json.dumps({"rgb_sha256": image_hash, "goal": goal, "result": result}),
                  file=out, flush=True)
            print(category, result["target_status"], [len(result[k]) for k in
                  ("target_entities", "context_entities", "visual_cues")], flush=True)


if __name__ == "__main__":
    main()
