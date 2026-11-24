"""Run two independent RGB requests against the same parsed chair goal."""
import argparse
import json
from pathlib import Path
from vln_perception import VLMClient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    rows = (root/"results/goal-27b-gpu3.jsonl").read_text().splitlines()
    goal = next(r["actual"] for r in map(json.loads, rows) if r["instruction"]=="Find a chair.")
    client = VLMClient(args.base_url, "qwen3.8:27b", timeout=180)
    with args.output.open("x") as out:
        for view in ("A","B"):
            result = client.perceive_scene(root/"evals"/"scene_pair"/(view+".jpg"),goal,
                                          task_id="find_chair",observation_id="obs_"+view,view_id=view)
            print(json.dumps(result,ensure_ascii=False),file=out,flush=True)
            print(view,result["target_status"],result["latency_ms"],flush=True)


if __name__ == "__main__":
    main()
