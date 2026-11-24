"""First same-view verification check using two previously detected table candidates."""
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
    root = Path(__file__).resolve().parents[1] / "results"
    rgb = root / "mp3d-2azQ1b91cZZ-chair-challenges-v1/multiple.png"
    rows = json.loads((root / "mp3d-three-goals-dining/dining-three-goals.json").read_text())
    record = next(row for row in rows if row["goal"]["instruction"] == "Find a table.")
    digest = hashlib.sha256(rgb.read_bytes()).hexdigest()
    if digest != record["rgb_sha256"]:
        raise ValueError("Source image differs from the detection frame")
    client = VLMClient(args.base_url, "qwen3.8:27b", timeout=240)
    with args.output.open("x", encoding="utf-8") as out:
        for candidate in record["result"]["target_entities"]:
            verdict = client.verify_target(rgb, record["goal"], record["result"], candidate["local_id"])
            entry = {"goal": record["goal"], "source_rgb_sha256": digest,
                     "candidate": candidate, "verification": verdict}
            print(json.dumps(entry, ensure_ascii=False), file=out, flush=True)
            print(candidate["local_id"], verdict["status"], verdict["latency_ms"], flush=True)


if __name__ == "__main__":
    main()
