"""Replay a recorded MP3D session, checking frame alignment and object history."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from vln_memory.online_memory import OnlineMemory


def replay(session):
    config = json.loads((session / "config.json").read_text())
    memory = OnlineMemory(config)
    counts, sources = Counter(), Counter()
    for frame_dir in sorted(session.glob("f[0-9]*")):
        step = json.loads((frame_dir / "step.json").read_text())
        vlm = json.loads((frame_dir / "vlm.json").read_text())
        digest = hashlib.sha256((frame_dir / "rgb.png").read_bytes()).hexdigest()
        assert digest == vlm["rgb_sha256"] == step["rgb_sha256"], frame_dir
        with np.load(frame_dir / "sensors.npz") as data:
            state = memory.apply(step, data, vlm["result"])
        json.dumps(state, allow_nan=False)
        evidence = state["object_memory"]["observations"]
        assert len(evidence) == sum(
            len(vlm["result"][k]) for k in ("target_entities", "context_entities")
        )
        ids = [v["object_id"] for v in evidence.values() if v["object_id"]]
        assert len(ids) == len(set(ids)), frame_dir
        for key, obs in evidence.items():
            if obs["object_id"]:
                assert (
                    key in memory.objects.get_object(obs["object_id"]).observation_ids
                )
        counts.update(v["association_status"] for v in evidence.values())
        sources[vlm["source"]] += 1
    if not sources:
        raise ValueError("No recorded frames")
    return dict(
        scene=config["scene"],
        frames=memory.last + 1,
        sources=sources,
        associations=counts,
        objects=len(memory.objects.records),
        tracks=Counter(o.track_status for o in memory.objects.records.values()),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    print(json.dumps(replay(parser.parse_args().session), indent=2))
