"""Check the verifier's boxes against held-out MP3D instance renders."""
import argparse, json, math, sys
from pathlib import Path
import habitat_sim as hs
import magnum as mn
import numpy as np
import quaternion

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("episode", type=Path)
parser.add_argument("--dataset", type=Path, required=True)
parser.add_argument("--gpu", type=int, default=0)
args = parser.parse_args()
root = args.dataset
scene = "2azQ1b91cZZ"
cfg = hs.SimulatorConfiguration()
cfg.scene_id = str(root / scene / (scene + ".glb"))
cfg.scene_dataset_config_file = str(root / "mp3d.scene_dataset_config.json")
cfg.gpu_device_id = args.gpu
cfg.enable_physics = False
agent = hs.agent.AgentConfiguration()
specs = []
for name, kind in (("sem", hs.SensorType.SEMANTIC), ("depth", hs.SensorType.DEPTH)):
    s = hs.CameraSensorSpec()
    s.uuid, s.sensor_type = name, kind
    s.resolution, s.hfov, s.position = (
        mn.Vector2i(480, 640),
        mn.Deg(90),
        mn.Vector3(0, 1.05, 0),
    )
    specs.append(s)
agent.sensor_specifications = specs
episode = args.episode
results = []
with hs.Simulator(hs.Configuration(cfg, [agent])) as sim:
    a = sim.initialize_agent(0)
    categories = {
        int(o.id.split("_")[-1]): o.category.name()
        for o in sim.semantic_scene.objects
        if o is not None and o.category is not None
    }
    objects = {
        int(o.id.split("_")[-1]): o
        for o in sim.semantic_scene.objects
        if o is not None and o.category is not None
    }
    for row in json.loads((episode / "trace.json").read_text())["actions"]:
        folder = episode / row["observation"]
        observed = json.loads((folder / "perception.json").read_text())
        sensors = np.load(folder / "sensors.npz")
        transform = sensors["camera_pose"]
        state = hs.AgentState()
        state.position = np.asarray(row["pose"], dtype=np.float32)
        state.rotation = quaternion.from_rotation_matrix(transform[:3, :3])
        a.set_state(state)
        raw = sim.get_sensor_observations()
        diff = np.abs(sensors["depth"] - raw["depth"])
        for verdict in row["verdicts"]:
            obj = next(
                it
                for it in observed["target_entities"]
                if it["local_id"] == verdict["candidate_id"]
            )
            l, t, r, b = np.asarray(obj["bbox"]) * [0.64, 0.48, 0.64, 0.48]
            regions = {}
            for scale in (1.0, 0.5):
                x, y = (l + r) / 2, (b + t) / 2
                dx, dy = (r - l) * scale / 2, (b - t) * scale / 2
                patch = raw["sem"][
                    int(y - dy) : math.ceil(y + dy), int(x - dx) : math.ceil(x + dx)
                ]
                ids, counts = np.unique(patch, return_counts=True)
                regions[str(scale)] = [
                    dict(
                        id=int(ids[i]),
                        label=categories.get(int(ids[i])),
                        fraction=round(float(counts[i] / patch.size), 3),
                    )
                    for i in np.argsort(counts)[::-1][:5]
                ]
            center = int(raw["sem"][int((b + t) / 2), int((l + r) / 2)])
            nearby = None
            if categories.get(center) == "chair":
                seat = objects[center].aabb
                choices = []
                for id_ in np.unique(raw["sem"]):
                    id_ = int(id_)
                    if categories.get(id_) not in {
                        "table",
                        "dining table",
                        "dining_table",
                    }:
                        continue
                    table = objects[id_].aabb
                    delta = np.abs(
                        np.asarray(table.center()) - np.asarray(seat.center())
                    )
                    gap = np.maximum(
                        delta
                        - (np.asarray(table.size()) + np.asarray(seat.size())) / 2,
                        0,
                    )
                    if gap[1] > 0.15:
                        continue
                    choices.append(
                        (
                            float(np.linalg.norm(gap[[0, 2]])),
                            float(np.linalg.norm(delta[[0, 2]])),
                            id_,
                        )
                    )
                if choices:
                    edge, centers, id_ = min(choices)
                    nearby = dict(
                        table_id=id_,
                        category=categories[id_],
                        aabb_gap_m=round(edge, 3),
                        centers_m=round(centers, 3),
                    )
            results.append(
                dict(
                    nearby_table=nearby,
                    obs=row["observation"],
                    local_id=obj["local_id"],
                    global_id=verdict["object"],
                    center_id=center,
                    center_label=categories.get(center),
                    regions=regions,
                    depth_match_pct=float(np.mean(diff < 0.001) * 100),
                )
            )
print(json.dumps(results, ensure_ascii=False, indent=2))
