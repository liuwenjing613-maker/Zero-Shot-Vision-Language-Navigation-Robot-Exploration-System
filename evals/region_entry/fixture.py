"""Test-fixture selection only: annotated floor polygons and Habitat navmesh."""
import json
import math
from pathlib import Path

import habitat_sim as hs
import magnum as mn
import numpy as np
from PIL import Image
from habitat_sim.utils.common import quat_from_angle_axis

SCENE = "2azQ1b91cZZ"
ROOT = Path("/data/chenkejun/datasets/mp3d_habitat/scenes")
W, H = 480, 360
HFOV = 80


def regions():
    surfaces, regions, polygons = {}, {}, {}
    for line in (ROOT / SCENE / (SCENE + ".house")).open():
        if line[:2] not in ("S ", "V ", "R "):
            continue
        t = line.split()
        if t[0] == "R":
            regions[int(t[1])] = (t[5], float(t[8]))
        elif t[0] == "S" and t[4] == "F":
            surfaces[int(t[1])] = int(t[2])
        elif t[0] == "V" and int(t[2]) in surfaces:
            polygons.setdefault(surfaces[int(t[2])], []).append(
                [float(t[4]), -float(t[5])]
            )
    return {k: (regions[k], np.array(v)) for k, v in polygons.items()}


def inside(p, poly):
    x, y = p
    a, b = poly, np.roll(poly, -1, axis=0)
    dy = b[:, 1] - a[:, 1]
    cross = ((a[:, 1] > y) != (b[:, 1] > y)) & (
        x < a[:, 0] + (b[:, 0] - a[:, 0]) * (y - a[:, 1]) / np.where(dy != 0, dy, 1)
    )
    return bool(cross.sum() % 2)


def simulator(gpu, *, manual_actions=False):
    cfg = hs.SimulatorConfiguration()
    cfg.scene_dataset_config_file = str(ROOT / "mp3d.scene_dataset_config.json")
    cfg.scene_id = str(ROOT / SCENE / (SCENE + ".glb"))
    cfg.gpu_device_id = gpu
    cfg.enable_physics = False
    ac = hs.agent.AgentConfiguration()
    if manual_actions:
        ac.action_space["move_backward"] = hs.agent.ActionSpec("move_backward", hs.agent.ActuationSpec(amount=.25))
        for name in ("turn_left", "turn_right"):
            ac.action_space[name].actuation.amount = 15.0
    specs = []
    for name, kind in [("rgb", hs.SensorType.COLOR), ("depth", hs.SensorType.DEPTH)]:
        sp = hs.CameraSensorSpec()
        sp.uuid = name
        sp.sensor_type = kind
        sp.resolution = mn.Vector2i(H, W)
        sp.hfov = mn.Deg(HFOV)
        sp.position = mn.Vector3(0, 1.05, 0)
        specs.append(sp)
    laser = hs.CameraSensorSpec()
    laser.uuid = "scan_depth"
    laser.sensor_type = hs.SensorType.DEPTH
    laser.resolution = mn.Vector2i(3, 1025)
    laser.hfov = mn.Deg(90)
    laser.position = mn.Vector3(0, 0.45, 0)
    specs.append(laser)
    ac.sensor_specifications = specs
    sim = hs.Simulator(hs.Configuration(cfg, [ac]))
    sim.pathfinder.load_nav_mesh(str(ROOT / SCENE / (SCENE + ".navmesh")))
    return sim


def orient(agent, xyz, normal):
    st = hs.AgentState()
    st.position = xyz
    yaw = math.atan2(-normal[0], -normal[1])
    st.rotation = quat_from_angle_axis(yaw, np.array([0, 1, 0]))
    agent.set_state(st)


def candidates(sim):
    pf = sim.pathfinder
    polys = regions()
    found = []
    for rid, ((label, floor), poly) in polys.items():
        if abs(floor) > 0.3:
            continue
        for a, b in zip(poly, np.roll(poly, -1, axis=0)):
            length = np.linalg.norm(b - a)
            if length < 0.6:
                continue
            t = (b - a) / length
            n = np.array([-t[1], t[0]])
            samples = np.linspace(a, b, math.ceil(length / 0.07) + 1)
            crossable = []
            for pt in samples:
                xyz = np.array([pt[0], floor + 0.12, pt[1]])
                snapped = np.asarray(pf.snap_point(xyz))
                crossable.append(
                    np.isfinite(snapped).all()
                    and np.linalg.norm(snapped[[0, 2]] - pt) < 0.045
                    and abs(snapped[1] - floor) < 0.3
                )
            group = np.split(
                np.flatnonzero(crossable),
                np.flatnonzero(np.diff(np.flatnonzero(crossable)) > 1) + 1,
            )
            for ids in group:
                if len(ids) < 3:
                    continue
                width = np.linalg.norm(samples[ids[-1]] - samples[ids[0]])
                if width > 1.9:
                    continue
                center = samples[ids].mean(axis=0)
                for sign in (-1, 1):
                    direction = n * sign
                    start, end = center - 1.5 * direction, center + 1.0 * direction
                    if not inside(start, poly):
                        continue
                    other = [
                        k
                        for k, ((_, f), q) in polys.items()
                        if k != rid and abs(f - floor) < 0.35 and inside(end, q)
                    ]
                    if len(other) != 1:
                        continue
                    xyz = np.asarray(
                        pf.snap_point(np.r_[start[0], floor + 0.12, start[1]])
                    )
                    if (
                        not np.isfinite(xyz).all()
                        or np.linalg.norm(xyz[[0, 2]] - start) > 0.035
                    ):
                        continue
                    pos = xyz.copy()
                    for _ in range(10):
                        goal = pos + np.r_[direction[0], 0, direction[1]] * 0.25
                        pos = np.asarray(pf.try_step(pos, goal))
                        if np.linalg.norm(pos[[0, 2]] - goal[[0, 2]]) > 0.02:
                            break
                    else:
                        if any(
                            np.linalg.norm(center - u["center"]) < 0.65
                            and u["from"] == rid
                            for u in found
                        ):
                            continue
                        found.append(
                            dict(
                                center=center.tolist(),
                                normal=direction.tolist(),
                                start=xyz.tolist(),
                                width=round(float(width), 3),
                                **{"from": rid, "to": other[0]},
                                labels=[label, polys[other[0]][0][0]],
                            )
                        )
    return sorted(found, key=lambda f: abs(f["width"] - 0.6))


if __name__ == "__main__":
    import sys

    out = Path(sys.argv[1])
    out.mkdir(exist_ok=True, parents=True)
    with simulator(3) as sim:
        entries = candidates(sim)
        agent = sim.initialize_agent(0)
        for i, f in enumerate(entries):
            orient(agent, f["start"], f["normal"])
            obs = sim.get_sensor_observations()
            Image.fromarray(obs["rgb"][:, :, :3]).save(out / f"view-{i}.png")
        (out / "fixtures.json").write_text(json.dumps(entries, indent=2))
        print(
            [
                (i, f["from"], f["to"], f["labels"], f["width"])
                for i, f in enumerate(entries)
            ]
        )
