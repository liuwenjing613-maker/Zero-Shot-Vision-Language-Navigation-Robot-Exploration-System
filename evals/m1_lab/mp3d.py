import hashlib
import json
import math
from pathlib import Path

import habitat_sim as hs
import magnum as mn
import numpy as np
import quaternion
from habitat_sim.utils.common import quat_from_angle_axis
from PIL import Image

from record import Recorder
from vln_memory.geometry_memory import Intrinsics, OccupancyMemory
from vln_memory.habitat_adapter import update_from_habitat
from vln_memory.topology_memory import TopologyMemory


def run(folder, dataset, archive, gpu):
    folder = Path(folder)
    root = Path(__file__).resolve().parents[2]
    scene = "2azQ1b91cZZ"
    pose = json.loads(
        (root / "results/verify-e02-new-view/e02-view-comparison.json").read_text()
    )
    record = json.loads(
        (root / "results/mp3d-three-goals-dining/dining-three-goals.json").read_text()
    )
    row = next(item for item in record if item["goal"]["target_category"] == "chair")
    evidence = row["result"]
    old = pose["old_view"]
    reference_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    if not (
        pose["scene"] == scene
        and row["rgb_sha256"] == old["rgb_sha256"] == reference_sha
    ):
        raise ValueError("Archived image/observation identity mismatch")
    source = np.asarray(Image.open(archive).convert("RGB"))
    start = np.array(old["position_m"])
    grid = OccupancyMemory((120, 120), start[[0, 2]] - [6.05, 6.05], 0.1)
    graph = TopologyMemory()
    rec = Recorder(folder, "mp3d", grid, graph)
    Image.fromarray(source).save(folder / "archive.jpg", quality=91)
    cfg = hs.SimulatorConfiguration()
    cfg.scene_dataset_config_file = str(dataset / "mp3d.scene_dataset_config.json")
    cfg.scene_id = str(dataset / scene / (scene + ".glb"))
    cfg.gpu_device_id = gpu
    cfg.enable_physics = False
    agent_cfg = hs.agent.AgentConfiguration()
    specs = []
    for key, kind in [
        ("rgb", hs.SensorType.COLOR),
        ("depth_sensor", hs.SensorType.DEPTH),
    ]:
        spec = hs.CameraSensorSpec()
        spec.uuid, spec.sensor_type = key, kind
        spec.resolution, spec.position, spec.hfov = (
            mn.Vector2i(480, 640),
            mn.Vector3(0, 1.35, 0),
            mn.Deg(80),
        )
        specs.append(spec)
    agent_cfg.sensor_specifications = specs
    K = Intrinsics.from_hfov(640, 480, 80)
    with hs.Simulator(hs.Configuration(cfg, [agent_cfg])) as sim:
        pf = sim.pathfinder
        if not pf.load_nav_mesh(
            str(dataset / scene / (scene + ".navmesh"))
        ) or not pf.is_navigable(start):
            raise RuntimeError("Invalid MP3D starting pose/navmesh")
        agent = sim.initialize_agent(0)
        state = hs.AgentState()
        state.position = start
        state.rotation = quat_from_angle_axis(
            math.radians(old["yaw_deg"]), np.array([0, 1, 0])
        )
        agent.set_state(state)
        obs_id = evidence["observation_id"]
        a = graph.visit("start", start[[0, 2]], 0, "start", [obs_id])
        path = [start]

        def capture(obs, action, extra_checks=(), semantics=None):
            i = len(rec.steps)
            live = agent.get_state()
            xyz = np.array(live.position)
            xy = xyz[[0, 2]]
            depth = obs["depth_sensor"]
            rgb = np.ascontiguousarray(obs["rgb"][:, :, :3])
            seg = [path[-1][[0, 2]], xy] if action == "move_forward" else None
            count_before = grid.visit_count.sum()
            # Reconstruct one sampled pixel independently of the mapper's matrix path.
            camera = live.sensor_states["depth_sensor"]
            z = float(depth[240, 320])
            witness = None
            if np.isfinite(z) and 0.05 < z < 8:
                optical = [(320 - K.cx) * z / K.fx, (K.cy - 240) * z / K.fy, -z]
                point = camera.position + quaternion.rotate_vectors(
                    camera.rotation, optical
                )
                cell = grid.cell(point[[0, 2]])
                if grid.contains(cell) and 0.12 <= point[1] - start[1] <= 1.65:
                    witness = dict(
                        cell=list(cell),
                        before=float(grid.log_odds[cell]),
                        ray=[
                            np.asarray(camera.position)[[0, 2]].tolist(),
                            point[[0, 2]].tolist(),
                        ],
                    )
            update_from_habitat(
                grid, obs, live, K, i, floor_y=start[1], stride=8, base_path=seg
            )
            path.append(xyz)
            checks = [
                (
                    "本次有效深度帧已融合",
                    grid.map_version == i + 1
                    and np.any(np.isfinite(depth) & (depth > 0.05) & (depth < 8)),
                )
            ]
            if witness:
                witness["after"] = float(grid.log_odds[tuple(witness["cell"])])
                checks.append(
                    (
                        "独立反投影命中格的证据增加 0.9，最高为 4",
                        np.isclose(witness["after"], min(4, witness["before"] + 0.9)),
                    )
                )
            desc = {
                "observe": "在感知实验的原始位姿读取 RGB-D，将已核对的历史观测 ID 挂到起点。",
                "turn_left": "执行 Habitat 原地左转，更新深度证据并检查底盘访问次数。",
                "move_forward": "执行 Habitat 前进动作，使用实际底盘位姿更新沿途访问格。",
            }[action]
            if action == "turn_left":
                checks.append(("原地转向未增加访问次数", grid.visit_count.sum() == count_before))
            if action == "move_forward":
                checks.append(
                    (
                        "前进约 0.25 米并进入新格",
                        abs(np.linalg.norm(xyz - path[-2]) - 0.25) < 0.015
                        and grid.visit_count.sum() > count_before,
                    )
                )
            if action == "observe":
                diff = float(np.abs(rgb.astype(float) - source.astype(float)).mean())
                checks.extend(
                    [
                        ("原图文件哈希匹配历史模型输入", reference_sha == row["rgb_sha256"]),
                        ("复现的 RGB 与历史原图像素平均差小于 2/255", diff < 2),
                        ("P01 包含感知观测 ID", obs_id in graph.places[a].observation_ids),
                    ]
                )
                semantics["pixel_mae"] = diff
            Image.fromarray(rgb).resize((320, 240)).save(
                folder / f"rgb-{i}.jpg", quality=87
            )
            meter = np.nan_to_num(depth, nan=0, posinf=8, neginf=0)
            Image.fromarray((np.clip(meter, 0, 8) * 255 / 8).astype("u1")).resize(
                (320, 240)
            ).save(folder / f"depth-{i}.png")
            checks.extend(extra_checks)
            rec.step(
                {"observe": "餐厅起点与感知证据", "turn_left": "原地左转", "move_forward": "沿路线前进"}[
                    action
                ],
                desc,
                checks,
                xy,
                rgb=f"rgb-{i}.jpg",
                depth=f"depth-{i}.png",
                semantic=semantics,
                action=action,
                center_m=round(z, 3),
                witness=witness,
                ray=witness["ray"] if witness else None,
            )

        sem = dict(
            id=obs_id,
            model=evidence["model_id"],
            goal=row["goal"]["instruction"],
            sha=reference_sha,
            entities=evidence["target_entities"],
            context=evidence["context_entities"],
            cues=evidence["visual_cues"],
        )
        capture(sim.get_sensor_observations(), "observe", semantics=sem)
        for turns in range(36):
            facing = quaternion.rotate_vectors(agent.get_state().rotation, [0, 0, -1])
            p = start.copy()
            for step in range(1, 5):
                goal = start + step * 0.25 * facing
                p = np.asarray(pf.try_step(p, goal))
                if np.linalg.norm(p - goal) > 0.02:
                    break
            else:
                break
            capture(sim.step("turn_left"), "turn_left")
        else:
            raise RuntimeError("No verified one-meter route")
        for step in range(4):
            obs = sim.step("move_forward")
            checks = []
            if step == 3:
                xyz = np.array(agent.get_state().position)
                length = float(
                    np.linalg.norm(np.diff(path + [xyz], axis=0), axis=1).sum()
                )
                b = graph.visit(
                    "arrival",
                    xyz[[0, 2]],
                    len(rec.steps),
                    "observation_place",
                    [f"depth-{len(rec.steps)}"],
                )
                e = graph.record_navigation("execution", a, b, "succeeded", length)
                checks.extend(
                    [
                        (
                            "实走一米并抵达计划终点",
                            abs(length - 1) < 0.03
                            and np.linalg.norm(xyz - goal) < 0.03,
                        ),
                        (
                            "两处 Place 与一条已走通的有向边",
                            len(graph.places) == 2
                            and len(graph.edges) == 1
                            and e.status == "TRAVERSED"
                            and e.traversal_count == 1,
                        ),
                    ]
                )
            capture(obs, "move_forward", checks)
        capture(sim.step("turn_left"), "turn_left")
