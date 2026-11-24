"""Emulate a horizontal spinning lidar with synchronized pinhole depth views."""
import json
import math
import sys
from pathlib import Path

import habitat_sim as hs
import magnum as mn
import numpy as np
from habitat_sim.utils.common import quat_from_angle_axis

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_m1_habitat import mp3d_triangles
from scan_lab import ANGLES, LIMIT, N, ScanExperiment
from vln_memory.geometry_memory import OccupancyMemory


def section(triangles, height, xy):
    """Intersect the original scan mesh with the laser plane for the evaluator."""
    y = triangles[:, :, 1]
    nearby = (y.min(axis=1) <= height) & (y.max(axis=1) > height)
    for axis, coordinate in zip((0, 2), xy):
        v = triangles[:, :, axis]
        nearby &= (v.min(axis=1) < coordinate + LIMIT + 0.2) & (
            v.max(axis=1) > coordinate - LIMIT - 0.2
        )
    t = triangles[nearby].astype(float)
    a, b = t, np.roll(t, -1, axis=1)
    crossing = (a[:, :, 1] <= height) != (b[:, :, 1] <= height)
    ia, ib = a[crossing], b[crossing]
    part = (height - ia[:, 1]) / (ib[:, 1] - ia[:, 1])
    points = ia[:, [0, 2]] + part[:, None] * (ib[:, [0, 2]] - ia[:, [0, 2]])
    segments = points.reshape(-1, 2, 2)
    # Preserve the wound front face; Habitat rasterization culls back faces.
    normal = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])[:, [0, 2]]
    edge = segments[:, 1] - segments[:, 0]
    flip = (normal[:, 0] * edge[:, 1] - normal[:, 1] * edge[:, 0]) < 0
    segments[flip] = segments[flip, ::-1]
    return segments


def scan(sim, agent, position, heading):
    width = 1025
    focal = width / 2
    ranges = np.full(N, np.nan)
    # Four 90-degree views share the exact same laser origin and timestamp.
    # Each azimuth belongs to one sector; no repeated 360-degree endpoint.
    face_of = np.floor((ANGLES + math.pi / 4) / (math.pi / 2)).astype(int) % 4
    for face in range(4):
        angle = heading + face * math.pi / 2
        state = hs.AgentState()
        state.position = position
        state.rotation = quat_from_angle_axis(-angle - math.pi / 2, np.array([0, 1, 0]))
        agent.set_state(state)
        take = face_of == face
        delta = (ANGLES[take] - face * math.pi / 2 + math.pi) % (2 * math.pi) - math.pi
        pixel = (
            np.rint((width - 1) / 2 + focal * np.tan(delta))
            .astype(int)
            .clip(0, width - 1)
        )
        axial = sim.get_sensor_observations()["scan_depth"][1, pixel].astype(float)
        radial = axial * np.hypot(1.0, (pixel - (width - 1) / 2) / focal)
        values = np.where(radial >= LIMIT, np.inf, radial)
        values[(~np.isfinite(axial)) | (axial <= 0) | (radial < 0.05)] = np.nan
        ranges[take] = values
    return ranges


def run(folder, dataset, gpu):
    scene = "2azQ1b91cZZ"
    glb = dataset / scene / (scene + ".glb")
    triangles = mp3d_triangles(glb)
    cfg = hs.SimulatorConfiguration()
    cfg.scene_dataset_config_file = str(dataset / "mp3d.scene_dataset_config.json")
    cfg.scene_id = str(glb)
    cfg.gpu_device_id = gpu
    cfg.enable_physics = False
    spec = hs.CameraSensorSpec()
    spec.uuid, spec.sensor_type = "scan_depth", hs.SensorType.DEPTH
    spec.resolution, spec.hfov, spec.position = (
        mn.Vector2i(3, 1025),
        mn.Deg(90),
        mn.Vector3(0, 0.45, 0),
    )
    ac = hs.agent.AgentConfiguration()
    ac.sensor_specifications = [spec]
    with hs.Simulator(hs.Configuration(cfg, [ac])) as sim:
        pf = sim.pathfinder
        if not pf.load_nav_mesh(str(glb.with_suffix(".navmesh"))):
            raise RuntimeError("MP3D navmesh unavailable")
        log = (
            Path(__file__).resolve().parents[2]
            / "results/verify-e02-new-view/e02-view-comparison.json"
        )
        start = np.array(json.loads(log.read_text())["old_view"]["position_m"])
        end = np.array(pf.snap_point(np.array([8.077029, start[1], 4.182217])))
        path = hs.ShortestPath()
        path.requested_start, path.requested_end = start, end
        if not pf.find_path(path) or path.geodesic_distance < 2:
            raise RuntimeError("Expected a reachable dining-room route longer than 2 m")
        stations = [start]
        for a, b in zip(path.points, path.points[1:]):
            a, b = np.asarray(a), np.asarray(b)
            length = np.linalg.norm(b - a)
            if length > 1e-5:
                stations.extend(np.linspace(a, b, int(math.ceil(length / 0.7)) + 1)[1:])
        lo = np.min(stations, axis=0)[[0, 2]] - LIMIT - 0.15
        hi = np.max(stations, axis=0)[[0, 2]] + LIMIT + 0.15
        grid = OccupancyMemory(np.ceil((hi - lo)[::-1] / 0.1).astype(int), lo, 0.1)
        ex = ScanExperiment(folder, "laser360", grid)
        agent = sim.initialize_agent(0)
        schedule = (
            [stations[0], stations[0]] + stations[1:] + list(reversed(stations[:-1]))
        )
        for i, pos in enumerate(schedule):
            xy = np.array(pos)[[0, 2]]
            delta = xy - ex.prev if i > 1 else None
            heading = math.atan2(delta[1], delta[0]) if i > 1 else i * math.pi / 3
            ranges = scan(sim, agent, pos, heading)
            ref = section(triangles, pos[1] + 0.45, xy)

            def visit(graph):
                if i in (0, len(stations), len(schedule) - 1):
                    p = graph.visit("s" + str(i), xy, i, "observation_place")
                    if i == len(stations) and p != "P01":
                        graph.record_navigation(
                            "navmesh-route",
                            "P01",
                            p,
                            "planned",
                            float(path.geodesic_distance),
                        )

            def check_pose(_grid):
                if i == len(schedule) - 1:
                    return [
                        (
                            "闭环回到同一 P01，端点共两个 Place",
                            len(ex.graph.places) == 2
                            and ex.graph.places["P01"].visit_count == 2
                            and np.linalg.norm(xy - start[[0, 2]]) < 0.001,
                        )
                    ]
                return []

            ex.sweep(
                xy,
                heading,
                ranges,
                ref,
                "餐厅起点 · 完整首圈"
                if i == 0
                else (
                    "原地转向 60°"
                    if i == 1
                    else ("返回起点" if i == len(schedule) - 1 else "沿导航网格换站")
                ),
                "MP3D 餐厅，已知位姿设站。水平面高出底盘 0.45 m，四个同心视角合成 360° 扫描；原始 GLB 截面独立给出参考首回波。",
                checks=check_pose,
                path=None if i == 0 else [ex.prev, xy],
                graph_update=visit,
            )
