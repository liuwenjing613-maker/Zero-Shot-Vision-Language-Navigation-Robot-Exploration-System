"""Replay a deterministic Habitat walk, compare depth with independent scene collision hits."""
import argparse
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import habitat_sim as hs
import magnum as mn
import numpy as np
import quaternion
from habitat_sim.utils.common import quat_from_angle_axis
from PIL import Image, ImageDraw

from vln_memory.geometry_memory import Intrinsics, OccupancyMemory
from vln_memory.habitat_adapter import update_from_habitat
from vln_memory.topology_memory import TopologyMemory


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_map(path, grid, topology):
    colors = np.empty((*grid.log_odds.shape, 3), dtype=np.uint8)
    cells = grid.grid()
    for value, color in [(-1, (138, 145, 151)), (0, (249, 249, 245)), (100, (38, 46, 54))]:
        colors[cells == value] = color
    colors[grid.visit_count > 0] = (230, 120, 35)
    image = Image.fromarray(colors).resize((600, 600), Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST)
    draw = ImageDraw.Draw(image)
    for p in topology.places.values():
        r, c = grid.cell((p.x, p.y))
        draw.text((5*c+3, 5*r+2), p.place_id, fill=(7, 93, 155))
    panel = Image.new("RGB", (600, 638), "white")
    panel.paste(image)
    ImageDraw.Draw(panel).text((8, 606), "White: free   Dark: occupied   Gray: unknown   Orange: base visits\n+X right, +Z down | 0.1 m per cell", fill="black")
    panel.save(path)


def prepare_replica_fixture(scene_dir, temporary_dir):
    """Align a temporary render mesh and build an independent front-face ray oracle."""
    mesh_path = scene_dir / "mesh.ply"
    with mesh_path.open("rb") as stream:
        header = []
        for raw in stream:
            line = raw.decode("ascii").strip()
            header.append(line)
            if line == "end_header":
                break
        vertices, faces = [int(next(s.split()[-1] for s in header if s.startswith("element " + kind + " ")))
                           for kind in ("vertex", "face")]
        props = [s for s in header if s.startswith("property ")]
        expected = ["property float " + key for key in ("x", "y", "z", "nx", "ny", "nz")]
        expected += ["property uchar " + key for key in ("red", "green", "blue")]
        expected.append("property list uint8 int vertex_indices")
        require("format binary_little_endian 1.0" in header and props == expected, "Unexpected Replica PLY layout")
        v = np.fromfile(stream, dtype=[("xyz", "<f4", 3), ("normal", "<f4", 3), ("rgb", "u1", 3)], count=vertices)
        # Replica contains polygon faces. Triangulate their ordered vertex lists.
        import struct
        data = stream.read()
        offset, indices = 0, []
        for _ in range(faces):
            size = data[offset]
            require(size >= 3, "Degenerate Replica polygon")
            face = struct.unpack_from("<" + "i" * size, data, offset+1)
            indices.extend((face[0], face[j], face[j+1]) for j in range(1, size-1))
            offset += 1 + 4*size
        require(offset == len(data), "Unexpected trailing PLY data")
    # office_1 dataset stage has up=(0,0,1), front=(0,1,0).
    for field in ("xyz", "normal"):
        original = v[field].copy()
        v[field][:, 1] = original[:, 2]
        v[field][:, 2] = -original[:, 1]
    triangles = v["xyz"][np.asarray(indices)]
    # This Habitat 0.3.3 general PLY loader renders the file in native axes.
    # Express the scene in navigation/world coordinates before loading it.
    aligned = temporary_dir / "office_1.ply"
    with aligned.open("wb") as stream:
        stream.write(("\n".join(header) + "\n").encode("ascii"))
        v.tofile(stream)
        stream.write(data)
    stage = temporary_dir / "office_1.stage_config.json"
    stage.write_text(json.dumps(dict(render_asset=str(aligned), up=[0, 1, 0], front=[0, 0, -1],
                                    nav_asset=str((scene_dir / "habitat/mesh_semantic.navmesh").resolve()))))

    def nearest(origin, direction):
        # All fixture rays are horizontal. Cull by height before ray/triangle intersection.
        y = triangles[:, :, 1]
        tri = triangles[(y.min(axis=1) <= origin[1]) & (y.max(axis=1) >= origin[1])]
        a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
        e1, e2 = b-a, c-a
        p = np.cross(direction, e2)
        det = np.einsum("ij,ij->i", e1, p)
        live = det > 1e-9  # The depth renderer culls back-facing triangles.
        inv = np.divide(1., det, where=live, out=np.zeros_like(det))
        s = origin-a
        u = np.einsum("ij,ij->i", s, p) * inv
        q = np.cross(s, e1)
        v = np.einsum("ij,j->i", q, direction) * inv
        dist = np.einsum("ij,ij->i", e2, q) * inv
        hits = dist[live & (u >= 0) & (v >= 0) & (u+v <= 1) & (dist > 0.05)]
        return float(hits.min()) if hits.size else math.inf
    return stage, nearest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path, help="Replica root containing replica.scene_dataset_config.json")
    parser.add_argument("output", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    cfg = hs.SimulatorConfiguration()
    temporary = TemporaryDirectory(prefix="vln-m1-replica-")
    stage, nearest_mesh_hit = prepare_replica_fixture(args.dataset / "office_1", Path(temporary.name))
    cfg.scene_id = str(stage)
    cfg.gpu_device_id = args.gpu
    cfg.enable_physics = False

    sp = hs.CameraSensorSpec()
    sp.uuid, sp.sensor_type = "depth_sensor", hs.SensorType.DEPTH
    sp.resolution, sp.hfov, sp.position = mn.Vector2i(121, 161), mn.Deg(80), mn.Vector3(0, 0.5, 0)
    agent_cfg = hs.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [sp]
    K = Intrinsics.from_hfov(161, 121, 80)
    graph = TopologyMemory()
    with hs.Simulator(hs.Configuration(cfg, [agent_cfg])) as sim:
        pf = sim.pathfinder
        require(pf.is_loaded, "Replica navmesh missing")
        agent = sim.initialize_agent(0)
        state = hs.AgentState()
        # Fixed start from the project's previously rendered office_1 view A.
        state.position = np.array([-0.2110783, -0.8312321, -0.6458088])
        start = np.array(state.position)
        require(pf.is_navigable(start), "Fixture start is not navigable")
        for yaw in range(0, 360, 15):
            state.rotation = quat_from_angle_axis(math.radians(yaw), np.array([0, 1, 0]))
            facing = quaternion.rotate_vectors(state.rotation, [0, 0, -1])
            goal = start + facing
            goal = np.asarray(pf.snap_point(goal))
            if np.linalg.norm(np.asarray(pf.try_step(start, goal)) - goal) > 0.02:
                continue
            if np.linalg.norm(goal[[0, 2]] - (start+facing)[[0, 2]]) > 0.02:
                continue
            agent.set_state(state)
            center = float(sim.get_sensor_observations()[sp.uuid][60, 80])
            if 1.3 < center < 7.5:
                break
        else:
            raise RuntimeError("No clear one-meter straight fixture route found")

        grid = OccupancyMemory((120, 120), start[[0, 2]] - [6.05, 6.05], 0.1)
        path = []
        reference_errors = []
        endpoint_errors = []
        def integrate(obs, tick):
            live = agent.get_state()
            xy = np.asarray(live.position)[[0, 2]]
            segment = [path[-1][[0, 2]], xy] if 1 <= tick <= 4 else None
            update_from_habitat(grid, obs, live, K, tick, floor_y=start[1], stride=4, base_path=segment)
            sensor = live.sensor_states[sp.uuid]
            direction = quaternion.rotate_vectors(sensor.rotation, [0, 0, -1])
            distance = nearest_mesh_hit(np.asarray(sensor.position), direction)
            measured = float(obs[sp.uuid][60, 80])
            if distance < 8.0 and 0.05 < measured < 8.0:
                reference_errors.append(abs(measured - distance))
                hit_xy = (np.asarray(sensor.position) + distance * direction)[[0, 2]]
                rc = grid.cell(hit_xy)
                if grid.contains(rc):
                    occupied = np.argwhere(grid.grid() == 100)
                    if occupied.size:
                        xy = occupied[:, ::-1] * grid.resolution + [grid.origin_x + 0.05, grid.origin_y + 0.05]
                        endpoint_errors.append(float(np.linalg.norm(xy - hit_xy, axis=1).min()))
            path.append(np.array(live.position))

        integrate(sim.get_sensor_observations(), 0)
        p1 = graph.visit("start", start[[0, 2]], 0, "start", ["obs0"])
        for tick in range(1, 5):
            integrate(sim.step("move_forward"), tick)
        length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
        require(abs(length - 1) < 0.05 and np.linalg.norm(path[-1]-goal) < 0.05, "Walk failed")
        p2 = graph.visit("arrival", path[-1][[0, 2]], 4, "observation_place", ["obs4"])
        graph.record_navigation("route-executed", p1, p2, "succeeded", length)
        for tick in range(5, 9):
            integrate(sim.step("turn_left"), tick)

    temporary.cleanup()
    counts = {name: int(np.sum(grid.grid() == val)) for name, val in
              [("unknown", -1), ("free", 0), ("occupied", 100)]}
    require(all(counts.values()) and grid.map_version == 9, "Missing geometry evidence")
    expected_visits = {grid.cell(p[[0, 2]]) for a, b in zip(path, path[1:]) for p in np.linspace(a, b, 1001)}
    require(set(zip(*np.nonzero(grid.visit_count))) == expected_visits and int(grid.visit_count.sum()) == len(expected_visits),
            "Coverage must follow actual base movement only")
    require(len(graph.places) == 2 and len(graph.edges) == 1 and
            graph.edges[p1, p2].traversal_count == 1, "Missing traversal evidence")
    require(len(reference_errors) == len(path) and max(reference_errors) < 0.1,
            "Center depth disagrees with collision oracle: {}".format(reference_errors))
    require(len(endpoint_errors) == len(path) and max(endpoint_errors) < 0.16,
            "Mapped endpoints disagree with collision oracle: {}".format(endpoint_errors))
    report = dict(scene="office_1", yaw_degrees=yaw, resolution_m=grid.resolution,
                  frames=grid.map_version, grid_counts=counts, visit_entries=int(grid.visit_count.sum()),
                  places=list(graph.places), edge=vars(graph.edges[p1, p2]), path_length_m=length,
                  mesh_ray_checks=len(reference_errors), max_depth_error_m=max(reference_errors),
                  mapped_endpoint_checks=len(endpoint_errors), max_endpoint_error_m=max(endpoint_errors))
    args.output.mkdir(parents=True, exist_ok=False)
    write_map(args.output / "occupancy.png", grid, graph)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
