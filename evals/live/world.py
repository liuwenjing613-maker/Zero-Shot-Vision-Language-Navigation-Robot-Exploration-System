"""Single-owner Habitat runtime with immutable per-action sensor frames."""
import hashlib, json, math, os, uuid, sys, time
from pathlib import Path
import numpy as np
import quaternion
from PIL import Image
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "region_entry"))
from fixture import simulator, orient, SCENE, W, H, HFOV

sys.path.insert(0, str(HERE.parent / "m1_lab"))
from scan_mp3d import scan
from vln_memory.online_memory import new_grid, fuse
from model import MODEL


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    tmp = path.with_name("." + path.name + "-" + uuid.uuid4().hex[:8])
    tmp.write_text(
        json.dumps(
            data, ensure_ascii=False, allow_nan=False, default=lambda x: x.item()
        )
    )
    os.replace(tmp, path)


def frames(folder):
    return sorted(
        p for p in folder.glob("f[0-9]*") if p.is_dir() and (p / "step.json").exists()
    )


def frontier_options(grid, xy):
    a = grid.grid()
    free = a == 0
    safe = ndi.distance_transform_edt(free) * grid.resolution >= 0.22
    labels, _ = ndi.label(safe)
    rc = grid.cell(xy)
    if not grid.contains(rc) or not labels[rc]:
        return []
    reachable = labels == labels[rc]
    groups, count = ndi.label(
        free & ndi.binary_dilation(a < 0), structure=np.ones((3, 3))
    )
    if not count:
        return []
    dist, indices = ndi.distance_transform_edt(~reachable, return_indices=True)
    options = []
    for i, box in enumerate(ndi.find_objects(groups), 1):
        if box is None:
            continue
        pts = np.argwhere(groups[box] == i)
        if len(pts) * grid.resolution < 0.45:
            continue
        pts += np.array([z.start for z in box])
        mid = pts.mean(0)
        for row, col in pts[np.argsort(((pts - mid) ** 2).sum(1))]:
            if dist[row, col] * grid.resolution > 0.8:
                continue
            r, c = indices[:, row, col]
            p = [
                grid.origin_x + (c + 0.5) * grid.resolution,
                grid.origin_y + (r + 0.5) * grid.resolution,
            ]
            delta = math.dist(p, xy)
            if delta < 0.55 or any(math.dist(p, f["xy"]) < 0.8 for f in options):
                continue
            options.append(dict(id=f"F{r}-{c}", xy=p, distance=round(delta, 2)))
            break
    return sorted(options, key=lambda f: f["distance"])[:8]


class World:
    def __init__(self, gpu):
        self.gpu = gpu
        self.sim = None
        self.folder = None

    def open(self, folder, preset=None):
        if self.folder == folder:
            return
        if self.sim is None:
            self.sim = simulator(self.gpu, manual_actions=True)
        self.agent = self.sim.initialize_agent(0)
        if preset:
            if preset == "junction":
                p = [16.52, 0.12711, 8.50]
                u = [0.0, 1.0]
            elif preset == "door":
                f = read(ROOT / "results/region-entry/1.json")["fixture"]
                p, u = f["start"], f["normal"]
            else:
                raise ValueError("Unknown starting position")
            pos = np.asarray(self.sim.pathfinder.snap_point(p))
            orient(self.agent, pos, u)
            lo, hi = self.sim.pathfinder.get_bounds()
            lo = np.floor(np.array(lo)[[0, 2]] - 7)
            hi = np.ceil(np.array(hi)[[0, 2]] + 7)
            res = 0.08
            cfg = dict(
                session=folder.name,
                scene=SCENE,
                preset=preset,
                origin=lo.tolist(),
                shape=np.ceil((hi - lo)[::-1] / res).astype(int).tolist(),
                resolution=res,
                radius=self.sim.config.agents[0].radius,
                camera=[W, H, HFOV],
                floor=float(pos[1]),
                start=pos.tolist(),
                direction=u,
                goal=read(
                    ROOT / "results/mp3d-three-goals-dining/dining-three-goals.json"
                )[0]["goal"],
                model=MODEL,
            )
            write(folder / "config.json", cfg)
        self.cfg = read(folder / "config.json")
        self.grid = new_grid(self.cfg)
        self.target = None
        self.folder = None
        orient(self.agent, self.cfg["start"], self.cfg["direction"])
        for f in frames(folder):
            s = read(f / "step.json")
            with np.load(f / "sensors.npz") as d:
                fuse(self.grid, s, d["ranges"])
            self.target = s["frontier"]
            self.restore(s)
        self.folder = folder

    def restore(self, s):
        import habitat_sim as hs

        st = hs.AgentState()
        st.position = s["xyz"]
        st.rotation = quaternion.from_float_array(s["rotation"])
        self.agent.set_state(st)

    def capture(self, folder, action, expected, choice=None):
        self.open(folder)
        fs = frames(folder)
        t = len(fs)
        if expected != t - 1:
            raise ValueError("Frame changed; refresh controls")
        previous = self.agent.get_state()
        before = np.asarray(previous.position).copy()
        if action in ("forward", "back", "left", "right"):
            self.sim.step(
                dict(
                    forward="move_forward",
                    back="move_backward",
                    left="turn_left",
                    right="turn_right",
                )[action]
            )
        elif action == "frontier":
            opts = read(fs[-1] / "step.json")["options"]
            target = next((f for f in opts if f["id"] == choice), None)
            if not target:
                raise ValueError("Select a current frontier")
            self.target = target
        elif action not in ("start", "observe"):
            raise ValueError("Invalid robot action")
        state = self.agent.get_state()
        floor_block = abs(state.position[1] - self.cfg["floor"]) > 0.28
        if floor_block:
            self.agent.set_state(previous)
            state = self.agent.get_state()
        xyz = np.array(state.position)
        xy = xyz[[0, 2]]
        delta = float(np.linalg.norm(xy - before[[0, 2]]))
        obs = self.sim.get_sensor_observations()
        rgb = Image.fromarray(obs["rgb"][:, :, :3].copy())
        depth = obs["depth"].copy()
        eye = state.sensor_states["depth"]
        T = np.eye(4)
        T[:3, :3] = quaternion.as_rotation_matrix(eye.rotation)
        T[:3, 3] = eye.position
        ranges = scan(self.sim, self.agent, xyz, 0.0)
        self.agent.set_state(state, reset_sensors=False, infer_sensor_states=False)
        tmp = folder / (".capture-" + uuid.uuid4().hex)
        tmp.mkdir()
        rgb.save(tmp / "rgb.png")
        rgb.save(tmp / "view.jpg", quality=85)
        s = dict(
            step=t,
            acquired_at=time.time(),
            session=folder.name,
            observation_id=f"{folder.name}-{t:05}",
            action=action,
            xy=xy.tolist(),
            xyz=xyz.tolist(),
            rotation=quaternion.as_float_array(state.rotation).tolist(),
            heading=math.degrees(math.atan2(-T[2, 2], -T[0, 2])),
            travel_m=delta,
            collision=bool(action in ("forward", "back") and delta < 0.02),
            floor_block=bool(floor_block),
            rgb_sha256=hashlib.sha256((tmp / "rgb.png").read_bytes()).hexdigest(),
            frontier=self.target,
        )
        fuse(self.grid, s, ranges)
        s["options"] = frontier_options(self.grid, xy)
        a = self.grid.grid()
        rr, cc = np.nonzero(a >= 0)
        r0, c0 = max(0, rr.min() - 3), max(0, cc.min() - 3)
        crop = a[r0 : rr.max() + 4, c0 : cc.max() + 4]
        pixels = np.empty((*crop.shape, 3), dtype=np.uint8)
        for value, color in [
            (-1, (165, 174, 177)),
            (0, (246, 245, 233)),
            (100, (42, 49, 52)),
        ]:
            pixels[crop == value] = color
        Image.fromarray(pixels).save(tmp / "map.png")
        s["window"] = [
            self.grid.origin_x + c0 * self.grid.resolution,
            self.grid.origin_y + r0 * self.grid.resolution,
            crop.shape[1] * self.grid.resolution,
            crop.shape[0] * self.grid.resolution,
        ]
        s["grid_cells"] = dict(
            free=int((a == 0).sum()),
            occupied=int((a == 100).sum()),
            visited=int((self.grid.visit_count > 0).sum()),
        )
        np.savez_compressed(
            tmp / "grid.npz",
            log_odds=self.grid.log_odds,
            visit_count=self.grid.visit_count,
            origin=self.cfg["origin"],
            resolution=self.grid.resolution,
        )
        np.savez_compressed(
            tmp / "sensors.npz", ranges=ranges, depth=depth, camera_pose=T
        )
        write(tmp / "step.json", s)
        tmp.rename(folder / f"f{t:05}")
        return s
