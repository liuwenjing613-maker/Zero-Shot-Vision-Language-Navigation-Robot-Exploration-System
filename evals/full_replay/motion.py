import base64, io, json, math, sys
from pathlib import Path
import numpy as np, quaternion
from PIL import Image
from options import options, REPO

args = options()
ROOT = args.runs_root
sys.path[:0] = [str(REPO / "code"), str(REPO / "src"), str(REPO / "evals")]
from execution.habitat_backend import HabitatBackend


def read(p):
    return json.loads(p.read_text())


def encode(im, quality):
    out = io.BytesIO()
    im.resize((384, 288), Image.Resampling.BILINEAR).save(
        out, "WEBP", quality=quality, method=2
    )
    return "data:image/webp;base64," + base64.b64encode(out.getvalue()).decode()


stage = args.dataset / "2azQ1b91cZZ/2azQ1b91cZZ.glb"
dataset = stage.parents[1] / "mp3d.scene_dataset_config.json"
assets = []
cache = {}
episodes = {}
checks = []
palette = np.array(
    [
        [21, 23, 67],
        [48, 55, 124],
        [39, 113, 151],
        [48, 171, 151],
        [146, 207, 111],
        [251, 232, 136],
    ]
)


class Agent:
    def __init__(self, base, owner):
        self.base = base
        self.owner = owner

    def __getattr__(self, k):
        return getattr(self.base, k)

    def act(self, a):
        result = self.base.act(a)
        self.owner.sample("turn")
        return result


class Backend(HabitatBackend):
    def _set_position(self, p, **kwargs):
        self.phase = "align"
        super()._set_position(p, **kwargs)
        if hasattr(self, "steps"):
            self.sample("move")

    def sample(self, kind):
        s = self._agent.get_state()
        pos = np.asarray(s.position).tolist()
        q = quaternion.as_rotation_matrix(s.rotation)
        h = (-q[:, 2])[[0, 2]].tolist()
        key = tuple(round(float(x), 5) for x in pos + h)
        if key not in cache:
            raw = self._sim.get_sensor_observations()
            depth = raw["depth_sensor"]
            rgb = encode(Image.fromarray(raw["color_sensor"][:, :, :3]), 73)
            ix = np.clip(np.nan_to_num(depth) / 6, 0, 1) * 5
            a = ix.astype(int)
            mix = (ix - a)[..., None]
            color = palette[a] * (1 - mix) + palette[np.minimum(5, a + 1)] * mix
            color[(~np.isfinite(depth)) | (depth <= 0)] = [15, 23, 31]
            cache[key] = len(assets)
            assets.append([rgb, encode(Image.fromarray(color.astype("uint8")), 65)])
        self.steps.append(dict(p=pos, h=h, img=cache[key], kind=kind, phase=self.phase))

    def scan(self):
        self.phase = "scan"
        return super().scan()

    def observe(self):
        packet = super().observe()
        self.marks[packet.observation_id] = len(self.steps)
        return packet


for ep in read(Path("dataset.json")):
    name = ep["id"]
    rows = [f["row"] for f in ep["frames"]]
    b = Backend(
        str(stage),
        start_position=rows[0]["pose"],
        camera_offset=1.05,
        enable_physics=False,
        navmesh=str(stage.with_suffix(".navmesh")),
        scene_dataset=str(dataset),
        gpu_device_id=args.gpu,
        align_with_travel=ep["align_with_travel"],
    )
    try:
        st = b._agent.get_state()
        origin_sensor = ROOT / name / rows[0]["observation"] / "sensors.npz"
        with np.load(origin_sensor) as first_packet:
            st.rotation = quaternion.from_rotation_matrix(
                first_packet["camera_pose"][:3, :3]
            )
        b._agent.set_state(st)
        b._agent = Agent(b._agent, b)
        b.steps = []
        b.marks = {}
        pending = []
        end = 0
        maxerr = 0
        maxpix = 0
        for f in ep["frames"]:
            r = f["row"]
            pkt = pending.pop(0) if pending else b.observe()
            assert pkt.observation_id == r["observation"], (name, pkt.observation_id, r)
            err = np.linalg.norm(np.array(pkt.robot_pose[0]) - r["pose"])
            assert err < 0.001, (name, err)
            pixels = np.asarray(
                Image.open(ROOT / name / r["observation"] / "rgb.png").convert("RGB")
            )
            mae = np.abs(pkt.rgb.astype(float) - pixels).mean()
            maxpix = max(maxpix, mae)
            assert mae < 1.5, (name, r["observation"], mae)
            end = b.marks[pkt.observation_id]
            action = r.get("action")
            if action in ("EXPLORE_FRONTIER", "APPROACH_TARGET"):
                pending.clear()
                if r.get("frontier"):
                    xy = next(
                        x["xy"] for x in f["frontiers"] if x["id"] == r["frontier"]
                    )
                    dest = np.array([xy[0], rows[0]["pose"][1], xy[1]])
                else:
                    xy = next(o["xy"] for o in f["objects"] if o["id"] == r["target"])
                    dest = b._snap(np.array([xy[0], rows[0]["pose"][1], xy[1]]))
                result = b.navigate_to(dest)
                err = np.linalg.norm(
                    np.asarray(result.final_pose) - r["execution"]["final_pose"]
                )
                maxerr = max(maxerr, err)
                assert (
                    result.status == r["execution"]["status"]
                    and err < 0.001
                    and abs(result.path_length - r["execution"]["path_length"]) < 0.001
                ), (name, r["observation"], err)
                end = len(b.steps)
                if (
                    action == "APPROACH_TARGET"
                    and result.status == "SUCCEEDED"
                    and not ep["planner_at_arrival"]
                ):
                    b.phase = "review"
                    b.face_towards(xy)
                    pending = b.scan()
            elif r.get("operation") == "BACKEND_SCAN" and not pending:
                pending = b.scan()
        marks = [b.marks[r["observation"]] for r in rows] + [end]
        assert marks == sorted(marks)
        episodes[name] = [b.steps[a:z] for a, z in zip(marks, marks[1:])]
        errors = []
        for row, segment in zip(rows, episodes[name]):
            prev = np.asarray(row["pose"])[[0, 2]]
            for step in segment:
                xy = np.asarray(step["p"])[[0, 2]]
                delta = xy - prev
                if step["kind"] == "move" and np.linalg.norm(delta) > 1.0e-5:
                    direction = delta / np.linalg.norm(delta)
                    heading = np.array(step["h"]) / np.linalg.norm(step["h"])
                    errors.append(
                        math.degrees(
                            math.acos(np.clip(np.dot(direction, heading), -1, 1))
                        )
                    )
                    step["heading_error"] = errors[-1]
                prev = xy
        if ep["align_with_travel"] and errors:
            assert max(errors) <= 5.01, (name, max(errors))
        checks.append(
            dict(
                moving_steps=len(errors),
                max_heading_error=max(errors, default=None),
                mean_heading_error=np.mean(errors) if errors else None,
                episode=name,
                steps=sum(map(len, episodes[name])),
                max_final_pose_error=maxerr,
                max_rgb_mae=maxpix,
            )
        )
        print(checks[-1], flush=True)
    finally:
        b.close()
Path("motion.json").write_text(
    json.dumps(dict(assets=assets, episodes=episodes), separators=(",", ":"))
)
Path("motion-checks.json").write_text(json.dumps(checks, indent=2))
