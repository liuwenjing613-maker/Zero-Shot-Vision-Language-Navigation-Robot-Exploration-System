"""Run a bounded chair-search episode on the repository's MP3D dining-room fixture."""
import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
import habitat_sim as hs
import magnum as mn
import numpy as np
import quaternion
from habitat_sim.utils.common import quat_from_angle_axis
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
for part in ("src", "code", "evals/live", "evals/region_entry", "evals/m1_lab"):
    sys.path.insert(0, str(ROOT / part))
from execution import TaskExecutor
from execution.habitat_backend import HabitatBackend
from exploration.explorer import FrontierExplorer
from exploration.viewpoint import ViewpointConfig
from fast_vision import FastVision
from model import HOST, MODEL
from scan_mp3d import scan
from v4.decision import HighLevelPlanner, planner_context_from_team_outputs
from v4.decision.adapters import goal_spec_from_lwj, perception_result_from_lwj
from v4.decision.execution_bridge import plan_tls_dispatch, execution_result_from_tls
from vln_memory.online_memory import OnlineMemory, new_grid, fuse
from vln_perception import VLMClient


def write(path, value):
    def native(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, set):
            return sorted(x)
        raise TypeError(type(x).__name__)

    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=native))


def reachable(backend, update, floor):
    positions = {}
    for key, obj in update["object_memory"]["objects"].items():
        xz = obj.get("map_position_xz")
        if xz is None:
            continue
        target = np.array([xz[0], floor, xz[1]])
        pose = backend._snap(target)
        if (
            pose is not None
            and np.linalg.norm(target - pose) <= 0.65
            and abs(pose[1] - floor) <= 0.25
        ):
            if backend._plan(pose) is not None:
                positions[key] = tuple(pose)
    return positions


def run(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    task = out.name
    client = VLMClient(HOST + "/v1", MODEL, timeout=180)
    parsed = client.parse_goal(args.instruction)
    write(out / "goal.json", parsed)
    goal = goal_spec_from_lwj(
        parsed,
        task_id=task,
        search_budget={
            "max_exploration_steps": args.exploration_budget,
            "max_navigation_failures": 5,
        },
    )
    vision = FastVision()
    start = json.loads(
        (ROOT / "results/verify-e02-new-view/e02-view-comparison.json").read_text()
    )["old_view"]
    laser = hs.CameraSensorSpec()
    laser.uuid, laser.sensor_type = "scan_depth", hs.SensorType.DEPTH
    laser.resolution, laser.hfov, laser.position = (
        mn.Vector2i(3, 1025),
        mn.Deg(90),
        mn.Vector3(0, 0.45, 0),
    )
    scene_path = args.dataset / "2azQ1b91cZZ/2azQ1b91cZZ.glb"
    b = HabitatBackend(
        str(scene_path),
        camera_offset=1.05,
        start_position=args.start_position or start["position_m"],
        enable_physics=False,
        scene_dataset=str(args.dataset / "mp3d.scene_dataset_config.json"),
        navmesh=str(scene_path.with_suffix(".navmesh")),
        gpu_device_id=args.gpu,
        extra_sensors=(laser,),
    )
    trace = {
        "terminal": "FRAME_LIMIT",
        "actions": [],
        "scene": scene_path.stem,
        "align_with_travel": b.align_with_travel,
        "planner_at_arrival": args.planner_at_arrival,
        "yaw_offset": args.yaw_offset,
        "task": task,
    }
    try:
        state = b._agent.get_state()
        state.rotation = quat_from_angle_axis(
            math.radians(start["yaw_deg"] + args.yaw_offset), np.array([0, 1, 0])
        )
        b._agent.set_state(state)
        floor = float(state.position[1])
        lo, hi = [np.asarray(x)[[0, 2]] for x in b._sim.pathfinder.get_bounds()]
        origin = np.floor(lo - 7)
        cfg = dict(
            shape=np.ceil((hi + 7 - origin)[::-1] / 0.08).astype(int),
            origin=origin,
            resolution=0.08,
            radius=b._agent.agent_config.radius,
            camera=[640, 480, 90],
            session=task,
            episode_path=out / "events.jsonl",
        )
        memory, preview = OnlineMemory(cfg), new_grid(cfg)
        explorer = FrontierExplorer(
            ViewpointConfig(
                sample_radius=1, sensor_range=2, num_rays=16, max_candidates=60
            )
        )
        planner = HighLevelPlanner()
        positions = {}
        executor = TaskExecutor(b, lambda kind, key: positions.get(key))
        prior, approached = None, None
        failures = 0
        frontier_attempts = 0
        pending = []
        for n in range(args.frames):
            packet = pending.pop(0) if pending else b.observe()
            xyz, rot = packet.robot_pose
            xy = [xyz[0], xyz[2]]
            frame = dict(
                session=task,
                step=n,
                observation_id=packet.observation_id,
                xy=xy,
                action="start" if n == 0 else "observe",
            )
            state = b._agent.get_state()
            try:
                ranges = scan(b._sim, b._agent, np.asarray(xyz), 0.0)
            finally:
                b._agent.set_state(
                    state, reset_sensors=False, infer_sensor_states=False
                )
            transform = np.eye(4)
            transform[:3, :3] = quaternion.as_rotation_matrix(
                quaternion.from_float_array(rot)
            )
            transform[:3, 3] = np.asarray(xyz) + [0, 1.05, 0]
            folder = out / packet.observation_id
            folder.mkdir()
            image = folder / "rgb.png"
            Image.fromarray(packet.rgb).save(image)
            np.savez_compressed(
                folder / "sensors.npz",
                depth=packet.depth,
                ranges=ranges,
                camera_pose=transform,
            )
            scene = vision.perceive_scene(
                str(image),
                parsed,
                task_id=task,
                observation_id=packet.observation_id,
                view_id="front",
            )
            write(folder / "perception.json", scene)
            fuse(preview, frame, ranges)
            snap = explorer.detect(
                preview.grid(),
                0.08,
                tuple(origin),
                preview.cell(xy),
                preview.map_version,
                floor_y=floor,
            )
            frame["frontier_snapshot"] = snap
            update = memory.apply(
                frame,
                dict(depth=packet.depth, ranges=ranges, camera_pose=transform),
                scene,
            )
            fm = memory.exploration.m3
            ranked = explorer.rank(snap, fm.records, fm.last[1])
            objects = reachable(b, update, floor)
            positions.clear()
            positions.update(objects)
            positions.update({f.frontier_id: f.safe_viewpoint for f in ranked})

            def context():
                return planner_context_from_team_outputs(
                    goal,
                    update,
                    ranked,
                    robot_pose=xyz,
                    floor_y=floor,
                    perception=perception_result_from_lwj(scene),
                    reachable_object_ids=objects,
                    last_execution_result=prior,
                    exploration_steps=frontier_attempts,
                    navigation_failures=failures,
                )

            first = planner.step(context())
            row = dict(
                frame=n,
                observation=packet.observation_id,
                map_version=update["map_version"],
                pose=xyz,
                frontiers=len(ranked),
                objects=len(update["object_memory"]["objects"]),
                initial_reason=first.reason,
                initial_evidence_ids=list(first.evidence_ids),
                initial_action=first.action_type.value,
                initial_target=first.target_id,
                verdicts=[],
            )
            trace["actions"].append(row)
            write(out / "trace.json", trace)
            print(
                n,
                packet.observation_id,
                first.action_type.value,
                first.target_id,
                flush=True,
            )
            fresh = set()
            for item in update["object_memory"]["observations"].values():
                key = item["object_id"]
                if (
                    not key
                    or item["perception_role"] != "target"
                    or key not in {first.target_id, approached}
                ):
                    continue
                verdict = client.verify_target(
                    str(image), parsed, scene, item["local_id"]
                )
                evidence = memory.apply_verification(verdict, n)
                row["verdicts"].append(dict(object=key, **verdict))
                if evidence.status == "VERIFIED":
                    fresh.add(key)
            update["object_memory"]["verification"] = [
                asdict(v) for v in memory.objects.verifications
            ]
            write(folder / "memory.json", update)
            if approached and approached in fresh and approached in objects:
                distance = math.dist(
                    xy,
                    update["object_memory"]["objects"][approached]["map_position_xz"],
                )
                row["distance_m"] = distance
                if distance <= 1.0:
                    row["stop_status"] = str(b.stop().status)
                    trace["terminal"] = (
                        "SUCCESS"
                        if row["stop_status"] == "SUCCEEDED"
                        else "STOP_FAILED"
                    )
                    break
            if approached and not args.planner_at_arrival:
                row["action"] = "VERIFY_AT_ARRIVAL"
                row["target"] = approached
                if not pending:
                    trace["terminal"] = "ARRIVAL_UNVERIFIED"
                    b.stop()
                    break
                continue
            decision = planner.step(context()) if row["verdicts"] else first
            dispatch = plan_tls_dispatch(decision)
            row.update(
                reason=decision.reason,
                evidence_ids=list(decision.evidence_ids),
                action=decision.action_type.value,
                operation=dispatch.operation.value,
                target=decision.target_id,
                frontier=decision.frontier_id,
            )
            if not planner.is_fresh(
                decision,
                map_version=memory.grid.map_version,
                observation_id=packet.observation_id,
            ):
                raise RuntimeError("Stale decision")
            if dispatch.operation.value == "TASK_EXECUTOR":
                pending.clear()
                if decision.frontier_id is not None:
                    fm.transition(
                        decision.frontier_id, "selected", decision.decision_id
                    )
                    fm.transition(decision.frontier_id, "started", decision.decision_id)
                prior = execution_result_from_tls(
                    executor.execute(dispatch.executor_payload)
                )
                row["execution"] = asdict(prior)
                if decision.frontier_id is not None:
                    frontier_attempts += 1
                    operation = {"SUCCEEDED": "reached", "CANCELED": "cancelled"}.get(
                        prior.status.value, "failed"
                    )
                    fm.transition(
                        decision.frontier_id,
                        operation,
                        decision.decision_id,
                        prior.error_code or prior.status.value,
                    )
                memory.exploration.m4.record(
                    decision.decision_id,
                    "EXECUTION_" + prior.status.value,
                    "executor",
                    n,
                    task_id=task,
                    observation_id=packet.observation_id,
                    action_id=decision.decision_id,
                    map_version=memory.grid.map_version,
                    details=dict(
                        action=decision.action_type.value,
                        target=decision.target_id,
                        frontier=decision.frontier_id,
                    ),
                )
                failures += int(prior.status.value in {"FAILED", "TIMEOUT"})
                if (
                    decision.action_type.value == "APPROACH_TARGET"
                    and prior.status.value == "SUCCEEDED"
                ):
                    approached = decision.target_id
                    if not args.planner_at_arrival:
                        estimate = update["object_memory"]["objects"][approached][
                            "map_position_xz"
                        ]
                        b.face_towards(estimate)
                        pending = b.scan()
                if decision.action_type.value == "STOP":
                    row["reason"] = decision.reason
                    trace["terminal"] = (
                        "SEARCH_EXHAUSTED"
                        if prior.status.value == "SUCCEEDED"
                        else "STOP_FAILED"
                    )
                    break
            elif dispatch.operation.value == "BACKEND_SCAN":
                if not pending:
                    pending = b.scan()
            elif dispatch.operation.value == "BACKEND_CANCEL":
                pending.clear()
                b.cancel_navigation()
                prior = None
            else:
                trace["terminal"] = dispatch.operation.value
                break
            write(out / "trace.json", trace)
    except Exception as exc:
        trace.update(terminal="ERROR", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write(out / "trace.json", trace)
        b.close()
    print(trace["terminal"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument(
        "--dataset",
        type=Path,
        default=Path("/data/chenkejun/datasets/mp3d_habitat/scenes"),
    )
    p.add_argument(
        "--start-position",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Alternative navigable starting position in this MP3D scene.",
    )
    p.add_argument("--gpu", type=int, default=7)
    p.add_argument("--exploration-budget", type=int, default=30)
    p.add_argument(
        "--yaw-offset",
        type=float,
        default=0,
        help="Degrees from the recorded starting heading.",
    )
    p.add_argument("--frames", type=int, default=8)
    p.add_argument(
        "--planner-at-arrival",
        action="store_true",
        help="Continue using the upstream planner after an approach to reproduce its termination behavior.",
    )
    p.add_argument("--instruction", default="Find a chair.")
    run(p.parse_args())
