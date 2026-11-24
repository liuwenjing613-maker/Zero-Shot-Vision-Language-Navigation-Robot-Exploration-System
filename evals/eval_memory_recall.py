"""Matched Habitat revisit trial from a completed, verified episode.

The shared detour is a scripted task waypoint. The planner chooses both return actions.
Only the fresh-camera M2 ablation and the separately labelled path-gate test alter inputs.
"""
import argparse
import base64
import io
from pathlib import Path
import json
import math
import numpy as np
import quaternion
from PIL import Image
import eval_tls_hyf_episode as episode


def read(path):
    return json.loads(path.read_text())


def thumbnail(path):
    buffer = io.BytesIO()
    with Image.open(path) as im:
        im.thumbnail((480, 360))
        im.save(buffer, "WEBP", quality=80)
    return "data:image/webp;base64," + base64.b64encode(buffer.getvalue()).decode()


class TrialBackend(episode.HabitatBackend):
    def _set_position(self, position, **kwargs):
        super()._set_position(position, **kwargs)
        if hasattr(self, "track"):
            self.track.append(self._position().tolist())


def main(source, dataset, output, gpu, replay_perception):
    trace = read(source / "trace.json")
    rows = trace["actions"]
    last = rows[-1]
    assert trace["terminal"] == "SUCCESS" and last["stop_status"] == "SUCCEEDED"
    target = last["initial_target"]
    assert any(
        v["object"] == target and v["status"] == "VERIFIED" for v in last["verdicts"]
    )
    parsed = read(source / "goal.json")
    task = trace["task"]
    goal = episode.goal_spec_from_lwj(
        parsed, task_id=task, search_budget={"max_exploration_steps": 30}
    )
    output.mkdir(parents=True, exist_ok=False)
    episode.write(output / "goal.json", parsed)
    stage = dataset / trace["scene"] / (trace["scene"] + ".glb")

    def sensor(name, kind, resolution, height):
        s = episode.hs.CameraSensorSpec()
        s.uuid, s.sensor_type, s.resolution = (
            name,
            kind,
            episode.mn.Vector2i(*resolution),
        )
        s.hfov, s.position = episode.mn.Deg(90), episode.mn.Vector3(0, height, 0)
        return s

    laser = sensor("scan_depth", episode.hs.SensorType.DEPTH, (3, 1025), 0.45)
    audit = sensor("trial_sem", episode.hs.SensorType.SEMANTIC, (480, 640), 1.05)
    b = TrialBackend(
        str(stage),
        start_position=last["pose"],
        camera_offset=1.05,
        enable_physics=False,
        scene_dataset=str(dataset / "mp3d.scene_dataset_config.json"),
        navmesh=str(stage.with_suffix(".navmesh")),
        gpu_device_id=gpu,
        extra_sensors=(laser, audit),
    )
    try:
        floor = float(last["pose"][1])
        lo, hi = [np.asarray(v)[[0, 2]] for v in b._sim.pathfinder.get_bounds()]
        origin = np.floor(lo - 7)
        cfg = dict(
            shape=np.ceil((hi + 7 - origin)[::-1] / 0.08).astype(int),
            origin=origin,
            resolution=0.08,
            radius=b._agent.agent_config.radius,
            camera=[640, 480, 90],
            session=task,
            episode_path=output / "events.jsonl",
        )
        memory, preview = episode.OnlineMemory(cfg), episode.new_grid(cfg)
        explorer = episode.FrontierExplorer(
            episode.ViewpointConfig(
                sample_radius=1, sensor_range=2, num_rays=16, max_candidates=60
            )
        )

        def integrate(n, ident, xyz, data, scene):
            xy = [xyz[0], xyz[2]]
            frame = dict(
                session=task,
                step=n,
                observation_id=ident,
                xy=xy,
                action="start" if n == 0 else "observe",
            )
            episode.fuse(preview, frame, data["ranges"])
            snap = explorer.detect(
                preview.grid(),
                0.08,
                tuple(origin),
                preview.cell(xy),
                preview.map_version,
                floor_y=floor,
            )
            frame["frontier_snapshot"] = snap
            update = memory.apply(frame, data, scene)
            m3 = memory.exploration.m3
            return frame, update, explorer.rank(snap, m3.records, m3.last[1])

        for n, row in enumerate(rows):
            folder = source / row["observation"]
            with np.load(folder / "sensors.npz") as data:
                frame, update, ranked = integrate(
                    n,
                    row["observation"],
                    row["pose"],
                    data,
                    read(folder / "perception.json"),
                )
                rotation = quaternion.from_rotation_matrix(data["camera_pose"][:3, :3])
                if n == 0:
                    entry_rotation = rotation
            for verdict in row["verdicts"]:
                evidence = memory.apply_verification(
                    {k: v for k, v in verdict.items() if k != "object"}, n
                )
                assert evidence.object_id == verdict["object"]
            if row.get("frontier") and row.get("execution"):
                record = row["execution"]
                key = row["frontier"]
                for state in (
                    "selected",
                    "started",
                    {"SUCCEEDED": "reached", "CANCELED": "cancelled"}.get(
                        record["status"], "failed"
                    ),
                ):
                    memory.exploration.m3.transition(key, state, record["action_id"])

        def orient(rot):
            state = b._agent.get_state()
            state.rotation = rot
            b._agent.set_state(state)

        orient(rotation)
        client = episode.VLMClient(episode.HOST + "/v1", episode.MODEL, timeout=180)
        vision = episode.FastVision()

        def capture(name):
            pkt = b.observe()
            xyz, rot = pkt.robot_pose
            matrix = np.eye(4)
            matrix[:3, :3] = quaternion.as_rotation_matrix(
                quaternion.from_float_array(rot)
            )
            matrix[:3, 3] = np.asarray(xyz) + [0, 1.05, 0]
            state = b._agent.get_state()
            try:
                ranges = episode.scan(b._sim, b._agent, np.asarray(xyz), 0.0)
            finally:
                b._agent.set_state(
                    state, reset_sensors=False, infer_sensor_states=False
                )
            path = output / (name + ".png")
            Image.fromarray(pkt.rgb).save(path)
            if replay_perception:
                if read(replay_perception / "goal.json") != parsed:
                    raise ValueError(
                        "Cached perception belongs to a different instruction"
                    )
                with Image.open(replay_perception / path.name) as previous:
                    np.testing.assert_array_equal(np.asarray(previous), pkt.rgb)
                scene = read(replay_perception / (name + ".json"))
                if scene["task_id"] != task or scene["observation_id"] != name:
                    raise ValueError(
                        "Cached perception does not belong to this task and frame"
                    )
            else:
                scene = vision.perceive_scene(
                    str(path),
                    parsed,
                    task_id=task,
                    observation_id=name,
                    view_id="front",
                )
            episode.write(output / (name + ".json"), scene)
            return (
                path,
                xyz,
                dict(camera_pose=matrix, ranges=ranges, depth=pkt.depth),
                scene,
            )

        def instance(local, scene):
            box = next(
                e["bbox"] for e in scene["target_entities"] if e["local_id"] == local
            )
            sem = b._sim.get_sensor_observations()["trial_sem"]
            x, y = np.clip(
                np.asarray(
                    [(box[0] + box[2]) / 2000 * 640, (box[1] + box[3]) / 2000 * 480],
                    int,
                ),
                [0, 0],
                [639, 479],
            )
            return int(sem[y, x])

        original_scene = read(source / last["observation"] / "perception.json")
        orig_id = instance(
            next(v["candidate_id"] for v in last["verdicts"] if v["object"] == target),
            original_scene,
        )
        b.track = [b._position().tolist()]
        detour = b.navigate_to(rows[0]["pose"])
        assert str(detour.status) == "SUCCEEDED"
        out_path = b.track
        orient(entry_rotation)
        setup_views = []
        for i in range(b.scan_views):
            name = "recall_query_" + str(i)
            photo, xyz, data, scene = capture(name)
            setup_views.append(scene["target_status"])
            if scene["target_status"] == "NOT_VISIBLE" and not scene["target_entities"]:
                break
            b._rotate(360 / b.scan_views)
        else:
            raise ValueError("No target-invisible query found at the entry waypoint")
        query_instance_pixels = int(
            np.count_nonzero(b._sim.get_sensor_observations()["trial_sem"] == orig_id)
        )
        query_state = b._agent.get_state()
        frame, full, ranked = integrate(len(rows), name, xyz, data, scene)
        freshmem = episode.OnlineMemory(dict(cfg, session=None))
        current = freshmem.apply(dict(frame, step=0, action="start"), data, scene)[
            "object_memory"
        ]

        def choose(update, paths):
            ctx = episode.planner_context_from_team_outputs(
                goal,
                update,
                ranked,
                robot_pose=xyz,
                floor_y=floor,
                reachable_object_ids=paths,
                perception=episode.perception_result_from_lwj(scene),
                exploration_steps=sum(bool(r.get("frontier")) for r in rows),
            )
            d = episode.HighLevelPlanner().step(ctx)
            return d, dict(
                reason=d.reason,
                evidence_ids=list(d.evidence_ids),
                action=d.action_type.value,
                target=d.target_id,
                frontier=d.frontier_id,
                object_count=len(ctx.world_state.objects),
            )

        def branch(update):
            paths = episode.reachable(b, update, floor)
            decision, result = choose(update, paths)
            dispatch = episode.plan_tls_dispatch(decision)
            positions = {**paths, **{f.frontier_id: f.safe_viewpoint for f in ranked}}
            b.track = [b._position().tolist()]
            if dispatch.operation.value == "TASK_EXECUTOR":
                run = episode.execution_result_from_tls(
                    episode.TaskExecutor(
                        b, lambda kind, key: positions.get(key)
                    ).execute(dispatch.executor_payload)
                )
                result.update(
                    status=run.status.value, length=run.path_length, path=b.track
                )
            return result

        positions = episode.reachable(b, full, floor)
        _, gate = choose(full, positions.keys() - {target})
        ablated = branch(dict(full, object_memory=current))
        b._agent.set_state(query_state)
        retained = branch(full)
        coord = full["object_memory"]["objects"][target]["map_position_xz"]
        report = dict(
            source=source.name,
            target=target,
            instance_before=orig_id,
            history_box=next(
                entity["bbox"]
                for entity in original_scene["target_entities"]
                if entity["local_id"]
                == next(
                    v["candidate_id"]
                    for v in last["verdicts"]
                    if v["object"] == target and v["status"] == "VERIFIED"
                )
            ),
            source_frame=last["observation"],
            history_image=thumbnail(source / last["observation"] / "rgb.png"),
            query_image=thumbnail(photo),
            visibility=scene["target_status"],
            setup_views=setup_views,
            query_instance_pixels=query_instance_pixels,
            query_pose=xyz,
            goal_xz=coord,
            frontiers=len(ranked),
            verification_frames=[
                v["observation_id"]
                for v in full["object_memory"]["verification"]
                if v["object_id"] == target and v["status"] == "VERIFIED"
            ],
            detour_path=out_path,
            detour_m=detour.path_length,
            full=retained,
            current=ablated,
            path_gate=gate,
            return_checks=[],
            same_instance=False,
        )
        if (
            retained["action"] == "APPROACH_TARGET"
            and retained["target"] == target
            and retained.get("status") == "SUCCEEDED"
        ):
            b.face_towards(coord)
            for i in range(b.scan_views):
                name = "return_" + str(i)
                path, end, arr, view = capture(name)
                _, updated, _ = integrate(len(rows) + 1 + i, name, end, arr, view)
                check = dict(view=name, visibility=view["target_status"], verdicts=[])
                check["candidates"] = [
                    dict(
                        bbox=r["bbox"],
                        local=r["local_id"],
                        associated_object=r["object_id"],
                        association=r["association_status"],
                        instance=instance(r["local_id"], view),
                    )
                    for r in updated["object_memory"]["observations"].values()
                    if r["perception_role"] == "target"
                ]
                check["image"] = thumbnail(path)
                for record in updated["object_memory"]["observations"].values():
                    if (
                        record["perception_role"] == "target"
                        and record["object_id"] == target
                    ):
                        verdict = client.verify_target(
                            str(path), parsed, view, record["local_id"]
                        )
                        memory.apply_verification(verdict, len(rows) + 1 + i)
                        distance = math.dist(
                            [end[0], end[2]],
                            updated["object_memory"]["objects"][target][
                                "map_position_xz"
                            ],
                        )
                        check["verdicts"].append(
                            dict(
                                status=verdict["status"],
                                instance=instance(record["local_id"], view),
                                distance_m=distance,
                            )
                        )
                        if verdict["status"] == "VERIFIED" and distance <= 1.0:
                            report["stop"] = str(b.stop().status)
                            report["same_instance"] = (
                                check["verdicts"][-1]["instance"] == orig_id
                            )
                            break
                report["return_checks"].append(check)
                if i == 0 or report.get("stop"):
                    report["return_image"] = check["image"]
                if report.get("stop"):
                    break
                b._rotate(360 / b.scan_views)
        episode.write(output / "report.json", report)
        print(
            {
                k: report[k]
                for k in (
                    "target",
                    "visibility",
                    "frontiers",
                    "same_instance",
                )
            },
            flush=True,
        )
    finally:
        b.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "dataset", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--replay-perception",
        type=Path,
        help="Use prior VLM scene outputs after verifying instruction, task, and exact RGB pixels",
    )
    main(**vars(parser.parse_args()))
