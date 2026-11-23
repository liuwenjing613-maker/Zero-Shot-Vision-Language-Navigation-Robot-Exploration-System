"""Exercise real Habitat actions; use MP3D polygons only for offline scoring."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import quaternion
from PIL import Image, ImageDraw

from fixture import H, W, HFOV, SCENE, inside, orient, regions, simulator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "m1_lab"))
from scan_mp3d import scan, N, LIMIT
from vln_memory.geometry_memory import Intrinsics, OccupancyMemory
from vln_memory.topology_memory import TopologyMemory
from vln_memory.portal_memory import RegionEntryDetector


def walk(sim, folder, index, output, repeat=False):
    fixture = json.loads((folder / "fixtures.json").read_text())[index]
    frame = folder / f"view-{index}.png"
    inference = json.loads((folder / f"qwen-{index}.json").read_text())
    assert hashlib.sha256(frame.read_bytes()).hexdigest() == inference["rgb_sha256"]
    cues = inference["result"]
    K = Intrinsics.from_hfov(W, H, HFOV)
    start = np.array(fixture["start"])
    direction = np.array(fixture["normal"])
    grid = OccupancyMemory((180, 180), start[[0, 2]] - 4.5, 0.05)
    graph = TopologyMemory()
    detector = RegionEntryDetector(graph, robot_radius=sim.config.agents[0].radius)
    polys = regions()
    agent = sim.initialize_agent(0)
    orient(agent, start, direction)
    trace = []
    events = []
    reports = []
    first_capture = {}

    def observe(phase):
        step = len(trace)
        state = agent.get_state()
        position = np.array(state.position)
        xy = position[[0, 2]]
        obs = sim.get_sensor_observations()
        rgb = Image.fromarray(obs["rgb"][:, :, :3])
        depth = obs["depth"].copy()
        ranges = scan(sim, agent, position, 0.0)
        agent.set_state(state, reset_sensors=False, infer_sensor_states=False)
        grid.update_scan(
            ranges,
            0.0,
            2 * np.pi / N,
            xy,
            0.0,
            xy,
            step,
            range_min=0.05,
            range_max=LIMIT,
        )
        rgb.save(output / f"{index}-rgb-{step}.jpg", quality=85)
        occupancy = grid.grid()
        palette = np.full((*occupancy.shape, 3), [158, 165, 169], dtype=np.uint8)
        palette[occupancy == 0] = [245, 245, 235]
        palette[occupancy == 100] = [28, 34, 39]
        Image.fromarray(palette).save(output / f"{index}-grid-{step}.png")
        sensor = state.sensor_states["depth"]
        transform = np.eye(4)
        transform[:3, :3] = quaternion.as_rotation_matrix(sensor.rotation)
        transform[:3, 3] = sensor.position
        if step == 0:
            np.savez_compressed(
                folder / f"geometry-{index}.npz",
                depth=depth,
                transform=transform,
                odds=grid.log_odds,
                origin=[grid.origin_x, grid.origin_y],
            )
        obsid = f"f{index}-{step}"
        if step == 0:
            assert np.array_equal(
                np.asarray(rgb), np.asarray(Image.open(frame))
            ), "Qwen frame mismatch"
            detector.start(xy, step, obsid)
            reports.extend(detector.perceive(cues, depth, K, transform, grid))
            first_capture.update(depth=depth, transform=transform)
            selected_rgb, selected_cues = rgb.copy(), cues
            draw = ImageDraw.Draw(selected_rgb)
            for cue in selected_cues["visual_cues"]:
                if cue["cue_type"] in (
                    "opening",
                    "doorway",
                    "blocked_passage",
                    "corridor",
                ):
                    bounds = np.array(cue["bbox"]) * [W, H, W, H] / 1000
                    draw.rectangle(bounds.tolist(), outline="orange", width=3)
            selected_rgb.save(output / f"{index}-start.jpg")
            event = None
        else:
            if step == 4:
                # The approach-retreat supplied more scan coverage. Recheck pending proposals
                # using their original RGB-D/pose, before attempting the actual traversal.
                prior = {
                    (r["observation_id"], r["cue_id"], r["status"]) for r in reports
                }
                for r in detector.perceive(
                    cues,
                    first_capture["depth"],
                    K,
                    first_capture["transform"],
                    grid,
                    base_xy=xy,
                ):
                    if (r["observation_id"], r["cue_id"], r["status"]) not in prior:
                        reports.append(dict(r, step=step))
            event = detector.advance(xy, step, grid, obsid)
        memberships = [
            k for k in (fixture["from"], fixture["to"]) if inside(xy, polys[k][1])
        ]
        row = dict(
            step=step,
            phase=phase,
            xy=xy.tolist(),
            regions=memberships,
            event=event,
            area=detector.area,
            known_portals=[
                q.portal_id
                for q in detector.portals.values()
                if q.status in ("VALIDATED", "CROSSED")
            ],
        )
        if event:
            rgb.save(output / f"{index}-{step}.jpg")
            events.append(row)
        trace.append(row)

    observe("start")
    schedule = [
        ("approach", 4, 1),
        ("retreat", 4, -1),
        ("outbound", 10, 1),
        ("return", 10, -1),
    ]
    if repeat:
        schedule.extend([("outbound_again", 10, 1), ("return_again", 10, -1)])
    for phase, count, sign in schedule:
        orient(agent, agent.get_state().position, direction * sign)
        for _ in range(count):
            previous = np.array(agent.get_state().position)
            sim.step("move_forward")
            now = np.array(agent.get_state().position)
            if (now - previous)[[0, 2]] @ (sign * direction) < 0.22:
                raise AssertionError(f"Motion blocked at {phase} {now}")
            observe(phase)

    # Score only after all predictions have been produced.
    def inside_margin(row, sign):
        return sign * (np.array(row["xy"]) - fixture["center"]) @ direction >= 0.10

    criteria = dict(
        negative=not any(r["phase"] in ("approach", "retreat") for r in events),
        outbound=sum(r["phase"] == "outbound" for r in events) == 1
        and any(
            r["phase"] == "outbound"
            and r["regions"] == [fixture["to"]]
            and inside_margin(r, 1)
            for r in events
        ),
        inbound=sum(r["phase"] == "return" for r in events) == 1
        and any(
            r["phase"] == "return"
            and r["regions"] == [fixture["from"]]
            and r["area"] == trace[0]["area"]
            and inside_margin(r, -1)
            for r in events
        ),
        same_portal=len(events) == (4 if repeat else 2)
        and len({r["event"]["portal_id"] for r in events}) == 1,
    )
    if repeat:
        criteria["repeat_reuses_places"] = len(events) == 4 and all(
            events[a]["event"]["place"] == events[b]["event"]["place"]
            and events[a]["area"] == events[b]["area"]
            and events[a]["regions"] == events[b]["regions"]
            and inside_margin(events[b], sign)
            for a, b, sign in ((0, 2, 1), (1, 3, -1))
        )
    portals = [
        dict(id=p.portal_id, a=p.a.tolist(), b=p.b.tolist(), width=p.width)
        for p in detector.portals.values()
    ]
    vals = grid.grid()
    pixels = np.full((*vals.shape, 3), [158, 165, 169], dtype=np.uint8)
    pixels[vals == 0] = [245, 245, 235]
    pixels[vals == 100] = [28, 34, 39]

    def pt(xy):
        return tuple(
            (
                np.rint(
                    (np.asarray(xy) - np.array([grid.origin_x, grid.origin_y]))
                    / grid.resolution
                    * 4
                )
            ).astype(int)
        )

    im = Image.fromarray(pixels).resize((720, 720), Image.Resampling.NEAREST)
    d = ImageDraw.Draw(im)
    for p in portals:
        d.line([pt(p["a"]), pt(p["b"])], fill="#b74000", width=6)
    d.line([pt(r["xy"]) for r in trace], fill="#367fad", width=4)
    for r in [trace[0], *events]:
        x, y = pt(r["xy"])
        d.ellipse([x - 6, y - 6, x + 6, y + 6], fill="green")
    im.save(output / f"{index}-map.png")
    result = dict(
        scene=SCENE,
        index=index,
        fixture=fixture,
        origin=[grid.origin_x, grid.origin_y],
        perception=cues,
        proposals=reports,
        portals=portals,
        trace=trace,
        criteria=criteria,
        places=[asdict(p) for p in graph.places.values()],
        edges=[asdict(e) for e in graph.edges.values()],
    )
    (output / f"{index}.json").write_text(json.dumps(result, indent=2))
    print(
        index,
        reports,
        criteria,
        [(r["step"], r["regions"]) for r in events],
        flush=True,
    )
    return all(criteria.values())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("folder", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("indices", type=int, nargs="+")
    p.add_argument("--gpu", type=int, default=3)
    p.add_argument("--repeat", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with simulator(args.gpu) as sim:
        passed = [
            walk(sim, args.folder, i, args.output, args.repeat) for i in args.indices
        ]
    raise SystemExit(0 if all(passed) else 1)
