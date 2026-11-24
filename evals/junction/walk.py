"""Run real robot actions, feed only scan/pose to M1, then score against held-out navmesh sectors."""
import json, sys, itertools
from dataclasses import asdict
from pathlib import Path
import numpy as np
from PIL import Image
from select_routes import simulator, gap
from fixture import orient, ROOT

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "m1_lab"))
from scan_mp3d import scan, N, LIMIT
from render_topdown import render
from vln_memory.geometry_memory import OccupancyMemory
from vln_memory.topology_memory import TopologyMemory
from vln_memory import JunctionDetector


def run(sim, kind, f, out):
    center = np.array(f["center"])
    axis = np.array(f["arrival_axis"])
    start = center + 1.5 * axis
    agent = sim.initialize_agent(0)
    pos = np.asarray(sim.pathfinder.snap_point([start[0], f["floor"], start[1]]))
    orient(agent, pos, -axis)
    grid = OccupancyMemory((144, 144), center - 5.76, 0.08)
    graph = TopologyMemory()
    start_place = graph.visit("start", pos[[0, 2]], -1, "start", area_id="A01")
    det = JunctionDetector(graph, sim.config.agents[0].radius)
    trace = []
    current = start_place
    travel = 0.0

    def observe(action):
        nonlocal current, travel
        t = len(trace)
        state = agent.get_state()
        p = np.array(state.position)
        xy = p[[0, 2]]
        rgb = sim.get_sensor_observations()["rgb"][:, :, :3].copy()
        ranges = scan(sim, agent, p, 0)
        agent.set_state(state, reset_sensors=False, infer_sensor_states=False)
        grid.update_scan(
            ranges, 0, 2 * np.pi / N, xy, 0, xy, t, range_min=0.05, range_max=LIMIT
        )
        place = det.update(grid, xy, t, kind + str(t), area_id="A01")
        if trace:
            travel += float(np.linalg.norm(xy - trace[-1]["xy"]))
        if place:
            if current != place:
                graph.record_navigation(
                    "exec:" + kind + str(t), current, place, "succeeded", travel
                )
            current = place
            travel = 0.0
        im = grid.grid()
        colors = np.zeros((*im.shape, 3), dtype=np.uint8)
        colors[im == -1] = [156, 166, 174]
        colors[im == 0] = [246, 245, 234]
        colors[im == 100] = [35, 42, 48]
        colors[det.skeleton] = [113, 167, 177]
        Image.fromarray(colors).save(out / f"grid-{t}.png")
        Image.fromarray(rgb).resize((360, 270)).save(out / f"rgb-{t}.jpg", quality=80)
        candidates = [asdict(c) for c in det.candidates]
        node = asdict(graph.places[place]) if place else None
        trace.append(
            dict(t=t, action=action, xy=xy.tolist(), candidates=candidates, node=node)
        )

    observe("start")
    for sign, action in [(-1, "approach"), (1, "leave"), (-1, "revisit")]:
        orient(agent, agent.get_state().position, axis * sign)
        for _ in range(6):
            before = np.array(agent.get_state().position)
            sim.step("move_forward")
            delta = (np.array(agent.get_state().position) - before)[[0, 2]]
            if delta @ (axis * sign) < 0.22:
                raise RuntimeError("Fixture route obstructed")
            observe(action)
        if sign == -1:
            observe("stop")
            observe("hold")
    events = [row for row in trace if row["node"]]
    positive = len(f["headings"]) >= 3

    def angular_error(node):
        predicted = node["branch_directions"]
        truth = f["headings"]
        if len(predicted) != len(truth):
            return None
        return min(
            max(gap(a, b) for a, b in zip(seq, truth))
            for seq in itertools.permutations(predicted)
        )

    errors = [angular_error(r["node"]) for r in events]
    checks = dict(
        approach=all(
            not r["node"] for r in trace if r["action"] not in ("hold", "stop")
        ),
        event_count=len(events) == (2 if positive else 0),
    )
    if positive:
        checks.update(
            position=len(events) == 2
            and all(np.linalg.norm(r["xy"] - center) < 0.5 for r in events),
            directions=len(errors) == 2
            and all(e is not None and e <= 28 for e in errors),
            revisit=len(events) == 2
            and events[0]["node"]["place_id"] == events[1]["node"]["place_id"]
            and events[1]["node"]["visit_count"] == 2,
        )
    record = dict(
        kind=kind,
        fixture=f,
        origin=[grid.origin_x, grid.origin_y],
        span=11.52,
        trace=trace,
        edges=[asdict(e) for e in graph.edges.values()],
        checks=checks,
        angle_errors=errors,
        places=[asdict(p) for p in graph.places.values()],
    )
    (out / "record.json").write_text(json.dumps(record))
    (out / "window.json").write_text(
        json.dumps(
            dict(
                origin=record["origin"],
                shape=[144, 144],
                resolution=0.08,
                xy=center.tolist(),
            )
        )
    )
    print(
        kind,
        checks,
        "events",
        [(r["t"], r["node"]["branch_directions"]) for r in events],
        "nearest",
        [
            (round(np.linalg.norm(center - j["xy"]), 2), len(j["branches"]))
            for j in trace[8]["candidates"]
            if np.linalg.norm(center - j["xy"]) < 2
        ],
        flush=True,
    )
    return all(checks.values())


if __name__ == "__main__":
    root = Path(sys.argv[1])
    fixtures = json.loads((root / "routes.json").read_text())
    names = sys.argv[2:] or list(fixtures)
    with simulator(3) as sim:
        results = []
        for kind in names:
            out = root / kind
            out.mkdir(exist_ok=True)
            results.append(run(sim, kind, fixtures[kind], out))
    for kind in names:
        out = root / kind
        render(ROOT, out / "window.json", out, 3, floor_y=fixtures[kind]["floor"])
    raise SystemExit(0 if all(results) else 1)
