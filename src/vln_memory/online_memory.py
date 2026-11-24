"""Consume one ordered, frame-matched scan, RGB-D pose and semantic observation."""
from dataclasses import asdict
import math
import numpy as np
from .geometry_memory import OccupancyMemory, Intrinsics
from .exploration_memory import ExplorationMemory
from .object_memory import ObjectMemory
from .junction_memory import JunctionDetector
from .portal_memory import RegionEntryDetector, free_footprint
from .topology_memory import TopologyMemory


def new_grid(config):
    return OccupancyMemory(config["shape"], config["origin"], config["resolution"])


def fuse(grid, frame, ranges):
    xy = frame["xy"]
    previous = grid._last_xy
    path = [previous, xy] if previous is not None else None
    grid.update_scan(
        ranges,
        0.0,
        2 * np.pi / len(ranges),
        xy,
        0.0,
        xy,
        frame["step"],
        range_min=0.05,
        range_max=6.0,
        base_path=path,
    )


class OnlineMemory:
    def __init__(self, config):
        self.grid = new_grid(config)
        self.graph = TopologyMemory()
        self.region = RegionEntryDetector(self.graph, config["radius"])
        self.junction = JunctionDetector(self.graph, config["radius"])
        self.camera = Intrinsics.from_hfov(*config["camera"])
        self.objects = ObjectMemory()
        self.exploration = ExplorationMemory(config["session"], config.get("episode_path")) if config.get("session") else None
        self.last = -1
        self.pending_views = []

    def _arrive(self, place, obs):
        r = self.region
        if r.place != place and r.continuous:
            self.graph.record_navigation(
                "walk:" + obs + ":" + place, r.place, place, "succeeded", r.path_length
            )
        r.place = place
        r.path_length = 0.0
        r.continuous = True

    def apply_verification(self, verdict, timestamp):
        evidence = self.objects.apply_verification(verdict, timestamp)
        if self.exploration:
            self.exploration.m4.record(
                f"{evidence.observation_id}:{evidence.status}:{timestamp}",
                "TARGET_" + evidence.status, "verifier", timestamp,
                task_id=evidence.task_id, observation_id=evidence.observation_id,
                object_id=evidence.object_id, map_version=self.grid.map_version)
        return evidence

    def apply(self, frame, data, perception):
        n, obs, xy = frame["step"], frame["observation_id"], frame["xy"]
        if (
            n != self.last + 1
            or obs != perception["observation_id"]
            or perception["task_id"] != frame["session"]
        ):
            raise ValueError("Perception and motion frame do not match")
        prepared = None
        if perception.get("schema_version") == "perception_three_groups_v1":
            prepared = self.objects.prepare(perception, data["depth"], self.camera,
                                            data["camera_pose"], frame.get("timestamp", n))
        fuse(self.grid, frame, data["ranges"])
        r = self.region
        events = []
        if n == 0:
            r.start(xy, n, obs)
            events.append(dict(reason="START", place=r.place))
        # Consume a previously validated crossing before integrating the new view.
        rechecked = []
        retained = []
        for older, depth, pose, base in self.pending_views:
            delta = math.dist(xy, base)
            if delta < 0.35:
                retained.append((older, depth, pose, base))
            elif delta <= 2.0:
                rechecked.extend(
                    r.perceive(older, depth, self.camera, pose, self.grid, base_xy=xy)
                )
        self.pending_views = retained
        if n:
            crossing = r.advance(xy, n, self.grid, obs)
            if crossing:
                events.append(dict(reason="REGION_ENTRY", **crossing))
        proposals = r.perceive(
            perception, data["depth"], self.camera, data["camera_pose"], self.grid
        )
        if any(
            c["cue_type"] in ("doorway", "opening", "corridor")
            for c in perception["visual_cues"]
        ) and (not proposals or any(p["status"] == "PROPOSED" for p in proposals)):
            self.pending_views = (
                self.pending_views
                + [(perception, data["depth"].copy(), data["camera_pose"].copy(), xy)]
            )[-4:]
        proposals = rechecked + proposals
        point = self.junction.update(self.grid, xy, n, obs, area_id=r.area)
        if point:
            self._arrive(point, obs)
            events.append(dict(reason="JUNCTION", place=point))
        target = frame.get("frontier")
        if target and self.exploration and frame["action"] == "observe":
            active = self.exploration.active
            target = dict(target, id=active[0]) if active else None
        if (
            target
            and frame["action"] == "observe"
            and math.dist(xy, target["xy"]) <= 0.35
            and target["id"] not in self.graph.frontier_places
            and free_footprint(self.grid, xy, r.clearance)
        ):
            point = self.graph.visit(
                "frontier:" + obs, xy, n, "frontier_observed", [obs], area_id=r.area
            )
            self.graph.link_frontier(target["id"], point)
            self._arrive(point, obs)
            events.append(
                dict(reason="FRONTIER_OBSERVED", place=point, frontier=target["id"])
            )
        # Attach objects only when this camera observation belongs to the place.
        p = self.graph.places[r.place]
        near = math.dist((p.x, p.y), xy) <= self.graph.merge_radius
        at_place = (
            p.area_id == r.area
            and near
            and (
                math.dist((p.x, p.y), xy) < 1e-6
                or all(
                    free_footprint(self.grid, q, r.clearance)
                    for q in np.linspace((p.x, p.y), xy, 8)
                )
            )
        )
        mapped = []
        # Legacy M1-only traces lack the calibrated perception schema.
        if prepared is not None:
            mapped = self.objects.integrate(prepared, r.place if at_place else None)
        if at_place:
            refs = [
                obs + "/" + e["local_id"]
                for e in perception["target_entities"] + perception["context_entities"]
            ]
            if prepared is not None:
                refs = [item.object_id for item in mapped if item.object_id]
            if obs not in p.observation_ids:
                p.observation_ids.append(obs)
            p.nearby_object_ids.extend(
                ref for ref in refs if ref not in p.nearby_object_ids
            )
        objects = self.objects.snapshot(mapped)
        exploration = self.exploration.update(frame,self.grid,events,objects) if self.exploration else None
        self.last = n
        return dict(
            exploration=exploration,
            step=n,
            observation_id=obs,
            events=events,
            area=r.area,
            place=r.place,
            proposals=proposals,
            portals=[
                dict(
                    id=q.portal_id,
                    a=q.a.tolist(),
                    b=q.b.tolist(),
                    status=q.status,
                    crossings=q.crossings,
                    areas=q.areas,
                )
                for q in r.portals.values()
            ],
            junctions=[
                dict(xy=j.xy, directions=[b.direction_deg for b in j.branches])
                for j in self.junction.candidates
            ],
            places=[asdict(p) for p in self.graph.places.values()],
            edges=[asdict(e) for e in self.graph.edges.values()],
            frontier_places=self.graph.frontier_places.copy(),
            object_memory=objects,
            map_version=self.grid.map_version,
        )
