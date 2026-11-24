"""Coordinate frontier state and chronological facts with the M1/M2 reducer."""
import math
from .frontier_memory import FrontierMemory
from .episodic_memory import EpisodicMemory
from .exploration_adapter import detect, bounded_cells


class ExplorationMemory:
    def __init__(self, session, path=None):
        self.m3 = FrontierMemory()
        self.m4 = EpisodicMemory(session, path)
        self.active = None
        self.conflicted = set()

    def update(self, frame, grid, m1, m2):
        fm = self.m3
        n = frame["step"]
        obs = frame["observation_id"]
        events = []

        def event(kind, source, **refs):
            entry = self.m4.record(
                f"{obs}:{len(events)}:{kind}",
                kind,
                source,
                n,
                task_id=frame["session"],
                observation_id=obs,
                map_version=grid.map_version,
                **refs,
            )
            events.append(entry)

        if frame["session"] != self.m4.session_id:
            raise ValueError("Session mismatch")
        if n == 0:
            event("TASK_STARTED", "session")
        if frame["action"] in ("forward", "back", "left", "right", "observe"):
            blocked = frame.get("collision") or frame.get("floor_block")
            event(
                "ACTION_FAILED" if blocked else "ACTION_SUCCEEDED",
                "executor",
                action_id=obs,
                details=dict(action=frame["action"], collision=bool(blocked)),
            )
        snapshot = frame.get("frontier_snapshot")
        if snapshot is None:
            snapshot = detect(grid, frame["xy"])
        if snapshot["map_version"] != grid.map_version:
            raise ValueError("Snapshot/map version mismatch")
        mapping = fm.update_snapshot(snapshot)
        if "regions" in snapshot:
            taken = set()
            for region in snapshot["regions"]:
                members = {mapping[k] for k in region["frontier_local_ids"]}
                cells = {tuple(c) for c in region["region_cells"]}
                if any(not grid.contains(c) for c in cells):
                    raise ValueError("Region outside grid")
                areas = region.get("area_id")
                ranked = sorted(
                    (
                        (len(r["frontiers"] & members), len(r["cells"] & cells), rid)
                        for rid, r in fm.regions.items()
                        if rid not in taken and r.get("area_id") == areas
                    ),
                    reverse=True,
                )
                rid = (
                    ranked[0][2]
                    if ranked and any(ranked[0][:2])
                    else "REG-" + str(len(fm.regions) + 1)
                )
                taken.add(rid)
                views = region.get("viewpoints")
                if views is None:
                    views = {
                        v["viewpoint_id"]: grid.cell((v["x"], v["y"]))
                        for v in region["representative_viewpoints"]
                    }
                if any(not grid.contains(c) for c in views.values()):
                    raise ValueError("View outside grid")
                fm.region(rid, members, cells, views)["area_id"] = areas
        else:
            for key in mapping.values():
                rid = "R" + key[1:]
                if rid not in fm.regions:
                    r = fm.records[key]
                    fm.region(
                        rid,
                        [key],
                        bounded_cells(grid, r["cells"]),
                        {"V1": grid.cell(r["safe_viewpoint_xy"])},
                    )
        target = frame.get("frontier")
        if frame["action"] == "frontier" and target:
            if self.active:
                old, aid = self.active
                fm.transition(old, "cancelled", aid)
                self.active = None
            choices = sorted(
                (math.dist(r["safe_viewpoint_xy"], target["xy"]), k)
                for k, r in fm.records.items()
                if r["last_seen"] == grid.map_version and r["status"] == "AVAILABLE"
            )
            if (
                choices
                and choices[0][0] <= 0.35
                and (len(choices) == 1 or choices[1][0] - choices[0][0] > 0.05)
            ):
                key = choices[0][1]
                fm.transition(key, "selected", obs)
                fm.transition(key, "started", obs)
                self.active = (key, obs)
        if self.active:
            for r in fm.regions.values():
                if self.active[0] in r["frontiers"]:
                    for v in r["views"].values():
                        if v["status"] == "PLANNED":
                            v["status"] = "NAVIGATING"
        for e in m1:
            kind = {
                "START": "PLACE_CREATED",
                "REGION_ENTRY": "REGION_ENTERED",
                "JUNCTION": "JUNCTION_REACHED",
            }.get(e["reason"], e["reason"])
            event(kind, "M1", place_id=e.get("place"), details=e)
            if kind == "FRONTIER_OBSERVED" and self.active:
                key, aid = self.active
                fm.transition(key, "reached", aid)
                self.active = None
                for region in fm.regions.values():
                    if key in region["frontiers"]:
                        for vid, v in region["views"].items():
                            r, c = v["cell"]
                            xy = (
                                grid.origin_x + (c + 0.5) * grid.resolution,
                                grid.origin_y + (r + 0.5) * grid.resolution,
                            )
                            if (
                                math.dist(xy, frame["xy"]) <= 0.35
                                and grid.last_observed_version[r, c] == grid.map_version
                            ):
                                fm.scan(
                                    next(
                                        k
                                        for k, val in fm.regions.items()
                                        if val is region
                                    ),
                                    vid,
                                    grid,
                                    frame["xy"],
                                    obs,
                                )
        for e in fm.events:
            refs = dict(e)
            kind = refs.pop("event_type")
            if "action_id" in refs["details"]:
                refs["action_id"] = refs["details"]["action_id"]
            event(kind, "M3", **refs)
        fm.events.clear()
        regions = {}
        for rid, r in fm.regions.items():
            before = r["status"]
            fm.evaluate(rid, grid, snapshot["complete"])
            if r["status"] != before:
                event(
                    "REGION_STATUS_CHANGED",
                    "M3",
                    details=dict(region_id=rid, before=before, after=r["status"]),
                )
            regions[rid] = {
                k: r[k]
                for k in (
                    "status",
                    "observed",
                    "eligible",
                    "coverage",
                    "remaining",
                    "checks",
                    "views",
                )
            }
        event("OBSERVATION_PROCESSED", "perception")
        for local, o in m2["observations"].items():
            kind = {
                "NEW": "OBJECT_CREATED",
                "MATCHED": "OBJECT_ASSOCIATED",
                "AMBIGUOUS": "OBJECT_ASSOCIATION_AMBIGUOUS",
                "UNLOCALIZED": "OBJECT_OBSERVATION_UNLOCALIZED",
            }[o["association_status"]]
            event(
                kind,
                "M2",
                object_id=o["object_id"],
                details=dict(local_id=o["local_id"], category=o["category_raw"]),
            )
            key = o["object_id"]
            if key:
                conflict = m2["objects"][key]["semantic_status"] == "CONFLICTED"
                if conflict and key not in self.conflicted:
                    event("OBJECT_SEMANTIC_CONFLICT", "M2", object_id=key)
                if conflict:
                    self.conflicted.add(key)
                else:
                    self.conflicted.discard(key)
        return dict(
            frontiers={
                k: {a: b for a, b in r.items() if a != "cells"}
                for k, r in fm.records.items()
            },
            regions=regions,
            events=events,
            complete=snapshot["complete"],
        )
