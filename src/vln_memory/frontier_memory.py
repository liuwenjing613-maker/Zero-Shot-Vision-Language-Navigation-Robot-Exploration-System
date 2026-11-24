import math
from .geometry_memory import _ray_cells


class FrontierMemory:
    def __init__(self):
        self.records = {}
        self.regions = {}
        self.version = -1
        self.actions = set()
        self.last = None
        self.events = []

    def emit(self, kind, key, **details):
        self.events.append(dict(event_type=kind, frontier_id=key, details=details))

    def update_snapshot(self, snapshot):
        version = snapshot["map_version"]
        if version == self.version and self.last[0] == snapshot:
            return self.last[1].copy()
        if version <= self.version:
            raise ValueError("Expected fresh snapshot")
        prepared, pairs = {}, []
        for item in snapshot["candidates"]:
            local = item["local_candidate_id"]
            cells = set(map(tuple, item["cluster_cells"]))
            xy = tuple(item["centroid_xy"])
            view = tuple(item["safe_viewpoint_xy"])
            if local in prepared or not cells or not all(map(math.isfinite, xy + view)):
                raise ValueError("Invalid candidate")
            prepared[local] = dict(cells=cells, centroid_xy=xy, safe_viewpoint_xy=view)
            for key, r in self.records.items():
                d = math.dist(xy, r["centroid_xy"])
                if d >= 0.5:
                    continue
                iou = len(cells & r["cells"]) / len(cells | r["cells"])
                if iou > 0.15 or d < 0.25:
                    pairs.append((0.7 * iou + 0.3 * math.exp(-d / 0.5), local, key))
        mapping, used = {}, set()
        for _, local, key in sorted(pairs, reverse=True):
            if local not in mapping and key not in used:
                mapping[local] = key
                used.add(key)
        for local, item in prepared.items():
            key = mapping.get(local)
            if key is None:
                key = mapping[local] = f"F{len(self.records)+1:04d}"
                self.records[key] = dict(
                    status="AVAILABLE",
                    first_seen=version,
                    selection_count=0,
                    attempt_count=0,
                    visit_count=0,
                    failed_count=0,
                    consecutive_failures=0,
                    action_id=None,
                )
                self.emit("FRONTIER_CREATED", key)
            r = self.records[key]
            if r["status"] in ("STALE", "RESOLVED", "REACHED"):
                r["status"] = "AVAILABLE"
                self.emit("FRONTIER_REACTIVATED", key)
            r.update(item, last_seen=version, missed_snapshots=0)
        if snapshot["complete"]:
            for key, r in self.records.items():
                if key in mapping.values():
                    continue
                r["missed_snapshots"] += 1
                if r["missed_snapshots"] >= 2:
                    state = {"AVAILABLE": "STALE", "REACHED": "RESOLVED"}.get(
                        r["status"]
                    )
                    if state:
                        r["status"] = state
                        self.emit("FRONTIER_" + state, key)
        self.version = version
        from copy import deepcopy

        self.last = (deepcopy(snapshot), mapping.copy())
        return mapping

    def transition(self, key, operation, action_id, reason=None):
        token = (key, operation, action_id)
        if token in self.actions:
            return
        r = self.records[key]
        allowed = {
            "selected": ("AVAILABLE",),
            "started": ("RESERVED",),
            "reached": ("IN_PROGRESS",),
            "failed": ("IN_PROGRESS",),
            "cancelled": ("RESERVED", "IN_PROGRESS"),
            "reactivate": ("BLOCKED", "STALE", "RESOLVED"),
            "resolved": ("REACHED", "STALE", "AVAILABLE"),
        }
        if r["status"] not in allowed[operation] or (
            operation in ("reached", "failed", "cancelled")
            and r["action_id"] != action_id
        ):
            raise ValueError("Invalid transition or action ID")
        if operation in ("resolved", "reactivate", "failed") and not reason:
            raise ValueError("Reason required")
        states = dict(
            selected="RESERVED",
            started="IN_PROGRESS",
            reached="REACHED",
            failed="AVAILABLE",
            cancelled="AVAILABLE",
            resolved="RESOLVED",
            reactivate="AVAILABLE",
        )
        state = states[operation]
        counter = dict(
            selected="selection_count",
            started="attempt_count",
            reached="visit_count",
            failed="failed_count",
        ).get(operation)
        if counter:
            r[counter] += 1
        if operation == "failed":
            r["consecutive_failures"] = (
                r["consecutive_failures"] + 1
                if reason in ("NO_PATH", "OBSTRUCTED", "REPEATED_STUCK")
                else 0
            )
            if r["consecutive_failures"] >= 2:
                state = "BLOCKED"
                r["blocked_at_map_version"] = self.version
        elif operation in ("reached", "reactivate"):
            r["consecutive_failures"] = 0
        r.update(status=state, action_id=action_id, last_reason=reason)
        self.actions.add(token)
        event = {"started": "NAVIGATION_STARTED", "reactivate": "REACTIVATED"}.get(
            operation, operation.upper()
        )
        self.emit("FRONTIER_" + event, key, action_id=action_id, reason=reason)
        if state == "BLOCKED":
            self.emit("FRONTIER_BLOCKED", key, action_id=action_id)

    def region(self, key, frontiers, cells, views):
        if key not in self.regions:
            self.regions[key] = dict(
                frontiers=set(),
                cells=set(),
                views={},
                status="UNEXPLORED",
                checks=0,
                last_version=-1,
            )
        r = self.regions[key]
        r["frontiers"].update(frontiers)
        r["cells"].update(cells)
        for vid, point in views.items():
            r["views"].setdefault(
                vid, dict(cell=tuple(point), status="PLANNED", observations=[])
            )
        return r

    def evaluate(self, key, grid, complete=True, max_range=6.0, min_cells=12):
        r = self.regions[key]
        eligible = set()
        for row, col in r["cells"]:
            for view in r["views"].values():
                vr, vc = view["cell"]
                if math.hypot(row - vr, col - vc) * grid.resolution > max_range:
                    continue
                ray = _ray_cells((vc + 0.5, vr + 0.5), (col + 0.5, row + 0.5))
                if not any(
                    grid.log_odds[cell] >= 0.8 for cell in ray if cell != (row, col)
                ):
                    eligible.add((row, col))
                    break
        r["eligible_cells"] = eligible
        observed = sum(bool(grid.observed_count[c]) for c in eligible)
        ratio = observed / len(eligible) if eligible else None
        remaining = sum(self.records[k]["status"] != "RESOLVED" for k in r["frontiers"])
        scanned = bool(r["views"]) and all(
            v["status"] == "SCANNED" for v in r["views"].values()
        )
        qualifies = (
            complete
            and scanned
            and len(eligible) >= min_cells
            and ratio >= 0.85
            and not remaining
        )
        if not qualifies:
            r["checks"] = 0
        elif grid.map_version != r["last_version"]:
            r["checks"] += 1
        r.update(
            eligible=len(eligible),
            observed=observed,
            coverage=ratio,
            remaining=remaining,
            last_version=grid.map_version,
        )
        r["status"] = (
            "RESOLVED"
            if r["checks"] >= 2
            else "IN_PROGRESS"
            if any(self.records[k]["status"] == "IN_PROGRESS" for k in r["frontiers"])
            else "PARTIALLY_EXPLORED"
            if observed
            else "UNEXPLORED"
        )
        states = {self.records[k]["status"] for k in r["frontiers"]}
        if states == {"BLOCKED"}:
            r["status"] = "BLOCKED"
        if states == {"STALE"}:
            r["status"] = "STALE"
        if "REACHED" in states and not scanned:
            r["status"] = "OBSERVING"
        return r

    def scan(self, key, vid, grid, xy, observation_id):
        view = self.regions[key]["views"][vid]
        vr, vc = view["cell"]
        target = (
            grid.origin_x + (vc + 0.5) * grid.resolution,
            grid.origin_y + (vr + 0.5) * grid.resolution,
        )
        if (
            math.dist(target, xy) > 0.35
            or grid.last_observed_version[vr, vc] != grid.map_version
        ):
            raise ValueError("Viewpoint requires an on-site current scan")
        if observation_id not in view["observations"]:
            eligible = self.evaluate(key, grid)["eligible_cells"]
            fresh = sum(
                grid.observed_count[c] == 1
                and grid.last_observed_version[c] == grid.map_version
                for c in eligible
            )
            view.update(
                coverage_gain=float(fresh) / len(eligible) if eligible else None,
                last_scan_version=grid.map_version,
            )
        view["status"] = "SCANNED"
        if observation_id not in view["observations"]:
            view["observations"].append(observation_id)
