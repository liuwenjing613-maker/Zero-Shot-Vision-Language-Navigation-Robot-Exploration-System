"""Sparse directed graph for one scene, updated by ordered navigation events."""
from dataclasses import dataclass, field
import math
import operator
from typing import List, Optional


@dataclass
class PlaceRecord:
    place_id: str
    x: float
    y: float
    visit_count: int
    created_at: float
    last_visit_time: float
    observation_ids: List[str] = field(default_factory=list)
    nearby_object_ids: List[str] = field(default_factory=list)
    area_id: Optional[str] = None
    node_types: List[str] = field(default_factory=list)
    entry_portal_ids: List[str] = field(default_factory=list)
    branch_directions: List[float] = field(default_factory=list)
    branch_lengths: List[float] = field(default_factory=list)
    branch_min_widths: List[float] = field(default_factory=list)


@dataclass
class NavigationEdge:
    source_id: str
    target_id: str
    path_cost: float
    status: str = "PLANNED"
    traversal_count: int = 0
    failure_count: int = 0
    consecutive_failures: int = 0
    portal_ids: List[str] = field(default_factory=list)


class TopologyMemory:
    def __init__(self, merge_radius=0.5, failures_to_block=2):
        self.merge_radius = float(merge_radius)
        self.failures_to_block = operator.index(failures_to_block)
        if not math.isfinite(self.merge_radius) or self.merge_radius <= 0 or self.failures_to_block < 1:
            raise ValueError("Invalid topology thresholds")
        self.places = {}
        self.edges = {}
        self.frontier_places = {}
        self._events = {}

    def _replay(self, event_id, payload):
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("A nonempty event ID is required")
        if event_id in self._events:
            previous, result = self._events[event_id]
            if previous != payload:
                raise ValueError("Event ID reused with different data")
            return result
        return None

    def visit(self, event_id, xy, timestamp, reason, observation_ids=(), nearby_object_ids=(),
              *, area_id=None, portal_id=None, reuse_place=None, branches=(), eligible_place_ids=None):
        """Call once on start, observed arrival, or another meaningful place event."""
        x, y = map(float, xy)
        if reason not in {"start", "frontier_observed", "new_region", "observation_place", "junction"}:
            raise ValueError("Unknown place event")
        if not all(map(math.isfinite, (x, y, timestamp))):
            raise ValueError("Invalid place pose/time")
        obs, objects = tuple(observation_ids), tuple(nearby_object_ids)
        branches = tuple(tuple(map(float, branch)) for branch in branches)
        if reason == "junction":
            if len(branches) < 3 or any(
                len(branch) != 3 or not all(map(math.isfinite, branch))
                or not (0 <= branch[0] < 360 and branch[1] + 1e-8 >= 1 and branch[2] > 0)
                for branch in branches
            ):
                raise ValueError("Expected three or more measured junction branches")
            headings = sorted(branch[0] for branch in branches)
            if min((b-a)%360 for a,b in zip(headings,headings[1:]+headings[:1])) < 40:
                raise ValueError("Expected distinct junction directions")
        elif branches:
            raise ValueError("Only a junction visit can supply branch evidence")
        allowed = None if eligible_place_ids is None else tuple(sorted(set(eligible_place_ids)))
        payload = ("visit", x, y, timestamp, reason, obs, objects, area_id, portal_id, reuse_place, branches, allowed)
        previous = self._replay(event_id, payload)
        if previous is not None:
            return previous
        eligible = [p for p in self.places.values() if area_id is None or p.area_id == area_id]
        if allowed is not None:
            eligible = [p for p in eligible if p.place_id in allowed]
        nearest = min(eligible, key=lambda p: math.hypot(x-p.x, y-p.y), default=None)
        if reuse_place is not None:
            if allowed is not None and reuse_place not in allowed:
                raise ValueError("Place is outside the eligible set")
            nearest = self.places[reuse_place]
            if nearest.area_id != area_id:
                raise ValueError("Cannot reuse a place across area IDs")
        if nearest is None or (reuse_place is None and math.hypot(x-nearest.x, y-nearest.y) > self.merge_radius):
            place_id = "P{:02d}".format(len(self.places) + 1)
            nearest = PlaceRecord(place_id, x, y, 0, timestamp, timestamp)
            nearest.area_id = area_id
            self.places[place_id] = nearest
        elif timestamp < nearest.last_visit_time:
            raise ValueError("Out-of-order place visit")
        kind = {"new_region": "REGION_ENTRY", "start": "START"}.get(reason, reason.upper())
        if kind not in nearest.node_types:
            nearest.node_types.append(kind)
        if portal_id and portal_id not in nearest.entry_portal_ids:
            nearest.entry_portal_ids.append(portal_id)
        if reason == "junction":
            directions, lengths, widths = map(list, zip(*branches))
            nearest.branch_directions = directions
            nearest.branch_lengths = lengths
            nearest.branch_min_widths = widths
        nearest.visit_count += 1
        nearest.last_visit_time = timestamp
        for ids, new in ((nearest.observation_ids, obs), (nearest.nearby_object_ids, objects)):
            ids.extend(item for item in new if item not in ids)
        self._events[event_id] = payload, nearest.place_id
        return nearest.place_id

    def record_navigation(self, event_id, source_id, target_id, outcome, path_cost=None):
        """One directed planner/executor report. Costs are path lengths in meters."""
        if source_id not in self.places or target_id not in self.places or source_id == target_id:
            raise ValueError("Expected two distinct known place IDs")
        if outcome not in {"planned", "succeeded", "failed", "blocked"}:
            raise ValueError("Unknown navigation outcome")
        key = (source_id, target_id)
        edge = self.edges.get(key)
        if outcome in {"planned", "succeeded"}:
            if path_cost is None or not math.isfinite(path_cost) or path_cost < 0:
                raise ValueError("Verified plans and traversals require a finite path cost")
        elif path_cost is not None or edge is None:
            raise ValueError("Failure requires an existing route and carries no new cost")
        payload = ("nav", key, outcome, path_cost)
        if self._replay(event_id, payload) is not None:
            return edge
        if edge is None:
            edge = NavigationEdge(source_id, target_id, float(path_cost))
            self.edges[key] = edge
        if outcome == "planned":
            if edge.status == "PLANNED":
                edge.path_cost = float(path_cost)
        elif outcome == "succeeded":
            edge.status = "TRAVERSED"
            edge.path_cost = float(path_cost)
            edge.traversal_count += 1
            edge.consecutive_failures = 0
        else:
            edge.failure_count += 1
            edge.consecutive_failures += 1
            if outcome == "blocked" or edge.consecutive_failures >= self.failures_to_block:
                edge.status = "BLOCKED"
        self._events[event_id] = payload, key
        return edge

    def link_frontier(self, frontier_id, place_id):
        """Link, reassign, or remove (place_id=None) the M3-owned frontier ID."""
        if not isinstance(frontier_id, str) or not frontier_id:
            raise ValueError("A nonempty frontier ID is required")
        if place_id is None:
            self.frontier_places.pop(frontier_id, None)
        elif place_id in self.places:
            self.frontier_places[frontier_id] = place_id
        else:
            raise ValueError("Unknown place")
