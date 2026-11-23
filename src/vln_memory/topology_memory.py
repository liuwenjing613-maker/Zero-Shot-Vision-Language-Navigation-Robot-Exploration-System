"""Sparse directed graph for one scene, updated by ordered navigation events."""
from dataclasses import dataclass, field
import math
import operator
from typing import List


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


@dataclass
class NavigationEdge:
    source_id: str
    target_id: str
    path_cost: float
    status: str = "PLANNED"
    traversal_count: int = 0
    failure_count: int = 0
    consecutive_failures: int = 0


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

    def visit(self, event_id, xy, timestamp, reason, observation_ids=(), nearby_object_ids=()):
        """Call once on start, observed arrival, or another meaningful place event."""
        x, y = map(float, xy)
        if reason not in {"start", "frontier_observed", "new_region", "observation_place"}:
            raise ValueError("Unknown place event")
        if not all(map(math.isfinite, (x, y, timestamp))):
            raise ValueError("Invalid place pose/time")
        obs, objects = tuple(observation_ids), tuple(nearby_object_ids)
        payload = ("visit", x, y, timestamp, reason, obs, objects)
        previous = self._replay(event_id, payload)
        if previous is not None:
            return previous
        nearest = min(self.places.values(), key=lambda p: math.hypot(x-p.x, y-p.y), default=None)
        if nearest is None or math.hypot(x-nearest.x, y-nearest.y) > self.merge_radius:
            place_id = "P{:02d}".format(len(self.places) + 1)
            nearest = PlaceRecord(place_id, x, y, 0, timestamp, timestamp)
            self.places[place_id] = nearest
        elif timestamp < nearest.last_visit_time:
            raise ValueError("Out-of-order place visit")
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
