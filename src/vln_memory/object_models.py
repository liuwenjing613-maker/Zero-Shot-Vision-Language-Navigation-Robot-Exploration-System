"""Evidence retained by one scene's object memory."""
from dataclasses import dataclass, field
from copy import deepcopy
from urllib.parse import quote
import math

from vln_perception.scene_perceptor import SCENE_FIELDS, validate_scene


def entity_key(task, frame, view, local):
    return "/".join(quote(value, safe="") for value in (task, frame, view, local))


@dataclass
class ObjectObservation:
    task_id: str
    frame_id: str
    view_id: str
    local_id: str
    perception_role: str
    category_raw: str
    attributes_raw: dict
    bbox: list
    model_confidence_raw: float
    visual_evidence: str
    timestamp: float
    place_id: str = None
    map_position_xz: tuple = None
    position_sigma_m: float = None
    geometry_quality: float = 0.0
    depth_valid_ratio: float = 0.0
    position_source: str = "UNLOCALIZED"
    camera_xyz: tuple = None
    forward: tuple = None
    bearing_world: tuple = None
    object_id: str = None
    association_status: str = "UNASSIGNED"
    candidates: list = field(default_factory=list)
    independent: bool = False

    @property
    def observation_id(self):
        return entity_key(self.task_id, self.frame_id, self.view_id, self.local_id)


@dataclass
class ObjectRecord:
    object_id: str
    first_seen: float
    last_seen: float
    map_position_xz: tuple = None
    position_sigma_m: float = None
    track_status: str = "PROVISIONAL"
    localization_status: str = "MAPPED"
    semantic_status: str = "UNRESOLVED"
    category_scores: dict = field(default_factory=dict)
    attribute_scores: dict = field(default_factory=dict)
    independent_ids: list = field(default_factory=list)
    observation_ids: list = field(default_factory=list)
    observed_from_places: list = field(default_factory=list)


@dataclass
class GoalMatchEvidence:
    task_id: str
    observation_id: str
    local_id: str
    object_id: str
    status: str
    supporting_evidence: list
    opposing_evidence: list
    uncertainty_reason: str
    timestamp: float


def normalize(result, timestamp, place_id=None):
    validate_scene({k: result[k] for k in SCENE_FIELDS})
    ids = [result[k] for k in ("task_id", "observation_id", "view_id")]
    if not all(isinstance(v, str) and v.strip() for v in ids) or not math.isfinite(
        timestamp
    ):
        raise ValueError("Invalid observation context")
    return [
        ObjectObservation(
            *ids,
            item["local_id"],
            role,
            item["category"],
            deepcopy(item["attributes"]),
            list(item["bbox"]),
            item["confidence"],
            item["evidence"],
            timestamp,
            place_id,
        )
        for role in ("target", "context")
        for item in result[role + "_entities"]
    ]
