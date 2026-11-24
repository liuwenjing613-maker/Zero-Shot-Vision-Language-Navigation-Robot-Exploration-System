"""Scene-scoped object memory. One writer consumes complete perception frames."""
from dataclasses import asdict
import math
import numpy as np

from vln_perception.target_verifier import validate_verdict
from .object_models import ObjectRecord, GoalMatchEvidence, entity_key, normalize
from .object_geometry import localize
from .object_association import associate, attributes, category


def independent(obs, old):
    return math.dist(obs.camera_xyz, old.camera_xyz) >= 0.25 or np.dot(
        obs.forward, old.forward
    ) <= math.cos(math.radians(20))


def add_vote(scores, key, weight):
    scores[key] = scores.get(key, 0.0) + weight


def fuse_record(obj, obs, history):
    obj.last_seen = max(obj.last_seen, obs.timestamp)
    obj.observation_ids.append(obs.observation_id)
    if obs.place_id and obs.place_id not in obj.observed_from_places:
        obj.observed_from_places.append(obs.place_id)
    prior = [history[key] for key in obj.independent_ids]
    if not all(independent(obs, old) for old in prior):
        return
    obs.independent = True
    obj.independent_ids.append(obs.observation_id)
    x, y, right, bottom = obs.bbox
    box_weight = min(1.0, math.sqrt((right - x) * (bottom - y)) / 100)
    if min(x, y, 1000 - right, 1000 - bottom) < 10:
        box_weight *= 0.65
    weight = box_weight * obs.geometry_quality
    add_vote(obj.category_scores, category(obs.category_raw), weight)
    for key, value in attributes(obs.attributes_raw).items():
        add_vote(obj.attribute_scores.setdefault(key, {}), value, weight)
    votes = sorted(obj.category_scores.values(), reverse=True)
    if any(len(values) > 1 for values in obj.attribute_scores.values()) or (
        len(votes) > 1 and votes[1] >= 0.65 * votes[0]
    ):
        obj.semantic_status = "CONFLICTED"
    elif len(votes) > 1 and votes[1] >= 0.25 * votes[0]:
        obj.semantic_status = "AMBIGUOUS"
    else:
        obj.semantic_status = "SUPPORTED"
    samples = prior + [obs]
    points = np.asarray([s.map_position_xz for s in samples])
    sigmas = np.array([s.position_sigma_m for s in samples])
    residual = np.linalg.norm(points - np.median(points, axis=0), axis=1)
    weights = [s.geometry_quality for s in samples] / np.square(sigmas)
    weights *= np.minimum(1.0, 0.30 / np.maximum(residual, 1e-8))
    center = np.average(points, axis=0, weights=weights)
    obj.map_position_xz = tuple(center)
    scatter = np.sum((points - center) ** 2, axis=1)
    obj.position_sigma_m = float(
        np.sqrt(np.average(sigmas**2 + scatter, weights=weights))
    )
    if prior:
        reliable = (
            obj.position_sigma_m <= 0.25
            and obs.candidates[0][1] >= 0.72
            and max(residual) <= 0.35
        )
        obj.track_status = "STABLE" if reliable else "TRACKED"


class ObjectMemory:
    def __init__(self):
        self.records = {}
        self.observations = {}
        self.verifications = []
        self.frames = {}

    def consume(self, result, depth, camera, pose, timestamp, place_id=None):
        prepared = self.prepare(result, depth, camera, pose, timestamp)
        return self.integrate(prepared, place_id)

    def prepare(self, result, depth, camera, pose, timestamp):
        frame = tuple(result[k] for k in ("task_id", "observation_id", "view_id"))
        obs = normalize(result, timestamp)
        if frame in self.frames:
            raise ValueError("Frame already consumed")
        localize(obs, depth, camera, pose)
        return frame, obs

    def integrate(self, prepared, place_id):
        frame, obs = prepared
        if frame in self.frames:
            raise ValueError("Frame already consumed")
        associate(obs, self.records)
        for item in obs:
            item.place_id = place_id
            if item.association_status == "NEW":
                item.object_id = f"O{len(self.records)+1:06d}"
                self.records[item.object_id] = ObjectRecord(
                    item.object_id, item.timestamp, item.timestamp
                )
            if item.object_id:
                fuse_record(self.records[item.object_id], item, self.observations)
            self.observations[item.observation_id] = item
        self.frames[frame] = [item.observation_id for item in obs]
        return obs

    def apply_verification(self, verdict, timestamp):
        fields = (
            "status",
            "supporting_evidence",
            "opposing_evidence",
            "uncertainty_reason",
        )
        validate_verdict({k: verdict[k] for k in fields})
        key = entity_key(
            *(
                verdict[k]
                for k in ("task_id", "observation_id", "view_id", "candidate_id")
            )
        )
        obs = self.observations[key]
        if obs.perception_role != "target":
            raise ValueError("Verification must reference a target candidate")
        evidence = GoalMatchEvidence(
            obs.task_id,
            key,
            obs.local_id,
            obs.object_id,
            verdict["status"],
            list(verdict["supporting_evidence"]),
            list(verdict["opposing_evidence"]),
            verdict["uncertainty_reason"],
            timestamp,
        )
        if evidence not in self.verifications:
            self.verifications.append(evidence)
        return evidence

    def get_object(self, object_id):
        return self.records[object_id]

    def query_goal_objects(self, goal, task_id=None):
        target = category(goal["target_category"])
        wanted = attributes(goal["target_attributes"])
        result = {}
        for key, obj in self.records.items():
            verdicts = {
                e.status
                for e in self.verifications
                if e.object_id == key and e.task_id == task_id
            }
            positive, negative = "VERIFIED" in verdicts, "REJECTED" in verdicts
            if positive or negative:
                result[key] = (
                    "POSSIBLE_MATCH"
                    if positive and negative
                    else "MATCH"
                    if positive
                    else "NON_MATCH"
                )
            elif not obj.category_scores:
                result[key] = "UNKNOWN"
            elif target not in obj.category_scores:
                result[key] = "NON_MATCH"
            else:
                votes = [obj.attribute_scores.get(k, {}) for k in wanted]
                if any(
                    v and value not in v for v, value in zip(votes, wanted.values())
                ):
                    result[key] = "NON_MATCH"
                elif (
                    goal.get("spatial_relations")
                    or len(obj.category_scores) > 1
                    or any(len(v) != 1 for v in votes)
                ):
                    result[key] = "POSSIBLE_MATCH"
                else:
                    result[key] = "MATCH"
        return result

    def snapshot(self, observations):
        return dict(
            objects={k: asdict(v) for k, v in self.records.items()},
            observations={o.observation_id: asdict(o) for o in observations},
            verification=[asdict(v) for v in self.verifications],
        )
