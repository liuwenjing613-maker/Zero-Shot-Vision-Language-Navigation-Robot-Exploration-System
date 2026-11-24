"""Goal parsing for single-object navigation, using a local Qwen endpoint."""
import json
from urllib.request import Request, urlopen


PROMPT = """Translate one single-object navigation instruction into a GoalSpec.
Output one JSON object with exactly these keys:
target_category: singular lowercase English noun naming the requested object.
target_attributes: object mapping stated attributes (color, material, size, etc.)
to lowercase English strings. Use {} if absent and "unknown" for an explicitly
unclear value. Keep attributes of a reference object out of target_attributes.
spatial_relations: array of objects with relation and reference_object (both
English strings). Use near for beside/next to, in for room membership, and
left_of/right_of for relative position. Keep explicit modifiers of the
reference in reference_object. Use [] when no spatial relation is given.
completion_conditions: object with booleans target_confirmed and approach_required.
By task convention both are true for a standard find command. If the user asks
only to identify it or explicitly says not to approach, approach_required is false.
Never infer a room, color, reference, or route from common sense.
Return JSON only."""


def _nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def _check_keys(value, keys):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"GoalSpec requires keys {sorted(keys)}")


def validate_goal(goal):
    _check_keys(goal, ("target_category", "target_attributes", "spatial_relations",
                       "completion_conditions"))
    if not _nonempty(goal["target_category"]):
        raise ValueError("Expected a target category")
    attrs = goal["target_attributes"]
    if not isinstance(attrs, dict) or not all(_nonempty(k) and _nonempty(v)
                                              for k, v in attrs.items()):
        raise ValueError("Expected string attributes")
    relations = goal["spatial_relations"]
    if not isinstance(relations, list):
        raise ValueError("Expected a relation list")
    for relation in relations:
        _check_keys(relation, ("relation", "reference_object"))
        if not all(_nonempty(v) for v in relation.values()):
            raise ValueError("Expected a relation and a reference object")
    conditions = goal["completion_conditions"]
    _check_keys(conditions, ("target_confirmed", "approach_required"))
    if not all(isinstance(v, bool) for v in conditions.values()):
        raise ValueError("Expected boolean completion conditions")


class VLMClient:
    def __init__(self, base_url, model, timeout=120):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.timeout = timeout

    def parse_goal(self, instruction):
        if not _nonempty(instruction):
            raise ValueError("Expected a nonempty instruction")
        goal = self._request_json([{"role": "system", "content": PROMPT},
                                   {"role": "user", "content": instruction}], max_tokens=768)
        validate_goal(goal)
        return {"instruction": instruction, **goal}

    def perceive_scene(self, image_path, goal_spec, *, task_id, observation_id, view_id):
        from .scene_perceptor import perceive_scene
        return perceive_scene(self, image_path, goal_spec, task_id=task_id,
                              observation_id=observation_id, view_id=view_id)

    def verify_target(self, image_path, goal_spec, scene_result, candidate_id):
        from .target_verifier import verify_target
        return verify_target(self, image_path, goal_spec, scene_result, candidate_id)

    def _request_json(self, messages, *, max_tokens):
        payload = {"model": self.model, "temperature": 0, "max_tokens": max_tokens,
                   "response_format": {"type": "json_object"}, "messages": messages}
        request = Request(self.url, data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=self.timeout) as response:
            choice = json.load(response)["choices"][0]
        if choice.get("finish_reason") == "length":
            raise ValueError("Model output truncated")
        return json.loads(choice["message"]["content"])
