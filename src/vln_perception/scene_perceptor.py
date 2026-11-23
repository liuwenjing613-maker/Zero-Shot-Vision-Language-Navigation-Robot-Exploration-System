"""Single-view RGB perception for object navigation."""
import base64
import json
from pathlib import Path
from time import perf_counter

from .goal_parser import validate_goal


PROMPT = """You are the visual perception module of a mobile robot. Inspect the single RGB
view and the GoalSpec. Return one JSON object with exactly four keys:
target_status, target_entities, context_entities, visual_cues.

First inspect the entire image for prominent visible objects useful for navigation
or future search (furniture, appliances, mirrors, windows, door panels, landmarks).
Put each matching or plausible goal instance in target_entities and other major
visible instances in context_entities. Keep visible instances distinct, including
non-target chairs, even when similar. Use observable attributes and visual evidence.
Keep the inventory of major objects useful when the goal changes. Do not repeat
one physical object in both lists or invent hidden objects. Before returning, recheck each context entity against the full goal: for an unqualified category goal every instance you identify as that category or a matching subtype belongs in target_entities, including secondary instances. A request for any
chair includes armchairs. Multiple acceptable generic matches are VISIBLE.

visual_cues are visible spatial/navigation structures only. Use doorway, corridor,
opening, stairs, ramp, or blocked_passage. Describe the visible opening separately
from a door panel only when both have distinct visual evidence. Never list furniture,
window, mirror, ceiling decoration or wall art as a cue. Do not infer reachability.
Return [] when no such item is visible. No quotas, no actions.

target_status: VISIBLE for a supported match; UNCERTAIN for insufficient evidence;
AMBIGUOUS for unresolved alternatives; NOT_VISIBLE for no candidate in this view.
For both entity lists use exactly:
{"local_id":"E01","category":"...","attributes":{},"bbox":[1,2,3,4],
 "confidence":0.8,"evidence":"short visible evidence"}.
For cues use exactly:
{"local_id":"V01","cue_type":"doorway","bbox":[1,2,3,4],
 "relative_direction":"left","visibility":"clear","evidence":"short visible evidence"}.
All attribute values must be nonempty JSON strings (e.g. "color":"brown"); omit unknown attributes. Use {} when none are clear. Use a unique local_id for every item in this observation. bbox is integer
[xmin,ymin,xmax,ymax] on a 0..1000 scale, origin top left. Cue visibility is
clear, partial, or uncertain. Use [] for empty arrays. JSON only."""


CUE_TYPES = {"doorway", "corridor", "opening", "stairs", "ramp", "blocked_passage"}
SCENE_FIELDS = ("target_status", "target_entities", "context_entities", "visual_cues")


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _check_item(item, fields, ids):
    if not isinstance(item, dict) or set(item) != fields:
        raise ValueError(f"Expected item fields {sorted(fields)}")
    local_id = item["local_id"]
    if not _text(local_id) or local_id in ids:
        raise ValueError("Missing or repeated observation-local ID")
    ids.add(local_id)
    if not _text(item["evidence"]):
        raise ValueError("Missing visual evidence")
    box = item["bbox"]
    if not isinstance(box, list) or len(box) != 4 or not all(type(v) is int for v in box):
        raise ValueError("Expected four integer bounding-box coordinates")
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
        raise ValueError("Invalid normalized bounding box")


def validate_scene(result):
    if not isinstance(result, dict) or set(result) != set(SCENE_FIELDS):
        raise ValueError("Invalid three-category perception schema")
    if not all(isinstance(result[k], list) for k in SCENE_FIELDS[1:]):
        raise ValueError("Expected target, context and cue arrays")
    status, targets = result["target_status"], result["target_entities"]
    if status not in ("VISIBLE", "NOT_VISIBLE", "UNCERTAIN", "AMBIGUOUS"):
        raise ValueError("Invalid target status")
    if ((status == "VISIBLE" and not targets) or (status == "NOT_VISIBLE" and targets)
            or (status == "AMBIGUOUS" and len(targets) < 2)):
        raise ValueError("Target status conflicts with target count")
    ids = set()
    entities_seen = set()
    for item in targets + result["context_entities"]:
        _check_item(item, {"local_id", "category", "attributes", "bbox", "confidence", "evidence"}, ids)
        if not _text(item["category"]):
            raise ValueError("Missing entity category")
        attrs = item["attributes"]
        if not isinstance(attrs, dict) or not all(_text(k) and _text(v) for k, v in attrs.items()):
            raise ValueError(f"Invalid attributes for {item['local_id']}: {attrs!r}")
        confidence = item["confidence"]
        if type(confidence) not in (int, float) or not 0 <= confidence <= 1:
            raise ValueError("Confidence must be in [0,1]")
        key = (item["category"].strip().casefold(), tuple(item["bbox"]))
        if key in entities_seen:
            raise ValueError("Duplicate entity with identical category and box")
        entities_seen.add(key)
    for cue in result["visual_cues"]:
        _check_item(cue, {"local_id", "cue_type", "bbox", "relative_direction", "visibility", "evidence"}, ids)
        if not _text(cue["cue_type"]) or cue["cue_type"] not in CUE_TYPES:
            raise ValueError("Unsupported navigation cue")
        if not _text(cue["relative_direction"]) or cue["visibility"] not in ("clear", "partial", "uncertain"):
            raise ValueError("Invalid cue direction or visibility")


def perceive_scene(client, image_path, goal_spec, *, task_id, observation_id, view_id):
    validate_goal({k:v for k,v in goal_spec.items() if k != "instruction"})
    if not all(isinstance(s,str) and s.strip() for s in (task_id,observation_id,view_id)):
        raise ValueError("Expected nonempty task, observation and view IDs")
    data = Path(image_path).read_bytes()
    if data.startswith(b"\xff\xd8\xff"):
        mime = "image/jpeg"
    elif data.startswith(b"\x89PNG\r\n\x1a\n"):
        mime = "image/png"
    else:
        raise ValueError("Expected a JPEG or PNG image")
    url = "data:" + mime + ";base64," + base64.b64encode(data).decode("ascii")
    messages = [{"role":"system","content":PROMPT},{"role":"user","content":[
        {"type":"text","text":"GoalSpec: " + json.dumps(goal_spec,ensure_ascii=False)},
        {"type":"image_url","image_url":{"url":url}}]}]
    start = perf_counter()
    result = client._request_json(messages, max_tokens=3200)
    elapsed = round(1000*(perf_counter()-start))
    validate_scene(result)
    return {"schema_version":"perception_three_groups_v1","task_id":task_id,"observation_id":observation_id,
            "view_id":view_id,"model_id":client.model,"latency_ms":elapsed,
            **{k:result[k] for k in SCENE_FIELDS}}
