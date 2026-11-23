"""Re-examine one perception candidate on its source RGB view."""
import base64
import io
import json
import math
from time import perf_counter

from PIL import Image, ImageDraw

from .goal_parser import validate_goal
from .scene_perceptor import SCENE_FIELDS, validate_scene


PROMPT = """Verify one proposed object for a robot navigation goal. Examine the RGB
view. The cyan rectangle indicates the only candidate to evaluate. Other objects,
even if they match, cannot verify this candidate. The prior description is a
hypothesis and can be incorrect. Independently identify visible distinguishing
features, assess the requested category and any explicit attributes or relations.
VERIFIED requires positive visual evidence for the requested match. REJECTED
requires visible contrary evidence. When occlusion, crop, or ambiguity leaves
the identity or a required property unresolved, return UNCERTAIN. Report only
image evidence. Never infer task completion or physical distance.
Return a JSON object with exactly four fields:
status (VERIFIED, REJECTED or UNCERTAIN),
supporting_evidence (list of short strings supporting the match),
opposing_evidence (list of short strings contradicting the match),
uncertainty_reason (a short string when uncertainty remains, otherwise empty).
Use empty lists when there is no evidence on that side."""


def marked_candidate_view(image_path, bbox):
    """Draw the supplied 0..1000 xyxy box on an RGB copy."""
    with Image.open(image_path) as src:
        view = src.convert("RGB")
    w, h = view.size
    x1, y1, x2, y2 = bbox
    pixel_box = (x1 * w // 1000, y1 * h // 1000,
                 math.ceil(x2 * w / 1000) - 1, math.ceil(y2 * h / 1000) - 1)
    ImageDraw.Draw(view).rectangle(pixel_box, outline="#00dada", width=2)
    return view


def validate_verdict(verdict):
    keys = {"status", "supporting_evidence", "opposing_evidence", "uncertainty_reason"}
    if not isinstance(verdict, dict) or set(verdict) != keys:
        raise ValueError("Invalid verification response fields")
    status = verdict["status"]
    if status not in ("VERIFIED", "REJECTED", "UNCERTAIN"):
        raise ValueError("Unknown verification status")
    for key in ("supporting_evidence", "opposing_evidence"):
        items = verdict[key]
        if not isinstance(items, list) or not all(isinstance(v, str) and v.strip() for v in items):
            raise ValueError(f"{key} must be a list of nonempty strings")
    reason = verdict["uncertainty_reason"]
    if not isinstance(reason, str) or (status == "UNCERTAIN" and not reason.strip()):
        raise ValueError("Uncertain verdict requires a reason")
    if status == "VERIFIED" and not verdict["supporting_evidence"]:
        raise ValueError("Verified verdict requires supporting evidence")
    if status == "REJECTED" and not verdict["opposing_evidence"]:
        raise ValueError("Rejected verdict requires opposing evidence")


def verify_target(client, image_path, goal_spec, scene_result, candidate_id):
    """image_path must be the original frame for scene_result, not a later pose."""
    validate_goal({k: v for k, v in goal_spec.items() if k != "instruction"})
    validate_scene({k: scene_result[k] for k in SCENE_FIELDS})
    source = {k: scene_result[k] for k in ("task_id", "observation_id", "view_id")}
    if not all(isinstance(v, str) and v.strip() for v in source.values()):
        raise ValueError("Missing source observation IDs")
    candidate = next((e for e in scene_result["target_entities"] if e["local_id"] == candidate_id), None)
    if candidate is None:
        raise ValueError("Candidate ID is not a target in this observation")
    marked = marked_candidate_view(image_path, candidate["bbox"])
    with io.BytesIO() as buffer:
        marked.save(buffer, format="PNG")
        url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    context = {"goal": goal_spec, "candidate_id": candidate_id, "bbox": candidate["bbox"],
               "prior_description": candidate["evidence"]}
    messages = [{"role": "system", "content": PROMPT}, {"role": "user", "content": [
        {"type": "text", "text": json.dumps(context, ensure_ascii=False)},
        {"type": "image_url", "image_url": {"url": url}}]}]
    start = perf_counter()
    verdict = client._request_json(messages, max_tokens=1200)
    elapsed_ms = round((perf_counter() - start) * 1000)
    validate_verdict(verdict)
    return {"schema_version": "target_verification_v1", **source, "candidate_id": candidate_id,
            "model_id": client.model, "latency_ms": elapsed_ms, **verdict}
