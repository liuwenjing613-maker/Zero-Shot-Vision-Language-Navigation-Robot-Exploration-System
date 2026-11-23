"""Evaluate GoalSpec output against six fixed bilingual cases."""
import argparse
import json
from pathlib import Path
from time import monotonic
from vln_perception import VLMClient


MODELS = {"qwen3.8:27b": 18181, "qwen3-vl:8b": 11434}
# instruction, category, attributes, (relation, reference), approach required
CASES = [
    ("Find a chair.", "chair", {}, [], True),
    ("请找到红色的椅子。", "chair", {"color": "red"}, [], True),
    ("Find the wooden table near the window.", "table", {"material": "wood"}, [("near", "window")], True),
    ("找到卧室里的床。", "bed", {}, [("in", "bedroom")], True),
    ("Find the lamp to the left of the blue sofa.", "lamp", {}, [("left_of", "blue sofa")], True),
    ("只需要确认是否看得到冰箱，不要靠近。", "refrigerator", {}, [], False),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=list(MODELS), help="Run one model (default: both)")
    parser.add_argument("--base-url", help="Override endpoint for a single selected model")
    args = parser.parse_args()
    if args.base_url and not args.model:
        parser.error("--base-url requires --model")
    with args.output.open("x", encoding="utf-8") as out:
        for model, port in MODELS.items():
            if args.model and model != args.model:
                continue
            client = VLMClient(args.base_url or f"http://127.0.0.1:{port}/v1", model)
            for instruction, category, attrs, relations, approach in CASES:
                expected = {"instruction": instruction, "target_category": category,
                            "target_attributes": attrs,
                            "spatial_relations": [{"relation": r, "reference_object": ref} for r, ref in relations],
                            "completion_conditions": {"target_confirmed": True, "approach_required": approach}}
                record = {"model": model, "instruction": instruction, "expected": expected}
                start = monotonic()
                try:
                    actual = client.parse_goal(instruction)
                    record.update(actual=actual, schema_ok=True, exact=(actual == expected))
                except Exception as exc:
                    record.update(error=f"{type(exc).__name__}: {exc}", schema_ok=False, exact=False)
                record["seconds"] = round(monotonic() - start, 3)
                print(json.dumps(record, ensure_ascii=False), file=out, flush=True)
                print(model, instruction, record["seconds"], record["exact"], record.get("error", ""), flush=True)


if __name__ == "__main__":
    main()
