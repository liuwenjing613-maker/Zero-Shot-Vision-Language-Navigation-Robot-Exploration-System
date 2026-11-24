"""Bound Ollama context and use structured decoding with the existing perception prompt."""
import json
from urllib.request import Request, urlopen
from vln_perception import VLMClient
from vln_perception.scene_perceptor import CUE_TYPES


def record(properties):
    return dict(
        type="object",
        properties=properties,
        required=list(properties),
        additionalProperties=False,
    )


string = dict(type="string")
box = dict(
    type="array",
    items=dict(type="integer", minimum=0, maximum=1000),
    minItems=4,
    maxItems=4,
)
entity = record(
    dict(
        local_id=string,
        category=string,
        attributes=dict(type="object", additionalProperties=string),
        bbox=box,
        confidence=dict(type="number"),
        evidence=string,
    )
)
cue = record(
    dict(
        local_id=string,
        cue_type=dict(enum=sorted(CUE_TYPES)),
        bbox=box,
        relative_direction=string,
        visibility=dict(enum=["clear", "partial", "uncertain"]),
        evidence=string,
    )
)
schema = record(
    dict(
        target_status=dict(enum=["VISIBLE", "NOT_VISIBLE", "AMBIGUOUS", "UNCERTAIN"]),
        target_entities=dict(type="array", items=entity),
        context_entities=dict(type="array", items=entity),
        visual_cues=dict(type="array", items=cue),
    )
)


class OllamaPerceptor(VLMClient):
    def _request_json(self, messages, *, max_tokens):
        parts = messages[1]["content"]
        user = dict(
            role="user",
            content=parts[0]["text"],
            images=[parts[1]["image_url"]["url"].split(",", 1)[1]],
        )
        url = self.url.split("/v1/")[0] + "/api/chat"
        req = dict(
            model=self.model,
            messages=[messages[0], user],
            format=schema,
            stream=False,
            think=False,
            options=dict(num_ctx=16384, num_predict=12288, temperature=0),
        )
        with urlopen(
            Request(
                url,
                data=json.dumps(req).encode(),
                headers={"Content-Type": "application/json"},
            ),
            timeout=self.timeout,
        ) as r:
            answer = json.load(r)
        if answer.get("done_reason") == "length":
            raise ValueError("Truncated Ollama response")
        return json.loads(answer["message"]["content"])
