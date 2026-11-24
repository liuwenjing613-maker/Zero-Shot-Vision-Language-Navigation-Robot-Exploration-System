"""Native Ollama vision transport for the project schema, with measured model timings."""
import json, time
from urllib.request import urlopen, Request
from vln_perception import VLMClient
from ollama_transport import schema
from model import HOST, MODEL


class FastVision(VLMClient):
    def __init__(self):
        super().__init__(HOST + "/v1", MODEL, timeout=240)
        self.timing = {}

    def _request_json(self, messages, *, max_tokens):
        parts = messages[1]["content"]
        user = dict(
            role="user",
            content=parts[0]["text"],
            images=[parts[1]["image_url"]["url"].split(",", 1)[1]],
        )
        payload = dict(
            model=MODEL,
            stream=False,
            think=False,
            keep_alive=-1,
            format=schema,
            messages=[messages[0], user],
            options=dict(num_ctx=4096, num_predict=max_tokens, temperature=0),
        )
        start = time.perf_counter()
        with urlopen(
            Request(
                HOST + "/api/chat",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            ),
            timeout=self.timeout,
        ) as r:
            d = json.load(r)
        self.timing = {
            k: d.get(k)
            for k in (
                "total_duration",
                "load_duration",
                "prompt_eval_count",
                "prompt_eval_duration",
                "eval_count",
                "eval_duration",
                "done_reason",
            )
        }
        self.timing.update(
            wall_s=round(time.perf_counter() - start, 3),
            thinking_chars=len(d["message"].get("thinking", "")),
        )
        if d.get("model") != MODEL:
            raise ValueError("Wrong model returned")
        if d.get("done_reason") == "length":
            raise ValueError("Vision response truncated")
        return json.loads(d["message"]["content"])
