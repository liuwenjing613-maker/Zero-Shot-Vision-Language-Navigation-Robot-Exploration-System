"""Single-writer session event ledger, recoverable from its JSONL file."""
import json
import os
from pathlib import Path


class EpisodicMemory:
    def __init__(self, session_id, path=None):
        self.session_id = session_id
        self.path = Path(path) if path else None
        self.events = {}
        if self.path and self.path.exists():
            for line in self.path.read_text().splitlines():
                event = json.loads(line)
                if (
                    event["session_id"] != session_id
                    or event["sequence"] != len(self.events) + 1
                    or event["event_id"] in self.events
                ):
                    raise ValueError("Invalid session ledger")
                self.events[event["event_id"]] = event

    def record(self, event_id, event_type, source, timestamp, **refs):
        event = dict(
            event_id=event_id,
            session_id=self.session_id,
            event_type=event_type,
            source=source,
            timestamp=timestamp,
            **refs,
        )
        old = self.events.get(event_id)
        event["sequence"] = old["sequence"] if old else len(self.events) + 1
        line = json.dumps(event, allow_nan=False, sort_keys=True)
        if old:
            if old != json.loads(line):
                raise ValueError("Conflicting event ID")
            return old
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())
        self.events[event_id] = json.loads(line)
        return self.events[event_id]

    def query(self, **refs):
        return [
            v
            for v in self.events.values()
            if all(v.get(k) == x for k, x in refs.items())
        ]

    def recent(self, limit=10, **refs):
        return self.query(**refs)[-limit:] if limit > 0 else []
