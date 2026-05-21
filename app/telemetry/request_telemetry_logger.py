import json
from datetime import datetime, timezone
from pathlib import Path


class RequestTelemetryLogger:
    def __init__(self, log_path: str = "logs/request_telemetry.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: dict) -> None:
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **event,
        }

        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, default=str) + "\n")