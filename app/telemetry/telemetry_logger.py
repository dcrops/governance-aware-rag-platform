import os
import json
from app.models.retrieval_log import RetrievalLog

class TelemetryLogger:
    """
    Persists RAG retrieval telemetry logs to a local JSONL file.
    """

    def __init__(self, log_path: str = "logs/retrieval_logs.jsonl") -> None:
        """
        Initialize the TelemetryLogger.

        Ensures the parent log directory exists.

        Args:
            log_path: Path to the JSONL log file.
        """
        self.log_path = log_path
        parent_dir = os.path.dirname(os.path.abspath(self.log_path))
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

    def log_retrieval(self, log: RetrievalLog) -> None:
        """
        Appends a RetrievalLog record as a JSON line to the log file.

        Args:
            log: A RetrievalLog object to be logged.

        Raises:
            ValueError: If the log object cannot be serialized.
        """
        if not hasattr(log, "dict") and not hasattr(log, "model_dump"):
            raise ValueError("log must provide a .dict() or .model_dump() method for serialization.")

        # Try model_dump (Pydantic v2), fallback to dict (Pydantic v1)
        if hasattr(log, "model_dump"):
            log_dict = log.model_dump()
        elif hasattr(log, "dict"):
            log_dict = log.dict()
        else:
            raise ValueError("Unable to serialize RetrievalLog object.")

        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_dict, ensure_ascii=False) + "\n")
        except Exception as e:
            raise RuntimeError(f"Failed to write retrieval log: {e}")