import csv
import os
import time
from typing import Dict


class CSVLogger:
    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["timestamp", "label", "shape", "color", "bin", "confidence"])

    def log(self, record: Dict[str, str]) -> None:
        with open(self.path, "a", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                record.get("label", ""),
                record.get("shape", ""),
                record.get("color", ""),
                record.get("bin", ""),
                record.get("confidence", ""),
            ])
