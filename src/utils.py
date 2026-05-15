from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"


def ensure_directories() -> None:
    """Create runtime directories if they do not exist."""
    for directory in (DATA_DIR, MODELS_DIR, LOGS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def setup_logging(name: str = "house_price") -> logging.Logger:
    """Configure application logger with console and file handlers."""
    ensure_directories()
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(LOGS_DIR / "pipeline.log", encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def save_json(data: dict[str, Any], path: Path) -> None:
    """Save dictionary as a pretty JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: Path) -> dict[str, Any]:
    """Load JSON file into dictionary."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class RunArtifacts:
    model_path: Path
    metrics_path: Path
    plots_dir: Path
    report_path: Path


def timestamp_now() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def read_env(name: str, default: str) -> str:
    return os.getenv(name, default)
