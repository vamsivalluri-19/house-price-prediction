from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Optional

from src.train import run_training_pipeline
from src.utils import save_json, setup_logging

logger = setup_logging("pipeline")


def automate_training(data_path: Optional[str] = None) -> dict[str, str]:
    """Run full training pipeline and emit a compact run summary."""
    artifacts = run_training_pipeline(data_path=data_path)
    summary = {
        "model_path": str(artifacts.model_path),
        "metrics_path": str(artifacts.metrics_path),
        "plots_dir": str(artifacts.plots_dir),
        "report_path": str(artifacts.report_path),
    }
    save_json(summary, Path(artifacts.report_path).with_name("latest_run.json"))
    logger.info("Automated pipeline run complete")
    return summary
