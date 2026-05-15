from __future__ import annotations

from pathlib import Path


class HouseImageAnalyzer:
    """Starter CNN-based image analyzer for house quality scoring.

    This module is intentionally lightweight for easy extension in production.
    """

    def __init__(self) -> None:
        self.model = None

    def load_pretrained(self) -> None:
        """Load a CNN backbone when TensorFlow/PyTorch is available."""
        # Hook for transfer learning setup.
        self.model = "placeholder_cnn"

    def score_image(self, image_path: str | Path) -> float:
        """Return a normalized quality score in [0, 1]."""
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        if self.model is None:
            self.load_pretrained()
        return 0.75
