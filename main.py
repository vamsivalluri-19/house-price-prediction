from __future__ import annotations

import argparse

import uvicorn

from src.pipeline import automate_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="House Price Prediction System")
    parser.add_argument(
        "command",
        choices=["train", "serve"],
        help="train: run model pipeline, serve: start FastAPI service",
    )
    parser.add_argument("--data-path", default=None, help="Optional CSV dataset path")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "train":
        summary = automate_training(data_path=args.data_path)
        print("Training completed")
        for key, value in summary.items():
            print(f"{key}: {value}")
    elif args.command == "serve":
        uvicorn.run("src.api:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
