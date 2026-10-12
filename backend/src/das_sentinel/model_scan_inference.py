"""Persistent, one-window-at-a-time inference in the existing Torch environment."""

import argparse
import json
import math
import sys
from pathlib import Path

from das_sentinel.window_inference import checked_metadata, file_sha


class ResidentPredictor:
    def __init__(self, folder, model):
        import torch

        from das_sentinel.window_benchmark import load_candidate
        self.metadata = checked_metadata(folder, model)
        self.snapshot = dict(model)
        torch.set_num_threads(2)
        self.model = load_candidate(folder)

    def predict(self, arrays):
        import torch

        from das_sentinel.window_benchmark import candidate_inputs
        with torch.inference_mode():
            score = self.model(*candidate_inputs(arrays, self.metadata)).sigmoid().item()
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("invalid model score")
        threshold = self.snapshot["threshold"]
        return {"status": "succeeded", "score": score, "threshold": threshold,
                "label": "present" if score >= threshold else "absent", "model_id": self.snapshot["id"]}


def serve(predictor, spool_root, source, output):
    import numpy as np
    root = Path(spool_root).resolve(strict=True)
    for line in source:
        request_id = None
        try:
            if len(line) > 16384:
                raise ValueError("request too large")
            request = json.loads(line)
            request_id = request["request_id"]
            path = Path(request["path"])
            if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root) or not path.is_file():
                raise ValueError("inference path outside spool")
            if file_sha(path) != request["sha256"]:
                raise ValueError("inference input hash mismatch")
            with np.load(path, allow_pickle=False) as arrays:
                result = predictor.predict(arrays)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            result = {"status": "failed", "error_code": "inference_failed", "error_message": str(error)[:1000]}
            if isinstance(error, RuntimeError):
                result["error_kind"] = "runtime"
        output.write(json.dumps(result | {"request_id": request_id}, allow_nan=False) + "\n")
        output.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["serve"])
    parser.add_argument("--model-folder", required=True, type=Path)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--spool-root", required=True, type=Path)
    args = parser.parse_args()
    predictor = ResidentPredictor(args.model_folder, json.loads(args.snapshot.read_text(encoding="utf8")))
    serve(predictor, args.spool_root, sys.stdin, sys.stdout)


if __name__ == "__main__":
    main()
