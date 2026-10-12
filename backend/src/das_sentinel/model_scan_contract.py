"""JSON identities shared by the API, reader and isolated Torch process."""

import hashlib
import itertools
import json
import math
from datetime import UTC, datetime

PROTOCOL = "model-scan-v1"
DIRECTORIES = (
    "jjk2023", "jjk20230919", "jjk20230926", "jjk20230930", "jjk20240127",
    "jjk20240416", "jjk20240424", "jjk20240513", "jjk20240521", "jjk20240527",
    "jjk20240608", "jjk2025_2026",
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def utc(value):
    stamp = datetime.fromisoformat(value) if isinstance(value, str) else value
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError("time must include timezone")
    return stamp.astimezone(UTC)


def excluded_two_metre_source(source):
    """User-excluded 2023 acquisition geometry, using inspected channel coordinates."""
    distance = source.get("distance_m", [])
    return (utc(source["start_utc"]).year == 2023 and len(distance) == 1663
            and all(math.isclose(right - left, 2.044989824295044, rel_tol=0, abs_tol=1e-7)
                    for left, right in itertools.pairwise(distance)))


def inference_key(site_id, start_utc, sources, model, input_protocol=PROTOCOL):
    return digest({"site_id": site_id, "start_utc": utc(start_utc).isoformat(),
                   "window_seconds": 30, "sources": sorted(sources, key=lambda s: s["path"]),
                   "input_protocol": input_protocol, "model_id": model["id"],
                   "weights_sha256": model["weights_sha256"],
                   "metadata_sha256": model["metadata_sha256"], "threshold": model["threshold"]})
