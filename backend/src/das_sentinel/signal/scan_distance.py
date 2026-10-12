from dataclasses import replace

import numpy as np

from das_sentinel.config import ProcessingProfile
from das_sentinel.domain import DasBlock


def scan_channel_mask(distances: np.ndarray, profile: ProcessingProfile) -> np.ndarray:
    """Select inclusive physical distances without renumbering source channels."""
    start, stop = profile.scan_distance_start_m, profile.scan_distance_stop_m
    if start is None and stop is None:
        return np.ones(len(distances), dtype=bool)
    selected = np.isfinite(distances)
    if start is not None:
        selected &= distances >= start
    if stop is not None:
        selected &= distances <= stop
    if not selected.any():
        raise ValueError("scan distance range does not contain any source channels")
    return selected


def mask_scan_block(block: DasBlock, profile: ProcessingProfile) -> DasBlock:
    """Exclude data before shared statistics, retaining original source indices."""
    selected = scan_channel_mask(block.metadata.channel_distance_m, profile)
    if selected.all():
        return block
    data = block.data.copy()
    data[:, ~selected] = 0.0
    return replace(block, data=data)
