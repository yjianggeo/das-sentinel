"""Display regions from qualified evidence; never infer class labels."""

import numpy as np


def distance_edges(distance):
    distance = np.asarray(distance, dtype=float)
    if len(distance) == 1:
        return np.array([distance[0] - 0.5, distance[0] + 0.5])
    return np.r_[distance[0] - (distance[1] - distance[0]) / 2,
                 (distance[:-1] + distance[1:]) / 2,
                 distance[-1] + (distance[-1] - distance[-2]) / 2]


def merge_regions(evidence, distance, recognition, sample_count, merge_gap_samples=100):
    edges = distance_edges(distance)
    rows = []
    for row in evidence:
        if row.get("qualified") is not True:
            continue
        start, stop = max(0, int(row["start_sample"])), min(sample_count, int(row["end_sample"]))
        first, last = max(0, int(row["channel_start"])), min(len(distance), int(row["channel_stop"]))
        peak = int(row["peak_sample"])
        if start >= stop or first >= last or not 0 <= peak < sample_count:
            continue
        selected = np.flatnonzero(recognition[first:last]) + first
        if not len(selected):
            continue
        rows.append({"start_sample": start, "end_sample": stop, "peak_sample": peak,
                         "channel_start": int(selected[0]), "channel_stop": int(selected[-1]) + 1,
                         "detector": str(row["detector"]), "score": float(row["score"])})
    rows.sort(key=lambda row: (row["start_sample"], row["end_sample"], row["channel_start"],
                              row["channel_stop"], row["peak_sample"], row["detector"], -row["score"]))
    groups = []
    for row in rows:
        for group in groups:
            # Complete-link overlap avoids bridging separate signals into one long box.
            if all(max(row["start_sample"], other["start_sample"]) <=
                   min(row["end_sample"], other["end_sample"]) + merge_gap_samples
                   and max(row["channel_start"], other["channel_start"]) <
                   min(row["channel_stop"], other["channel_stop"]) for other in group):
                group.append(row)
                break
        else:
            groups.append([row])
    regions = []
    for index, group in enumerate(groups, 1):
        strongest = min(group, key=lambda row: (-row["score"], row["peak_sample"]))
        first, last = min(row["channel_start"] for row in group), max(row["channel_stop"] for row in group)
        regions.append({"id": index, "start_sample": min(row["start_sample"] for row in group),
                            "end_sample": max(row["end_sample"] for row in group),
                            "peak_sample": strongest["peak_sample"], "channel_start": first, "channel_stop": last,
                            "distance_start_m": float(edges[first]), "distance_stop_m": float(edges[last]),
                            "score": strongest["score"], "detectors": sorted({row["detector"] for row in group}),
                            "evidence_count": len(group)})
    return regions


def preview_bounds(region, sample_count, channel_count):
    start, stop = region["start_sample"], region["end_sample"]
    width = min(sample_count, max(300, stop - start + 100))
    first = max(0, min(sample_count - width, (start + stop - width) // 2))
    return (first, first + width, max(0, region["channel_start"] - 1),
            min(channel_count, region["channel_stop"] + 1))
