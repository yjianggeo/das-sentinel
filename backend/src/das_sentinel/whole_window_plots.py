"""Full-distance and local waterfalls from cached 100 Hz samples."""

import io
from zoneinfo import ZoneInfo

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from das_sentinel.window_regions import distance_edges, merge_regions, preview_bounds

# Normalized image coordinates, measured from its top-left corner.
PLOT_AXES = [0.085, 0.12, 0.78, 0.73]


def render_waterfall_preview(data, distance, start, recognition, unit, *, filtered, regions=(),
                             site_timezone="Asia/Shanghai", sample_offset=0):
    figure = Figure(figsize=(12, 5), dpi=130)
    FigureCanvasAgg(figure)
    left, top, width, height = PLOT_AXES
    ax = figure.add_axes([left, 1 - top - height, width, height])
    reference = start.astimezone(ZoneInfo(site_timezone)).strftime("%Y-%m-%d %H:%M:%S %Z %z")
    ax.set_title(f"{'0.5-40 Hz bandpass' if filtered else '100 Hz, no bandpass'} | Reference: {reference}")
    seconds = (sample_offset + np.arange(len(data) + 1)) / 100
    edges = distance_edges(distance)
    scale_data = data[:, recognition] if np.any(recognition) else data
    limit = max(float(np.percentile(np.abs(scale_data), 99)), 1e-20)
    mesh = ax.pcolormesh(seconds, edges, data.T, cmap="RdBu_r", vmin=-limit, vmax=limit,
                         shading="flat", rasterized=True)
    figure.colorbar(mesh, cax=figure.add_axes([0.885, 1 - top - height, 0.018, height]),
                    label=unit + "\n99% display clip in recognition range")
    for index in np.flatnonzero(~recognition):
        ax.axhspan(edges[index], edges[index + 1], color="grey", alpha=0.10)
    for region in regions:
        x = region["start_sample"] / 100
        y = region["distance_start_m"]
        ax.add_patch(Rectangle((x, y), (region["end_sample"] - region["start_sample"]) / 100,
                               region["distance_stop_m"] - y, fill=False,
                               edgecolor="#cf8a00", linewidth=1.2))
        ax.text(x, region["distance_stop_m"], str(region["id"]), color="#7a4700", fontsize=8,
                va="top", clip_on=True, bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1})
    ax.set(xlabel="Seconds from reference", ylabel="Native distance (m)",
           xlim=(seconds[0], seconds[-1]), ylim=(edges[0], edges[-1]))
    payload = io.BytesIO()
    figure.savefig(payload, format="png", dpi=130)
    figure.clear()
    return payload.getvalue()


def create_visualization(raw, filtered, distance, start, recognition, unit, evidence, *,
                         merge_gap_seconds, site_timezone="Asia/Shanghai"):
    regions = merge_regions(evidence, distance, recognition, len(raw), round(merge_gap_seconds * 100))
    edges = distance_edges(distance)
    metadata = {"visualization_version": 2, "waterfall_regions_baked": False, "detection_regions": regions,
                    "plot_geometry": {"axes": PLOT_AXES.copy(), "distance_min_m": float(edges[0]),
                                       "distance_max_m": float(edges[-1])}}
    filtered = np.asarray(filtered, dtype=np.float32)
    if filtered.shape != raw.shape or not np.isfinite(filtered).all():
        raise ValueError("invalid filtered visualization array")
    buffer = io.BytesIO()
    np.savez_compressed(buffer, waveform=filtered.T)
    files = {"filtered_waveform_v2.npz": buffer.getvalue()}
    for name, data, is_filtered in (("waterfall_raw_v2.png", raw, False),
                                    ("waterfall_filtered_v2.png", filtered, True)):
        files[name] = render_waterfall_preview(data, distance, start, recognition, unit, filtered=is_filtered,
                                 site_timezone=site_timezone)
    return metadata, files


def render_region_preview(data, distance, start, recognition, unit, region, *,
                          filtered, site_timezone="Asia/Shanghai", show_regions=True):
    a, b, c, d = preview_bounds(region, len(data), len(distance))
    return render_waterfall_preview(data[a:b, c:d], distance[c:d], start, recognition[c:d], unit,
                      filtered=filtered, regions=[region] if show_regions else (), site_timezone=site_timezone,
                      sample_offset=a)
