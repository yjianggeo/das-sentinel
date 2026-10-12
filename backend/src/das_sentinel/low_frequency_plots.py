from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FormatStrFormatter
from scipy.signal import spectrogram, welch


def render_low_frequency_plots(
    window, analysis, directory: Path, *, timezone_name: str = "Asia/Shanghai"
) -> dict[str, Path]:
    """All four plots use the same continuous, trimmed, unnormalised low-frequency data."""
    fs = window.block.metadata.sample_rate_hz
    data = analysis.filtered_data
    valid = analysis.valid_channel_mask
    distances = window.block.metadata.channel_distance_m
    start = window.block.metadata.start_time_utc + timedelta(
        seconds=window.analysis_start / fs
    )
    reference = start.astimezone(ZoneInfo(timezone_name)).strftime(
        "%Y-%m-%d %H:%M:%S.%f"
    )[:-3]
    duration = len(data) / fs
    seam_positions = [
        (datetime.fromisoformat(source["boundary_utc"]) - start).total_seconds()
        for source in window.sources
        if source["previous_seam_seconds"]
    ]
    seam_positions = [seconds for seconds in seam_positions if 0 <= seconds < duration]
    unit = window.block.metadata.unit
    values = data[:, valid]
    if values.shape[1] == 0:
        raise ValueError("所选距离内没有有效低频通道")
    plots = {}

    def figure(title):
        fig = Figure(figsize=(11, 5), dpi=120, layout="constrained")
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        fig.suptitle(
            f"{title} | 0.01-0.4 Hz | {duration:g} s\nLocal reference (t=0): {reference} ({timezone_name})",
            fontsize=11,
        )
        return fig, ax

    def time_axis(ax):
        ax.set_xlabel("Time relative to reference (s)")
        ax.set_xlim(0, duration)
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        for item in analysis.candidates:
            ax.axvspan(
                item["start_seconds"], item["end_seconds"], color="orange", alpha=0.12
            )
        for seconds in seam_positions:
            ax.axvline(seconds, color="black", linestyle="--", linewidth=0.6, alpha=0.3)
        if seam_positions:
            ax.text(
                0.01,
                0.98,
                "Dashed: source clock seams",
                transform=ax.transAxes,
                va="top",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
            )

    def save(fig, name):
        path = directory / f"{name}.png"
        fig.savefig(path)
        fig.clear()
        plots[name] = path

    fig, ax = figure("Continuous low-frequency waterfall")
    stride = max(1, int(np.ceil(len(data) / 1600)))
    displayed = data[::stride].copy()
    displayed[:, ~valid] = np.nan
    scale = float(np.quantile(np.abs(values), 0.99)) or 1.0
    if len(distances) > 1:
        distance_edges = np.r_[
            distances[0] - (distances[1] - distances[0]) / 2,
            (distances[:-1] + distances[1:]) / 2,
            distances[-1] + (distances[-1] - distances[-2]) / 2,
        ]
    else:
        distance_edges = np.array([distances[0] - 0.5, distances[0] + 0.5])
    time_edges = np.r_[np.arange(0, len(data), stride) / fs, duration]
    mesh = ax.pcolormesh(
        time_edges,
        distance_edges,
        displayed.T,
        cmap="RdBu_r",
        vmin=-scale,
        vmax=scale,
        shading="flat",
    )
    fig.colorbar(mesh, ax=ax, label=unit)
    ax.set_ylabel("Cable distance (m)")
    time_axis(ax)
    save(fig, "waterfall")

    fig, ax = figure("Low-frequency waveform")
    power = np.mean(values**2, axis=0)
    representative = int(np.argsort(power)[len(power) // 2])
    times = np.arange(len(data)) / fs
    distance = distances[valid][representative]
    ax.plot(
        times,
        values[:, representative],
        linewidth=0.8,
        alpha=0.8,
        label=f"Channel at {distance:g} m",
    )
    ax.plot(
        times,
        np.median(values, axis=1),
        linewidth=0.9,
        label="Median across valid channels",
    )
    ax.set_ylabel(unit)
    ax.legend(loc="upper right")
    time_axis(ax)
    save(fig, "waveform")

    nperseg = min(len(data), round(300 * fs))
    frequencies, density = welch(values, fs=fs, nperseg=nperseg, axis=0)
    density = np.mean(density, axis=1)
    fig, ax = figure(f"Low-frequency PSD | frequency spacing {fs / nperseg:.4f} Hz")
    selected = (frequencies >= 0.005) & (frequencies <= 0.8)
    ax.semilogy(
        frequencies[selected], np.maximum(density[selected], np.finfo(float).tiny)
    )
    ax.axvspan(0.01, 0.4, color="steelblue", alpha=0.08)
    ax.set_xlim(0.005, 0.5)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(f"PSD ({unit})^2 / Hz")
    ax.grid(alpha=0.2)
    save(fig, "psd")

    nperseg = min(len(data), round(200 * fs))
    frequencies, times, density = spectrogram(
        values, fs=fs, nperseg=nperseg, noverlap=round(nperseg * 0.8), axis=0
    )
    density = np.mean(density, axis=1)
    selected = (frequencies >= 0.005) & (frequencies <= 0.4)
    fig, ax = figure(f"Low-frequency spectrogram | window {nperseg / fs:g} s")
    mesh = ax.pcolormesh(
        times,
        frequencies[selected],
        10 * np.log10(np.maximum(density[selected], np.finfo(float).tiny)),
        shading="nearest",
        cmap="viridis",
    )
    fig.colorbar(mesh, ax=ax, label="PSD (dB re 1 input-unit^2 / Hz)")
    ax.set_ylabel("Frequency (Hz)")
    time_axis(ax)
    save(fig, "stft")
    return plots
