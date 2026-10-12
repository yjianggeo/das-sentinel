from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
from das_sentinel.domain import DasBlock, DasMetadata
from das_sentinel.low_frequency_plots import render_low_frequency_plots
from matplotlib.figure import Figure


def test_low_frequency_plots_keep_nonuniform_distance_and_site_reference(
    tmp_path, monkeypatch
):
    data = np.tile(
        np.sin(np.arange(6000) * 2 * np.pi * 0.03 / 5)[:, None], (1, 3)
    ).astype(np.float32)
    metadata = DasMetadata(
        "source",
        datetime(2026, 6, 16, 9, tzinfo=UTC),
        5.0,
        np.array([500.0, 504.0, 800.0]),
        "input-unit",
    )
    window = SimpleNamespace(
        block=DasBlock(data, metadata, 0),
        analysis_start=1500,
        analysis_stop=4500,
        sources=(
            {
                "boundary_utc": "2026-06-16T09:06:00+00:00",
                "previous_seam_seconds": 0.01,
            },
            {"boundary_utc": "2026-06-16T09:07:00+00:00", "previous_seam_seconds": 0.0},
            {
                "boundary_utc": "2026-06-16T09:08:00+00:00",
                "previous_seam_seconds": -0.01,
            },
            {
                "boundary_utc": "2026-06-16T09:01:00+00:00",
                "previous_seam_seconds": 0.01,
            },
        ),
    )
    analysis = SimpleNamespace(
        filtered_data=data[1500:4500],
        valid_channel_mask=np.ones(3, dtype=bool),
        candidates=(),
    )
    figures = []
    seam_positions = []
    original = Figure.savefig

    def save(fig, *args, **kwargs):
        seam_positions.append(
            [
                float(line.get_xdata()[0])
                for line in fig.axes[0].lines
                if line.get_linestyle() == "--"
            ]
        )
        if not figures:
            figures.append(
                (
                    fig.axes[0].collections[0].get_coordinates().copy(),
                    fig._suptitle.get_text(),
                )
            )
        return original(fig, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", save)
    plots = render_low_frequency_plots(window, analysis, tmp_path, timezone_name="UTC")
    assert set(plots) == {"waterfall", "waveform", "psd", "stft"}
    coordinates, title = figures[0]
    np.testing.assert_allclose(coordinates[:, 0, 1], [498.0, 502.0, 652.0, 948.0])
    assert "09:05:00" in title
    assert seam_positions == [[60.0, 180.0], [60.0, 180.0], [], [60.0, 180.0]]
    for path in plots.values():
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
