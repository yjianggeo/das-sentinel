"""Spatial coverage of cached, filtered DAS waveforms, not earthquake probability."""

import numpy as np

BOREHOLE_COVERAGE_VERSION = "borehole-coverage-v3"
BOREHOLE_COVERAGE_READ_VERSIONS = (
    "borehole-coverage-v1", "borehole-coverage-v2", BOREHOLE_COVERAGE_VERSION,
)


def _rolling_mean(data, width):
    total = np.concatenate((np.zeros_like(data[:1]), np.cumsum(data, axis=0)), axis=0)
    return (total[width:] - total[:-width]) / width


def _lag_observation(first, second, max_lag, rate):
    """Integer-sample lag with overlap normalization and separated alias checks."""
    first, second = first - first.mean(), second - second.mean()
    count = len(first)
    limit = min(max_lag, count - max(3, round(0.5 * rate)))
    lags = np.arange(-limit, limit + 1)
    fft_size = 1 << (2 * count - 1).bit_length()
    products = np.fft.irfft(
        np.fft.rfft(second, fft_size) * np.conj(np.fft.rfft(first, fft_size)), fft_size
    )[lags % fft_size]
    first_power = np.concatenate(([0.0], np.cumsum(first ** 2)))
    second_power = np.concatenate(([0.0], np.cumsum(second ** 2)))
    left_start, right_start = np.maximum(0, -lags), np.maximum(0, lags)
    length = count - np.abs(lags)
    denominator = np.sqrt(
        (first_power[left_start + length] - first_power[left_start])
        * (second_power[right_start + length] - second_power[right_start])
    )
    correlation = np.zeros_like(products)
    np.divide(products, denominator, out=correlation, where=denominator > 0)
    strength = np.abs(correlation)
    peak = int(np.argmax(strength))
    separated = np.abs(lags - lags[peak]) > max(2, round(0.03 * rate))
    local_peak = np.zeros(len(lags), dtype=bool)
    local_peak[1:-1] = (strength[1:-1] >= strength[:-2]) & (strength[1:-1] >= strength[2:])
    local_peak[[0, -1]] = True
    ambiguous = bool(np.any(separated & local_peak & (strength >= 0.9 * strength[peak])))
    return int(lags[peak]), float(strength[peak]), ambiguous, abs(lags[peak]) == limit


def _propagation_diagnostics(samples, distance, rate, varying, normalized, native_spacing):
    result = {
        "propagation_slowness_s_per_m": None, "apparent_velocity_mps": None,
        "propagation_residual_s": None, "propagation_direction": None,
        "propagation_confidence": None, "propagation_status": "insufficient_support",
        "common_mode_status": "unknown", "common_mode_penalty": 0.0,
    }
    usable = np.flatnonzero(varying)
    if len(usable) < 6 or np.max(np.diff(distance[usable]), initial=0) > 1.5 * native_spacing:
        return result
    targets = np.linspace(distance[usable[0]], distance[usable[-1]], 12)
    anchors = np.unique([usable[np.argmin(np.abs(distance[usable] - target))] for target in targets])
    positions = distance[anchors]
    traces = samples[:, anchors] - np.mean(samples[:, anchors], axis=0)
    traces /= np.sqrt(np.mean(traces ** 2, axis=0))
    width = int(2 * rate)
    if width < 3:
        return result
    powers = _rolling_mean(traces ** 2, width)
    start = int(np.argmax(powers.mean(axis=1)))
    stop = start + width
    if start < width or stop + width > len(traces):
        result["propagation_status"] = "edge_or_boundary"
        return result
    pairs = [(left, right) for left in range(len(anchors)) for right in range(left + 1, len(anchors))
             if 500 <= positions[right] - positions[left] <= 1000]
    if len(pairs) > 40:
        pairs = [pairs[index] for index in np.linspace(0, len(pairs) - 1, 40, dtype=int)]
    if len(pairs) < 5 or len({index for pair in pairs for index in pair}) < 6:
        return result
    observations, aliases, boundaries = [], 0, 0
    for left, right in pairs:
        baseline = positions[right] - positions[left]
        lag, correlation, ambiguous, boundary = _lag_observation(
            traces[start:stop, left], traces[start:stop, right],
            max(1, int(0.002 * baseline * rate)), rate,
        )
        if correlation < 0.6:
            continue
        aliases += ambiguous
        boundaries += boundary
        if not ambiguous and not boundary:
            observations.append((baseline, lag / rate, correlation))
    if aliases >= max(2, len(pairs) / 4):
        result["propagation_status"] = "ambiguous_periodic"
        return result
    if len(observations) < max(5, 0.6 * len(pairs)):
        result["propagation_status"] = "edge_or_boundary" if boundaries else "low_correlation"
        return result
    baseline, delays, correlations = np.asarray(observations).T
    if np.all(delays == 0):
        result["propagation_status"] = "unresolved_zero_lag"
        if np.median(correlations) < 0.9 or len(observations) < 0.8 * len(pairs):
            return result
        # Check original normalized channel energy: opposite polarities cannot cancel an onset.
        channel_energy = normalized[:, varying]
        shared_energy = np.median(channel_energy, axis=1)
        channel_excess = channel_energy / np.maximum(shared_energy[:, None], 1e-12)
        if np.max(channel_energy) >= 4 or np.max(channel_excess) >= 1.5:
            result["common_mode_status"] = "not_supported"
            return result
        native_power = _rolling_mean(samples ** 2, width)
        # Incomplete edge backgrounds can veto a penalty, never support one.
        if (np.any(native_power[:width] >= 1.5 * native_power[width:2 * width])
                or np.any(native_power[-width:] >= 1.5 * native_power[-2 * width:-width])):
            return result
        peaks = np.argmax(native_power, axis=0)
        if np.any(peaks < width) or np.any(peaks + 2 * width > len(samples)):
            return result
        background = np.minimum(native_power[:-2 * width], native_power[2 * width:])
        if np.any(background <= 0):
            return result
        if np.any(native_power[width:-width] >= 1.5 * background):
            result["common_mode_status"] = "not_supported"
            return result
        polarity = np.sign(traces.T @ traces[:, 0])
        aligned = traces * polarity
        beam_power = _rolling_mean(np.mean(aligned, axis=1) ** 2, width)
        individual_power = powers.mean(axis=1)
        coherence = beam_power / individual_power - 1 / len(anchors)
        prominence = coherence[width:-width] - np.maximum(coherence[:-2 * width], coherence[2 * width:])
        background = np.minimum(beam_power[:-2 * width], beam_power[2 * width:])
        if (np.all(background > 0) and np.mean(coherence >= 0.8) >= 0.8
                and np.all(prominence < 0.04) and np.all(beam_power[width:-width] < 1.5 * background)):
            result["common_mode_status"] = "persistent_supported"
        else:
            result["common_mode_status"] = "not_supported"
        return result
    if np.count_nonzero(delays) <= len(delays) / 2:
        result["propagation_status"] = "insufficient_support"
        return result
    slope = float(np.sum(correlations * baseline * delays) / np.sum(correlations * baseline ** 2))
    residual = float(np.sqrt(np.average((delays - slope * baseline) ** 2, weights=correlations)))
    consistency = float(np.mean(np.sign(delays[delays != 0]) == np.sign(slope)))
    confidence = float(np.median(correlations) * len(observations) / len(pairs)
                       * max(0, 1 - residual / (2 / rate)))
    if (abs(slope) < 0.5 / (rate * np.ptp(positions)) or abs(slope) > 0.002
            or residual > 1.5 / rate or consistency < 0.8 or confidence < 0.6):
        result["propagation_status"] = "insufficient_support"
        return result
    result.update(
        propagation_slowness_s_per_m=slope, apparent_velocity_mps=1 / abs(slope),
        propagation_residual_s=residual, propagation_direction="+" if slope > 0 else "-",
        propagation_confidence=float(np.clip(confidence, 0, 1)), propagation_status="resolved",
        common_mode_status="not_supported",
    )
    return result


def _coverage_measurement(samples, selected_positions, positions, rate, spacing, span):
    """The original coverage calculation, also used on an in-memory residual."""
    seconds = samples.shape[0] // int(rate)
    energy = np.mean(samples.reshape(seconds, int(rate), -1) ** 2, axis=1)
    peak_energy = np.max(energy, axis=0)
    baseline = np.maximum(np.median(energy, axis=0), peak_energy * 1e-12)
    varying = (peak_energy > 0) & (np.ptp(energy, axis=0) > peak_energy * 1e-6)
    normalized = np.zeros_like(energy)
    if not varying.any() or not np.isfinite(energy).all():
        return {}, varying, normalized
    normalized[:, varying] = energy[:, varying] / baseline[varying]
    active = varying & (np.max(normalized, axis=0) > 1.5)

    envelopes = []
    segment_ids = np.floor((selected_positions - 500) / 500).astype(int)
    for segment in np.unique(segment_ids):
        keep = (segment_ids == segment) & varying
        if segment < 1 or keep.sum() < 3:
            continue
        envelope = np.median(normalized[:, keep], axis=1)
        if np.ptp(envelope) > 1e-6:
            envelopes.append(envelope)
    if len(envelopes) < 3:
        return {}, varying, normalized
    correlations = np.corrcoef(envelopes)
    pairs = correlations[np.triu_indices(len(envelopes), 1)]
    if not np.isfinite(pairs).all():
        return {}, varying, normalized
    median_correlation = float(np.median(pairs))

    active_positions = selected_positions[active]
    runs = np.split(
        active_positions,
        np.flatnonzero(np.diff(active_positions) > 1.5 * spacing) + 1,
    )
    continuous_span = max(
        (float(run[-1] - run[0] + spacing) for run in runs if len(run)),
        default=0.0,
    )
    coverage = min(1.0, float(active.sum() * spacing / span))
    deep = positions >= 1000
    deep_span = float(positions[-1] - positions[deep][0] + spacing)
    deep_coverage = min(
        1.0, float(np.sum(active & (selected_positions >= 1000)) * spacing / deep_span)
    )
    score = 100 * max(0.0, median_correlation) * (
        coverage + deep_coverage + min(1.0, continuous_span / span)
    ) / 3
    return {
        "coverage_score": float(np.clip(score, 0, 100)),
        "coverage_fraction": coverage,
        "deep_coverage_fraction": deep_coverage,
        "continuous_span_m": continuous_span,
        "median_correlation": median_correlation,
    }, varying, normalized


def _common_component_diagnostics(samples, selected_positions, positions, rate, spacing, span):
    result = {
        "common_mode_energy_fraction": None, "common_mode_channel_fraction": None,
        "residual_coverage_score": None, "common_mode_status": "unknown",
    }
    if len(samples) < 32:
        return result, False
    centered = samples - np.mean(samples, axis=0)
    power = np.einsum("tc,tc->c", centered, centered)
    usable = np.isfinite(power) & (power > 0) & (np.ptp(samples, axis=0) > 0)
    if usable.sum() < 6:
        return result, False
    normalized = centered[:, usable] / np.sqrt(power[usable] / len(samples))
    template = np.median(normalized, axis=1)
    template_power = float(np.einsum("t,t->", template, template))
    if not np.isfinite(template_power) or template_power <= 1e-12:
        return result, False
    products = np.einsum("tc,t->c", centered, template)[usable]
    correlation = np.clip(products / np.sqrt(power[usable] * template_power), -1, 1)
    energy_fraction = float(np.mean(np.maximum(correlation, 0) ** 2))
    channel_fraction = float(np.mean(correlation >= 0.2))
    result.update(
        common_mode_energy_fraction=energy_fraction,
        common_mode_channel_fraction=channel_fraction,
        common_mode_status="component_supported" if energy_fraction > 0.02 and channel_fraction > 0.6
        else "not_supported",
    )
    coefficients = np.zeros(samples.shape[1])
    coefficients[usable] = np.maximum(products / template_power, 0)
    residual = centered - template[:, None] * coefficients
    residual_power = np.einsum("tc,tc->c", residual, residual)
    # Numerical cancellation cannot establish measurable residual coverage.
    if not np.isfinite(residual_power).all() or residual_power.sum() <= 1e-12 * power.sum():
        return result, False
    residual_metrics, _, _ = _coverage_measurement(
        residual, selected_positions, positions, rate, spacing, span,
    )
    result["residual_coverage_score"] = residual_metrics.get("coverage_score")
    independent_pulse = False
    if energy_fraction > 0.95:
        # Keep a pronounced native-channel onset in an otherwise nearly pure component.
        width = max(2, round(0.2 * rate))
        native_power = _rolling_mean(residual ** 2, width)
        background = np.median(native_power, axis=0)
        independent_pulse = bool(np.any(np.max(native_power, axis=0) > 8 * background))
    return result, independent_pulse


def measure_borehole_coverage(data, distance_m, valid_channel_mask, sample_rate_hz):
    """Measure >=500 m coverage from a caller-filtered (time, channels) array.

    coverage_score retains the original one-second spatial coverage measurement.
    A signed median template independently quantifies common waveform energy;
    only positive projections are removed from a private copy for residual
    coverage. Strong common components with weak residual coverage may reduce
    candidate score by at most 50%. Broad residual coverage, reliable propagation,
    and inadequate evidence protect ranking. These are development-calibrated
    ranking diagnostics, never class probabilities or automatic noise labels.
    """
    result = {
        "algorithm_version": BOREHOLE_COVERAGE_VERSION,
        "score": None,
        "coverage_score": None,
        "coverage_fraction": None,
        "deep_coverage_fraction": None,
        "continuous_span_m": None,
        "median_correlation": None,
        "valid_coverage_fraction": 0.0,
        "propagation_slowness_s_per_m": None,
        "apparent_velocity_mps": None,
        "propagation_residual_s": None,
        "propagation_direction": None,
        "propagation_confidence": None,
        "propagation_status": "insufficient_support",
        "common_mode_status": "unknown",
        "common_mode_penalty": 0.0,
        "common_mode_energy_fraction": None,
        "common_mode_channel_fraction": None,
        "residual_coverage_score": None,
    }
    try:
        values = np.asarray(data, dtype=np.float64)
        distance = np.asarray(distance_m, dtype=np.float64)
        valid = np.asarray(valid_channel_mask, dtype=bool)
        rate = float(sample_rate_hz)
    except (TypeError, ValueError):
        return result
    if (
        values.ndim != 2
        or distance.shape != (values.shape[1],)
        or valid.shape != distance.shape
        or not np.isfinite(distance).all()
        or not np.all(np.diff(distance) > 0)
        or not np.isfinite(rate)
        or rate < 1
        or not rate.is_integer()
    ):
        return result

    borehole = distance >= 500
    positions = distance[borehole]
    if not len(positions):
        return result
    if len(positions) == 1:
        result["valid_coverage_fraction"] = float(valid[borehole].mean())
        return result
    spacing = float(np.median(np.diff(positions)))
    span = float(positions[-1] - positions[0] + spacing)
    selected = borehole & valid
    selected_positions = distance[selected]
    valid_fraction = min(1.0, float(selected.sum() * spacing / span))
    result["valid_coverage_fraction"] = valid_fraction
    if (
        len(positions) < 9
        or values.shape[0] < 8 * rate
        or valid_fraction < 0.5
        or not np.isfinite(values[:, selected]).all()
    ):
        return result

    # The cache was filtered with source context; retain edge-window signals.
    samples_per_second = int(rate)
    samples = values[:, selected]
    seconds = samples.shape[0] // samples_per_second
    samples = samples[:seconds * samples_per_second]
    coverage, varying, normalized = _coverage_measurement(
        samples, selected_positions, positions, rate, spacing, span,
    )
    result.update(coverage)
    result["score"] = result["coverage_score"]
    if result["coverage_score"] is not None:
        result.update(_propagation_diagnostics(samples, selected_positions, rate, varying, normalized, spacing))
    component, independent_pulse = _common_component_diagnostics(
        samples, selected_positions, positions, rate, spacing, span,
    )
    result.update(component)
    raw_score, residual_score = result["coverage_score"], result["residual_coverage_score"]
    if (raw_score is not None and raw_score > 0 and residual_score is not None
            and result["common_mode_status"] == "component_supported"
            and result["propagation_status"] != "resolved" and not independent_pulse):
        retained = min(1.0, residual_score / raw_score)
        if retained < 0.9:
            # Development examples calibrate these soft risk ramps; they are not class cutoffs.
            strength = np.clip((result["common_mode_channel_fraction"] - 0.6) / 0.3, 0, 1)
            shared = np.clip((result["common_mode_energy_fraction"] - 0.02) / 0.1, 0, 1)
            risk = float(strength * shared * (1 - retained))
            result["score"] = raw_score * (1 - 0.5 * risk)
            result["common_mode_penalty"] = raw_score - result["score"]
    return result
