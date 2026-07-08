from __future__ import annotations

from typing import List, Tuple

import numpy as np

from eeg_report_multiagent.schemas.measurement import MeasurementValue, StatusSemantic
from eeg_report_multiagent.tools.common import (
    make_categorical_measurement,
    make_distribution_measurement,
    make_exact_measurement,
    make_provenance,
    make_range_measurement,
    make_status_measurement,
    make_upper_bound_measurement,
)

LEFT_CHANNELS = {"Fp1", "F3", "F7", "C3", "T3", "T5", "P3", "O1", "A1"}
RIGHT_CHANNELS = {"Fp2", "F4", "F8", "C4", "T4", "T6", "P4", "O2", "A2"}
FRONTAL_CHANNELS = {"Fp1", "Fp2", "Fpz", "F3", "F4", "F7", "F8", "Fz"}
TEMPORAL_CHANNELS = {"F7", "F8", "T3", "T4", "T5", "T6"}
POSTERIOR_CHANNELS = {"O1", "O2", "P3", "P4", "Pz", "T5", "T6"}
LEFT_TEMPORAL_CHANNELS = {"F7", "T3", "T5"}
RIGHT_TEMPORAL_CHANNELS = {"F8", "T4", "T6"}
LEFT_FRONTAL_CHANNELS = {"Fp1", "F3", "F7"}
RIGHT_FRONTAL_CHANNELS = {"Fp2", "F4", "F8"}


def _infer_uv_scale(signal_nct: np.ndarray) -> float:
    return 1_000_000.0 if float(np.percentile(np.abs(signal_nct), 95.0)) < 1.0 else 1.0


def transient_candidate_score(signal_nct: np.ndarray, source_ref: str) -> List[MeasurementValue]:
    # per-window score from robust max amplitude + derivative energy
    deriv = np.diff(signal_nct, axis=-1)
    amp = np.percentile(np.abs(signal_nct), 99, axis=(1, 2))
    sharp = np.percentile(np.abs(deriv), 99, axis=(1, 2))
    score = (amp / (np.median(amp) + 1e-12) + sharp / (np.median(sharp) + 1e-12)) / 2.0

    return [
        make_distribution_measurement(
            measurement_id="m_event_candidate_score_distribution",
            measurement_name="event_candidate_score_distribution",
            values=[float(x) for x in score],
            unit="score",
            provenance=make_provenance(
                tool_name="transient_candidate_score",
                function_name="transient_candidate_score",
                source_ref=source_ref,
                window_indices=range(signal_nct.shape[0]),
            ),
        ),
        make_exact_measurement(
            measurement_id="m_event_candidate_burden_ratio",
            measurement_name="event_candidate_burden_ratio",
            value=float(np.mean(score > np.percentile(score, 90))),
            unit="ratio",
            provenance=make_provenance(
                tool_name="transient_candidate_score",
                function_name="transient_candidate_score",
                source_ref=source_ref,
                window_indices=range(signal_nct.shape[0]),
            ),
        ),
    ]


def spike_wave_candidate_score(signal_nct: np.ndarray, window_seconds: int, source_ref: str) -> List[MeasurementValue]:
    """Rank windows for rhythmic spike-wave-like activity before numeric extraction.

    The legacy transient score mostly finds large or sharp windows. This score
    adds a bounded rhythmic prior so waveform numeric extraction is less likely
    to summarize slow drifts or high-amplitude artifacts.
    """
    if signal_nct.size == 0:
        return []
    n_windows, _n_channels, n_samples = signal_nct.shape
    fs = float(n_samples) / float(window_seconds) if window_seconds > 0 else 0.0
    if fs <= 0:
        return []

    x = signal_nct.astype(np.float32) * _infer_uv_scale(signal_nct)
    x = x - np.median(x, axis=-1, keepdims=True)
    window = np.hanning(n_samples).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(x * window[None, None, :], axis=-1)) ** 2
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)

    def band_power(low: float, high: float) -> np.ndarray:
        mask = (freqs >= low) & (freqs <= high)
        if not np.any(mask):
            return np.zeros((n_windows, signal_nct.shape[1]), dtype=float)
        return np.mean(spectrum[:, :, mask], axis=-1)

    spike_wave_power = band_power(2.0, 7.0)
    slow_power = band_power(0.5, 1.5)
    alpha_beta_power = band_power(8.0, 20.0)
    band_ratio = spike_wave_power / (slow_power + alpha_beta_power + 1e-12)
    top_band_ratio = np.percentile(band_ratio, 90, axis=1)

    deriv = np.diff(x, axis=-1)
    sharpness = np.percentile(np.abs(deriv), 95, axis=(1, 2))
    p2p = np.percentile(x, 95, axis=-1) - np.percentile(x, 5, axis=-1)
    top_p2p = np.percentile(p2p, 95, axis=1)
    typical_p2p = np.median(p2p, axis=1)
    field_balance = typical_p2p / (top_p2p + 1e-6)

    def robust_z(values: np.ndarray) -> np.ndarray:
        med = np.median(values)
        mad = np.median(np.abs(values - med)) + 1e-6
        return (values - med) / mad

    artifact_penalty = 1.0 / (1.0 + np.maximum(0.0, (top_p2p - 1000.0) / 500.0) ** 2)
    rhythmic_score = np.maximum(0.0, robust_z(np.log1p(top_band_ratio)))
    sharp_score = np.maximum(0.0, robust_z(np.log1p(sharpness)))
    field_score = np.clip(field_balance, 0.0, 1.0)
    score = (0.60 * rhythmic_score + 0.25 * sharp_score + 0.15 * field_score) * artifact_penalty

    return [
        make_distribution_measurement(
            measurement_id="m_spike_wave_candidate_score_distribution",
            measurement_name="spike_wave_candidate_score_distribution",
            values=[float(v) for v in score],
            unit="score",
            provenance=make_provenance(
                tool_name="spike_wave_candidate_score",
                function_name="spike_wave_candidate_score",
                source_ref=source_ref,
                window_indices=range(signal_nct.shape[0]),
            ),
        ),
        make_exact_measurement(
            measurement_id="m_spike_wave_candidate_burden_ratio",
            measurement_name="spike_wave_candidate_burden_ratio",
            value=float(np.mean(score > np.percentile(score, 90))) if score.size else 0.0,
            unit="ratio",
            provenance=make_provenance(
                tool_name="spike_wave_candidate_score",
                function_name="spike_wave_candidate_score",
                source_ref=source_ref,
                window_indices=range(signal_nct.shape[0]),
            ),
        ),
    ]


def _contiguous_runs(indices: np.ndarray) -> List[Tuple[int, int]]:
    if len(indices) == 0:
        return []
    runs = []
    start = int(indices[0])
    prev = int(indices[0])
    for idx in indices[1:]:
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
            continue
        runs.append((start, prev))
        start = idx
        prev = idx
    runs.append((start, prev))
    return runs


def burst_train_duration_estimate(score_distribution: np.ndarray, window_seconds: int, source_ref: str) -> List[MeasurementValue]:
    suspicious = np.where(score_distribution > np.percentile(score_distribution, 90))[0]
    runs = _contiguous_runs(suspicious)
    durations = [float((end - start + 1) * window_seconds) for start, end in runs]
    max_dur = max(durations) if durations else 0.0

    return [
        make_upper_bound_measurement(
            measurement_id="m_event_train_duration_upper",
            measurement_name="event_train_duration_upper_sec",
            upper=max_dur,
            unit="sec",
            provenance=make_provenance(
                tool_name="burst_train_duration_estimate",
                function_name="burst_train_duration_estimate",
                source_ref=source_ref,
                window_indices=[int(x) for x in suspicious.tolist()],
            ),
        ),
        make_distribution_measurement(
            measurement_id="m_event_train_duration_distribution",
            measurement_name="event_train_duration_distribution_sec",
            values=durations if durations else [0.0],
            unit="sec",
            provenance=make_provenance(
                tool_name="burst_train_duration_estimate",
                function_name="burst_train_duration_estimate",
                source_ref=source_ref,
                window_indices=[int(x) for x in suspicious.tolist()],
            ),
        ),
    ]


def channel_spread_laterality_summary(
    signal_nct: np.ndarray,
    channels: List[str],
    suspicious_windows: List[int],
    source_ref: str,
) -> List[MeasurementValue]:
    if not suspicious_windows:
        suspicious_windows = list(range(signal_nct.shape[0]))

    ch_to_idx = {c: i for i, c in enumerate(channels)}
    left_idx = [ch_to_idx[c] for c in channels if c in LEFT_CHANNELS]
    right_idx = [ch_to_idx[c] for c in channels if c in RIGHT_CHANNELS]

    sliced = signal_nct[suspicious_windows]
    left_energy = float(np.mean(np.abs(sliced[:, left_idx, :])) if left_idx else 0.0)
    right_energy = float(np.mean(np.abs(sliced[:, right_idx, :])) if right_idx else 0.0)
    li = (left_energy - right_energy) / (left_energy + right_energy + 1e-12)

    laterality = "midline"
    if li > 0.1:
        laterality = "left"
    elif li < -0.1:
        laterality = "right"

    return [
        make_exact_measurement(
            measurement_id="m_event_laterality_index",
            measurement_name="event_laterality_index",
            value=float(li),
            unit="ratio",
            provenance=make_provenance(
                tool_name="channel_spread_laterality_summary",
                function_name="channel_spread_laterality_summary",
                source_ref=source_ref,
                window_indices=suspicious_windows,
                channels=channels,
                laterality=laterality,
            ),
        )
    ]


def focality_bifrontal_summary(
    signal_nct: np.ndarray,
    channels: List[str],
    suspicious_windows: List[int],
    source_ref: str,
) -> List[MeasurementValue]:
    if not suspicious_windows:
        suspicious_windows = list(range(signal_nct.shape[0]))

    ch_to_idx = {c: i for i, c in enumerate(channels)}
    frontal_idx = [ch_to_idx[c] for c in channels if c in FRONTAL_CHANNELS]
    non_frontal_idx = [idx for ch, idx in ch_to_idx.items() if ch not in FRONTAL_CHANNELS]

    sliced = signal_nct[suspicious_windows]
    frontal = float(np.mean(np.abs(sliced[:, frontal_idx, :])) if frontal_idx else 0.0)
    non_frontal = float(np.mean(np.abs(sliced[:, non_frontal_idx, :])) if non_frontal_idx else 0.0)
    bifrontal_ratio = frontal / (non_frontal + 1e-12)

    return [
        make_exact_measurement(
            measurement_id="m_event_bifrontal_ratio",
            measurement_name="event_bifrontal_ratio",
            value=bifrontal_ratio,
            unit="ratio",
            provenance=make_provenance(
                tool_name="focality_bifrontal_summary",
                function_name="focality_bifrontal_summary",
                source_ref=source_ref,
                window_indices=suspicious_windows,
                channels=channels,
                region="frontal-bifrontal",
            ),
        )
    ]


def _region_label_from_channels(top_channels: List[str]) -> str:
    if not top_channels:
        return "unknown"
    top = set(top_channels)
    left = len(top & LEFT_CHANNELS)
    right = len(top & RIGHT_CHANNELS)
    frontal = len(top & FRONTAL_CHANNELS)
    temporal = len(top & TEMPORAL_CHANNELS)
    posterior = len(top & POSTERIOR_CHANNELS)

    laterality = "bilateral"
    if left > right:
        laterality = "left"
    elif right > left:
        laterality = "right"

    if temporal >= 2 and frontal >= 1:
        region = "frontotemporal"
    elif temporal >= 2:
        region = "temporal"
    elif frontal >= 2:
        region = "frontal"
    elif posterior >= 2:
        region = "posterior"
    else:
        region = "multiregional"
    return f"{laterality}_{region}"


def _sanitize_windows(suspicious_windows: List[int], n_windows: int) -> List[int]:
    if not suspicious_windows:
        suspicious_windows = list(range(n_windows))
    out = sorted(set(int(i) for i in suspicious_windows if 0 <= int(i) < n_windows))
    return out if out else list(range(n_windows))


def _topography_label_from_event_field(top_channels: List[str], active_channels: List[str], laterality_index: float) -> str:
    """Map peak-centered field channels to a clinical topography label.

    This is intentionally conservative: it names a field pattern when channel
    evidence is clear, and otherwise falls back to a broad multiregional label.
    """
    if not top_channels:
        return "unknown"

    top = set(top_channels)
    active = set(active_channels)
    left_temporal = len(top & LEFT_TEMPORAL_CHANNELS)
    right_temporal = len(top & RIGHT_TEMPORAL_CHANNELS)
    left_frontal = len(top & LEFT_FRONTAL_CHANNELS)
    right_frontal = len(top & RIGHT_FRONTAL_CHANNELS)
    frontal = len(top & FRONTAL_CHANNELS)
    temporal = len(top & TEMPORAL_CHANNELS)
    posterior = len(top & POSTERIOR_CHANNELS)
    bilateral_support = bool(active & LEFT_CHANNELS) and bool(active & RIGHT_CHANNELS)

    if laterality_index > 0.15:
        side = "left"
    elif laterality_index < -0.15:
        side = "right"
    else:
        side = "bilateral"

    if bilateral_support and frontal >= 3 and len(active) >= 6:
        return "generalized_frontal_predominance"
    if side == "left":
        if {"F7", "T3"}.issubset(top) or {"F7", "T5"}.issubset(top):
            return "left_temporal"
        if {"F3", "F7"}.issubset(top):
            return "left_frontal_frontotemporal"
        if left_temporal >= 2:
            return "left_temporal"
        if left_frontal >= 2:
            return "left_frontal"
    if side == "right":
        if {"F8", "T4"}.issubset(top) or {"F8", "T6"}.issubset(top):
            return "right_temporal"
        if {"F4", "F8"}.issubset(top):
            return "right_frontal_frontotemporal"
        if right_temporal >= 2:
            return "right_temporal"
        if right_frontal >= 2:
            return "right_frontal"
    if temporal >= 2 and frontal >= 1:
        return f"{side}_frontotemporal"
    if temporal >= 2:
        return f"{side}_temporal"
    if frontal >= 2:
        return f"{side}_frontal"
    if posterior >= 2:
        return f"{side}_posterior"
    return f"{side}_multiregional"


def _region_from_label(label: str) -> str:
    for region in ("frontotemporal", "temporal", "frontal", "posterior", "multiregional"):
        if region in label:
            return region
    return "unknown"


def _laterality_from_label(label: str) -> str:
    if label.startswith("left"):
        return "left"
    if label.startswith("right"):
        return "right"
    if label.startswith("bilateral") or label.startswith("generalized"):
        return "bilateral"
    return "unknown"


def _spatial_pattern_phrase(laterality: str, region: str, electrode_maxima: List[str]) -> str:
    if not electrode_maxima or region == "unknown":
        return "spatial field not localizable"
    side = {
        "left": "left",
        "right": "right",
        "bilateral": "bilateral",
    }.get(laterality, "")
    prefix = f"{side} {region}".strip()
    maxima = "/".join(electrode_maxima[:3])
    return f"{prefix} predominance, maximal at {maxima}"


def _peak_field_summary(
    signal_nct: np.ndarray,
    channels: List[str],
    suspicious_windows: List[int],
    *,
    peak_context_samples: int = 50,
    max_peaks: int = 12,
) -> tuple[List[int], List[int], List[str], List[str], np.ndarray]:
    focused_windows = _sanitize_windows(suspicious_windows, signal_nct.shape[0])
    x = signal_nct[focused_windows].astype(np.float32) * _infer_uv_scale(signal_nct)
    x = x - np.median(x, axis=-1, keepdims=True)
    if x.size == 0:
        return [], [], [], [], np.zeros((len(channels),), dtype=float)

    dx = np.diff(x, axis=-1, prepend=x[:, :, :1])
    amplitude_score = np.max(np.abs(x), axis=1)
    sharpness_score = np.max(np.abs(dx), axis=1)
    sample_score = amplitude_score + 0.50 * sharpness_score

    peak_rows: List[Tuple[int, int, float]] = []
    for local_idx, window_idx in enumerate(focused_windows):
        peak_sample = int(np.argmax(sample_score[local_idx]))
        peak_strength = float(sample_score[local_idx, peak_sample])
        peak_rows.append((local_idx, int(window_idx), peak_strength))
    peak_rows = sorted(peak_rows, key=lambda item: item[2], reverse=True)[: max(1, int(max_peaks))]

    event_fields = []
    peak_window_indices: List[int] = []
    peak_sample_indices: List[int] = []
    for local_idx, window_idx, _strength in peak_rows:
        peak_sample = int(np.argmax(sample_score[local_idx]))
        start = max(0, peak_sample - int(peak_context_samples))
        end = min(x.shape[-1], peak_sample + int(peak_context_samples) + 1)
        epoch = x[local_idx, :, start:end]
        peak_abs = np.abs(x[local_idx, :, peak_sample])
        peak_to_peak = np.max(epoch, axis=-1) - np.min(epoch, axis=-1)
        event_fields.append(0.60 * peak_abs + 0.40 * peak_to_peak)
        peak_window_indices.append(window_idx)
        peak_sample_indices.append(peak_sample)

    field = np.median(np.stack(event_fields, axis=0), axis=0) if event_fields else np.zeros((len(channels),), dtype=float)
    top_indices = np.argsort(field)[-5:][::-1].tolist() if field.size else []
    top_channels = [channels[i] for i in top_indices if i < len(channels)]
    max_field = float(np.max(field)) if field.size else 0.0
    active_indices = np.where(field >= max_field * 0.45)[0].tolist() if max_field > 0 else []
    active_channels = [channels[i] for i in active_indices if i < len(channels)]
    return peak_window_indices, peak_sample_indices, top_channels, active_channels, field


def _safe_spatial_descriptor(
    signal_nct: np.ndarray,
    channels: List[str],
    suspicious_windows: List[int],
) -> dict[str, object]:
    peak_window_indices, peak_sample_indices, top_channels, active_channels, field = _peak_field_summary(
        signal_nct,
        channels,
        suspicious_windows,
    )
    ch_to_idx = {c: i for i, c in enumerate(channels)}
    left_idx = [ch_to_idx[c] for c in channels if c in LEFT_CHANNELS]
    right_idx = [ch_to_idx[c] for c in channels if c in RIGHT_CHANNELS]
    left_field = float(np.sum(field[left_idx])) if left_idx else 0.0
    right_field = float(np.sum(field[right_idx])) if right_idx else 0.0
    laterality_index = (left_field - right_field) / (left_field + right_field + 1e-6)
    label = _topography_label_from_event_field(top_channels, active_channels, laterality_index)
    laterality = _laterality_from_label(label)
    region = _region_from_label(label)
    electrode_maxima = top_channels[:3]
    return {
        "label": label,
        "laterality": laterality,
        "region": region,
        "electrode_maxima": electrode_maxima,
        "active_channels": active_channels[:10],
        "peak_window_indices": peak_window_indices,
        "peak_sample_indices": peak_sample_indices,
        "spatial_pattern": _spatial_pattern_phrase(laterality, region, electrode_maxima),
        "field": field,
    }


def _event_morphology_descriptor(
    signal_nct: np.ndarray,
    channels: List[str],
    focused_idx: np.ndarray,
    spatial: dict[str, object],
    *,
    window_seconds: int = 10,
) -> dict[str, object]:
    """Return clinical-shape descriptors without exposing numeric scores.

    The goal is not to diagnose epileptiform activity directly. It separates
    broad transients from spike-wave-like waveforms by requiring a sharp peak,
    a following slow component, and rhythmic repetition support.
    """
    n_windows, _n_channels, n_samples = signal_nct.shape
    if focused_idx.size == 0 or n_samples < 8:
        return {
            "descriptor": "insufficient_morphology",
            "sharp_component": "unknown",
            "slow_wave_follow": "unknown",
            "rhythmicity": "unknown",
        }
    fs = float(n_samples) / float(window_seconds) if window_seconds > 0 else 0.0
    if fs <= 0.0:
        fs = 200.0
    channel_index = {channel: idx for idx, channel in enumerate(channels)}
    electrode_maxima = [str(ch) for ch in spatial.get("electrode_maxima", []) if str(ch) in channel_index]
    selected_channels = electrode_maxima[:3] or [
        str(ch) for ch in spatial.get("active_channels", []) if str(ch) in channel_index
    ][:5]
    if not selected_channels:
        selected_channels = list(channels[: min(5, len(channels))])
    selected_indices = [channel_index[ch] for ch in selected_channels if ch in channel_index]
    if not selected_indices:
        return {
            "descriptor": "insufficient_morphology",
            "sharp_component": "unknown",
            "slow_wave_follow": "unknown",
            "rhythmicity": "unknown",
        }

    focused = {int(window_idx): local_idx for local_idx, window_idx in enumerate(focused_idx.tolist())}
    peak_windows = [int(x) for x in spatial.get("peak_window_indices", [])]
    peak_samples = [int(x) for x in spatial.get("peak_sample_indices", [])]
    x = signal_nct[focused_idx].astype(np.float32) * _infer_uv_scale(signal_nct)
    x = x - np.median(x, axis=-1, keepdims=True)

    sharp_votes = 0
    slow_votes = 0
    rhythmic_votes = 0
    evaluated = 0
    peak_to_peak_values: list[float] = []
    for window_idx, peak_sample in zip(peak_windows, peak_samples):
        if window_idx not in focused:
            continue
        local_idx = focused[window_idx]
        peak_sample = int(np.clip(peak_sample, 1, n_samples - 2))
        traces = x[local_idx, selected_indices, :]
        if traces.size == 0:
            continue
        peak_channel = int(np.argmax(np.abs(traces[:, peak_sample])))
        y = _linear_detrend_1d(traces[peak_channel].astype(np.float64))
        baseline_start, baseline_end = _centered_slice(peak_sample, n_samples, int(round(0.30 * fs)))
        baseline = float(np.median(y[baseline_start:baseline_end])) if baseline_end > baseline_start else float(np.median(y))
        peak_amp = abs(float(y[peak_sample] - baseline))
        local_start, local_end = _centered_slice(peak_sample, n_samples, int(round(0.12 * fs)))
        local = y[local_start:local_end]
        local_p2p = float(np.percentile(local, 95) - np.percentile(local, 5)) if local.size else 0.0
        peak_to_peak_values.append(local_p2p)

        curvature = abs(float(y[peak_sample + 1] - 2.0 * y[peak_sample] + y[peak_sample - 1]))
        sharp = peak_amp >= 25.0 and curvature >= 8.0
        if sharp:
            sharp_votes += 1

        post_start = min(n_samples, peak_sample + int(round(0.08 * fs)))
        post_end = min(n_samples, peak_sample + int(round(0.60 * fs)))
        post = y[post_start:post_end]
        slow_follow = False
        if post.size:
            post_dev = float(np.max(np.abs(post - baseline)))
            slow_follow = post_dev >= max(10.0, 0.25 * peak_amp)
        if slow_follow:
            slow_votes += 1

        freq_start, freq_end = _centered_slice(peak_sample, n_samples, int(round(1.50 * fs)))
        freq = _dominant_frequency_hz(y[freq_start:freq_end], fs)
        if freq is not None and 2.0 <= float(freq) <= 7.0:
            rhythmic_votes += 1
        evaluated += 1

    if evaluated == 0:
        return {
            "descriptor": "insufficient_morphology",
            "sharp_component": "unknown",
            "slow_wave_follow": "unknown",
            "rhythmicity": "unknown",
        }

    sharp_component = sharp_votes >= max(1, int(np.ceil(0.30 * evaluated)))
    slow_wave_follow = slow_votes >= max(1, int(np.ceil(0.30 * evaluated)))
    rhythmicity = rhythmic_votes >= max(1, int(np.ceil(0.30 * evaluated)))
    broad_field = (
        str(spatial.get("laterality")) == "bilateral"
        and len([str(ch) for ch in spatial.get("active_channels", [])]) >= 6
    ) or str(spatial.get("label", "")).startswith("generalized")

    descriptor = "insufficient_morphology"
    if sharp_component and slow_wave_follow and rhythmicity and broad_field:
        descriptor = "generalized_spike_wave_like"
    elif sharp_component and slow_wave_follow and rhythmicity:
        descriptor = "spike_wave_like"
    elif sharp_component and slow_wave_follow:
        descriptor = "sharp_wave_like"
    elif sharp_component:
        descriptor = "sharp_transient_like"
    elif peak_to_peak_values and float(np.median(peak_to_peak_values)) >= 25.0:
        descriptor = "nonspecific_transient_like"

    return {
        "descriptor": descriptor,
        "sharp_component": "present" if sharp_component else "absent",
        "slow_wave_follow": "present" if slow_wave_follow else "absent",
        "rhythmicity": "present" if rhythmicity else "absent",
        "field_extent": "broad_bilateral" if broad_field else "focal_or_regional",
    }


def _linear_detrend_1d(y: np.ndarray) -> np.ndarray:
    if y.size < 3:
        return y - np.mean(y) if y.size else y
    x = np.linspace(-1.0, 1.0, y.size)
    slope, intercept = np.polyfit(x, y.astype(np.float64), 1)
    return y.astype(np.float64) - (slope * x + intercept)


def _dominant_frequency_hz(epoch: np.ndarray, fs: float) -> float | None:
    if epoch.size < 8 or fs <= 0:
        return None
    y = _linear_detrend_1d(epoch.astype(np.float64))
    y = y - np.mean(y)
    # Event frequency should reflect discharge repetition rate. Autocorrelation
    # is a better first pass than a plain FFT here because sharp transients can
    # otherwise make the spectral peak drift toward high-frequency sharpness.
    denom = float(np.dot(y, y))
    if denom > 1e-12:
        corr = np.correlate(y, y, mode="full")[y.size - 1 :] / denom
        min_lag = max(1, int(round(fs / min(15.0, fs / 2.0))))
        max_lag = min(corr.size - 2, int(round(fs / 1.0)))
        if max_lag > min_lag:
            lag_slice = corr[min_lag : max_lag + 1]
            peak_lags: list[int] = []
            for offset in range(1, lag_slice.size - 1):
                if lag_slice[offset] > lag_slice[offset - 1] and lag_slice[offset] >= lag_slice[offset + 1]:
                    peak_lags.append(min_lag + offset)
            if peak_lags:
                peak_lags = sorted(peak_lags, key=lambda lag: float(corr[lag]), reverse=True)
                best_lag = peak_lags[0]
                best_freq = float(fs / best_lag)
                if best_freq <= 1.25:
                    for lag in peak_lags[1:]:
                        freq = float(fs / lag)
                        if freq >= 1.5 and float(corr[lag]) >= 0.60 * float(corr[best_lag]):
                            return freq
                return best_freq

    window = np.hanning(y.size)
    n_fft = max(y.size, int(round(fs * 4.0)))
    spectrum = np.abs(np.fft.rfft(y * window, n=n_fft)) ** 2
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    mask = (freqs >= 1.0) & (freqs <= min(15.0, fs / 2.0))
    if not np.any(mask):
        return None
    band_power = spectrum[mask]
    if not np.any(np.isfinite(band_power)) or float(np.max(band_power)) <= 0.0:
        return None
    return float(freqs[mask][int(np.argmax(band_power))])


def _centered_slice(sample_idx: int, n_samples: int, half_width: int) -> tuple[int, int]:
    start = max(0, int(sample_idx) - int(half_width))
    end = min(n_samples, int(sample_idx) + int(half_width) + 1)
    return start, end


def _peak_envelope_duration_sec(y: np.ndarray, peak_sample: int, fs: float) -> float | None:
    if y.size < 8 or fs <= 0:
        return None
    detrended = _linear_detrend_1d(y.astype(np.float64))
    envelope = np.abs(detrended)
    smooth_len = max(3, int(round(fs * 0.50)))
    if smooth_len % 2 == 0:
        smooth_len += 1
    kernel = np.ones(smooth_len, dtype=float) / float(smooth_len)
    smooth = np.convolve(envelope, kernel, mode="same")
    peak_sample = int(np.clip(peak_sample, 0, smooth.size - 1))
    baseline = float(np.median(smooth))
    mad = float(np.median(np.abs(smooth - baseline))) + 1e-6
    peak_value = float(smooth[peak_sample])
    if peak_value <= baseline:
        return None
    threshold = max(baseline + 3.0 * mad, baseline + 0.25 * (peak_value - baseline))
    left = peak_sample
    right = peak_sample
    allowed_gap = max(1, int(round(fs * 0.25)))
    gap = 0
    while left > 0:
        if smooth[left - 1] >= threshold:
            gap = 0
            left -= 1
            continue
        gap += 1
        if gap > allowed_gap:
            break
        left -= 1
    gap = 0
    while right < smooth.size - 1:
        if smooth[right + 1] >= threshold:
            gap = 0
            right += 1
            continue
        gap += 1
        if gap > allowed_gap:
            break
        right += 1
    return float(max(1, right - left + 1) / fs)


def event_waveform_numeric_v2(
    signal_nct: np.ndarray,
    channels: List[str],
    suspicious_windows: List[int],
    window_seconds: int,
    source_ref: str,
    peak_context_samples: int = 50,
    max_peaks: int = 12,
) -> List[MeasurementValue]:
    """Estimate traceable waveform numerics from focused event-candidate windows.

    These measurements are not seizure evidence and do not make a definitive
    epileptiform claim. They provide bounded amplitude/frequency/duration values
    that later stages may use only with linked provenance and caveats.
    """
    n_windows, _n_channels, n_samples = signal_nct.shape
    focused = [int(i) for i in sorted(set(suspicious_windows)) if 0 <= int(i) < n_windows]
    if not focused:
        return []

    context = max(8, min(int(peak_context_samples), max(8, n_samples // 20)))
    peak_window_indices, peak_sample_indices, top_channels, active_channels, _field = _peak_field_summary(
        signal_nct,
        channels,
        focused,
        peak_context_samples=context,
        max_peaks=max_peaks,
    )
    if not peak_window_indices:
        return []
    peak_window_indices = [int(window_idx) for window_idx in peak_window_indices]
    peak_sample_indices = [int(sample_idx) for sample_idx in peak_sample_indices]

    channel_index = {channel: idx for idx, channel in enumerate(channels)}
    selected_channels = [ch for ch in (top_channels or active_channels) if ch in channel_index][:5]
    if not selected_channels:
        selected_channels = list(channels[: min(5, len(channels))])
    selected_indices = [channel_index[ch] for ch in selected_channels]

    x = signal_nct[peak_window_indices].astype(np.float32) * _infer_uv_scale(signal_nct)
    x = x - np.median(x, axis=-1, keepdims=True)
    fs = float(n_samples) / float(window_seconds) if window_seconds > 0 else 0.0

    amplitude_values: list[float] = []
    frequency_values: list[float] = []
    duration_values: list[float] = []
    amp_half_width = max(context, int(round(fs * 0.25))) if fs > 0 else context
    freq_half_width = max(context, int(round(fs * 1.50))) if fs > 0 else context
    for local_idx, peak_sample in enumerate(peak_sample_indices):
        amp_start, amp_end = _centered_slice(peak_sample, n_samples, amp_half_width)
        epoch = x[local_idx, selected_indices, amp_start:amp_end]
        if epoch.size:
            per_channel_p2p = np.percentile(epoch, 95, axis=-1) - np.percentile(epoch, 5, axis=-1)
            if per_channel_p2p.size:
                top_p2p = np.sort(per_channel_p2p)[-min(2, per_channel_p2p.size) :]
                amplitude_values.append(float(np.median(top_p2p)))

        freq_start, freq_end = _centered_slice(peak_sample, n_samples, freq_half_width)
        local_epoch = x[local_idx, selected_indices, freq_start:freq_end]
        if local_epoch.size and fs > 0:
            channel_p2p = np.percentile(local_epoch, 95, axis=-1) - np.percentile(local_epoch, 5, axis=-1)
            best_channel = int(np.argmax(channel_p2p)) if channel_p2p.size else 0
            freq = _dominant_frequency_hz(local_epoch[best_channel], fs)
            if freq is not None:
                frequency_values.append(freq)
            dur = _peak_envelope_duration_sec(x[local_idx, selected_indices[best_channel], :], peak_sample, fs)
            if dur is not None and np.isfinite(dur):
                duration_values.append(dur)

    # Focused event numerics should not turn large movement/electrode artifacts
    # into reportable waveform values. Keep the measurement bounded to a broad
    # physiologic EEG range; discarded values remain absent rather than surfaced.
    amplitude_values = [
        float(value)
        for value in amplitude_values
        if np.isfinite(value) and 2.0 <= float(value) <= 1000.0
    ]
    if not amplitude_values:
        return []

    provenance = make_provenance(
        tool_name="event_waveform_numeric_v2",
        function_name="event_waveform_numeric_v2",
        source_ref=source_ref,
        window_indices=peak_window_indices,
        channels=selected_channels,
        region=_region_label_from_channels(selected_channels),
        laterality=_laterality_from_label(_region_label_from_channels(selected_channels)),
        reason="focused_event_candidate_waveform_numeric_not_seizure_evidence",
    )
    if peak_sample_indices:
        provenance.value_span = (float(min(peak_sample_indices)), float(max(peak_sample_indices)))

    metadata = {
        "event_waveform_numeric_v2": "true",
        "candidate_context_only": "true",
        "not_seizure_evidence": "true",
        "top_channels": ",".join(selected_channels),
        "peak_window_indices": ",".join(str(i) for i in peak_window_indices),
        "peak_sample_indices": ",".join(str(i) for i in peak_sample_indices),
        "duration_source": "peak_centered_envelope_duration",
        "frequency_source": "peak_centered_local_segment_fft",
        "amplitude_source": "peak_centered_local_segment_p95_minus_p5",
    }

    measurements: list[MeasurementValue] = []
    if amplitude_values:
        amp = np.asarray(amplitude_values, dtype=float)
        typical = float(np.median(amp))
        lo = float(np.percentile(amp, 25))
        hi = float(np.percentile(amp, 75))
        measurements.extend(
            [
                make_exact_measurement(
                    measurement_id="m_event_waveform_amplitude_peak_to_peak_typical",
                    measurement_name="event_waveform_amplitude_peak_to_peak_typical_uv",
                    value=typical,
                    unit="uV",
                    provenance=provenance,
                ),
                make_range_measurement(
                    measurement_id="m_event_waveform_amplitude_peak_to_peak_range",
                    measurement_name="event_waveform_amplitude_peak_to_peak_range_uv",
                    lower=lo,
                    upper=hi,
                    unit="uV",
                    provenance=provenance,
                ),
            ]
        )
    if frequency_values:
        measurements.append(
            make_exact_measurement(
                measurement_id="m_event_waveform_dominant_frequency",
                measurement_name="event_waveform_dominant_frequency_hz",
                value=float(np.median(np.asarray(frequency_values, dtype=float))),
                unit="Hz",
                provenance=provenance,
            )
        )
    if duration_values:
        duration_array = np.asarray(duration_values, dtype=float)
        measurements.extend(
            [
                make_exact_measurement(
                    measurement_id="m_event_waveform_duration_typical",
                    measurement_name="event_waveform_duration_typical_sec",
                    value=float(np.median(duration_array)),
                    unit="sec",
                    provenance=make_provenance(
                        tool_name="event_waveform_numeric_v2",
                        function_name="event_waveform_numeric_v2",
                        source_ref=source_ref,
                        window_indices=peak_window_indices,
                        channels=selected_channels,
                        reason="peak_centered_envelope_duration_not_seizure_duration",
                    ),
                ),
                make_upper_bound_measurement(
                    measurement_id="m_event_waveform_duration_upper",
                    measurement_name="event_waveform_duration_upper_sec",
                    upper=float(np.percentile(duration_array, 90)),
                    unit="sec",
                    provenance=make_provenance(
                        tool_name="event_waveform_numeric_v2",
                        function_name="event_waveform_numeric_v2",
                        source_ref=source_ref,
                        window_indices=peak_window_indices,
                        channels=selected_channels,
                        reason="peak_centered_envelope_duration_upper_bound_not_seizure_duration",
                    ),
                ),
            ]
        )

    for measurement in measurements:
        measurement.metadata.update(metadata)
    return measurements


def event_spatiomorphology_v2(
    signal_nct: np.ndarray,
    channels: List[str],
    suspicious_windows: List[int],
    source_ref: str,
    window_seconds: int = 10,
) -> List[MeasurementValue]:
    """Build trace-safe spatial and morphology descriptors for event candidates.

    This tool intentionally emits clinical descriptors, not internal ratios or
    scores. Raw support values remain available only through the legacy proxy
    measurements.
    """
    if not suspicious_windows:
        suspicious_windows = list(range(signal_nct.shape[0]))
    focused_idx = np.asarray(sorted(set(int(i) for i in suspicious_windows)), dtype=int)
    focused_idx = focused_idx[(focused_idx >= 0) & (focused_idx < signal_nct.shape[0])]
    if focused_idx.size == 0:
        focused_idx = np.arange(signal_nct.shape[0], dtype=int)

    spatial = _safe_spatial_descriptor(signal_nct, channels, [int(x) for x in focused_idx.tolist()])
    electrode_maxima = [str(ch) for ch in spatial["electrode_maxima"]]
    region = str(spatial["region"])
    laterality = str(spatial["laterality"])
    spatial_pattern = str(spatial["spatial_pattern"])
    morphology = _event_morphology_descriptor(
        signal_nct,
        channels,
        focused_idx,
        spatial,
        window_seconds=window_seconds,
    )
    morphology_descriptor = str(morphology["descriptor"])
    field_descriptor = (
        f"event field shows {spatial_pattern}"
        if spatial_pattern != "spatial field not localizable"
        else "event field not localizable"
    )

    provenance = make_provenance(
        tool_name="event_spatiomorphology_v2",
        function_name="event_spatiomorphology_v2",
        source_ref=source_ref,
        window_indices=[int(x) for x in focused_idx.tolist()],
        channels=electrode_maxima or channels,
        region=region,
        laterality=laterality,
        reason="safe_event_spatial_morphology_descriptor_no_scores_or_ratios",
    )
    metadata = {
        "active_channels": ",".join(str(ch) for ch in spatial["active_channels"]),
        "peak_window_indices": ",".join(str(i) for i in spatial["peak_window_indices"]),
        "peak_sample_indices": ",".join(str(i) for i in spatial["peak_sample_indices"]),
        "sharp_component": str(morphology["sharp_component"]),
        "slow_wave_follow": str(morphology["slow_wave_follow"]),
        "rhythmicity": str(morphology["rhythmicity"]),
        "field_extent": str(morphology["field_extent"]),
        "safe_spatiomorphology_v2": "true",
        "internal_scores_suppressed": "true",
    }

    measurements = [
        make_categorical_measurement(
            measurement_id="m_event_electrode_maxima_v2",
            measurement_name="event_electrode_maxima_v2",
            value=",".join(electrode_maxima) if electrode_maxima else "unknown",
            provenance=provenance,
        ),
        make_categorical_measurement(
            measurement_id="m_event_region_v2",
            measurement_name="event_region_v2",
            value=region,
            provenance=provenance,
        ),
        make_categorical_measurement(
            measurement_id="m_event_laterality_v2",
            measurement_name="event_laterality_v2",
            value=laterality,
            provenance=provenance,
        ),
        make_categorical_measurement(
            measurement_id="m_event_spatial_pattern_v2",
            measurement_name="event_spatial_pattern_v2",
            value=spatial_pattern,
            provenance=provenance,
        ),
        make_categorical_measurement(
            measurement_id="m_event_field_descriptor_v2",
            measurement_name="event_field_descriptor_v2",
            value=field_descriptor,
            provenance=provenance,
        ),
        make_categorical_measurement(
            measurement_id="m_event_morphology_descriptor_v2",
            measurement_name="event_morphology_descriptor_v2",
            value=morphology_descriptor,
            provenance=provenance,
        ),
    ]
    for measurement in measurements:
        measurement.metadata.update(metadata)
    return measurements


def event_localization_normalizer(
    signal_nct: np.ndarray,
    channels: List[str],
    suspicious_windows: List[int],
    source_ref: str,
) -> List[MeasurementValue]:
    """Convert channel-energy evidence into a coarse clinical localization label.

    This is a normalizer over local signal features, not a definitive clinical
    localization adjudicator. It preserves asymmetry in the categorical value
    and keeps top channels in provenance metadata.
    """
    if not suspicious_windows:
        suspicious_windows = list(range(signal_nct.shape[0]))
    suspicious_windows = [int(i) for i in suspicious_windows if 0 <= int(i) < signal_nct.shape[0]]
    if not suspicious_windows:
        suspicious_windows = list(range(signal_nct.shape[0]))

    x = signal_nct[suspicious_windows].astype(np.float32) * _infer_uv_scale(signal_nct)
    channel_energy = np.mean(np.abs(x), axis=(0, 2))
    top_indices = np.argsort(channel_energy)[-5:][::-1].tolist() if channel_energy.size else []
    top_channels = [channels[i] for i in top_indices if i < len(channels)]
    label = _region_label_from_channels(top_channels)
    concentration = float(np.max(channel_energy) / (np.median(channel_energy) + 1e-6)) if channel_energy.size else 0.0

    provenance = make_provenance(
        tool_name="event_localization_normalizer",
        function_name="event_localization_normalizer",
        source_ref=source_ref,
        window_indices=suspicious_windows,
        channels=top_channels or channels,
        region=label,
        laterality=label.split("_", 1)[0] if "_" in label else None,
        reason="coarse_channel_energy_to_clinical_region_label",
    )
    label_measurement = make_categorical_measurement(
        measurement_id="m_event_clinical_localization_label",
        measurement_name="event_clinical_localization_label",
        value=label,
        provenance=provenance,
    )
    label_measurement.metadata.update({"top_channels": ",".join(top_channels), "localization_is_proxy": "true"})
    concentration_measurement = make_exact_measurement(
        measurement_id="m_event_localization_concentration_ratio",
        measurement_name="event_localization_concentration_ratio",
        value=concentration,
        unit="ratio",
        provenance=provenance,
    )
    concentration_measurement.metadata.update(label_measurement.metadata)
    return [label_measurement, concentration_measurement]


def event_peak_topography_localizer(
    signal_nct: np.ndarray,
    channels: List[str],
    suspicious_windows: List[int],
    source_ref: str,
    peak_context_samples: int = 50,
    max_peaks: int = 12,
) -> List[MeasurementValue]:
    """Localize event candidates from peak-centered topographic fields.

    Unlike the legacy 10-second mean-energy normalizer, this tool first finds
    event-like peaks inside focused windows and then summarizes the spatial
    field at those peaks. It still produces proxy evidence, not a definitive
    clinical localization claim.
    """
    peak_window_indices, peak_sample_indices, top_channels, active_channels, field = _peak_field_summary(
        signal_nct,
        channels,
        suspicious_windows,
        peak_context_samples=peak_context_samples,
        max_peaks=max_peaks,
    )
    if not peak_window_indices:
        provenance = make_provenance(
            tool_name="event_peak_topography_localizer",
            function_name="event_peak_topography_localizer",
            source_ref=source_ref,
            window_indices=[],
            reason="no_signal_for_peak_centered_topography",
        )
        return [
            make_categorical_measurement(
                measurement_id="m_event_peak_localization_label",
                measurement_name="event_peak_localization_label",
                value="unknown",
                provenance=provenance,
            )
        ]
    max_field = float(np.max(field)) if field.size else 0.0

    ch_to_idx = {c: i for i, c in enumerate(channels)}
    left_idx = [ch_to_idx[c] for c in channels if c in LEFT_CHANNELS]
    right_idx = [ch_to_idx[c] for c in channels if c in RIGHT_CHANNELS]
    left_field = float(np.sum(field[left_idx])) if left_idx else 0.0
    right_field = float(np.sum(field[right_idx])) if right_idx else 0.0
    laterality_index = (left_field - right_field) / (left_field + right_field + 1e-6)
    label = _topography_label_from_event_field(top_channels, active_channels, laterality_index)
    concentration = float(max_field / (np.median(field) + 1e-6)) if field.size else 0.0

    laterality = "left" if laterality_index > 0.15 else "right" if laterality_index < -0.15 else "bilateral"
    provenance = make_provenance(
        tool_name="event_peak_topography_localizer",
        function_name="event_peak_topography_localizer",
        source_ref=source_ref,
        window_indices=peak_window_indices,
        channels=top_channels or active_channels or channels,
        region=label,
        laterality=laterality,
        reason="event_peak_centered_topographic_field_proxy",
    )
    if peak_sample_indices:
        provenance.value_span = (float(min(peak_sample_indices)), float(max(peak_sample_indices)))

    metadata = {
        "top_channels": ",".join(top_channels),
        "active_channels": ",".join(active_channels[:10]),
        "peak_window_indices": ",".join(str(i) for i in peak_window_indices),
        "peak_sample_indices": ",".join(str(i) for i in peak_sample_indices),
        "localization_is_peak_centered": "true",
        "localization_is_proxy": "true",
    }
    label_measurement = make_categorical_measurement(
        measurement_id="m_event_peak_localization_label",
        measurement_name="event_peak_localization_label",
        value=label,
        provenance=provenance,
    )
    label_measurement.metadata.update(metadata)
    concentration_measurement = make_exact_measurement(
        measurement_id="m_event_peak_field_concentration_ratio",
        measurement_name="event_peak_field_concentration_ratio",
        value=concentration,
        unit="ratio",
        provenance=provenance,
    )
    concentration_measurement.metadata.update(metadata)
    laterality_measurement = make_exact_measurement(
        measurement_id="m_event_peak_laterality_index",
        measurement_name="event_peak_laterality_index",
        value=laterality_index,
        unit="ratio",
        provenance=provenance,
    )
    laterality_measurement.metadata.update(metadata)
    return [label_measurement, concentration_measurement, laterality_measurement]


def morphology_feature_encoder(
    signal_nct: np.ndarray,
    channels: List[str],
    suspicious_windows: List[int],
    source_ref: str,
) -> List[MeasurementValue]:
    """Local morphology-feature encoder proxy for method E.

    This is not a clinical epileptiform classifier. It creates bounded typed
    measurements that summarize sharpness, curvature, and field concentration
    around focused candidate windows so downstream agents can reason about
    whether event candidates have any morphology-oriented support.
    """
    if not suspicious_windows:
        suspicious_windows = list(range(signal_nct.shape[0]))

    focused_idx = np.asarray(sorted(set(int(i) for i in suspicious_windows)), dtype=int)
    focused_idx = focused_idx[(focused_idx >= 0) & (focused_idx < signal_nct.shape[0])]
    if focused_idx.size == 0:
        focused_idx = np.arange(signal_nct.shape[0], dtype=int)

    # Method E is a focused-pass tool: avoid re-encoding every 10 s window in
    # long recordings when event scout has already selected candidate windows.
    x = signal_nct[focused_idx].astype(np.float32) * _infer_uv_scale(signal_nct)
    x = x - np.median(x, axis=-1, keepdims=True)
    dx = np.diff(x, axis=-1)
    ddx = np.diff(dx, axis=-1)

    line_length = np.mean(np.abs(dx), axis=(1, 2))
    curvature = np.mean(np.abs(ddx), axis=(1, 2))
    peak_to_peak = np.percentile(x, 99, axis=(1, 2)) - np.percentile(x, 1, axis=(1, 2))
    channel_energy = np.mean(np.abs(x), axis=-1)
    field_concentration = np.max(channel_energy, axis=1) / (np.median(channel_energy, axis=1) + 1e-6)

    def robust_z(v: np.ndarray) -> np.ndarray:
        med = np.median(v)
        mad = np.median(np.abs(v - med)) + 1e-6
        return (v - med) / (1.4826 * mad)

    focused_scores = (
        0.35 * robust_z(line_length)
        + 0.35 * robust_z(curvature)
        + 0.15 * robust_z(peak_to_peak)
        + 0.15 * robust_z(field_concentration)
    )
    support_score = float(np.percentile(focused_scores, 90)) if focused_scores.size else 0.0

    ch_energy_focused = np.mean(channel_energy, axis=0)
    top_channel_indices = np.argsort(ch_energy_focused)[-4:][::-1].tolist() if ch_energy_focused.size else []
    top_channels = [channels[i] for i in top_channel_indices if i < len(channels)]

    morphology_class = "insufficient_morphology_evidence"
    if support_score >= 2.0 and float(np.percentile(field_concentration, 90)) >= 1.5:
        morphology_class = "sharp_transient_candidate"
    elif support_score >= 1.0:
        morphology_class = "nonspecific_transient_candidate"

    class_provenance = make_provenance(
        tool_name="morphology_feature_encoder",
        function_name="morphology_feature_encoder",
        source_ref=source_ref,
        window_indices=[int(x) for x in focused_idx.tolist()],
        channels=top_channels or channels,
        reason="bounded_morphology_proxy_class_not_definitive_epileptiform_label",
    )

    return [
        make_categorical_measurement(
            measurement_id="m_event_morphology_proxy_class",
            measurement_name="event_morphology_proxy_class",
            value=morphology_class,
            provenance=class_provenance,
        ),
        make_distribution_measurement(
            measurement_id="m_event_morphology_proxy_score_distribution",
            measurement_name="event_morphology_proxy_score_distribution",
            values=[float(x) for x in focused_scores],
            unit="score",
            provenance=make_provenance(
                tool_name="morphology_feature_encoder",
                function_name="morphology_feature_encoder",
                source_ref=source_ref,
                window_indices=[int(x) for x in focused_idx.tolist()],
                channels=channels,
                reason="local_feature_encoder_v0_line_length_curvature_field_concentration",
            ),
        ),
        make_exact_measurement(
            measurement_id="m_event_morphology_support_score",
            measurement_name="event_morphology_support_score",
            value=support_score,
            unit="score",
            provenance=make_provenance(
                tool_name="morphology_feature_encoder",
                function_name="morphology_feature_encoder",
                source_ref=source_ref,
                window_indices=[int(x) for x in focused_idx.tolist()],
                channels=top_channels or channels,
                reason="focused_candidate_windows_top_decile_morphology_proxy_score",
            ),
        ),
        make_exact_measurement(
            measurement_id="m_event_field_concentration_ratio",
            measurement_name="event_field_concentration_ratio",
            value=float(np.percentile(field_concentration, 90)) if field_concentration.size else 0.0,
            unit="ratio",
            provenance=make_provenance(
                tool_name="morphology_feature_encoder",
                function_name="morphology_feature_encoder",
                source_ref=source_ref,
                window_indices=[int(x) for x in focused_idx.tolist()],
                channels=top_channels or channels,
                reason="focused_candidate_windows_spatial_field_proxy",
            ),
        ),
    ]


def event_type_separation_classifier(
    signal_nct: np.ndarray,
    channels: List[str],
    suspicious_windows: List[int],
    score_distribution: np.ndarray,
    window_seconds: int,
    source_ref: str,
) -> List[MeasurementValue]:
    """Bounded local classifier separating candidate burden from seizure claims.

    This is a conservative heuristic. It can raise an epileptiform-candidate
    likelihood, but seizure likelihood requires sustained candidate runs and
    rhythmicity-like concentration. It does not create a definitive diagnosis.
    """
    if not suspicious_windows and score_distribution.size:
        suspicious_windows = np.where(score_distribution > np.percentile(score_distribution, 90))[0].tolist()
    suspicious_windows = sorted(set(int(i) for i in suspicious_windows if 0 <= int(i) < signal_nct.shape[0]))
    if not suspicious_windows:
        suspicious_windows = []

    runs = _contiguous_runs(np.asarray(suspicious_windows, dtype=int)) if suspicious_windows else []
    max_run_sec = max([float((end - start + 1) * window_seconds) for start, end in runs], default=0.0)
    burden = float(len(suspicious_windows) / max(signal_nct.shape[0], 1))

    if suspicious_windows:
        x = signal_nct[suspicious_windows].astype(np.float32) * _infer_uv_scale(signal_nct)
        x = x - np.median(x, axis=-1, keepdims=True)
        dx = np.diff(x, axis=-1)
        ddx = np.diff(dx, axis=-1)
        sharpness = float(np.percentile(np.abs(ddx), 95))
        amplitude = float(np.percentile(np.abs(x), 95))
        rhythmicity = float(np.mean(np.abs(np.diff(np.signbit(dx), axis=-1))))
    else:
        sharpness = 0.0
        amplitude = 0.0
        rhythmicity = 0.0

    if score_distribution.size:
        score_prominence = float(np.percentile(score_distribution, 95) / (np.median(score_distribution) + 1e-12))
    else:
        score_prominence = 0.0

    epileptiform_likelihood = float(
        min(
            1.0,
            0.35 * min(score_prominence / 3.0, 1.0)
            + 0.25 * min(sharpness / 50.0, 1.0)
            + 0.20 * min(amplitude / 100.0, 1.0)
            + 0.20 * min(burden / 0.10, 1.0),
        )
    )
    seizure_gate = (
        min(max_run_sec / 120.0, 1.0)
        * min(burden / 0.30, 1.0)
        * min(rhythmicity / 0.50, 1.0)
        * min(score_prominence / 6.0, 1.0)
    )
    # Candidate runs alone should not trigger seizure language. Keep the score
    # low unless multiple seizure-oriented conditions are jointly present.
    seizure_likelihood = float(min(1.0, seizure_gate))
    seizure_pattern_status = (
        StatusSemantic.PRESENT
        if seizure_likelihood >= 0.65 and max_run_sec >= 120.0 and burden >= 0.30
        else StatusSemantic.NOT_OBSERVED
        if seizure_likelihood <= 0.05 and max_run_sec < 30.0 and burden <= 0.10
        else StatusSemantic.UNKNOWN
    )

    provenance = make_provenance(
        tool_name="event_type_separation_classifier",
        function_name="event_type_separation_classifier",
        source_ref=source_ref,
        window_indices=suspicious_windows,
        channels=channels,
        reason="conservative_local_likelihoods_from_candidate_runs_sharpness_rhythmicity_burden",
    )
    return [
        make_exact_measurement(
            measurement_id="m_epileptiform_candidate_likelihood",
            measurement_name="epileptiform_candidate_likelihood_score",
            value=epileptiform_likelihood,
            unit="score",
            provenance=provenance,
        ),
        make_exact_measurement(
            measurement_id="m_electrographic_seizure_likelihood",
            measurement_name="electrographic_seizure_likelihood_score",
            value=seizure_likelihood,
            unit="score",
            provenance=provenance,
        ),
        make_status_measurement(
            measurement_id="m_electrographic_seizure_pattern_status",
            measurement_name="electrographic_seizure_pattern_status",
            status=seizure_pattern_status,
            provenance=provenance,
            reason=(
                "sustained_rhythmic_candidate_run_gate"
                if seizure_pattern_status == StatusSemantic.PRESENT
                else "sustained_rhythmic_evolving_pattern_not_observed"
                if seizure_pattern_status == StatusSemantic.NOT_OBSERVED
                else "seizure_pattern_gate_indeterminate"
            ),
        ),
    ]
