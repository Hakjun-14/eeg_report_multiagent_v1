from __future__ import annotations

from typing import List, Tuple

import numpy as np

from eeg_report_multiagent.schemas.measurement import MeasurementValue
from eeg_report_multiagent.tools.common import (
    make_categorical_measurement,
    make_distribution_measurement,
    make_exact_measurement,
    make_provenance,
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
    focused_windows = _sanitize_windows(suspicious_windows, signal_nct.shape[0])
    x = signal_nct[focused_windows].astype(np.float32) * _infer_uv_scale(signal_nct)
    x = x - np.median(x, axis=-1, keepdims=True)
    if x.size == 0:
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
    ]
