from __future__ import annotations

from typing import List

import numpy as np

from eeg_report_multiagent.schemas.measurement import MeasurementValue
from eeg_report_multiagent.schemas.measurement import StatusSemantic
from eeg_report_multiagent.tools.common import (
    make_exact_measurement,
    make_provenance,
    make_range_measurement,
    make_status_measurement,
)


BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (12.0, 30.0),
}
PHYSIOLOGIC_BAND = (0.5, 30.0)
POSTERIOR_CHANNELS = {"O1", "O2", "P3", "P4", "Pz"}
ANTERIOR_CHANNELS = {"Fp1", "Fp2", "Fpz", "F3", "F4", "F7", "F8", "Fz"}


def _detrend(signal_nct: np.ndarray) -> np.ndarray:
    return signal_nct - np.mean(signal_nct, axis=-1, keepdims=True)


def _infer_voltage_scale(signal_nct: np.ndarray) -> tuple[float, str, str]:
    """Return multiplier to microvolts plus a traceable scale assumption."""
    p95 = float(np.percentile(np.abs(signal_nct), 95.0))
    if p95 < 1.0:
        return 1_000_000.0, "uV", "input_magnitude_lt_1_assumed_volts"
    return 1.0, "uV", "input_magnitude_ge_1_assumed_microvolts"


def _compute_psd(signal_nct: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    # signal_nct: (N, C, T)
    signal_nct = _detrend(signal_nct)
    n = signal_nct.shape[-1]
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    fft = np.fft.rfft(signal_nct, axis=-1)
    psd = (np.abs(fft) ** 2) / n
    return freqs, psd


def _bandpower(freqs: np.ndarray, psd_ncf: np.ndarray, band: tuple[float, float], reduce: str = "sum") -> np.ndarray:
    lo, hi = band
    mask = (freqs >= lo) & (freqs < hi)
    if not np.any(mask):
        return np.zeros(psd_ncf.shape[:-1], dtype=np.float32)
    band_psd = psd_ncf[..., mask]
    if reduce == "mean":
        return band_psd.mean(axis=-1)
    return band_psd.sum(axis=-1)


def psd_power_spectrum_summary(signal_nct: np.ndarray, fs: int, source_ref: str) -> List[MeasurementValue]:
    freqs, psd = _compute_psd(signal_nct, fs=fs)
    mean_psd = psd.mean(axis=(0, 1))
    mask = (freqs >= PHYSIOLOGIC_BAND[0]) & (freqs <= PHYSIOLOGIC_BAND[1])
    dom_freq = float(freqs[mask][int(np.argmax(mean_psd[mask]))]) if np.any(mask) else 0.0
    provenance = make_provenance(
        tool_name="psd_power_spectrum_summary",
        function_name="psd_power_spectrum_summary",
        source_ref=source_ref,
        window_indices=range(signal_nct.shape[0]),
        reason="detrended_psd_argmax_restricted_to_0p5_30_hz",
    )
    m = make_exact_measurement(
        measurement_id="m_background_dominant_frequency",
        measurement_name="background_dominant_frequency_hz",
        value=dom_freq,
        unit="Hz",
        provenance=provenance,
        confidence=0.7,
    )
    m.metadata.update({"frequency_search_hz": "0.5-30", "preprocessing": "per_window_channel_mean_removed"})
    return [m]


def posterior_dominant_rhythm_candidate(
    signal_nct: np.ndarray,
    fs: int,
    source_ref: str,
    channels: List[str] | None = None,
) -> List[MeasurementValue]:
    """Estimate a posterior alpha/PDR candidate without using the 0.5 Hz global argmax.

    This tool intentionally searches posterior channels in the 8-13 Hz range
    and reports a confidence score. It is not a full PDR/reactivity detector.
    """
    if channels is None:
        channels = [str(i) for i in range(signal_nct.shape[1])]
    ch_to_idx = {c: i for i, c in enumerate(channels)}
    posterior_idx = [ch_to_idx[c] for c in channels if c in POSTERIOR_CHANNELS]
    anterior_idx = [ch_to_idx[c] for c in channels if c in ANTERIOR_CHANNELS]
    if not posterior_idx:
        posterior_idx = list(range(signal_nct.shape[1]))

    freqs, psd = _compute_psd(signal_nct[:, posterior_idx, :], fs=fs)
    posterior_psd = psd.mean(axis=(0, 1))
    alpha_mask = (freqs >= 8.0) & (freqs <= 13.0)
    if not np.any(alpha_mask):
        peak_hz = 0.0
        confidence = 0.0
        posterior_alpha_ratio = 0.0
        ap_ratio = 0.0
        symmetry_score = 0.0
    else:
        alpha_freqs = freqs[alpha_mask]
        alpha_psd = posterior_psd[alpha_mask]
        peak_idx = int(np.argmax(alpha_psd))
        peak_hz = float(alpha_freqs[peak_idx])
        posterior_total = float(_bandpower(freqs, psd, PHYSIOLOGIC_BAND).mean()) + 1e-12
        posterior_alpha = float(_bandpower(freqs, psd, (8.0, 13.0)).mean())
        posterior_alpha_ratio = posterior_alpha / posterior_total

        if anterior_idx:
            _, ant_psd = _compute_psd(signal_nct[:, anterior_idx, :], fs=fs)
            anterior_alpha = float(_bandpower(freqs, ant_psd, (8.0, 13.0)).mean())
            ap_ratio = posterior_alpha / (anterior_alpha + 1e-12)
        else:
            ap_ratio = 1.0

        if len(posterior_idx) >= 2:
            posterior_channel_alpha = _bandpower(freqs, psd, (8.0, 13.0)).mean(axis=0)
            left_vals = [
                float(posterior_channel_alpha[i])
                for i, channel_idx in enumerate(posterior_idx)
                if channels[channel_idx] in {"O1", "P3"}
            ]
            right_vals = [
                float(posterior_channel_alpha[i])
                for i, channel_idx in enumerate(posterior_idx)
                if channels[channel_idx] in {"O2", "P4"}
            ]
            if left_vals and right_vals:
                left_alpha = float(np.mean(left_vals))
                right_alpha = float(np.mean(right_vals))
                symmetry_score = float(1.0 - abs(left_alpha - right_alpha) / (left_alpha + right_alpha + 1e-12))
                symmetry_score = max(0.0, min(1.0, symmetry_score))
            else:
                symmetry_score = 0.0
        else:
            symmetry_score = 0.0

        prominence = float(np.max(alpha_psd) / (np.median(alpha_psd) + 1e-12))
        confidence = float(
            min(
                1.0,
                0.45 * min(posterior_alpha_ratio / 0.25, 1.0)
                + 0.35 * min(prominence / 3.0, 1.0)
                + 0.20 * min(ap_ratio / 1.5, 1.0),
            )
        )

    pdr_supported = 8.0 <= peak_hz <= 13.0 and confidence >= 0.35
    provenance = make_provenance(
        tool_name="posterior_dominant_rhythm_candidate",
        function_name="posterior_dominant_rhythm_candidate",
        source_ref=source_ref,
        window_indices=range(signal_nct.shape[0]),
        channels=[channels[i] for i in posterior_idx if i < len(channels)],
        region="posterior",
        reason="posterior_channel_alpha_peak_search_8_13_hz_not_global_psd_argmax",
    )
    freq = make_exact_measurement(
        measurement_id="m_pdr_candidate_frequency",
        measurement_name="pdr_candidate_frequency_hz",
        value=peak_hz,
        unit="Hz",
        provenance=provenance,
        confidence=confidence,
    )
    freq.metadata.update(
        {
            "pdr_supported": str(pdr_supported).lower(),
            "posterior_alpha_ratio": f"{posterior_alpha_ratio:.6f}",
            "posterior_anterior_alpha_ratio": f"{ap_ratio:.6f}",
            "posterior_alpha_symmetry_score": f"{symmetry_score:.6f}",
            "search_hz": "8-13",
        }
    )
    score = make_exact_measurement(
        measurement_id="m_pdr_candidate_confidence",
        measurement_name="pdr_candidate_confidence_score",
        value=confidence,
        unit="score",
        provenance=provenance,
        confidence=confidence,
    )
    score.metadata.update(freq.metadata)
    ap = make_exact_measurement(
        measurement_id="m_pdr_posterior_anterior_alpha_ratio",
        measurement_name="pdr_posterior_anterior_alpha_ratio",
        value=ap_ratio,
        unit="ratio",
        provenance=provenance,
        confidence=confidence,
    )
    symmetry = make_exact_measurement(
        measurement_id="m_pdr_symmetry_score",
        measurement_name="pdr_symmetry_score",
        value=symmetry_score,
        unit="score",
        provenance=provenance,
        confidence=confidence,
    )
    ap.metadata.update(freq.metadata)
    symmetry.metadata.update(freq.metadata)
    return [freq, score, ap, symmetry]


def background_organization_proxy(
    signal_nct: np.ndarray,
    fs: int,
    source_ref: str,
    channels: List[str] | None = None,
) -> List[MeasurementValue]:
    """Typed proxy for anterior-posterior organization from alpha topography."""
    if channels is None:
        channels = [str(i) for i in range(signal_nct.shape[1])]
    ch_to_idx = {c: i for i, c in enumerate(channels)}
    posterior_idx = [ch_to_idx[c] for c in channels if c in POSTERIOR_CHANNELS]
    anterior_idx = [ch_to_idx[c] for c in channels if c in ANTERIOR_CHANNELS]
    if not posterior_idx or not anterior_idx:
        score = 0.0
        region_channels = channels
        reason = "missing_standard_anterior_or_posterior_channels"
    else:
        freqs, post_psd = _compute_psd(signal_nct[:, posterior_idx, :], fs=fs)
        _, ant_psd = _compute_psd(signal_nct[:, anterior_idx, :], fs=fs)
        posterior_alpha = float(_bandpower(freqs, post_psd, (8.0, 13.0)).mean())
        anterior_alpha = float(_bandpower(freqs, ant_psd, (8.0, 13.0)).mean())
        ratio = posterior_alpha / (anterior_alpha + 1e-12)
        score = float(min(1.0, ratio / 2.0))
        region_channels = [channels[i] for i in posterior_idx + anterior_idx if i < len(channels)]
        reason = "posterior_over_anterior_alpha_ratio_proxy"

    return [
        make_exact_measurement(
            measurement_id="m_background_ap_organization_score",
            measurement_name="background_ap_organization_score",
            value=score,
            unit="score",
            provenance=make_provenance(
                tool_name="background_organization_proxy",
                function_name="background_organization_proxy",
                source_ref=source_ref,
                window_indices=range(signal_nct.shape[0]),
                channels=region_channels,
                region="anterior-posterior",
                reason=reason,
            ),
            confidence=0.45,
        )
    ]


def background_unavailable_slot_status(source_ref: str) -> List[MeasurementValue]:
    """Declare clinically important background slots that v1 cannot measure yet."""
    provenance = make_provenance(
        tool_name="background_unavailable_slot_status",
        function_name="background_unavailable_slot_status",
        source_ref=source_ref,
        reason="no_eye_opening_markers_sleep_stage_classifier_or_protocol_events_in_processed_pkl",
    )
    return [
        make_status_measurement(
            measurement_id="m_background_reactivity_status",
            measurement_name="background_reactivity_status",
            status=StatusSemantic.UNKNOWN,
            provenance=provenance,
            reason="reactivity requires activation/eye-opening markers not available to v1 signal tools",
            confidence=0.9,
        ),
        make_status_measurement(
            measurement_id="m_sleep_architecture_status",
            measurement_name="sleep_architecture_status",
            status=StatusSemantic.UNKNOWN,
            provenance=provenance,
            reason="sleep architecture requires a validated sleep/state detector not available in v1",
            confidence=0.9,
        ),
    ]


def bandpower_summary(signal_nct: np.ndarray, fs: int, source_ref: str) -> List[MeasurementValue]:
    freqs, psd = _compute_psd(signal_nct, fs=fs)
    total_power = _bandpower(freqs, psd, PHYSIOLOGIC_BAND).mean() + 1e-12
    out: List[MeasurementValue] = []
    for band_name, band in BANDS.items():
        bp = float(_bandpower(freqs, psd, band).mean())
        rel = bp / total_power
        m = make_exact_measurement(
            measurement_id=f"m_bandpower_{band_name}",
            measurement_name=f"relative_bandpower_{band_name}",
            value=rel,
            unit="ratio",
            provenance=make_provenance(
                tool_name="bandpower_summary",
                function_name="bandpower_summary",
                source_ref=source_ref,
                window_indices=range(signal_nct.shape[0]),
                reason="band_power_sum_divided_by_total_0p5_30_hz_power_after_detrending",
            ),
            confidence=0.7,
        )
        m.metadata.update({"band_hz": f"{band[0]}-{band[1]}", "denominator_hz": "0.5-30"})
        out.append(m)
    return out


def amplitude_summary(signal_nct: np.ndarray, source_ref: str) -> List[MeasurementValue]:
    # robust amplitude range from global 5th-95th percentile of absolute amplitude
    scale, unit, scale_assumption = _infer_voltage_scale(signal_nct)
    abs_sig = np.abs(signal_nct * scale)
    lo = float(np.percentile(abs_sig, 5.0))
    hi = float(np.percentile(abs_sig, 95.0))
    m = make_range_measurement(
        measurement_id="m_background_amplitude_range",
        measurement_name="background_amplitude_range_uv",
        lower=lo,
        upper=hi,
        unit=unit,
        provenance=make_provenance(
            tool_name="amplitude_summary",
            function_name="amplitude_summary",
            source_ref=source_ref,
            window_indices=range(signal_nct.shape[0]),
            reason=scale_assumption,
        ),
        confidence=0.65,
    )
    m.metadata.update({"scale_assumption": scale_assumption, "percentile_range": "abs_signal_p5_p95"})
    return [m]


def slowing_score(signal_nct: np.ndarray, fs: int, source_ref: str) -> List[MeasurementValue]:
    freqs, psd = _compute_psd(signal_nct, fs=fs)
    delta = float(_bandpower(freqs, psd, BANDS["delta"]).mean())
    theta = float(_bandpower(freqs, psd, BANDS["theta"]).mean())
    alpha = float(_bandpower(freqs, psd, BANDS["alpha"]).mean())
    beta = float(_bandpower(freqs, psd, BANDS["beta"]).mean())
    score = float((delta + theta) / (alpha + beta + 1e-12))
    return [
        make_exact_measurement(
            measurement_id="m_slowing_score",
            measurement_name="slowing_score",
            value=score,
            unit="ratio",
            provenance=make_provenance(
                tool_name="slowing_score",
                function_name="slowing_score",
                source_ref=source_ref,
                window_indices=range(signal_nct.shape[0]),
            ),
            confidence=0.72,
        )
    ]


def beta_excess_score(signal_nct: np.ndarray, fs: int, source_ref: str) -> List[MeasurementValue]:
    freqs, psd = _compute_psd(signal_nct, fs=fs)
    beta = float(_bandpower(freqs, psd, BANDS["beta"]).mean())
    others = (
        float(_bandpower(freqs, psd, BANDS["delta"]).mean())
        + float(_bandpower(freqs, psd, BANDS["theta"]).mean())
        + float(_bandpower(freqs, psd, BANDS["alpha"]).mean())
    )
    score = float(beta / (others + 1e-12))
    return [
        make_exact_measurement(
            measurement_id="m_beta_excess_score",
            measurement_name="beta_excess_score",
            value=score,
            unit="ratio",
            provenance=make_provenance(
                tool_name="beta_excess_score",
                function_name="beta_excess_score",
                source_ref=source_ref,
                window_indices=range(signal_nct.shape[0]),
            ),
            confidence=0.7,
        )
    ]
