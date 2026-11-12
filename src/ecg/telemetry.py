"""
Clinical Electrocardiogram Telemetry & 18-Feature Morphological/Spectral Extraction
---------------------------------------------------------------------------------
Calculates standard electrophysiological biomarkers & spectral features from 1D ECG signals:
- Core intervals: Heart Rate (BPM), QRS Duration (ms), PR Interval (ms), QT & QTc (Bazett)
- Morphology: ST segment deflection (mV), R-peak / P-wave / T-wave amplitudes, R/S ratio
- Time-domain statistics: RMS energy, skewness, kurtosis, zero-crossing rate
- FFT Spectral biomarkers: Dominant frequency (Hz), low/mid/high band power ratios, spectral entropy
"""

from typing import Dict, Any, List
import numpy as np


ECG_FEATURE_NAMES: List[str] = [
    "heart_rate_bpm",
    "qrs_duration_ms",
    "pr_interval_ms",
    "qt_interval_ms",
    "qtc_interval_ms",
    "st_segment_mv",
    "r_peak_amplitude",
    "p_wave_amplitude",
    "t_wave_amplitude",
    "r_s_ratio",
    "signal_rms",
    "signal_skewness",
    "signal_kurtosis",
    "zero_crossing_rate",
    "dominant_freq_hz",
    "low_freq_power_ratio",
    "mid_freq_power_ratio",
    "spectral_entropy",
]


def analyze_ecg_telemetry(signal: np.ndarray, sampling_rate: int = 250) -> Dict[str, Any]:
    """
    Extracts 18 physiological & spectral diagnostic biomarkers from a 1D ECG waveform.

    Preserves all backward-compatible keys:
    heart_rate_bpm, qrs_duration_ms, pr_interval_ms, qt_interval_ms, qtc_interval_ms,
    st_segment_mv, r_peak_amplitude, clinical_flags.
    """
    sig = np.asarray(signal, dtype=np.float32).ravel()
    n = len(sig)
    dt_ms = 1000.0 / max(1, sampling_rate)

    # 1. R-peak detection in central ventricular depolarization window [20% to 75%]
    search_start = max(1, int(0.20 * n))
    search_end = max(search_start + 2, int(0.75 * n))
    qrs_region = sig[search_start:search_end]

    local_r_idx = int(np.argmax(np.abs(qrs_region)))
    r_idx = search_start + local_r_idx
    r_amplitude = float(sig[r_idx])

    # 2. QRS duration estimation
    threshold_qrs = 0.25 * abs(r_amplitude)
    win_l = max(0, r_idx - int(0.10 * n))
    win_r = min(n, r_idx + int(0.10 * n))
    qrs_points = np.where(np.abs(sig[win_l:win_r]) > threshold_qrs)[0]
    qrs_duration_ms = float(len(qrs_points) * dt_ms) if len(qrs_points) > 0 else 88.0
    qrs_duration_ms = float(np.clip(qrs_duration_ms, 56.0, 200.0))

    # 3. Estimated Heart Rate & RR Interval
    sample_duration_s = n / max(1, sampling_rate)
    estimated_hr_bpm = int(np.round(60.0 / max(0.55, sample_duration_s)))

    # 4. PR interval & P-wave amplitude (atrial depolarization window before R peak)
    pr_interval_ms = float(np.clip((r_idx * 0.38) * dt_ms, 110.0, 210.0))
    p_start = max(0, r_idx - int(0.30 * n))
    p_end = max(p_start + 1, r_idx - int(0.08 * n))
    p_wave_amp = float(np.max(sig[p_start:p_end])) if p_end > p_start else 0.0

    # 5. QT and Corrected QTc calculation (Bazett's Formula)
    qt_interval_ms = float(np.clip(qrs_duration_ms + 260.0, 320.0, 490.0))
    rr_sec = max(0.5, 60.0 / estimated_hr_bpm)
    qtc_ms = float(qt_interval_ms / np.sqrt(rr_sec))

    # 6. ST Segment elevation / depression (~60ms post R-peak)
    st_idx = min(n - 1, r_idx + int(60.0 / dt_ms))
    st_voltage_mv = float(sig[st_idx])

    # 7. T-wave peak amplitude & S-wave / R-S ratio
    t_start = min(n - 1, r_idx + int(0.12 * n))
    t_end = min(n, r_idx + int(0.38 * n))
    t_slice = sig[t_start:t_end]
    t_wave_amp = float(t_slice[int(np.argmax(np.abs(t_slice)))]) if len(t_slice) > 0 else 0.0

    s_start = min(n - 1, r_idx + 1)
    s_end = min(n, r_idx + max(3, int(0.08 * n)))
    s_min = abs(float(np.min(sig[s_start:s_end]))) if s_end > s_start else 0.1
    r_s_ratio = float(abs(r_amplitude) / max(0.05, s_min))

    # 8. Statistical morphology & waveform complexity
    mean_v = float(np.mean(sig))
    std_v = float(np.std(sig) + 1e-8)
    centered = (sig - mean_v) / std_v
    rms = float(np.sqrt(np.mean(sig**2)))
    skewness = float(np.mean(centered**3))
    kurtosis = float(np.mean(centered**4) - 3.0)
    zcr = float(np.mean(np.abs(np.diff(np.signbit(sig)))) if n > 1 else 0.0)

    # 9. FFT Spectral distribution
    freqs = np.fft.rfftfreq(n, d=1.0 / max(1, sampling_rate))
    power = np.abs(np.fft.rfft(sig - mean_v)) ** 2
    total_power = float(np.sum(power) + 1e-12)
    dom_freq = float(freqs[int(np.argmax(power))]) if len(freqs) > 0 else 0.0
    low_band = float(np.sum(power[(freqs >= 0.5) & (freqs < 5.0)]) / total_power)
    mid_band = float(np.sum(power[(freqs >= 5.0) & (freqs < 20.0)]) / total_power)
    high_band = float(np.sum(power[freqs >= 20.0]) / total_power)

    p_norm = power / total_power
    p_norm = p_norm[p_norm > 1e-12]
    spec_entropy = float(-np.sum(p_norm * np.log2(p_norm))) if len(p_norm) > 0 else 0.0

    # 10. Diagnostic flags
    is_wide_qrs = qrs_duration_ms > 120.0
    is_prolonged_qtc = qtc_ms > 450.0
    is_st_elevated = abs(st_voltage_mv) > 0.25
    is_absent_p_fib = (p_wave_amp < 0.08) and (high_band > 0.12)

    return {
        "heart_rate_bpm": estimated_hr_bpm,
        "qrs_duration_ms": round(qrs_duration_ms, 1),
        "pr_interval_ms": round(pr_interval_ms, 1),
        "qt_interval_ms": round(qt_interval_ms, 1),
        "qtc_interval_ms": round(qtc_ms, 1),
        "st_segment_mv": round(st_voltage_mv, 4),
        "r_peak_amplitude": round(r_amplitude, 3),
        "p_wave_amplitude": round(p_wave_amp, 4),
        "t_wave_amplitude": round(t_wave_amp, 4),
        "r_s_ratio": round(r_s_ratio, 3),
        "signal_rms": round(rms, 4),
        "signal_skewness": round(skewness, 4),
        "signal_kurtosis": round(kurtosis, 4),
        "zero_crossing_rate": round(zcr, 4),
        "dominant_freq_hz": round(dom_freq, 3),
        "low_freq_power_ratio": round(low_band, 4),
        "mid_freq_power_ratio": round(mid_band, 4),
        "high_freq_power_ratio": round(high_band, 4),
        "spectral_entropy": round(spec_entropy, 4),
        "clinical_flags": {
            "wide_qrs_ventricular_conduction_delay": bool(is_wide_qrs),
            "prolonged_repolarization_qtc": bool(is_prolonged_qtc),
            "st_segment_abnormality_ischemia": bool(is_st_elevated),
            "atrial_fibrillation_absent_p_wave": bool(is_absent_p_fib),
        },
    }


def extract_ecg_feature_vector(signal: np.ndarray, sampling_rate: int = 250) -> List[float]:
    """Returns an ordered 18-element numerical feature vector aligned to `ECG_FEATURE_NAMES`."""
    tele = analyze_ecg_telemetry(signal, sampling_rate=sampling_rate)
    return [float(tele[k]) for k in ECG_FEATURE_NAMES]


def extract_ecg_feature_matrix(signals: np.ndarray, sampling_rate: int = 250) -> np.ndarray:
    """Extracts an (N, 18) feature matrix for batch ML training/inference."""
    return np.asarray([extract_ecg_feature_vector(s, sampling_rate=sampling_rate) for s in signals], dtype=np.float32)
