"""Transcribe the drum stem into separate kit pieces.

ADTOF's Frame RNN (PyTorch) hears five families on the separated drum stem:
kick, snare, hi-hat, tom, and cymbal. A short look at each hit then splits
hi-hat into closed/open, cymbal into ride/crash, and tom into high/mid/floor.
Those names map to General MIDI drum notes. If the model cannot be loaded,
a simpler spectrum classifier is used so a half-finished install still runs.
"""

from __future__ import annotations

import logging
import tempfile
import threading
from pathlib import Path

import numpy as np

from app.audio_io import save_wav, to_mono

logger = logging.getLogger(__name__)

# Rows the interface shows, and the General MIDI note written into the MIDI file.
DRUM_MIDI = {
    "kick": 36,
    "snare": 38,
    "hat_closed": 42,
    "hat_open": 46,
    "ride": 51,
    "crash": 49,
    "tom_high": 50,
    "tom_mid": 47,
    "tom_floor": 43,
}
DRUM_KINDS = tuple(DRUM_MIDI)

# ADTOF's five outputs, as General MIDI note numbers.
_ADTOF_FAMILY = {
    35: "kick",
    38: "snare",
    42: "hat",
    47: "tom",
    49: "cymbal",
}

_model = None
_model_device = None
_model_lock = threading.Lock()


def _decay_seconds(segment: np.ndarray, sample_rate: int, ratio: float = 0.2) -> float:
    if segment.size < 8:
        return 0.0
    hop = max(1, sample_rate // 200)
    envelope = np.array(
        [float(np.sqrt(np.mean(segment[index : index + hop] ** 2))) for index in range(0, segment.size, hop)],
        dtype=np.float32,
    )
    peak = float(envelope.max()) if envelope.size else 0.0
    if peak <= 1e-6:
        return 0.0
    peak_at = int(np.argmax(envelope))
    below = np.where(envelope[peak_at:] < peak * ratio)[0]
    if below.size == 0:
        return float(segment.size) / sample_rate
    return float((peak_at + int(below[0])) * hop) / sample_rate


def _spectral_flatness(segment: np.ndarray) -> float:
    if segment.size < 16:
        return 0.0
    window = segment * np.hanning(segment.size)
    spectrum = np.abs(np.fft.rfft(window)) + 1e-8
    geometric = float(np.exp(np.mean(np.log(spectrum))))
    arithmetic = float(np.mean(spectrum))
    return geometric / arithmetic


def _fundamental(segment: np.ndarray, sample_rate: int) -> float:
    clip = np.asarray(segment[: int(0.09 * sample_rate)], dtype=np.float32)
    if clip.size < 64:
        return 0.0
    clip = clip - float(clip.mean())
    energy = float(np.dot(clip, clip))
    if energy <= 1e-8:
        return 0.0
    correlation = np.correlate(clip, clip, mode="full")[clip.size - 1 :]
    low_hz, high_hz = 60.0, 400.0
    min_lag = max(1, int(sample_rate / high_hz))
    max_lag = min(correlation.size - 1, int(sample_rate / low_hz))
    if max_lag <= min_lag:
        return 0.0
    lag = min_lag + int(np.argmax(correlation[min_lag : max_lag + 1]))
    if correlation[lag] < 0.25 * correlation[0]:
        return 0.0
    return float(sample_rate) / float(lag)


def split_family(family: str, segment: np.ndarray, sample_rate: int) -> str:
    """Turn an ADTOF family into one kit piece."""
    if family == "kick":
        return "kick"
    if family == "snare":
        return "snare"
    if family == "hat":
        return "hat_open" if _decay_seconds(segment, sample_rate) >= 0.09 else "hat_closed"
    if family == "cymbal":
        decay = _decay_seconds(segment, sample_rate, ratio=0.18)
        flatness = _spectral_flatness(segment[: min(segment.size, int(0.12 * sample_rate))])
        if decay >= 0.35 or (decay >= 0.22 and flatness >= 0.28):
            return "crash"
        return "ride"
    if family == "tom":
        pitch = _fundamental(segment, sample_rate)
        if pitch >= 150:
            return "tom_high"
        if pitch >= 95:
            return "tom_mid"
        return "tom_floor"
    return "snare"


def _band_energy(segment: np.ndarray, sample_rate: int) -> tuple[float, float, float, float]:
    window = segment * np.hanning(segment.size)
    spectrum = np.abs(np.fft.rfft(window)) ** 2
    frequencies = np.fft.rfftfreq(window.size, 1.0 / sample_rate)
    if spectrum.sum() <= 0:
        return 0.0, 0.0, 0.0, 0.0

    def band(low: float, high: float) -> float:
        chosen = (frequencies >= low) & (frequencies < high)
        if not np.any(chosen):
            return 0.0
        return float(spectrum[chosen].sum())

    low = band(20, 180)
    mid = band(180, 4000)
    nyquist = sample_rate / 2 - 50
    high = band(6000, max(6500, nyquist))
    centroid = float(np.sum(frequencies * spectrum) / (np.sum(spectrum) + 1e-12))
    return low, mid, high, centroid


def classify_hit(segment: np.ndarray, sample_rate: int) -> list[str]:
    """Fallback labels when the drum model is unavailable.

    Kick and a closed hat can happen together. The window is short, so an
    open hat, ride, crash, or tom is not guessed here.
    """
    if segment.size < 16:
        return []
    low, mid, high, centroid = _band_energy(segment, sample_rate)
    labels: list[str] = []
    if low > mid * 1.15 and low > high * 1.15 and low > 0:
        labels.append("kick")
    if high > low * 1.4 and high > mid * 0.75 and high > 0 and centroid > 2500:
        labels.append("hat_closed")
    if mid > low * 0.9 and mid > high * 0.65 and mid > 0 and centroid < 4500 and "kick" not in labels:
        labels.append("snare")
    if not labels:
        dominant = max((low, "kick"), (mid, "snare"), (high, "hat_closed"))
        if dominant[0] <= 0:
            return []
        labels.append(dominant[1])
    return labels


def detect_drums_by_spectrum(audio: np.ndarray, sample_rate: int) -> list[dict]:
    """Onset detector plus the simple spectrum labels. Used when ADTOF is absent."""
    import librosa

    mono = to_mono(audio)
    if mono.size < sample_rate // 2:
        return []
    if float(np.sqrt(np.mean(mono**2))) < 1e-4:
        return []
    pad = int(0.05 * sample_rate)
    mono = np.concatenate([np.zeros(pad, dtype=np.float32), mono])

    hop = 512
    onset_env = librosa.onset.onset_strength(y=mono, sr=sample_rate, hop_length=hop)
    frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sample_rate,
        hop_length=hop,
        units="frames",
        backtrack=False,
        delta=0.15,
        wait=3,
    )
    if frames.size == 0:
        return []

    strengths = onset_env[np.clip(frames, 0, onset_env.shape[0] - 1)]
    positive = strengths[strengths > 0]
    ceiling = float(np.percentile(positive, 95)) if positive.size else 1.0
    ceiling = max(ceiling, 1e-6)

    window = int(0.045 * sample_rate)
    hits: list[dict] = []
    min_gap = 0.05
    for frame, strength in zip(frames, strengths):
        start = int(librosa.frames_to_samples(int(frame), hop_length=hop))
        segment = mono[start : start + window]
        kinds = classify_hit(segment, sample_rate)
        velocity = int(np.clip(round(40 + 87 * (float(strength) / ceiling)), 1, 127))
        when = max(0.0, float(start - pad) / sample_rate)
        for kind in kinds:
            if hits and hits[-1]["kind"] == kind and when - hits[-1]["time"] < min_gap:
                if velocity > hits[-1]["velocity"]:
                    hits[-1]["velocity"] = velocity
                    hits[-1]["time"] = round(when, 4)
                continue
            hits.append({"time": round(when, 4), "kind": kind, "velocity": velocity})
    hits.sort(key=lambda hit: (hit["time"], hit["kind"]))
    return hits


def _load_adtof(device: str):
    global _model, _model_device
    with _model_lock:
        if _model is not None and _model_device == device:
            return _model
        from adtof_pytorch import (
            calculate_n_bins,
            create_frame_rnn_model,
            get_default_weights_path,
            load_pytorch_weights,
        )

        weights = get_default_weights_path()
        if not weights or not Path(weights).exists():
            raise FileNotFoundError("没有找到 ADTOF 鼓点模型的权重文件。")
        model = create_frame_rnn_model(calculate_n_bins())
        model.eval()
        model = load_pytorch_weights(model, weights, strict=False)
        model.to(device)
        _model = model
        _model_device = device
        return _model


def _detect_with_adtof(audio: np.ndarray, sample_rate: int) -> list[dict]:
    import librosa
    import torch
    from adtof_pytorch import FRAME_RNN_THRESHOLDS, LABELS_5, PeakPicker, load_audio_for_model

    mono = to_mono(audio)
    if mono.size < sample_rate // 2 or float(np.sqrt(np.mean(mono**2))) < 1e-4:
        return []
    model_rate = 44100
    if sample_rate != model_rate:
        mono = librosa.resample(mono, orig_sr=sample_rate, target_sr=model_rate).astype(np.float32)
        sample_rate = model_rate

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_adtof(device)
    with tempfile.TemporaryDirectory() as folder:
        wav_path = Path(folder) / "drums.wav"
        save_wav(wav_path, mono, sample_rate)
        features = load_audio_for_model(str(wav_path)).to(device)
    with torch.no_grad():
        activations = model(features).detach().cpu().numpy()
    if activations.ndim == 2:
        activations = activations[None, ...]

    picker = PeakPicker(thresholds=FRAME_RNN_THRESHOLDS, fps=100)
    peaks = picker.pick(activations, labels=LABELS_5, label_offset=0)[0]
    hits: list[dict] = []
    duration = float(mono.size) / sample_rate
    for index, pitch in enumerate(LABELS_5):
        family = _ADTOF_FAMILY.get(int(pitch))
        if family is None:
            continue
        column = activations[0, :, index]
        for when in peaks.get(int(pitch), []):
            when = float(when)
            if when < 0 or when > duration + 0.02:
                continue
            when = min(duration, max(0.0, when))
            frame = int(np.clip(round(when * 100), 0, column.shape[0] - 1))
            strength = float(column[frame])
            start = int(when * sample_rate)
            segment = mono[start : start + int(0.5 * sample_rate)]
            kind = split_family(family, segment, sample_rate)
            velocity = int(np.clip(round(28 + 99 * strength), 1, 127))
            hits.append({"time": round(when, 4), "kind": kind, "velocity": velocity})
    hits.sort(key=lambda hit: (hit["time"], hit["kind"]))
    return hits


def detect_drums(audio: np.ndarray, sample_rate: int) -> list[dict]:
    """Return hits as {time, kind, velocity}, ordered in time."""
    try:
        hits = _detect_with_adtof(audio, sample_rate)
    except Exception:
        logger.exception("鼓点模型没有跑起来，改用简单的频谱分类。")
        return detect_drums_by_spectrum(audio, sample_rate)
    if hits:
        return hits
    # A very quiet or unusual stem can leave the model with nothing to say.
    return detect_drums_by_spectrum(audio, sample_rate)
