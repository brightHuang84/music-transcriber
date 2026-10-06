"""Find drum hits and label them kick, snare, or hi-hat.

The labels come from the balance of low, mid, and high energy around each
onset. That is less accurate than a dedicated drum model, but it installs
cleanly on CPU and is understandable when it makes a mistake.
"""

from __future__ import annotations

import numpy as np

from app.audio_io import to_mono

DRUM_KINDS = ("kick", "snare", "hihat", "other")


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
    """Return one or more drum names heard in a short window.

    Kick and hi-hat often sound together, so both can be returned.
    """
    if segment.size < 16:
        return []
    low, mid, high, centroid = _band_energy(segment, sample_rate)
    labels: list[str] = []
    if low > mid * 1.15 and low > high * 1.15 and low > 0:
        labels.append("kick")
    if high > low * 1.4 and high > mid * 0.75 and high > 0 and centroid > 2500:
        labels.append("hihat")
    # A snare is mostly mid noise. Don't also call a pure kick a snare.
    if mid > low * 0.9 and mid > high * 0.65 and mid > 0 and centroid < 4500 and "kick" not in labels:
        labels.append("snare")
    if not labels:
        dominant = max((low, "kick"), (mid, "snare"), (high, "hihat"))
        if dominant[0] <= 0:
            return []
        labels.append(dominant[1])
    return labels


def detect_drums(audio: np.ndarray, sample_rate: int) -> list[dict]:
    """Return hits as {time, kind, velocity}, ordered in time."""
    import librosa

    mono = to_mono(audio)
    if mono.size < sample_rate // 2:
        return []
    if float(np.sqrt(np.mean(mono**2))) < 1e-4:
        return []
    # A short silence lets an onset on the very first sample still be detected.
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
        when = float(start - pad) / sample_rate
        if when < -0.03:
            continue
        when = max(0.0, when)
        for kind in kinds:
            if hits and hits[-1]["kind"] == kind and when - hits[-1]["time"] < min_gap:
                if velocity > hits[-1]["velocity"]:
                    hits[-1]["velocity"] = velocity
                    hits[-1]["time"] = when
                continue
            hits.append({"time": round(when, 4), "kind": kind, "velocity": velocity})
    hits.sort(key=lambda hit: (hit["time"], hit["kind"]))
    return hits
