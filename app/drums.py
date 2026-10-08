"""Transcribe the drum stem into separate kit pieces.

ADTOF's Frame RNN (PyTorch) hears five families on the separated drum stem:
kick, snare, hi-hat, tom, and cymbal. Each family has its own threshold, so
a kick and a hat at the same moment both come through. A short look at each
hit then splits hi-hat into closed/open, cymbal into ride/crash, and tom into
high/mid/floor. A hat or a cymbal stays only when the sound is actually bright,
and it stays next to the kick or snare underneath it. Two tom names, or a tom
that is really the same low thud as the kick, collapse to one drum. A low thud
is not called a high tom. Those names map to General MIDI drum notes. If the
model cannot be loaded, a simpler spectrum classifier is used so a
half-finished install still runs.
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
    if envelope.size == 0:
        return 0.0
    # The hit being named is at the start. A louder drum later in the window
    # must not become the peak: that made every closed hat before a snare,
    # and every 16th-note hat, look like an open hat.
    attack = max(1, int(round(0.04 * sample_rate / hop)))
    peak_at = int(np.argmax(envelope[:attack]))
    peak = float(envelope[peak_at])
    if peak <= 1e-6:
        return 0.0
    below = np.where(envelope[peak_at:] < peak * ratio)[0]
    if below.size == 0:
        return float(segment.size) / sample_rate
    return float((peak_at + int(below[0])) * hop) / sample_rate


def _high_passed(segment: np.ndarray, sample_rate: int, cutoff: float) -> np.ndarray:
    """Drop low drums so a kick under a hat does not stretch the hat's decay."""
    if segment.size < 32:
        return segment
    spectrum = np.fft.rfft(np.asarray(segment, dtype=np.float32))
    frequencies = np.fft.rfftfreq(segment.size, 1.0 / sample_rate)
    spectrum[frequencies < cutoff] = 0
    return np.fft.irfft(spectrum, n=segment.size).astype(np.float32)


def _spectral_flatness(segment: np.ndarray) -> float:
    if segment.size < 16:
        return 0.0
    window = segment * np.hanning(segment.size)
    spectrum = np.abs(np.fft.rfft(window)) + 1e-8
    geometric = float(np.exp(np.mean(np.log(spectrum))))
    arithmetic = float(np.mean(spectrum))
    return geometric / arithmetic


def _fundamental(segment: np.ndarray, sample_rate: int) -> float:
    """Lowest strong partial between 45 and 320 Hz.

    Autocorrelation on a low thud peaks at the shortest lag it is allowed to
    look at, which is about 400 Hz, and that was labeling bass drums as high
    toms. The spectral peak stays on the low tone.
    """
    clip = np.asarray(segment[: int(0.09 * sample_rate)], dtype=np.float32)
    if clip.size < 64:
        return 0.0
    clip = clip - float(clip.mean())
    window = clip * np.hanning(clip.size)
    spectrum = np.abs(np.fft.rfft(window))
    frequencies = np.fft.rfftfreq(window.size, 1.0 / sample_rate)
    band = (frequencies >= 45.0) & (frequencies <= 320.0)
    if not np.any(band):
        return 0.0
    peak = float(spectrum[band].max())
    if peak <= 1e-8:
        return 0.0
    strong = np.where(band & (spectrum >= peak * 0.55))[0]
    if strong.size == 0:
        return 0.0
    return float(frequencies[int(strong[0])])


def split_family(family: str, segment: np.ndarray, sample_rate: int) -> str:
    """Turn an ADTOF family into one kit piece."""
    if family == "kick":
        return "kick"
    if family == "snare":
        return "snare"
    if family == "hat":
        bright = _high_passed(segment, sample_rate, 3000.0)
        return "hat_open" if _decay_seconds(bright, sample_rate) >= 0.09 else "hat_closed"
    if family == "cymbal":
        # The kick under a crash is loud and short. Judging the ring on the
        # unfiltered mix called that crash a ride.
        bright = _high_passed(segment, sample_rate, 3000.0)
        decay = _decay_seconds(bright, sample_rate, ratio=0.18)
        window = bright[: min(bright.size, int(0.12 * sample_rate))]
        flatness = _spectral_flatness(window)
        # 0.30 s, not 0.35: a snare on the same onset steals the first peak
        # and shortens the measured ring of a crash.
        if decay >= 0.30 or (decay >= 0.22 and flatness >= 0.28):
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


_LOW_KINDS = {"kick", "snare", "tom_high", "tom_mid", "tom_floor"}
_BRIGHT_KINDS = {"hat_closed", "hat_open", "ride", "crash"}


def _attack_profile(segment: np.ndarray, sample_rate: int) -> tuple[float, float, float]:
    """Noise ratio (700–5000 Hz), air ratio (above 5 kHz), and low pitch."""
    clip = np.asarray(segment[: max(16, int(0.08 * sample_rate))], dtype=np.float32)
    if clip.size < 16:
        return 0.0, 0.0, 0.0
    clip = clip - float(np.mean(clip))
    window = clip * np.hanning(clip.size)
    spectrum = np.abs(np.fft.rfft(window)) ** 2
    frequencies = np.fft.rfftfreq(window.size, 1.0 / sample_rate)
    total = float(spectrum.sum()) + 1e-12
    noise = float(spectrum[(frequencies >= 700.0) & (frequencies < 5000.0)].sum()) / total
    air = float(spectrum[frequencies >= 5000.0].sum()) / total
    return noise, air, _fundamental(clip, sample_rate)


def _choose_low(
    kinds: list[str],
    segment: np.ndarray,
    sample_rate: int,
    noise: float,
    fundamental: float,
) -> str | None:
    """One low-drum name for an onset the model may have labeled several times."""
    if not any(kind in _LOW_KINDS for kind in kinds):
        return None
    # Snare wires are noisy. A dark hit around 60 Hz is a kick or a low tom,
    # even when the model also said snare.
    if "snare" in kinds and noise >= 0.18:
        return "snare"
    pitched = fundamental if fundamental > 0 else _fundamental(segment, sample_rate)
    if "kick" in kinds and noise < 0.18 and (pitched <= 0 or pitched < 105):
        return "kick"
    if pitched >= 70:
        return split_family("tom", segment, sample_rate)
    if "kick" in kinds or noise < 0.12:
        return "kick"
    if "snare" in kinds:
        return "snare"
    return "tom_floor"


def _band_rms(segment: np.ndarray, sample_rate: int, low_hz: float) -> float:
    """Loudness above low_hz. A quiet hat under a kick still shows up here."""
    clip = np.asarray(segment[: max(32, int(0.08 * sample_rate))], dtype=np.float32)
    if clip.size < 32:
        return 0.0
    spectrum = np.fft.rfft(clip)
    frequencies = np.fft.rfftfreq(clip.size, 1.0 / sample_rate)
    spectrum = spectrum.copy()
    spectrum[frequencies < low_hz] = 0
    wave = np.fft.irfft(spectrum, n=clip.size)
    return float(np.sqrt(np.mean(np.square(wave))))


def _metal_is_real(air: float, bright: float, air_min: float, bright_min: float) -> bool:
    return air >= air_min or bright >= bright_min


def clarify_kinds(kinds: list[str], segment: np.ndarray, sample_rate: int) -> list[str]:
    """Keep each class the onset evidence supports.

    ADTOF already thresholds every family on its own, so kick and hat can
    arrive together and both stay. A cymbal stays beside a snare when the
    high end is really there. The low drum is still one name: two toms are
    the same hit, and a kick-shaped thud is not also a floor tom.
    """
    if segment.size < 16 or not kinds:
        return []
    noise, air, fundamental = _attack_profile(segment, sample_rate)
    labels: list[str] = []
    named_hat = any(kind.startswith("hat_") for kind in kinds)
    named_cymbal = any(kind in {"ride", "crash"} for kind in kinds)
    # Ratio alone misses a hat buried under a loud kick. Absolute brightness
    # catches that hat, and stays above the hiss of a dark kick.
    if named_hat and _metal_is_real(air, _band_rms(segment, sample_rate, 6000.0), 0.02, 0.012):
        labels.append(split_family("hat", segment, sample_rate))
    # A dark kick can leak a little energy above 5 kHz. Real crashes in the
    # simultaneous-hit check sit higher than that leak on both measures.
    if named_cymbal and _metal_is_real(air, _band_rms(segment, sample_rate, 5000.0), 0.05, 0.07):
        labels.append(split_family("cymbal", segment, sample_rate))
    low = _choose_low(kinds, segment, sample_rate, noise, fundamental)
    if low:
        labels.append(low)
    # A snare on top of a kick is two drums. _choose_low keeps the snare;
    # the kick stays when the model named it and the body is still low.
    if (
        low == "snare"
        and "kick" in kinds
        and noise >= 0.18
        and (fundamental <= 0 or fundamental < 105)
    ):
        labels.insert(0, "kick")
    if not labels:
        fallback = low or _choose_low(["kick"], segment, sample_rate, noise, fundamental) or "kick"
        labels.append(fallback)
    unique: list[str] = []
    for label in labels:
        if label not in unique and label in DRUM_MIDI:
            unique.append(label)
    return unique


def _clarify_hits(hits: list[dict], mono: np.ndarray, sample_rate: int) -> list[dict]:
    if not hits:
        return []
    ordered = sorted(hits, key=lambda hit: (hit["time"], hit["kind"]))
    groups: list[list[dict]] = [[ordered[0]]]
    for hit in ordered[1:]:
        if hit["time"] - groups[-1][0]["time"] <= 0.045:
            groups[-1].append(hit)
        else:
            groups.append([hit])
    clarified: list[dict] = []
    for group in groups:
        when = min(hit["time"] for hit in group)
        start = int(when * sample_rate)
        segment = mono[start : start + int(0.5 * sample_rate)]
        velocity = max(int(hit["velocity"]) for hit in group)
        for kind in clarify_kinds([hit["kind"] for hit in group], segment, sample_rate):
            clarified.append({"time": round(float(when), 4), "kind": kind, "velocity": velocity})
    clarified.sort(key=lambda hit: (hit["time"], hit["kind"]))
    return clarified


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
    return _clarify_hits(hits, mono, sample_rate)


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
