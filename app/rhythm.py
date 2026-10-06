"""Tempo, beats, and a 4/4 downbeat guess."""

from __future__ import annotations

import numpy as np

from app.audio_io import rms, to_mono


def _as_float(value) -> float:
    array = np.atleast_1d(np.asarray(value, dtype=float))
    if array.size == 0 or not np.isfinite(array[0]):
        return 0.0
    return float(array[0])


def _track_beats(audio: np.ndarray, sample_rate: int) -> tuple[float, list[float]]:
    import librosa

    mono = to_mono(audio)
    tempo, beat_frames = librosa.beat.beat_track(y=mono, sr=sample_rate, trim=False, units="frames")
    beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate).astype(float)
    times = [round(float(t), 4) for t in beat_times if np.isfinite(t)]
    if len(times) >= 2:
        interval = float(np.median(np.diff(times)))
        bpm = 60.0 / interval if interval > 1e-3 else _as_float(tempo)
    else:
        bpm = _as_float(tempo)
    return bpm, times


def _prefer_tappable_tempo(bpm: float, beats: list[float]) -> tuple[float, list[float]]:
    """Keep the beat grid near a tempo a beginner would tap, roughly 70–160 BPM."""
    if bpm <= 0 or not beats:
        return bpm, beats
    beats = list(beats)
    guard = 0
    while bpm > 160 and len(beats) >= 2 and guard < 4:
        bpm /= 2
        beats = beats[::2]
        guard += 1
    while bpm < 70 and len(beats) >= 2 and guard < 8:
        bpm *= 2
        middles = [round((left + right) / 2, 4) for left, right in zip(beats, beats[1:])]
        beats = sorted(beats + middles)
        guard += 1
    return round(float(bpm), 1), beats


def _low_energy(audio: np.ndarray, sample_rate: int, time_seconds: float) -> float:
    mono = to_mono(audio)
    # Look slightly before the reported beat. Trackers often peak a little late.
    start = max(0, int((time_seconds - 0.12) * sample_rate))
    window = mono[start : start + int(0.16 * sample_rate)]
    if window.size < 8:
        return 0.0
    spectrum = np.abs(np.fft.rfft(window * np.hanning(window.size)))
    frequencies = np.fft.rfftfreq(window.size, 1.0 / sample_rate)
    chosen = frequencies < 150
    if not np.any(chosen):
        return 0.0
    return float(np.sum(spectrum[chosen] ** 2))


def _restore_opening_beat(audio: np.ndarray, sample_rate: int, bpm: float, beats: list[float]) -> list[float]:
    """If the tracker skipped a downbeat sitting on the first sample, put it back.

    A real pickup (silence, then the first beat) has no low-frequency accent at 0,
    so it is left alone.
    """
    if not beats or bpm <= 0:
        return beats
    interval = 60.0 / bpm
    if not (0.35 * interval < beats[0] < 1.35 * interval):
        return beats
    opening = _low_energy(audio, sample_rate, 0.02)
    first = _low_energy(audio, sample_rate, beats[0])
    if opening > max(first, 1e-9) * 0.5:
        return [0.0] + beats
    return beats


def estimate_downbeats(audio: np.ndarray, sample_rate: int, beats: list[float]) -> list[float]:
    """Pick which beat of each 4/4 bar is beat 1, using low-frequency accents (often the kick)."""
    if len(beats) < 4:
        return list(beats[:1])
    energies = [_low_energy(audio, sample_rate, beat) for beat in beats]
    best_phase = 0
    best_score = -1.0
    for phase in range(4):
        chosen = energies[phase::4]
        if not chosen:
            continue
        score = float(np.mean(chosen))
        if score > best_score:
            best_score = score
            best_phase = phase
    return [beats[index] for index in range(best_phase, len(beats), 4)]


def analyze_rhythm(mix: np.ndarray, drums: np.ndarray, sample_rate: int) -> dict:
    source = drums if rms(drums) > max(rms(mix) * 0.02, 1e-4) else mix
    bpm, beats = _track_beats(source, sample_rate)
    if len(beats) < 2:
        bpm, beats = _track_beats(mix, sample_rate)
    bpm, beats = _prefer_tappable_tempo(bpm, beats)
    if bpm <= 0:
        bpm = 120.0
    anchor = drums if rms(drums) > 1e-4 else mix
    beats = _restore_opening_beat(anchor, sample_rate, bpm, beats)
    downbeats = estimate_downbeats(anchor, sample_rate, beats)
    return {
        "bpm": bpm,
        "beats": beats,
        "downbeats": downbeats,
        "time_signature": "4/4",
    }
