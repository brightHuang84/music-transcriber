"""A short, known piece used by the demo button and the automated tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.audio_io import save_wav

SAMPLE_RATE = 44100
BPM = 120.0
# Two bars of 4/4. Beat length is 0.5 seconds.
DURATION = 4.0


def _silence(seconds: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    return np.zeros(int(seconds * sample_rate), dtype=np.float32)


def _add(buffer: np.ndarray, start_seconds: float, clip: np.ndarray) -> None:
    start = int(start_seconds * SAMPLE_RATE)
    end = min(buffer.shape[-1], start + clip.shape[-1])
    if start >= buffer.shape[-1] or end <= start:
        return
    buffer[start:end] += clip[: end - start]


def _tone(frequency: float, seconds: float, harmonics: tuple[float, ...] = (1.0, 0.45, 0.2, 0.08)) -> np.ndarray:
    count = int(seconds * SAMPLE_RATE)
    if count <= 0:
        return np.zeros(0, dtype=np.float32)
    time = np.arange(count, dtype=np.float32) / SAMPLE_RATE
    # Fast attack, exponential decay: closer to a plucked note than a bare sine.
    envelope = np.exp(-time / max(seconds * 0.55, 0.05))
    envelope[: min(80, count)] *= np.linspace(0.0, 1.0, min(80, count))
    wave = np.zeros(count, dtype=np.float32)
    for index, weight in enumerate(harmonics, start=1):
        wave += np.float32(weight) * np.sin(2 * np.pi * frequency * index * time)
    peak = np.max(np.abs(wave)) + 1e-8
    return (wave / peak * envelope).astype(np.float32)


def _shaped_noise(seconds: float, low_hz: float | None, high_hz: float | None, decay: float) -> np.ndarray:
    count = max(8, int(seconds * SAMPLE_RATE))
    noise = np.random.default_rng(0).standard_normal(count).astype(np.float32)
    spectrum = np.fft.rfft(noise)
    frequencies = np.fft.rfftfreq(count, 1.0 / SAMPLE_RATE)
    mask = np.ones(frequencies.shape, dtype=np.float32)
    if low_hz is not None:
        mask[frequencies < low_hz] = 0
    if high_hz is not None:
        mask[frequencies > high_hz] = 0
    shaped = np.fft.irfft(spectrum * mask, n=count).astype(np.float32)
    time = np.arange(count, dtype=np.float32) / SAMPLE_RATE
    shaped *= np.exp(-time / decay)
    peak = np.max(np.abs(shaped)) + 1e-8
    return shaped / peak


def synthesize_example(path: Path | None = None) -> dict:
    """Write a 4 second clip and return what each instrument is supposed to play.

    Drums are on their own beats so a simple classifier can be checked:
    kick on beats 1 and 3, snare on beats 2 and 4, hi-hat on the offbeats.
    """
    frames = int(DURATION * SAMPLE_RATE)
    drums = np.zeros(frames, dtype=np.float32)
    bass = np.zeros(frames, dtype=np.float32)
    melody = np.zeros(frames, dtype=np.float32)

    # A short click gives the kick and snare a real attack, which is what
    # beat trackers lock onto. The body underneath keeps the kick low.
    kick_click = _shaped_noise(0.012, 400, 3500, 0.003) * 0.9
    kick = _tone(55, 0.22, harmonics=(1.0, 0.4, 0.08)) * 0.95
    kick[: kick_click.size] += kick_click
    snare_click = _shaped_noise(0.01, 800, 5000, 0.003) * 0.55
    snare = _shaped_noise(0.16, 180, 3200, 0.045) * 0.75
    snare[: snare_click.size] += snare_click
    hat = _shaped_noise(0.04, 7000, 15000, 0.01) * 0.28

    drum_hits: list[dict] = []
    for bar in range(2):
        bar_start = bar * 2.0
        for beat, kind, clip in (
            (0, "kick", kick),
            (1, "snare", snare),
            (2, "kick", kick),
            (3, "snare", snare),
        ):
            when = bar_start + beat * 0.5
            _add(drums, when, clip)
            drum_hits.append({"time": when, "kind": kind})
        for off in (0.25, 0.75, 1.25, 1.75):
            when = bar_start + off
            _add(drums, when, hat)
            drum_hits.append({"time": when, "kind": "hat_closed"})

    bass_notes = [
        (0.0, 0.95, 36),  # C2
        (1.0, 1.95, 43),  # G2
        (2.0, 2.95, 36),
        (3.0, 3.9, 43),
    ]
    for start, end, midi in bass_notes:
        frequency = 440.0 * 2 ** ((midi - 69) / 12)
        _add(bass, start, _tone(frequency, end - start, harmonics=(1.0, 0.55, 0.22)) * 0.7)

    melody_notes = [
        (0.0, 0.48, 64),  # E4
        (0.5, 0.98, 67),  # G4
        (1.0, 1.48, 72),  # C5
        (1.5, 1.98, 71),  # B4
        (2.0, 2.48, 69),  # A4
        (2.5, 2.98, 67),  # G4
        (3.0, 3.48, 64),  # E4
        (3.5, 3.95, 60),  # C4
    ]
    for start, end, midi in melody_notes:
        frequency = 440.0 * 2 ** ((midi - 69) / 12)
        _add(melody, start, _tone(frequency, end - start) * 0.55)

    mix = (0.9 * melody + 0.85 * bass + drums).astype(np.float32)
    peak = float(np.max(np.abs(mix)) + 1e-8)
    mix = np.clip(mix / peak * 0.9, -1, 1)
    stereo = np.stack([mix, mix], axis=0)
    if path is not None:
        save_wav(path, stereo, SAMPLE_RATE)
    return {
        "sample_rate": SAMPLE_RATE,
        "bpm": BPM,
        "duration": DURATION,
        "melody": melody_notes,
        "bass": bass_notes,
        "drums": drum_hits,
        "audio": stereo,
        "drum_audio": np.stack([drums, drums], axis=0),
        "melody_audio": np.stack([melody, melody], axis=0),
        "bass_audio": np.stack([bass, bass], axis=0),
        "chord_audio": _chord_audio(),
    }


def _chord_audio() -> np.ndarray:
    """Two seconds of a C major triad, used by the chord test."""
    frames = int(2.0 * SAMPLE_RATE)
    wave = np.zeros(frames, dtype=np.float32)
    for midi in (60, 64, 67):
        frequency = 440.0 * 2 ** ((midi - 69) / 12)
        wave += _tone(frequency, 2.0, harmonics=(1.0, 0.25, 0.08))
    wave *= 0.3
    return np.stack([wave, wave], axis=0)
