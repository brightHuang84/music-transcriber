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


def _piano_tone(frequency: float, seconds: float) -> np.ndarray:
    """Fast attack. Higher partials die first, and the fundamental stays clear."""
    count = int(seconds * SAMPLE_RATE)
    if count <= 0:
        return np.zeros(0, dtype=np.float32)
    time = np.arange(count, dtype=np.float32) / SAMPLE_RATE
    wave = np.zeros(count, dtype=np.float32)
    for partial in range(1, 7):
        envelope = np.exp(-time / (0.9 / partial))
        wave += np.float32(0.55 / partial) * np.sin(2 * np.pi * frequency * partial * time) * envelope
    # The fundamental is louder and rings longer than the hammer partials.
    wave += np.sin(2 * np.pi * frequency * time) * np.exp(-time / 1.1)
    peak = np.max(np.abs(wave)) + 1e-8
    return (wave / peak).astype(np.float32)


def _bowed_tone(frequency: float, seconds: float) -> np.ndarray:
    """Slow attack, light vibrato, and a sustained body, like a violin."""
    count = int(seconds * SAMPLE_RATE)
    if count <= 0:
        return np.zeros(0, dtype=np.float32)
    time = np.arange(count, dtype=np.float32) / SAMPLE_RATE
    vibrato = np.sin(2 * np.pi * 5.1 * time)
    instantaneous = frequency * (1.0 + 0.007 * vibrato)
    phase = np.cumsum(instantaneous) / SAMPLE_RATE
    wave = np.zeros(count, dtype=np.float32)
    for partial, weight in enumerate((1.0, 0.62, 0.34, 0.16, 0.08), start=1):
        wave += np.float32(weight) * np.sin(2 * np.pi * partial * phase)
    attack = np.minimum(1.0, time / 0.09)
    release = np.ones(count, dtype=np.float32)
    release_samples = min(count, int(0.12 * SAMPLE_RATE))
    if release_samples:
        release[-release_samples:] = np.linspace(1.0, 0.0, release_samples, dtype=np.float32)
    wave *= attack * release
    peak = np.max(np.abs(wave)) + 1e-8
    return (wave / peak).astype(np.float32)


def _pluck_tone(frequency: float, seconds: float) -> np.ndarray:
    """A short plucked note, like a guitar."""
    body = _tone(frequency, seconds, harmonics=(1.0, 0.5, 0.28, 0.12, 0.05))
    click = _shaped_noise(min(0.012, seconds), 500, 4000, 0.004) * 0.35
    if click.size:
        body = body.copy()
        body[: click.size] += click
    peak = np.max(np.abs(body)) + 1e-8
    return (body / peak).astype(np.float32)


def synthesize_ensemble(path: Path | None = None) -> dict:
    """A short clip of piano, violin, and guitar on known pitches.

    The ranges do not overlap: piano stays at or below C4, guitar sits in the
    middle, violin stays at or above D5. That makes a later stem check able to
    say which instrument a transcribed note belonged to.
    """
    # A short rest keeps the first chord off the very first sample. Basic Pitch
    # often drops a note that starts at time zero.
    offset = 0.25
    duration = 8.5
    frames = int(duration * SAMPLE_RATE)
    piano = np.zeros(frames, dtype=np.float32)
    violin = np.zeros(frames, dtype=np.float32)
    guitar = np.zeros(frames, dtype=np.float32)

    # One note at a time. A block chord's inner voices are easy for Basic Pitch
    # to swallow, which would make the later stem check look worse than the
    # separation really is.
    piano_notes = [
        (offset + 0.00, offset + 0.70, 48),  # C3
        (offset + 0.85, offset + 1.55, 52),  # E3
        (offset + 1.70, offset + 2.40, 55),  # G3
        (offset + 2.55, offset + 3.25, 60),  # C4
        (offset + 3.40, offset + 4.10, 57),  # A3
        (offset + 4.25, offset + 4.95, 53),  # F3
        (offset + 5.10, offset + 5.80, 51),  # D#3
        (offset + 5.95, offset + 6.65, 54),  # F#3
        (offset + 6.80, offset + 7.50, 55),  # G3
        (offset + 7.65, offset + 8.15, 52),  # E3
    ]
    violin_notes = [
        (offset + 0.25, offset + 1.85, 76),  # E5
        (offset + 2.25, offset + 3.85, 79),  # G5
        (offset + 4.25, offset + 5.85, 83),  # B5
        (offset + 6.25, offset + 7.7, 74),  # D5
    ]
    guitar_notes = []
    guitar_pattern = (64, 67, 71, 67)  # E4 G4 B4 G4
    for bar in range(4):
        for step, midi in enumerate(guitar_pattern):
            start = offset + bar * 2.0 + step * 0.5
            guitar_notes.append((round(start, 4), round(start + 0.42, 4), midi))

    for start, end, midi in piano_notes:
        frequency = 440.0 * 2 ** ((midi - 69) / 12)
        _add(piano, start, _piano_tone(frequency, end - start) * 0.42)
    for start, end, midi in violin_notes:
        frequency = 440.0 * 2 ** ((midi - 69) / 12)
        _add(violin, start, _bowed_tone(frequency, end - start) * 0.38)
    for start, end, midi in guitar_notes:
        frequency = 440.0 * 2 ** ((midi - 69) / 12)
        _add(guitar, start, _pluck_tone(frequency, end - start) * 0.34)

    mix = (piano + violin + guitar).astype(np.float32)
    peak = float(np.max(np.abs(mix)) + 1e-8)
    scale = np.float32(0.9 / peak)
    piano = piano * scale
    violin = violin * scale
    guitar = guitar * scale
    mix = (piano + violin + guitar).astype(np.float32)

    def stereo(mono: np.ndarray) -> np.ndarray:
        return np.stack([mono, mono], axis=0)

    if path is not None:
        save_wav(path, stereo(mix), SAMPLE_RATE)
    return {
        "sample_rate": SAMPLE_RATE,
        "bpm": BPM,
        "duration": duration,
        "piano": piano_notes,
        "violin": violin_notes,
        "guitar": guitar_notes,
        "audio": stereo(mix),
        "piano_audio": stereo(piano),
        "violin_audio": stereo(violin),
        "guitar_audio": stereo(guitar),
    }


def synthesize_lead_sheet(path: Path | None = None) -> dict:
    """A known melody above block chords. The melody is always the top note.

    Nothing starts at time zero, because Basic Pitch often drops that note.
    Chord tones stay at or below G4. The tune stays at or above C5.
    """
    offset = 0.30
    duration = 8.6
    frames = int(duration * SAMPLE_RATE)
    chords = np.zeros(frames, dtype=np.float32)
    melody = np.zeros(frames, dtype=np.float32)
    # One bar each: C, F, G, C. Inner voices stay under the tune.
    chord_plan = [
        (offset + 0.0, offset + 1.95, (60, 64, 67)),
        (offset + 2.0, offset + 3.95, (53, 57, 60)),
        (offset + 4.0, offset + 5.95, (55, 59, 62)),
        (offset + 6.0, offset + 7.95, (60, 64, 67)),
    ]
    melody_notes = [
        (offset + 0.00, offset + 0.90, 76),  # E5
        (offset + 1.00, offset + 1.90, 79),  # G5
        (offset + 2.00, offset + 2.90, 81),  # A5
        (offset + 3.00, offset + 3.90, 77),  # F5
        (offset + 4.00, offset + 4.90, 83),  # B5
        (offset + 5.00, offset + 5.90, 79),  # G5
        (offset + 6.00, offset + 6.90, 76),  # E5
        (offset + 7.00, offset + 7.90, 72),  # C5
    ]
    for start, end, pitches in chord_plan:
        for midi in pitches:
            frequency = 440.0 * 2 ** ((midi - 69) / 12)
            _add(chords, start, _piano_tone(frequency, end - start) * 0.18)
    for start, end, midi in melody_notes:
        frequency = 440.0 * 2 ** ((midi - 69) / 12)
        _add(melody, start, _tone(frequency, end - start, harmonics=(1.0, 0.28, 0.08)) * 0.62)
    mix = (chords + melody).astype(np.float32)
    peak = float(np.max(np.abs(mix)) + 1e-8)
    mix = np.clip(mix / peak * 0.9, -1, 1)

    def stereo(mono: np.ndarray) -> np.ndarray:
        return np.stack([mono, mono], axis=0)

    if path is not None:
        save_wav(path, stereo(mix), SAMPLE_RATE)
    return {
        "sample_rate": SAMPLE_RATE,
        "bpm": BPM,
        "duration": duration,
        "melody": melody_notes,
        "chords": chord_plan,
        "audio": stereo(mix),
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
