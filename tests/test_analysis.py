"""Checks that do not download the Demucs model."""

import time
import wave
from pathlib import Path

import numpy as np
import pretty_midi
import pytest
from fastapi.testclient import TestClient

from app.errors import UserFacingError

from app.audio_io import convert_to_wav, load_audio, save_wav
from app.chords import detect_chords
from app.drums import DRUM_MIDI, _clarify_hits, clarify_kinds, detect_drums, detect_drums_by_spectrum, split_family
from app.midi_export import write_combined_midi, write_stem_midi
from app.notes import midi_to_name, midi_to_solfege
from app.rhythm import analyze_rhythm
from app.server import app
from app.synth import synthesize_example
from app.transcription import _drop_inaudible, transcribe


def test_note_names():
    assert midi_to_name(60) == "C4"
    assert midi_to_name(64) == "E4"
    assert midi_to_name(36) == "C2"
    assert midi_to_solfege(64) == "mi"
    assert midi_to_solfege(61) == "#do"


def test_drum_hits_match_the_pattern():
    example = synthesize_example()
    hits = detect_drums_by_spectrum(example["drum_audio"], example["sample_rate"])
    assert hits, "expected at least one drum hit"
    for hit in hits:
        assert hit["kind"] in {"kick", "snare", "hat_closed"}
        assert 1 <= hit["velocity"] <= 127
        assert hit["time"] >= 0
    for expected in example["drums"]:
        matched = [
            hit
            for hit in hits
            if hit["kind"] == expected["kind"] and abs(hit["time"] - expected["time"]) <= 0.08
        ]
        assert matched, f"missing {expected} in {hits}"


def _tone_burst(frequency: float, seconds: float, sample_rate: int = 44100) -> np.ndarray:
    count = int(seconds * sample_rate)
    time = np.arange(count, dtype=np.float32) / sample_rate
    wave = np.sin(2 * np.pi * frequency * time).astype(np.float32)
    wave *= np.exp(-time / 0.12)
    return wave


def test_open_hat_rings_longer_than_a_closed_hat():
    sample_rate = 44100
    closed = np.zeros(int(0.4 * sample_rate), dtype=np.float32)
    noise = np.random.default_rng(1).standard_normal(int(0.03 * sample_rate)).astype(np.float32)
    closed[: noise.size] = noise
    opened = np.random.default_rng(2).standard_normal(int(0.4 * sample_rate)).astype(np.float32)
    opened *= np.exp(-np.arange(opened.size, dtype=np.float32) / sample_rate / 0.18)
    assert split_family("hat", closed, sample_rate) == "hat_closed"
    assert split_family("hat", opened, sample_rate) == "hat_open"


def test_cymbal_and_tom_splits():
    sample_rate = 44100
    ride = _tone_burst(480, 0.16, sample_rate)
    crash = np.random.default_rng(3).standard_normal(int(0.5 * sample_rate)).astype(np.float32)
    crash *= np.exp(-np.arange(crash.size, dtype=np.float32) / sample_rate / 0.28)
    assert split_family("cymbal", ride, sample_rate) == "ride"
    assert split_family("cymbal", crash, sample_rate) == "crash"
    assert split_family("tom", _tone_burst(180, 0.3, sample_rate), sample_rate) == "tom_high"
    assert split_family("tom", _tone_burst(120, 0.3, sample_rate), sample_rate) == "tom_mid"
    assert split_family("tom", _tone_burst(80, 0.35, sample_rate), sample_rate) == "tom_floor"


def test_low_thud_is_not_called_a_high_tom():
    sample_rate = 44100
    thud = _tone_burst(55, 0.3, sample_rate)
    click = np.random.default_rng(4).standard_normal(int(0.008 * sample_rate)).astype(np.float32)
    thud = thud.copy()
    thud[: click.size] += click * 0.8
    assert split_family("tom", thud, sample_rate) in {"tom_floor", "tom_mid"}
    found = clarify_kinds(["tom_high"], thud, sample_rate)
    assert found in (["kick"], ["tom_floor"])
    assert "tom_high" not in found


def test_closed_hats_stay_closed_when_the_next_hit_is_in_the_window():
    sample_rate = 44100
    hat = np.random.default_rng(5).standard_normal(int(0.12 * sample_rate)).astype(np.float32)
    hat *= np.exp(-np.arange(hat.size, dtype=np.float32) / sample_rate / 0.03)
    window = np.zeros(int(0.5 * sample_rate), dtype=np.float32)
    for start in (0.0, 0.125, 0.25, 0.375):
        index = int(start * sample_rate)
        window[index : index + hat.size] += hat[: window.size - index]
    assert split_family("hat", hat, sample_rate) == "hat_closed"
    assert split_family("hat", window, sample_rate) == "hat_closed"
    # A snare landing while the hat window is still open must not flip the hat.
    snare = np.random.default_rng(6).standard_normal(int(0.18 * sample_rate)).astype(np.float32)
    snare *= np.exp(-np.arange(snare.size, dtype=np.float32) / sample_rate / 0.05)
    mixed = np.zeros(int(0.5 * sample_rate), dtype=np.float32)
    mixed[: hat.size] += hat
    at = int(0.12 * sample_rate)
    mixed[at : at + snare.size] += snare
    assert split_family("hat", mixed, sample_rate) == "hat_closed"


def test_same_onset_is_not_kick_and_tom_and_a_dark_hit_is_not_a_cymbal():
    sample_rate = 44100
    kick = _tone_burst(60, 0.25, sample_rate)
    assert clarify_kinds(["kick", "snare", "tom_floor", "tom_high"], kick, sample_rate) == ["kick"]
    assert "ride" not in clarify_kinds(["kick", "ride", "snare"], kick, sample_rate)
    assert "crash" not in clarify_kinds(["crash"], kick, sample_rate)
    noisy = np.random.default_rng(7).standard_normal(int(0.12 * sample_rate)).astype(np.float32)
    noisy *= np.exp(-np.arange(noisy.size, dtype=np.float32) / sample_rate / 0.04)
    # Band-limit toward the snare's noise, leaving the lows quiet.
    spectrum = np.fft.rfft(noisy)
    freqs = np.fft.rfftfreq(noisy.size, 1.0 / sample_rate)
    spectrum[freqs < 800] = 0
    snare = np.fft.irfft(spectrum, n=noisy.size).astype(np.float32)
    assert clarify_kinds(["snare", "tom_floor"], snare, sample_rate) == ["snare"]


def _noise_burst(seconds, low_hz, high_hz, decay, seed, sample_rate=44100):
    count = max(8, int(seconds * sample_rate))
    noise = np.random.default_rng(seed).standard_normal(count).astype(np.float32)
    spectrum = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(count, 1.0 / sample_rate)
    spectrum[(freqs < low_hz) | (freqs > high_hz)] = 0
    wave = np.fft.irfft(spectrum, n=count).astype(np.float32)
    time = np.arange(count, dtype=np.float32) / sample_rate
    wave *= np.exp(-time / decay)
    peak = float(np.max(np.abs(wave))) + 1e-8
    return (wave / peak).astype(np.float32)


def _place(buffer, when, clip, sample_rate):
    start = int(when * sample_rate)
    end = min(buffer.size, start + clip.size)
    if end > start:
        buffer[start:end] += clip[: end - start]


def _score_hits(expected, predicted, tolerance=0.06):
    expected_by_kind = {}
    predicted_by_kind = {}
    for hit in expected:
        expected_by_kind.setdefault(hit["kind"], []).append(float(hit["time"]))
    for hit in predicted:
        predicted_by_kind.setdefault(hit["kind"], []).append(float(hit["time"]))
    stats = {}
    for kind in sorted(set(expected_by_kind) | set(predicted_by_kind)):
        wanted = sorted(expected_by_kind.get(kind, []))
        found = sorted(predicted_by_kind.get(kind, []))
        used = [False] * len(found)
        matched = 0
        for when in wanted:
            for index, got in enumerate(found):
                if used[index] or abs(got - when) > tolerance:
                    continue
                used[index] = True
                matched += 1
                break
        false_pos = len(found) - matched
        false_neg = len(wanted) - matched
        stats[kind] = {
            "precision": matched / (matched + false_pos) if matched + false_pos else 1.0,
            "recall": matched / (matched + false_neg) if matched + false_neg else 1.0,
            "tp": matched,
            "fp": false_pos,
            "fn": false_neg,
        }
    return stats


def _simultaneous_kit(sample_rate=44100):
    """Kick+hat, snare+hat, kick+crash, snare+crash, each twice."""
    duration = 8.0
    audio = np.zeros(int(duration * sample_rate), dtype=np.float32)
    kick = _tone_burst(55, 0.2, sample_rate) * 0.9
    snare = _noise_burst(0.18, 180, 4000, 0.045, seed=11, sample_rate=sample_rate) * 0.85
    hat = _noise_burst(0.045, 7000, 16000, 0.012, seed=12, sample_rate=sample_rate) * 0.55
    crash = _noise_burst(0.7, 3000, 16000, 0.22, seed=13, sample_rate=sample_rate) * 0.7
    pattern = [
        (1.0, ("kick", "hat_closed"), (kick, hat)),
        (2.0, ("snare", "hat_closed"), (snare, hat)),
        (3.0, ("kick", "crash"), (kick, crash)),
        (4.0, ("snare", "crash"), (snare, crash)),
        (5.0, ("kick", "hat_closed"), (kick, hat)),
        (6.0, ("snare", "hat_closed"), (snare, hat)),
    ]
    expected = []
    proposed = []
    for when, kinds, clips in pattern:
        for kind, clip in zip(kinds, clips):
            _place(audio, when, clip, sample_rate)
            expected.append({"time": when, "kind": kind})
            proposed.append({"time": when, "kind": kind, "velocity": 90})
    peak = float(np.max(np.abs(audio))) + 1e-8
    return audio / peak * 0.8, expected, proposed, sample_rate


def test_simultaneous_drums_keep_each_class_and_drop_duplicate_toms():
    sample_rate = 44100
    audio, expected, proposed, _sr = _simultaneous_kit(sample_rate)
    clarified = _clarify_hits(proposed, audio, sample_rate)
    stats = _score_hits(expected, clarified)
    for kind in ("kick", "snare", "hat_closed", "crash"):
        assert stats[kind]["precision"] == 1.0, stats
        assert stats[kind]["recall"] == 1.0, stats
    # The same low hit labeled as every tom is still one tom.
    tom = _tone_burst(120, 0.3, sample_rate)
    assert clarify_kinds(["tom_high", "tom_mid", "tom_floor"], tom, sample_rate) == ["tom_mid"]
    kick = _tone_burst(55, 0.25, sample_rate)
    assert clarify_kinds(["kick", "tom_floor", "tom_high"], kick, sample_rate) == ["kick"]
    # Full detector on the same pattern. ADTOF was trained on real kits, so
    # synthesized snares and crashes are only partly heard. A pair it does
    # hear must survive as two notes, and kicks must not grow a floor tom.
    detected = detect_drums(audio.astype(np.float32), sample_rate)
    model_stats = _score_hits(expected, detected, tolerance=0.1)
    assert model_stats["kick"]["precision"] == 1.0, model_stats
    assert model_stats["kick"]["recall"] == 1.0, model_stats
    by_time = {}
    for hit in detected:
        by_time.setdefault(round(hit["time"], 1), set()).add(hit["kind"])
    assert any("kick" in kinds and any(kind.startswith("hat_") for kind in kinds) for kinds in by_time.values()), model_stats
    assert any("snare" in kinds and "crash" in kinds for kinds in by_time.values()), model_stats
    assert all(not kind.startswith("tom_") for kinds in by_time.values() for kind in kinds), model_stats


def test_drum_midi_notes_follow_general_midi():
    assert DRUM_MIDI["kick"] == 36
    assert DRUM_MIDI["snare"] == 38
    assert DRUM_MIDI["hat_closed"] == 42
    assert DRUM_MIDI["hat_open"] == 46
    assert DRUM_MIDI["ride"] == 51
    assert DRUM_MIDI["crash"] == 49
    assert DRUM_MIDI["tom_high"] == 50
    assert DRUM_MIDI["tom_mid"] == 47
    assert DRUM_MIDI["tom_floor"] == 43


def test_tempo_is_near_120_and_downbeats_land_on_kicks():
    example = synthesize_example()
    rhythm = analyze_rhythm(example["audio"], example["drum_audio"], example["sample_rate"])
    assert 100 <= rhythm["bpm"] <= 140
    assert len(rhythm["beats"]) >= 6
    assert rhythm["downbeats"], rhythm
    assert rhythm["downbeats"][0] <= 0.3
    if len(rhythm["downbeats"]) > 1:
        gap = rhythm["downbeats"][1] - rhythm["downbeats"][0]
        assert abs(gap - 2.0) < 0.2


def test_c_major_chord_and_key():
    example = synthesize_example()
    found = detect_chords(example["chord_audio"], example["sample_rate"], [0.0, 1.0], 2.0)
    symbols = [chord["symbol"] for chord in found["chords"]]
    assert "C" in symbols, found
    assert found["key"] is not None
    assert found["key"]["tonic"] == "C"
    assert found["key"]["mode"] == "major"


def test_notes_on_silence_are_dropped(tmp_path: Path):
    sample_rate = 44100
    audio = np.zeros((2, sample_rate * 2), dtype=np.float32)
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    audio[:, sample_rate:] = 0.2 * np.sin(2 * np.pi * 440 * time)
    path = tmp_path / "gated.wav"
    save_wav(path, audio, sample_rate)
    notes = [
        {"pitch": 69, "start": 0.1, "end": 0.4, "velocity": 80, "name": "A4", "solfege": "la", "duration": 0.3},
        {"pitch": 69, "start": 1.1, "end": 1.5, "velocity": 80, "name": "A4", "solfege": "la", "duration": 0.4},
    ]
    kept = _drop_inaudible(notes, path)
    assert [note["start"] for note in kept] == [1.1]


def test_melody_transcription_finds_the_written_pitches(tmp_path: Path):
    example = synthesize_example()
    path = tmp_path / "melody.wav"
    save_wav(path, example["melody_audio"], example["sample_rate"])
    notes = transcribe(path, "vocals", midi_tempo=120)
    for start, _end, pitch in example["melody"]:
        matched = [
            note
            for note in notes
            if note["pitch"] == pitch and abs(note["start"] - start) <= 0.12
        ]
        assert matched, f"missing pitch {pitch} at {start}: {notes}"
        assert matched[0]["name"]
        assert matched[0]["velocity"] >= 1
        assert matched[0]["duration"] > 0


def test_download_bar_reports_bytes():
    from app.separation import _download_bar

    seen: list[tuple[float, str]] = []
    bar_type = _download_bar(lambda fraction, message: seen.append((fraction, message)), "精细", 55)
    bar = bar_type(total=20_000_000, desc="weights", unit="B", unit_scale=True)
    bar.update(5_000_000)
    bar.close()
    assert seen[-1][0] == pytest.approx(0.25)
    assert "5/20 MB" in seen[-1][1]
    assert "精细" in seen[-1][1]


def test_instrument_programs_match_general_midi():
    from app.stems import MODES, STEMS, mode_spec

    assert STEMS["piano"]["program"] == 0
    assert STEMS["guitar"]["program"] == 25
    assert STEMS["vocals"]["program"] == 53
    assert STEMS["bass"]["program"] == 33
    assert STEMS["other"]["program"] == 48
    assert "violin" not in STEMS
    assert MODES["fine"]["model"] == "htdemucs_6s"
    assert "piano" in MODES["fine"]["stems"]
    assert "guitar" in MODES["fine"]["stems"]
    assert "piano" not in MODES["fast"]["stems"]
    assert mode_spec("fine")["id"] == "fine"
    with pytest.raises(UserFacingError):
        mode_spec("studio")


def test_isolated_piano_and_violin_notes_transcribe(tmp_path: Path):
    from app.synth import synthesize_ensemble

    clip = synthesize_ensemble()
    piano_path = tmp_path / "piano.wav"
    violin_path = tmp_path / "violin.wav"
    save_wav(piano_path, clip["piano_audio"], clip["sample_rate"])
    save_wav(violin_path, clip["violin_audio"], clip["sample_rate"])
    piano_notes = transcribe(piano_path, "piano", midi_tempo=120)
    violin_notes = transcribe(violin_path, "other", midi_tempo=120)
    for start, _end, pitch in clip["piano"]:
        assert any(note["pitch"] == pitch and abs(note["start"] - start) <= 0.2 for note in piano_notes), (
            pitch,
            piano_notes,
        )
    for start, _end, pitch in clip["violin"]:
        assert any(note["pitch"] == pitch and abs(note["start"] - start) <= 0.25 for note in violin_notes), (
            pitch,
            violin_notes,
        )


def test_midi_files_round_trip(tmp_path: Path):
    notes = [
        {"pitch": 64, "start": 0.0, "end": 0.4, "velocity": 80, "name": "E4", "solfege": "mi", "duration": 0.4}
    ]
    bass = [
        {"pitch": 36, "start": 0.0, "end": 0.4, "velocity": 70, "name": "C2", "solfege": "do", "duration": 0.4}
    ]
    hits = [
        {"time": 0.5, "kind": "snare", "velocity": 100},
        {"time": 1.0, "kind": "hat_open", "velocity": 90},
        {"time": 1.5, "kind": "crash", "velocity": 110},
    ]
    write_stem_midi(tmp_path / "vocals.mid", "vocals", 120, notes=notes)
    write_stem_midi(tmp_path / "piano.mid", "piano", 120, notes=notes)
    write_stem_midi(tmp_path / "guitar.mid", "guitar", 120, notes=notes)
    write_stem_midi(tmp_path / "drums.mid", "drums", 120, hits=hits)
    write_combined_midi(
        tmp_path / "all.mid",
        120,
        {"vocals": notes, "piano": notes, "guitar": notes, "bass": bass, "other": []},
        hits,
    )
    vocals = pretty_midi.PrettyMIDI(str(tmp_path / "vocals.mid"))
    assert vocals.instruments[0].notes[0].pitch == 64
    assert vocals.instruments[0].program == 53
    piano = pretty_midi.PrettyMIDI(str(tmp_path / "piano.mid"))
    assert piano.instruments[0].program == 0
    guitar = pretty_midi.PrettyMIDI(str(tmp_path / "guitar.mid"))
    assert guitar.instruments[0].program == 25
    drums = pretty_midi.PrettyMIDI(str(tmp_path / "drums.mid"))
    assert drums.instruments[0].is_drum
    assert [note.pitch for note in drums.instruments[0].notes] == [38, 46, 49]
    combined = pretty_midi.PrettyMIDI(str(tmp_path / "all.mid"))
    names = {instrument.name for instrument in combined.instruments}
    assert {"Vocals", "Piano", "Guitar", "Bass", "Drums"} <= names


def test_ffmpeg_reads_mp3(tmp_path: Path):
    wav_path = tmp_path / "tone.wav"
    sample_rate = 44100
    frames = sample_rate
    tone = (0.2 * np.sin(2 * np.pi * 440 * np.arange(frames) / sample_rate)).astype(np.float32)
    with wave.open(str(wav_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes((tone * 32767).astype(np.int16).tobytes())
    mp3_path = tmp_path / "tone.mp3"
    import subprocess

    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(wav_path), str(mp3_path)],
        check=True,
    )
    converted = tmp_path / "back.wav"
    convert_to_wav(mp3_path, converted)
    audio, rate = load_audio(converted)
    assert rate == 44100
    assert audio.shape[0] == 2
    assert audio.shape[1] > sample_rate // 2


def test_api_rejects_unknown_extension(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MUSIC_ANALYZER_DATA", str(tmp_path))
    client = TestClient(app)
    response = client.post("/api/analyze", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 400
    assert "mp3" in response.json()["detail"]


def test_api_short_audio_has_a_friendly_error(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MUSIC_ANALYZER_DATA", str(tmp_path))
    wav_path = tmp_path / "short.wav"
    sample_rate = 44100
    frames = int(0.4 * sample_rate)
    tone = (0.2 * np.sin(2 * np.pi * 440 * np.arange(frames) / sample_rate)).astype(np.float32)
    with wave.open(str(wav_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes((tone * 32767).astype(np.int16).tobytes())
    client = TestClient(app)
    with wav_path.open("rb") as handle:
        response = client.post("/api/analyze", files={"file": ("short.wav", handle, "audio/wav")})
    assert response.status_code == 200
    job_id = response.json()["id"]
    status = {}
    for _ in range(80):
        status = client.get(f"/api/jobs/{job_id}").json()
        if status["status"] in {"done", "error"}:
            break
        time.sleep(0.1)
    assert status["status"] == "error"
    assert "太短" in status["message"]


def test_home_page_is_chinese():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "听音识谱" in response.text
