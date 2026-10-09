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
    assert STEMS["strings"]["program"] == 48
    assert STEMS["strings"]["label"] == "弦乐"
    assert "violin" not in STEMS
    assert MODES["fine"]["model"] == "htdemucs_6s"
    assert "piano" in MODES["fine"]["stems"]
    assert "guitar" in MODES["fine"]["stems"]
    assert "strings" not in MODES["fine"]["stems"]
    assert "strings" in MODES["best"]["stems"]
    assert "piano" not in MODES["fast"]["stems"]
    assert mode_spec("fine")["id"] == "fine"
    assert mode_spec(None)["id"] == "best"
    assert MODES["best"]["drum_kit"] is True
    with pytest.raises(UserFacingError):
        mode_spec("studio")


def test_quality_assemble_keeps_roformer_instruments_and_supplied_vocals():
    from app.roformer import assemble_quality_stems

    mix = np.ones((2, 8), dtype=np.float32)
    roformer = {name: np.full((2, 8), 0.1, dtype=np.float32) for name in ("drums", "bass", "piano", "guitar", "vocals", "other")}
    vocals = np.full((2, 8), 0.2, dtype=np.float32)
    stems = assemble_quality_stems(mix, roformer, vocals)
    assert stems["vocals"][0, 0] == pytest.approx(0.2)
    assert stems["piano"][0, 0] == pytest.approx(0.1)
    assert stems["guitar"][0, 0] == pytest.approx(0.1)
    # Mix minus vocals and the four instrument stems: 1 - 0.2 - 0.4.
    assert stems["other"][0, 0] == pytest.approx(0.4)
    assert "strings" not in stems
    bowed = np.full((2, 8), 0.15, dtype=np.float32)
    with_strings = assemble_quality_stems(mix, roformer, vocals, bowed)
    assert with_strings["strings"][0, 0] == pytest.approx(0.15)
    assert with_strings["other"][0, 0] == pytest.approx(0.25)


def test_kit_pieces_keep_simultaneous_kick_and_snare():
    from app.drums import hits_from_kit_pieces

    sample_rate = 44100
    silence = np.zeros(sample_rate * 2, dtype=np.float32)

    def burst(at: float, seconds: float, seed: int) -> np.ndarray:
        wave = silence.copy()
        start = int(at * sample_rate)
        count = int(seconds * sample_rate)
        rng = np.random.default_rng(seed)
        wave[start : start + count] = rng.standard_normal(count).astype(np.float32) * 0.5
        return wave

    pieces = {
        "kick": burst(0.50, 0.09, 1),
        "snare": burst(0.50, 0.09, 2),
        "toms": silence,
        "hh": burst(1.20, 0.04, 3),
        "ride": silence,
        "crash": silence,
    }
    hits = hits_from_kit_pieces(pieces, sample_rate)
    groups: dict[float, set[str]] = {}
    for hit in hits:
        groups.setdefault(round(hit["time"], 1), set()).add(hit["kind"])
    assert any("kick" in kinds and "snare" in kinds for kinds in groups.values()), hits


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
    assert "五线谱" in response.text


def _note(start, end, pitch, velocity=80):
    return {
        "start": start,
        "end": end,
        "pitch": pitch,
        "velocity": velocity,
        "name": "C4",
        "solfege": "do",
        "duration": round(end - start, 4),
    }


def test_extract_melody_keeps_the_top_line():
    from app.melody import extract_melody

    melody_line = [76, 79, 81, 79]
    notes = []
    for index, pitch in enumerate(melody_line):
        start = index * 0.5
        for chord in (60, 64, 67, pitch):
            notes.append(_note(start, start + 0.45, chord))
    melody, harmony = extract_melody(notes)
    assert [note["pitch"] for note in melody] == melody_line
    assert len(melody) + len(harmony) == len(notes)
    assert all(note["pitch"] != 76 or note in melody for note in notes)


def test_octave_leap_prefers_the_smoother_note():
    from app.melody import extract_melody

    notes = [
        _note(0.0, 0.4, 72),
        _note(0.5, 0.9, 84),
        _note(0.51, 0.9, 74),
    ]
    melody, _harmony = extract_melody(notes)
    assert [note["pitch"] for note in melody] == [72, 74]
    wild = [
        _note(0.0, 0.4, 72),
        _note(0.5, 0.9, 84),
        _note(0.5, 0.9, 60),
    ]
    assert [note["pitch"] for note in extract_melody(wild)[0]] == [72, 84]


def test_bass_is_not_the_song_melody():
    from app.melody import split_performance

    piano = []
    for index, pitch in enumerate((67, 69, 71, 72)):
        start = index * 0.5
        piano.append(_note(start, start + 0.45, pitch, 90))
        piano.append(_note(start, start + 0.45, pitch - 12, 60))
    performance = split_performance(
        {
            "bass": [_note(0.0, 1.0, 36), _note(1.0, 2.0, 43)],
            "piano": piano,
        },
        vocal_rms=0.0,
    )
    assert all(note["role"] == "bass" for note in performance["stems"]["bass"]["notes"])
    assert performance["stems"]["bass"]["melody"] == []
    song = performance["melody"] + performance["harmony"]
    assert song
    assert all(note["pitch"] >= 48 for note in song)
    assert all(note.get("source") != "bass" for note in song)


def test_quiet_vocals_do_not_become_the_song_melody():
    from app.melody import split_performance

    vocals = [_note(index * 0.5, index * 0.5 + 0.4, 84) for index in range(10)]
    piano = []
    for index in range(10):
        start = index * 0.5
        piano.append(_note(start, start + 0.4, 76, 90))
        piano.append(_note(start, start + 0.4, 60, 50))
        piano.append(_note(start, start + 0.4, 64, 50))
    performance = split_performance({"vocals": vocals, "piano": piano}, vocal_rms=0.006)
    assert performance["melody_source"] == "skyline"
    assert [note["pitch"] for note in performance["melody"]] == [76] * 10
    assert all(note.get("source") != "vocals" for note in performance["melody"] + performance["harmony"])
    # The vocal stem still has its own line, for the 人声 tab.
    assert len(performance["stems"]["vocals"]["melody"]) == 10


def test_clear_vocals_are_the_song_melody():
    from app.melody import split_performance

    vocals = [_note(index * 0.5, index * 0.5 + 0.4, 72 + (index % 3)) for index in range(8)]
    piano = [_note(index * 0.5, index * 0.5 + 0.4, 60) for index in range(8)]
    performance = split_performance({"vocals": vocals, "piano": piano}, vocal_rms=0.05)
    assert performance["melody_source"] == "vocals"
    assert [note["pitch"] for note in performance["melody"]] == [note["pitch"] for note in vocals]
    assert all(note["source"] == "vocals" for note in performance["melody"])
    assert any(note["source"] == "piano" for note in performance["harmony"])
    assert all(not (note["source"] == "vocals" and note["role"] == "melody") for note in performance["harmony"])


def test_musicxml_is_quantized_in_the_key():
    import xml.etree.ElementTree as ET

    from app.notation import build_musicxml

    notes = [
        _note(0.0, 0.5, 67),
        _note(0.5, 1.0, 70),
        _note(1.5, 2.2, 64),
        _note(3.0, 3.03, 72),
    ]
    for note in notes:
        note["role"] = "melody"
    xml = build_musicxml(
        [{"name": "旋律", "notes": notes, "clef": "treble", "program": 73}],
        bpm=120,
        time_signature="4/4",
        key={"tonic": "G", "mode": "minor", "name": "G 小调"},
        chords=[{"start": 0.0, "end": 2.0, "symbol": "Gm"}],
        duration=4.0,
    )
    assert "32nd" not in xml
    assert "64th" not in xml
    root = ET.fromstring(xml)
    fifths = root.find("./part/measure/attributes/key/fifths")
    assert fifths is not None and fifths.text == "-2"
    beats = root.find("./part/measure/attributes/time/beats")
    assert beats is not None and beats.text == "4"
    sign = root.find("./part/measure/attributes/clef/sign")
    assert sign is not None and sign.text == "G"
    kinds = [item.text for item in root.findall(".//kind")]
    assert "minor" in kinds
    # A# is spelled Bb, which the key signature already covers, so it is not reprinted.
    alters = [item.text for item in root.findall(".//pitch/alter")]
    assert "-1" in alters
    assert any(item.text == "natural" for item in root.findall(".//accidental"))
    assert any(item.get("type") == "start" for item in root.findall(".//tied"))


def test_piano_grand_staff_and_drum_clef():
    import xml.etree.ElementTree as ET

    from app.notation import build_musicxml

    piano = build_musicxml(
        [{
            "name": "钢琴",
            "clef": "grand",
            "program": 0,
            "notes": [_note(0.0, 0.5, 48, 70), _note(0.0, 0.5, 72, 90)],
        }],
        bpm=120,
        key={"tonic": "C", "mode": "major"},
        chords=[{"start": 0.0, "end": 1.0, "symbol": "C"}],
        duration=2.0,
    )
    assert "<staves>2</staves>" in piano
    assert piano.count("<sign>G</sign>") == 1
    assert "<sign>F</sign>" in piano
    root = ET.fromstring(piano)
    staves = [item.text for item in root.findall(".//staff")]
    assert "1" in staves and "2" in staves
    drums = build_musicxml(
        [{"name": "鼓", "clef": "percussion", "hits": [
            {"time": 0.0, "kind": "kick", "velocity": 100},
            {"time": 0.0, "kind": "hat_closed", "velocity": 70},
            {"time": 0.5, "kind": "snare", "velocity": 90},
        ]}],
        bpm=120,
        duration=2.0,
    )
    assert "<sign>percussion</sign>" in drums
    assert "<notehead>x</notehead>" in drums
    assert "32nd" not in drums and "64th" not in drums


def test_melody_midi_is_separate_from_the_stems(tmp_path: Path):
    from app.midi_export import write_combined_midi, write_lead_midi, write_part_midi

    melody = [_note(0.0, 0.4, 76)]
    harmony = [_note(0.0, 0.4, 60), _note(0.0, 0.4, 64)]
    bass = [_note(0.0, 0.4, 36)]
    hits = [{"time": 0.0, "kind": "kick", "velocity": 100}]
    write_part_midi(tmp_path / "melody.mid", 120, melody, 73, "Melody")
    write_part_midi(tmp_path / "harmony.mid", 120, harmony, 0, "Harmony")
    write_lead_midi(tmp_path / "lead.mid", 120, melody, harmony, bass, hits)
    write_combined_midi(tmp_path / "all.mid", 120, {"piano": melody, "bass": bass}, hits)
    melody_midi = pretty_midi.PrettyMIDI(str(tmp_path / "melody.mid"))
    assert melody_midi.instruments[0].program == 73
    assert melody_midi.instruments[0].name == "Melody"
    assert [note.pitch for note in melody_midi.instruments[0].notes] == [76]
    harmony_midi = pretty_midi.PrettyMIDI(str(tmp_path / "harmony.mid"))
    assert harmony_midi.instruments[0].program == 0
    lead = pretty_midi.PrettyMIDI(str(tmp_path / "lead.mid"))
    names = {instrument.name for instrument in lead.instruments}
    assert {"Melody", "Harmony", "Bass", "Drums"} <= names
    combined = {instrument.name for instrument in pretty_midi.PrettyMIDI(str(tmp_path / "all.mid")).instruments}
    assert "Melody" not in combined
    assert "Harmony" not in combined


def test_known_lead_sheet_notes_extract_exactly():
    from app.melody import extract_melody
    from app.synth import synthesize_lead_sheet

    clip = synthesize_lead_sheet()
    notes = []
    for start, end, pitches in clip["chords"]:
        for pitch in pitches:
            notes.append(_note(start, end, pitch, 60))
    for start, end, pitch in clip["melody"]:
        notes.append(_note(start, end, pitch, 100))
    melody, _harmony = extract_melody(notes)
    assert [note["pitch"] for note in melody] == [pitch for _start, _end, pitch in clip["melody"]]


def test_musicxml_route_explains_an_old_result(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MUSIC_ANALYZER_DATA", str(tmp_path))
    client = TestClient(app)
    folder = tmp_path / "jobs" / "abc123"
    folder.mkdir(parents=True)
    missing = client.get("/api/jobs/abc123/musicxml/song")
    assert missing.status_code == 404
    assert "重新分析" in missing.json()["detail"]
    (folder / "notation").mkdir()
    (folder / "notation" / "song.musicxml").write_text("<score-partwise/>", encoding="utf-8")
    found = client.get("/api/jobs/abc123/musicxml/song")
    assert found.status_code == 200
    assert b"score-partwise" in found.content


def test_lead_sheet_melody_matches_the_written_line(tmp_path: Path):
    from app.melody import split_performance
    from app.notation import build_musicxml
    from app.synth import synthesize_lead_sheet

    clip = synthesize_lead_sheet(tmp_path / "lead.wav")
    notes = transcribe(tmp_path / "lead.wav", "piano", midi_tempo=120)
    performance = split_performance({"piano": notes}, vocal_rms=0.0)
    melody = performance["melody"]

    def matched(expected, pool):
        used = set()
        hits = 0
        for start, _end, pitch in expected:
            for index, note in enumerate(pool):
                if index in used:
                    continue
                if note["pitch"] == pitch and abs(note["start"] - start) <= 0.2:
                    used.add(index)
                    hits += 1
                    break
        return hits

    expected = clip["melody"]
    recall = matched(expected, melody) / len(expected)
    precision_hits = matched([(note["start"], note["end"], note["pitch"]) for note in melody], [
        {"start": start, "end": end, "pitch": pitch} for start, end, pitch in expected
    ])
    # matched() expects pool items with pitch/start. The second call swaps roles:
    # each extracted note must land on a written melody note.
    precision = precision_hits / max(1, len(melody))
    assert recall >= 0.7, (recall, melody, expected)
    assert precision >= 0.7, (precision, melody, expected)
    xml = build_musicxml(
        [{"name": "旋律", "notes": melody, "clef": "treble", "program": 73}],
        bpm=120,
        time_signature="4/4",
        key={"tonic": "C", "mode": "major"},
        chords=[{"start": 0.3, "end": 2.2, "symbol": "C"}],
        duration=clip["duration"],
    )
    assert "32nd" not in xml and "64th" not in xml
    assert "<sign>G</sign>" in xml
    assert "<kind" in xml


def _xml_root(xml: str):
    import xml.etree.ElementTree as ET

    return ET.fromstring(xml)


def test_same_pitch_across_a_tiny_gap_becomes_one_note():
    from app.notation import build_musicxml

    xml = build_musicxml(
        [{"name": "旋律", "clef": "treble", "program": 73, "monophonic": True, "notes": [
            _note(0.0, 0.45, 60),
            _note(0.50, 1.0, 60),
        ]}],
        bpm=120,
        duration=2.0,
    )
    root = _xml_root(xml)
    pitches = root.findall(".//pitch")
    assert len(pitches) == 1
    assert root.find(".//type").text == "half"
    assert root.find(".//tied") is None


def test_a_short_blip_is_dropped_at_the_standard_level():
    from app.notation import build_musicxml

    xml = build_musicxml(
        [{"name": "旋律", "clef": "treble", "program": 73, "monophonic": True, "notes": [
            _note(0.0, 0.5, 60),
            _note(1.5, 1.53, 72),
        ]}],
        bpm=120,
        duration=2.0,
    )
    root = _xml_root(xml)
    octaves = [item.text for item in root.findall(".//pitch/octave")]
    assert octaves == ["4"]
    assert "16th" not in xml


def test_three_beats_are_one_dotted_half():
    from app.notation import build_musicxml

    xml = build_musicxml(
        [{"name": "旋律", "clef": "treble", "program": 73, "monophonic": True, "notes": [
            _note(0.0, 1.5, 67),
        ]}],
        bpm=120,
        duration=2.0,
    )
    root = _xml_root(xml)
    assert [item.text for item in root.findall(".//pitch/step")] == ["G"]
    assert root.find(".//type").text == "half"
    assert root.find(".//dot") is not None
    assert root.find(".//tied") is None


def test_an_empty_measure_is_one_full_rest():
    from app.notation import build_musicxml

    xml = build_musicxml(
        [{"name": "旋律", "clef": "treble", "program": 73, "monophonic": True, "notes": [
            _note(0.0, 0.5, 60),
        ]}],
        bpm=120,
        duration=4.0,
    )
    root = _xml_root(xml)
    rests = root.findall(".//rest")
    assert any(rest.get("measure") == "yes" for rest in rests)


def test_near_simultaneous_notes_become_one_chord():
    from app.notation import build_musicxml

    xml = build_musicxml(
        [{"name": "钢琴", "clef": "treble", "program": 0, "kind": "piano", "notes": [
            _note(0.00, 0.50, 60, 80),
            _note(0.03, 0.52, 64, 75),
            _note(0.04, 0.48, 67, 90),
        ]}],
        bpm=120,
        duration=2.0,
    )
    root = _xml_root(xml)
    assert len(root.findall(".//chord")) == 2
    assert {item.text for item in root.findall(".//voice")} == {"1"}
    assert [item.text for item in root.findall(".//pitch/step")] == ["C", "E", "G"]
    assert "16th" not in xml


def test_harmony_reattacks_become_one_sustained_chord():
    from app.notation import build_musicxml

    notes = []
    for index in range(4):
        start = index * 0.5
        for pitch in (60, 64, 67):
            note = _note(start, start + 0.45, pitch, 70)
            note["role"] = "harmony"
            notes.append(note)
    xml = build_musicxml(
        [{"name": "和声", "clef": "treble", "program": 0, "kind": "harmony", "notes": notes}],
        bpm=120,
        chords=[{"start": 0.0, "end": 2.0, "symbol": "C"}],
        duration=2.0,
    )
    root = _xml_root(xml)
    assert len(root.findall(".//pitch")) == 3
    assert len(root.findall(".//chord")) == 2
    assert root.find(".//type").text == "whole"
    assert {item.text for item in root.findall(".//voice")} == {"1"}


def test_a_moving_melody_over_a_held_chord_uses_a_second_voice():
    from app.notation import build_musicxml

    notes = []
    for pitch in (60, 64):
        note = _note(0.0, 2.0, pitch, 60)
        note["role"] = "harmony"
        notes.append(note)
    for index, pitch in enumerate((72, 74, 76, 77)):
        note = _note(index * 0.5, index * 0.5 + 0.45, pitch, 100)
        note["role"] = "melody"
        notes.append(note)
    xml = build_musicxml(
        [{"name": "钢琴", "clef": "treble", "program": 0, "kind": "piano", "notes": notes}],
        bpm=120,
        chords=[{"start": 0.0, "end": 2.0, "symbol": "C"}],
        duration=2.0,
    )
    root = _xml_root(xml)
    voices = {item.text for item in root.findall(".//voice")}
    assert voices == {"1", "2"}
    assert len(root.findall(".//chord")) == 1


def test_simple_writes_fewer_notes_than_detail():
    from app.notation import build_musicxml

    notes = [_note(index * 0.125, index * 0.125 + 0.12, 60 + (index % 5), 80) for index in range(16)]
    detail = build_musicxml(
        [{"name": "旋律", "clef": "treble", "program": 73, "monophonic": True, "notes": notes}],
        bpm=120,
        duration=2.0,
        simplify="detail",
    )
    simple = build_musicxml(
        [{"name": "旋律", "clef": "treble", "program": 73, "monophonic": True, "notes": notes}],
        bpm=120,
        duration=2.0,
        simplify="simple",
    )
    assert len(_xml_root(simple).findall(".//pitch")) < len(_xml_root(detail).findall(".//pitch"))


def test_a_clear_triplet_is_marked():
    from app.notation import build_musicxml

    notes = [
        _note(0.00, 0.15, 60),
        _note(0.167, 0.31, 62),
        _note(0.333, 0.48, 64),
    ]
    xml = build_musicxml(
        [{"name": "旋律", "clef": "treble", "program": 73, "monophonic": True, "notes": notes}],
        bpm=120,
        duration=2.0,
        simplify="detail",
    )
    root = _xml_root(xml)
    assert [item.text for item in root.findall(".//actual-notes")] == ["3", "3", "3"]
    assert "16th" not in xml


def test_musicxml_route_rebuilds_from_the_analysis(monkeypatch, tmp_path: Path):
    import json

    monkeypatch.setenv("MUSIC_ANALYZER_DATA", str(tmp_path))
    client = TestClient(app)
    folder = tmp_path / "jobs" / "abc123"
    folder.mkdir(parents=True)
    (folder / "result.json").write_text(json.dumps({
        "bpm": 120,
        "time_signature": "4/4",
        "duration": 2.0,
        "key": {"tonic": "C", "mode": "major", "name": "C 大调"},
        "chords": [{"start": 0.0, "end": 2.0, "symbol": "C"}],
        "melody": [],
        "harmony": [
            {**_note(0.0, 0.5, 60), "role": "harmony"},
            {**_note(0.02, 0.5, 64), "role": "harmony"},
            {**_note(0.04, 0.48, 67), "role": "harmony"},
        ],
        "stems": {},
    }), encoding="utf-8")
    found = client.get("/api/jobs/abc123/musicxml/song_harmony?simplify=standard")
    assert found.status_code == 200
    assert b"<chord" in found.content
    assert found.content.count(b"<chord") == 2
