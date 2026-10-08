"""Write one MIDI file per stem, plus a combined file."""

from __future__ import annotations

from pathlib import Path

import pretty_midi

from app.drums import DRUM_MIDI
from app.stems import STEMS, pitched_names

DRUM_PITCH = dict(DRUM_MIDI)

# General MIDI programs. Track names stay ASCII so strict MIDI readers do not
# reject them. Vocals are Voice Oohs (53), not piano, once a real piano track exists.
STEM_PROGRAM = {name: int(meta["program"]) for name, meta in STEMS.items() if meta["pitched"]}
STEM_TRACK = {name: str(meta["track"]) for name, meta in STEMS.items()}


def _new_midi(bpm: float) -> pretty_midi.PrettyMIDI:
    midi = pretty_midi.PrettyMIDI(initial_tempo=float(bpm) if bpm and bpm > 0 else 120.0)
    midi.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0))
    return midi


def _pitched_instrument(stem: str, notes: list[dict]) -> pretty_midi.Instrument:
    instrument = pretty_midi.Instrument(program=STEM_PROGRAM[stem], is_drum=False, name=STEM_TRACK[stem])
    for note in notes:
        end = max(float(note["end"]), float(note["start"]) + 0.05)
        instrument.notes.append(
            pretty_midi.Note(
                velocity=int(note["velocity"]),
                pitch=int(note["pitch"]),
                start=float(note["start"]),
                end=end,
            )
        )
    return instrument


def _drum_instrument(hits: list[dict]) -> pretty_midi.Instrument:
    instrument = pretty_midi.Instrument(program=0, is_drum=True, name=STEM_TRACK["drums"])
    for hit in hits:
        pitch = DRUM_PITCH.get(hit["kind"], 38)
        start = float(hit["time"])
        instrument.notes.append(
            pretty_midi.Note(
                velocity=int(hit["velocity"]),
                pitch=pitch,
                start=start,
                end=start + 0.08,
            )
        )
    return instrument


def write_stem_midi(path: Path, stem: str, bpm: float, notes: list[dict] | None = None, hits: list[dict] | None = None) -> None:
    midi = _new_midi(bpm)
    if stem == "drums":
        midi.instruments.append(_drum_instrument(hits or []))
    else:
        midi.instruments.append(_pitched_instrument(stem, notes or []))
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(path))


def write_combined_midi(
    path: Path,
    bpm: float,
    notes_by_stem: dict[str, list[dict]],
    hits: list[dict],
) -> None:
    midi = _new_midi(bpm)
    for stem in pitched_names(tuple(notes_by_stem)):
        midi.instruments.append(_pitched_instrument(stem, notes_by_stem.get(stem, [])))
    midi.instruments.append(_drum_instrument(hits))
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(path))
