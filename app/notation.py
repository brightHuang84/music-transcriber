"""Write beginner sheet music as MusicXML.

Pitched rhythms are quantized to eighth notes from the detected tempo, then
spelled with ordinary values (whole, half, quarter, eighth, and the dotted
ones). A shorter blip is written as an eighth, so a noisy transcription does
not become a page of sixteenths. Drum hits stay on a sixteenth grid. The window
draws the file with OpenSheetMusicDisplay. MuseScore opens the same file.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from app.stems import STEMS, label_for

DIVISIONS = 4
MELODY_COLOR = "#C24B2C"
HARMONY_COLOR = "#2D6D9A"
DEFAULT_COLOR = "#241C16"

# Longest first. The dotted values are the ones a beginner already sees in a method book.
SPELL = (
    (16, "whole", False),
    (12, "half", True),
    (8, "half", False),
    (6, "quarter", True),
    (4, "quarter", False),
    (3, "eighth", True),
    (2, "eighth", False),
    (1, "16th", False),
)

SHARP_SPELL = (
    ("C", 0), ("C", 1), ("D", 0), ("D", 1), ("E", 0), ("F", 0),
    ("F", 1), ("G", 0), ("G", 1), ("A", 0), ("A", 1), ("B", 0),
)
FLAT_SPELL = (
    ("C", 0), ("D", -1), ("D", 0), ("E", -1), ("E", 0), ("F", 0),
    ("G", -1), ("G", 0), ("A", -1), ("A", 0), ("B", -1), ("B", 0),
)
ACCIDENTAL = {-2: "flat-flat", -1: "flat", 0: "natural", 1: "sharp", 2: "double-sharp"}

PITCH_CLASS = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5,
    "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}

DRUM_LINE = {
    "kick": ("F", 4, None, 36, "底鼓"),
    "snare": ("C", 5, None, 38, "军鼓"),
    "tom_floor": ("A", 4, None, 43, "落地通鼓"),
    "tom_mid": ("D", 5, None, 47, "中通鼓"),
    "tom_high": ("E", 5, None, 50, "高通鼓"),
    "hat_closed": ("G", 5, "x", 42, "闭镲"),
    "hat_open": ("G", 5, "circle-x", 46, "开镲"),
    "ride": ("F", 5, "x", 51, "叮叮镲"),
    "crash": ("A", 5, "x", 49, "吊镲"),
}


def key_fifths(key: dict | None) -> tuple[int, str]:
    """Return MusicXML fifths and mode. Minor uses its relative major signature."""
    if not key or not key.get("tonic"):
        return 0, "major"
    mode = key.get("mode") or "major"
    tonic = PITCH_CLASS.get(str(key["tonic"]), 0)
    major_pc = (tonic + 3) % 12 if mode == "minor" else tonic
    raw = (major_pc * 7) % 12
    if raw > 7:
        raw -= 12
    return raw, "minor" if mode == "minor" else "major"


def clef_for(kind: str, notes: list[dict]) -> str:
    base = kind
    for suffix in ("_melody", "_harmony"):
        if kind.endswith(suffix):
            base = kind[: -len(suffix)]
            break
    if base == "piano":
        return "grand"
    if base == "bass":
        return "bass"
    if base in {"vocals", "guitar"}:
        return "treble"
    pitches = sorted(int(note["pitch"]) for note in notes if note.get("pitch") is not None)
    if not pitches:
        return "treble"
    low, high = pitches[0], pitches[-1]
    median = pitches[len(pitches) // 2]
    if base in {"other", "harmony", "melody"} and low < 55 and high >= 65:
        return "grand"
    if median < 55:
        return "bass"
    return "treble"


def _parse_time(signature: str | None) -> tuple[int, int]:
    if signature and "/" in signature:
        beats, beat_type = signature.split("/", 1)
        try:
            return max(1, int(beats)), max(1, int(beat_type))
        except ValueError:
            return 4, 4
    return 4, 4


def _bar_divisions(beats: int, beat_type: int) -> int:
    unit = 16 // beat_type if beat_type in {1, 2, 4, 8, 16} else 4
    return max(1, beats * unit)


def _spell(length: int) -> list[tuple[int, str, bool]]:
    pieces: list[tuple[int, str, bool]] = []
    left = length
    while left > 0:
        picked = None
        for size, name, dotted in SPELL:
            if size <= left:
                picked = (size, name, dotted)
                break
        if picked is None:
            raise ValueError(f"cannot spell {length}")
        pieces.append(picked)
        left -= picked[0]
    return pieces


def _iter_pieces(start: int, length: int, bar_len: int):
    position = start
    left = length
    while left > 0:
        room = bar_len - (position % bar_len)
        if room <= 0:
            room = bar_len
        chunk = min(left, room)
        for size, name, dotted in _spell(chunk):
            yield position, size, name, dotted
            position += size
        left -= chunk


def _color(note: dict) -> str:
    role = note.get("role")
    if role == "melody":
        return MELODY_COLOR
    if role in {"harmony", "bass"}:
        return HARMONY_COLOR
    return DEFAULT_COLOR


def _monophonize(notes: list[dict]) -> list[dict]:
    """One sounding pitch at a time, so a melody staff is a single line."""
    ordered = sorted(
        (note for note in notes if note.get("pitch") is not None and note.get("start") is not None),
        key=lambda note: (float(note["start"]), -int(note["pitch"])),
    )
    line: list[dict] = []
    for note in ordered:
        item = dict(note)
        if line and float(item["start"]) < float(line[-1]["end"]):
            previous = dict(line[-1])
            previous["end"] = float(item["start"])
            if float(previous["end"]) - float(previous["start"]) < 0.05:
                line.pop()
            else:
                line[-1] = previous
        line.append(item)
    return line


def _snap_even(div: int) -> int:
    """Nearest eighth note. An eighth is two sixteenth-divisions."""
    return int(round(div / 2.0)) * 2


def _occupancy(notes: list[dict], step: float, total: int) -> list[dict[int, dict]]:
    slots: list[dict[int, dict]] = [{} for _ in range(total)]
    for note in notes:
        if note.get("pitch") is None or note.get("start") is None:
            continue
        start = _snap_even(int(round(float(note["start"]) / step)))
        end = _snap_even(int(round(float(note.get("end", note["start"])) / step)))
        if end <= start:
            end = start + 2
        if start >= total:
            continue
        start = max(0, start)
        end = min(total, max(start + 2, end))
        pitch = int(note["pitch"])
        for index in range(start, end):
            current = slots[index].get(pitch)
            if current is None:
                slots[index][pitch] = note
                continue
            prefer_new = note.get("role") == "melody" and current.get("role") != "melody"
            if prefer_new:
                slots[index][pitch] = note
    return slots


def _runs_from_slots(slots: list[dict[int, dict]]) -> list[dict]:
    runs: list[dict] = []
    index = 0
    total = len(slots)
    while index < total:
        if not slots[index]:
            index += 1
            continue
        signature = tuple(sorted(slots[index]))
        end = index + 1
        while end < total and tuple(sorted(slots[end])) == signature:
            end += 1
        runs.append({"start": index, "length": end - index, "notes": dict(slots[index])})
        index = end
    return runs


def _pieces_from_runs(runs: list[dict], total: int, bar_len: int) -> list[dict]:
    items: list[dict] = []
    cursor = 0
    for run in runs:
        if run["start"] > cursor:
            items.append({"start": cursor, "length": run["start"] - cursor, "notes": {}})
        items.append(run)
        cursor = run["start"] + run["length"]
    if cursor < total:
        items.append({"start": cursor, "length": total - cursor, "notes": {}})
    pieces: list[dict] = []
    for item in items:
        for position, size, name, dotted in _iter_pieces(item["start"], item["length"], bar_len):
            pieces.append(
                {
                    "div": position,
                    "duration": size,
                    "type": name,
                    "dotted": dotted,
                    "notes": dict(item["notes"]),
                }
            )
    for index, piece in enumerate(pieces):
        previous = pieces[index - 1]["notes"] if index else {}
        following = pieces[index + 1]["notes"] if index + 1 < len(pieces) else {}
        piece["tie_stop"] = set()
        piece["tie_start"] = set()
        for pitch in piece["notes"]:
            if pitch in previous:
                piece["tie_stop"].add(pitch)
            if pitch in following:
                piece["tie_start"].add(pitch)
    return pieces


def _chord_divisions(chords: list[dict], step: float) -> list[tuple[int, str]]:
    placed = []
    for chord in chords or []:
        symbol = chord.get("symbol")
        if not symbol or chord.get("start") is None:
            continue
        placed.append((int(round(float(chord["start"]) / step)), str(symbol)))
    placed.sort()
    return placed


def _attach_chords(pieces: list[dict], chords: list[tuple[int, str]]) -> None:
    for piece in pieces:
        piece["chords"] = []
    pending = list(chords)
    for piece in pieces:
        start = piece["div"]
        end = start + piece["duration"]
        while pending and pending[0][0] < end:
            div, symbol = pending.pop(0)
            if div >= start:
                piece["chords"].append(symbol)


def _key_alters(fifths: int) -> dict[str, int]:
    alters = {step: 0 for step in "CDEFGAB"}
    if fifths > 0:
        for step in "FCGDAEB"[:fifths]:
            alters[step] = 1
    elif fifths < 0:
        for step in "BEADGCF"[: -fifths]:
            alters[step] = -1
    return alters


def _spell_pitch(midi: int, fifths: int) -> tuple[str, int, int]:
    table = FLAT_SPELL if fifths < 0 else SHARP_SPELL
    step, alter = table[int(midi) % 12]
    return step, alter, int(midi) // 12 - 1


def _add_pitch(parent, step: str, alter: int, octave: int) -> None:
    pitch = ET.SubElement(parent, "pitch")
    ET.SubElement(pitch, "step").text = step
    if alter:
        ET.SubElement(pitch, "alter").text = str(alter)
    ET.SubElement(pitch, "octave").text = str(octave)


def _add_accidental(parent, step: str, alter: int, octave: int, state: dict, key_alters: dict[str, int]) -> None:
    shown = state.get((step, octave), key_alters.get(step, 0))
    if alter == shown:
        return
    state[(step, octave)] = alter
    name = ACCIDENTAL.get(alter)
    if name:
        ET.SubElement(parent, "accidental").text = name


def _add_ties(parent, start: bool, stop: bool) -> None:
    if stop:
        ET.SubElement(parent, "tie").set("type", "stop")
    if start:
        ET.SubElement(parent, "tie").set("type", "start")


def _add_notations(parent, start: bool, stop: bool) -> None:
    if not start and not stop:
        return
    notations = ET.SubElement(parent, "notations")
    if stop:
        ET.SubElement(notations, "tied").set("type", "stop")
    if start:
        ET.SubElement(notations, "tied").set("type", "start")


def _add_harmony(parent, symbol: str) -> None:
    minor = symbol.endswith("m") and not symbol.endswith("dim")
    root = symbol[:-1] if minor else symbol
    if not root or root[0] not in "ABCDEFG":
        return
    harmony = ET.SubElement(parent, "harmony")
    root_el = ET.SubElement(harmony, "root")
    ET.SubElement(root_el, "root-step").text = root[0]
    if len(root) > 1 and root[1] == "#":
        ET.SubElement(root_el, "root-alter").text = "1"
    elif len(root) > 1 and root[1] == "b":
        ET.SubElement(root_el, "root-alter").text = "-1"
    kind = ET.SubElement(harmony, "kind")
    kind.text = "minor" if minor else "major"
    if minor:
        kind.set("text", "m")


def _emit_pitched_piece(measure, piece, *, fifths, key_alters, states, staves, instrument_id) -> None:
    for symbol in piece.get("chords") or []:
        _add_harmony(measure, symbol)
    staff_count = 2 if staves == 2 else 1
    for staff in range(1, staff_count + 1):
        pitches = []
        for midi, note in sorted(piece["notes"].items()):
            on_treble = int(midi) >= 60
            if staves == 2 and staff == 1 and not on_treble:
                continue
            if staves == 2 and staff == 2 and on_treble:
                continue
            pitches.append((int(midi), note))
        if not pitches:
            note_el = ET.SubElement(measure, "note")
            ET.SubElement(note_el, "rest")
            ET.SubElement(note_el, "duration").text = str(piece["duration"])
            ET.SubElement(note_el, "voice").text = "1"
            ET.SubElement(note_el, "type").text = piece["type"]
            if piece["dotted"]:
                ET.SubElement(note_el, "dot")
            if staves == 2:
                ET.SubElement(note_el, "staff").text = str(staff)
        else:
            for index, (midi, note) in enumerate(pitches):
                note_el = ET.SubElement(measure, "note")
                note_el.set("color", _color(note))
                if index:
                    ET.SubElement(note_el, "chord")
                step, alter, octave = _spell_pitch(midi, fifths)
                _add_pitch(note_el, step, alter, octave)
                if index == 0:
                    ET.SubElement(note_el, "duration").text = str(piece["duration"])
                    start = midi in piece["tie_start"]
                    stop = midi in piece["tie_stop"]
                    _add_ties(note_el, start, stop)
                ET.SubElement(note_el, "instrument").set("id", instrument_id)
                ET.SubElement(note_el, "voice").text = "1"
                ET.SubElement(note_el, "type").text = piece["type"]
                if piece["dotted"]:
                    ET.SubElement(note_el, "dot")
                _add_accidental(note_el, step, alter, octave, states[staff], key_alters)
                if staves == 2:
                    ET.SubElement(note_el, "staff").text = str(staff)
                if index == 0:
                    start = midi in piece["tie_start"]
                    stop = midi in piece["tie_stop"]
                    _add_notations(note_el, start, stop)
                else:
                    start = midi in piece["tie_start"]
                    stop = midi in piece["tie_stop"]
                    if start or stop:
                        _add_ties(note_el, start, stop)
                        _add_notations(note_el, start, stop)
        if staves == 2 and staff == 1:
            backup = ET.SubElement(measure, "backup")
            ET.SubElement(backup, "duration").text = str(piece["duration"])


def _drum_pieces(hits: list[dict], step: float, total: int, bar_len: int) -> list[dict]:
    slots: dict[int, list[dict]] = {}
    for hit in hits or []:
        if hit.get("time") is None or not hit.get("kind"):
            continue
        div = int(round(float(hit["time"]) / step))
        if div < 0 or div >= total:
            continue
        kind = str(hit["kind"])
        if kind not in DRUM_LINE:
            continue
        current = slots.setdefault(div, [])
        if any(item["kind"] == kind for item in current):
            continue
        current.append(hit)
    pieces: list[dict] = []
    cursor = 0
    for div in sorted(slots):
        if div > cursor:
            for position, size, name, dotted in _iter_pieces(cursor, div - cursor, bar_len):
                pieces.append({"div": position, "duration": size, "type": name, "dotted": dotted, "hits": []})
        pieces.append({"div": div, "duration": 1, "type": "16th", "dotted": False, "hits": slots[div]})
        cursor = div + 1
    if cursor < total:
        for position, size, name, dotted in _iter_pieces(cursor, total - cursor, bar_len):
            pieces.append({"div": position, "duration": size, "type": name, "dotted": dotted, "hits": []})
    return pieces


def _emit_drum_piece(measure, piece, instrument_prefix: str) -> None:
    hits = piece["hits"]
    if not hits:
        note_el = ET.SubElement(measure, "note")
        ET.SubElement(note_el, "rest")
        ET.SubElement(note_el, "duration").text = str(piece["duration"])
        ET.SubElement(note_el, "voice").text = "1"
        ET.SubElement(note_el, "type").text = piece["type"]
        if piece["dotted"]:
            ET.SubElement(note_el, "dot")
        return
    ordered = sorted(hits, key=lambda hit: DRUM_LINE[hit["kind"]][3])
    for index, hit in enumerate(ordered):
        step, octave, notehead, midi, _label = DRUM_LINE[hit["kind"]]
        note_el = ET.SubElement(measure, "note")
        if index:
            ET.SubElement(note_el, "chord")
        unpitched = ET.SubElement(note_el, "unpitched")
        ET.SubElement(unpitched, "display-step").text = step
        ET.SubElement(unpitched, "display-octave").text = str(octave)
        if index == 0:
            ET.SubElement(note_el, "duration").text = str(piece["duration"])
        ET.SubElement(note_el, "instrument").set("id", f"{instrument_prefix}-I{midi}")
        ET.SubElement(note_el, "voice").text = "1"
        ET.SubElement(note_el, "type").text = piece["type"]
        if piece["dotted"]:
            ET.SubElement(note_el, "dot")
        if notehead:
            ET.SubElement(note_el, "notehead").text = notehead


def _attributes(parent, *, fifths, mode, beats, beat_type, clef) -> None:
    attributes = ET.SubElement(parent, "attributes")
    ET.SubElement(attributes, "divisions").text = str(DIVISIONS)
    key = ET.SubElement(attributes, "key")
    ET.SubElement(key, "fifths").text = str(fifths)
    ET.SubElement(key, "mode").text = mode
    time_el = ET.SubElement(attributes, "time")
    ET.SubElement(time_el, "beats").text = str(beats)
    ET.SubElement(time_el, "beat-type").text = str(beat_type)
    if clef == "grand":
        ET.SubElement(attributes, "staves").text = "2"
        treble = ET.SubElement(attributes, "clef")
        treble.set("number", "1")
        ET.SubElement(treble, "sign").text = "G"
        ET.SubElement(treble, "line").text = "2"
        bass = ET.SubElement(attributes, "clef")
        bass.set("number", "2")
        ET.SubElement(bass, "sign").text = "F"
        ET.SubElement(bass, "line").text = "4"
    elif clef == "bass":
        clef_el = ET.SubElement(attributes, "clef")
        ET.SubElement(clef_el, "sign").text = "F"
        ET.SubElement(clef_el, "line").text = "4"
    elif clef == "percussion":
        clef_el = ET.SubElement(attributes, "clef")
        ET.SubElement(clef_el, "sign").text = "percussion"
    else:
        clef_el = ET.SubElement(attributes, "clef")
        ET.SubElement(clef_el, "sign").text = "G"
        ET.SubElement(clef_el, "line").text = "2"


def _fill_part(part_el, part: dict, *, fifths, mode, beats, beat_type, bar_len, step, total, chords) -> None:
    clef = part.get("clef") or "treble"
    if part.get("monophonic"):
        part = dict(part)
        part["notes"] = _monophonize(part.get("notes") or [])
    if clef == "percussion":
        pieces = _drum_pieces(part.get("hits") or [], step, total, bar_len)
        _attach_chords(pieces, chords if part.get("chords", True) else [])
    else:
        slots = _occupancy(part.get("notes") or [], step, total)
        pieces = _pieces_from_runs(_runs_from_slots(slots), total, bar_len)
        _attach_chords(pieces, chords if part.get("chords", True) else [])
    key_alters = _key_alters(fifths)
    measure_notes: list[list[dict]] = []
    current: list[dict] = []
    filled = 0
    for piece in pieces:
        current.append(piece)
        filled += piece["duration"]
        if filled == bar_len:
            measure_notes.append(current)
            current = []
            filled = 0
        elif filled > bar_len:
            raise ValueError("a notation measure ran past the barline")
    if current:
        raise ValueError("the last notation measure was left short")
    part_id = part_el.get("id") or "P1"
    for number, group in enumerate(measure_notes, start=1):
        measure = ET.SubElement(part_el, "measure")
        measure.set("number", str(number))
        if number == 1:
            _attributes(measure, fifths=fifths, mode=mode, beats=beats, beat_type=beat_type, clef=clef)
        states = {1: {}, 2: {}}
        for piece in group:
            if clef == "percussion":
                for symbol in piece.get("chords") or []:
                    _add_harmony(measure, symbol)
                _emit_drum_piece(measure, piece, part_id)
            else:
                _emit_pitched_piece(
                    measure,
                    piece,
                    fifths=fifths,
                    key_alters=key_alters,
                    states=states,
                    staves=2 if clef == "grand" else 1,
                    instrument_id=f"{part_id}-I1",
                )


def build_musicxml(
    parts: list[dict],
    bpm: float,
    time_signature: str = "4/4",
    key: dict | None = None,
    chords: list[dict] | None = None,
    duration: float = 4.0,
) -> str:
    """Return a MusicXML 3.1 score. `parts` are melody, harmony, or one instrument."""
    tempo = float(bpm) if bpm and float(bpm) > 0 else 120.0
    beats, beat_type = _parse_time(time_signature)
    bar_len = _bar_divisions(beats, beat_type)
    fifths, mode = key_fifths(key)
    step = (60.0 / tempo) / 4.0
    total = int(round(float(duration) / step)) if duration else bar_len
    total = max(bar_len, total)
    if total % bar_len:
        total += bar_len - (total % bar_len)
    chord_marks = _chord_divisions(chords or [], step)
    # Chord symbols sit on the first part only, so a two-part score is not labeled twice.
    prepared = []
    for index, part in enumerate(parts):
        item = dict(part)
        item["chords"] = index == 0
        prepared.append(item)

    root = ET.Element("score-partwise")
    root.set("version", "3.1")
    work = ET.SubElement(root, "work")
    ET.SubElement(work, "work-title").text = prepared[0].get("name") or "五线谱"
    identification = ET.SubElement(root, "identification")
    encoding = ET.SubElement(identification, "encoding")
    ET.SubElement(encoding, "software").text = "听音识谱"
    part_list = ET.SubElement(root, "part-list")
    for index, part in enumerate(prepared, start=1):
        part_id = f"P{index}"
        part["id"] = part_id
        score_part = ET.SubElement(part_list, "score-part")
        score_part.set("id", part_id)
        ET.SubElement(score_part, "part-name").text = part.get("name") or part_id
        if part.get("clef") == "percussion":
            for _kind, (_step, _octave, _head, midi, label) in DRUM_LINE.items():
                instrument_id = f"{part_id}-I{midi}"
                score_instrument = ET.SubElement(score_part, "score-instrument")
                score_instrument.set("id", instrument_id)
                ET.SubElement(score_instrument, "instrument-name").text = label
                midi_instrument = ET.SubElement(score_part, "midi-instrument")
                midi_instrument.set("id", instrument_id)
                ET.SubElement(midi_instrument, "midi-channel").text = "10"
                ET.SubElement(midi_instrument, "midi-unpitched").text = str(midi)
        else:
            instrument_id = f"{part_id}-I1"
            score_instrument = ET.SubElement(score_part, "score-instrument")
            score_instrument.set("id", instrument_id)
            ET.SubElement(score_instrument, "instrument-name").text = part.get("name") or part_id
            midi_instrument = ET.SubElement(score_part, "midi-instrument")
            midi_instrument.set("id", instrument_id)
            channel = index if index < 10 else index + 1
            ET.SubElement(midi_instrument, "midi-channel").text = str(channel)
            program = int(part.get("program") if part.get("program") is not None else 0)
            ET.SubElement(midi_instrument, "midi-program").text = str(program + 1)
    for part in prepared:
        part_el = ET.SubElement(root, "part")
        part_el.set("id", part["id"])
        marks = chord_marks if part.get("chords") else []
        _fill_part(
            part_el,
            part,
            fifths=fifths,
            mode=mode,
            beats=beats,
            beat_type=beat_type,
            bar_len=bar_len,
            step=step,
            total=total,
            chords=marks,
        )
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode")


def _part(name: str, notes: list[dict], clef: str, program: int, monophonic: bool = False) -> dict:
    return {"name": name, "notes": notes, "clef": clef, "program": program, "monophonic": monophonic}


def write_notation(
    folder: Path,
    *,
    bpm: float,
    time_signature: str,
    key: dict | None,
    chords: list[dict],
    duration: float,
    stems: dict[str, list[dict]],
    melody: list[dict],
    harmony: list[dict],
    hits: list[dict],
) -> None:
    """Write one MusicXML file per view the window can show."""
    folder.mkdir(parents=True, exist_ok=True)
    common = {
        "bpm": bpm,
        "time_signature": time_signature or "4/4",
        "key": key,
        "chords": chords or [],
        "duration": duration,
    }

    def save(filename: str, parts: list[dict]) -> None:
        xml = build_musicxml(parts, **common)
        (folder / filename).write_text(xml, encoding="utf-8")

    save(
        "song.musicxml",
        [
            _part("旋律", melody, clef_for("melody", melody), 73, monophonic=True),
            _part("和声", harmony, clef_for("harmony", harmony), 0),
        ],
    )
    save("song_melody.musicxml", [_part("旋律", melody, clef_for("melody", melody), 73, monophonic=True)])
    save("song_harmony.musicxml", [_part("和声", harmony, clef_for("harmony", harmony), 0)])
    for stem, notes in stems.items():
        if stem == "drums" or stem not in STEMS:
            continue
        program = int(STEMS[stem]["program"])
        label = label_for(stem)
        save(f"{stem}.musicxml", [_part(label, notes, clef_for(stem, notes), program)])
        if stem == "bass":
            continue
        melody_notes = [note for note in notes if note.get("role") == "melody"]
        harmony_notes = [note for note in notes if note.get("role") == "harmony"]
        save(
            f"{stem}_melody.musicxml",
            [_part(f"{label}旋律", melody_notes, clef_for(f"{stem}_melody", melody_notes), program, monophonic=True)],
        )
        save(
            f"{stem}_harmony.musicxml",
            [_part(f"{label}和声", harmony_notes, clef_for(f"{stem}_harmony", harmony_notes), program)],
        )
    save("drums.musicxml", [{"name": "鼓", "clef": "percussion", "hits": hits, "program": 0}])
