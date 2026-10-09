"""Write beginner sheet music as MusicXML.

Pitched notes are cleaned before they are written. The 简化程度 level chooses
the grid: 详细 keeps sixteenths, 标准 (the default) uses eighths, and 简化
uses quarters. The same pitch is joined across a tiny gap, a very short blip
is dropped, and notes that start in the same grid slot with a similar length
become one chord (MusicXML <chord/>, one stem). Piano, guitar, strings, and
harmony staves also hold a chord for the length of the chord symbol instead of
restriking it. A second voice is used only when a rhythm really is different,
such as a moving melody over a held chord. Durations are the longest ordinary
value that fits, including dotted notes, and ties appear only where a single
value cannot. Empty measures are one full-measure rest. Drum hits stay on a
sixteenth grid. The window draws the file with OpenSheetMusicDisplay.
"""

from __future__ import annotations

import math
from pathlib import Path
import xml.etree.ElementTree as ET

from app.stems import STEMS, label_for

DIVISIONS = 12
QUARTER = 12
DEFAULT_SIMPLIFY = "standard"

MELODY_COLOR = "#C24B2C"
HARMONY_COLOR = "#2D6D9A"
DEFAULT_COLOR = "#241C16"

# Longest first. A value is used only when it can start on a legal beat.
_VALUES = (
    (48, "whole", False),
    (36, "half", True),
    (24, "half", False),
    (18, "quarter", True),
    (12, "quarter", False),
    (9, "eighth", True),
    (6, "eighth", False),
    (3, "16th", False),
)

# grid: quantization slot. min_len: shortest written value. gap: same-pitch
# join, in divisions. slack: durations this close still share a stem.
_LEVELS = {
    "detail": {
        "grid": 3,
        "min_len": 3,
        "min_ratio": 0.45,
        "gap": 3,
        "sustain": False,
        "max_tones": 6,
        "allow_16": True,
        "triplets": True,
        "ratio": 2 / 3,
        "slack": 3,
    },
    "standard": {
        "grid": 6,
        "min_len": 6,
        "min_ratio": 0.8,
        "gap": 3,
        "sustain": True,
        "max_tones": 4,
        "allow_16": False,
        "triplets": True,
        "ratio": 2 / 3,
        "slack": 6,
    },
    "simple": {
        "grid": 12,
        "min_len": 12,
        "min_ratio": 0.8,
        "gap": 6,
        "sustain": True,
        "max_tones": 3,
        "allow_16": False,
        "triplets": False,
        "ratio": 0.5,
        "slack": 12,
    },
}

# These staves stack chord tones and, at 标准/简化, hold the chord.
CHORDAL_KINDS = {"piano", "guitar", "strings", "other", "harmony"}

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


def normalize_simplify(name: str | None) -> str:
    key = (name or DEFAULT_SIMPLIFY).strip().lower()
    if key not in _LEVELS:
        return DEFAULT_SIMPLIFY
    return key


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
    if base in {"other", "strings", "harmony", "melody"} and low < 55 and high >= 65:
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
    if beat_type not in {1, 2, 4, 8, 16}:
        beat_type = 4
    return max(QUARTER, beats * QUARTER * 4 // beat_type)


def _round_div(value: float) -> int:
    return int(math.floor(float(value) + 0.5))


def _div_at(seconds: float, step: float) -> int:
    if step <= 0:
        return 0
    return _round_div(float(seconds) / step)


def _snap_div(div: int, grid: int) -> int:
    if grid <= 1:
        return int(div)
    return _round_div(div / grid) * grid


def _color(note: dict) -> str:
    role = note.get("role")
    if role == "melody":
        return MELODY_COLOR
    if role in {"harmony", "bass"}:
        return HARMONY_COLOR
    return DEFAULT_COLOR


def _prepare_notes(notes: list[dict]) -> list[dict]:
    prepared = []
    for note in notes or []:
        if note.get("pitch") is None or note.get("start") is None:
            continue
        start = float(note["start"])
        if note.get("end") is None:
            end = start + float(note.get("duration") or 0)
        else:
            end = float(note["end"])
        if end <= start:
            continue
        item = dict(note)
        item["start"] = start
        item["end"] = end
        item["pitch"] = int(note["pitch"])
        prepared.append(item)
    return prepared


def _monophonize(notes: list[dict], prefer_low: bool = False) -> list[dict]:
    """One sounding pitch. Melody keeps the top note; bass keeps the bottom."""
    ordered = sorted(
        notes,
        key=lambda note: (
            float(note["start"]),
            int(note["pitch"]) if prefer_low else -int(note["pitch"]),
        ),
    )
    line: list[dict] = []
    for note in ordered:
        item = dict(note)
        if line and float(item["start"]) < float(line[-1]["end"]) - 1e-9:
            if abs(float(item["start"]) - float(line[-1]["start"])) < 1e-4:
                continue
            previous = dict(line[-1])
            previous["end"] = float(item["start"])
            if float(previous["end"]) - float(previous["start"]) < 0.05:
                line[-1] = item
                continue
            line[-1] = previous
        line.append(item)
    return line


def _merge_same_pitch(notes: list[dict], gap: float) -> list[dict]:
    """Join repeated pitches across a short silence. Triplet slots stay put."""
    groups: dict[tuple, list[dict]] = {}
    for note in notes:
        if note.get("_triplet"):
            groups.setdefault(("triplet", id(note)), []).append(note)
            continue
        key = (int(note["pitch"]), note.get("role") or "")
        groups.setdefault(key, []).append(dict(note))
    merged: list[dict] = []
    for group in groups.values():
        group.sort(key=lambda note: float(note["start"]))
        current = dict(group[0])
        for note in group[1:]:
            if float(note["start"]) <= float(current["end"]) + gap:
                current["end"] = max(float(current["end"]), float(note["end"]))
                if float(note.get("velocity") or 0) >= float(current.get("velocity") or 0):
                    current["velocity"] = note.get("velocity")
            else:
                merged.append(current)
                current = dict(note)
        merged.append(current)
    return merged


def _time_overlap(left: dict, right: dict) -> float:
    start = max(float(left["start"]), float(right["start"]))
    end = min(float(left["end"]), float(right["end"]))
    return max(0.0, end - start)


def _mark_triplets(notes: list[dict], step: float) -> None:
    """Mark a beat that is clearly three notes, not two straight eighths."""
    beat_seconds = QUARTER * step
    if beat_seconds <= 0:
        return
    groups: dict[int, list[dict]] = {}
    for note in notes:
        beat_index = int(float(note["start"]) / beat_seconds + 1e-9)
        groups.setdefault(beat_index, []).append(note)
    for beat_index, group in groups.items():
        group.sort(key=lambda note: float(note["start"]))
        onsets: list[list] = []
        for note in group:
            phase = float(note["start"]) / beat_seconds - beat_index
            if onsets and abs(phase - onsets[-1][0]) <= 0.05:
                onsets[-1][1].append(note)
            else:
                onsets.append([phase, [note]])
        if len(onsets) != 3:
            continue
        phases = [item[0] for item in onsets]
        targets = (0.0, 1.0 / 3.0, 2.0 / 3.0)
        if not all(abs(phase - target) <= 0.08 for phase, target in zip(phases, targets)):
            continue
        if abs(phases[1] - 0.5) < 0.08:
            continue
        durations = [float(note["end"]) - float(note["start"]) for _phase, cluster in onsets for note in cluster]
        if any(dur > beat_seconds * 0.45 for dur in durations):
            continue
        for slot, (_phase, cluster) in enumerate(onsets):
            for note in cluster:
                note["_triplet"] = True
                note["_triplet_beat"] = beat_index
                note["_triplet_slot"] = slot


def _simplify_lengths(notes: list[dict], step: float, level: dict, total: int) -> list[dict]:
    """Drop a lone blip. Keep a run of short notes as one note per grid cell."""
    grid = level["grid"]
    grid_seconds = grid * step
    min_seconds = level["min_ratio"] * grid_seconds
    triplets = [note for note in notes if note.get("_triplet")]
    long_notes = []
    short_notes = []
    for note in notes:
        if note.get("_triplet"):
            continue
        duration = float(note["end"]) - float(note["start"])
        velocity = note.get("velocity")
        if velocity is not None and float(velocity) < 12 and duration < grid_seconds:
            continue
        if duration >= min_seconds:
            long_notes.append(note)
        else:
            short_notes.append(note)
    kept_shorts = []
    for note in short_notes:
        if any(_time_overlap(note, other) > 0 for other in long_notes):
            continue
        kept_shorts.append(note)
    cells: dict[int, list[dict]] = {}
    for note in kept_shorts:
        duration = float(note["end"]) - float(note["start"])
        neighbors = [
            other
            for other in kept_shorts
            if other is not note and abs(float(other["start"]) - float(note["start"])) <= grid_seconds
        ]
        if not neighbors and duration < max(0.06, grid_seconds * 0.5):
            continue
        div = _snap_div(_div_at(note["start"], step), grid)
        if div < 0 or div >= total:
            continue
        cells.setdefault(div, []).append(note)
    promoted: list[dict] = []
    for div, group in cells.items():
        group.sort(key=lambda note: float(note["start"]))
        clusters: list[list[dict]] = []
        for note in group:
            if not clusters:
                clusters.append([note])
                continue
            previous = clusters[-1][-1]
            if float(note["start"]) < float(previous["end"]) - 1e-6:
                clusters[-1].append(note)
            else:
                clusters.append([note])
        if len(clusters) > 1:
            def strength(cluster: list[dict]) -> float:
                return sum(
                    (float(item["end"]) - float(item["start"])) * float(item.get("velocity") or 60)
                    for item in cluster
                )

            clusters = [max(clusters, key=strength)]
        cell_end = min(total, div + grid)
        for note in clusters[0]:
            item = dict(note)
            item["_forced_div"] = (div, cell_end)
            promoted.append(item)
    return long_notes + promoted + triplets


def _quantize_events(notes: list[dict], step: float, level: dict, total: int) -> list[dict]:
    events = []
    for note in notes:
        tuplet = bool(note.get("_triplet"))
        if tuplet:
            start = int(note["_triplet_beat"]) * QUARTER + int(note["_triplet_slot"]) * 4
            end = start + 4
        elif note.get("_forced_div"):
            start, end = note["_forced_div"]
        else:
            start = _snap_div(_div_at(note["start"], step), level["grid"])
            end = _snap_div(_div_at(note["end"], step), level["grid"])
            if end <= start:
                end = start + level["grid"]
        if start >= total or end <= 0:
            continue
        start = max(0, start)
        end = min(total, end)
        if end <= start:
            continue
        if not tuplet and end - start < level["min_len"]:
            continue
        events.append(
            {
                "start": start,
                "end": end,
                "pitch": int(note["pitch"]),
                "note": note,
                "tuplet": tuplet,
                "tuplet_slot": note.get("_triplet_slot"),
                "tuplet_beat": note.get("_triplet_beat"),
            }
        )
    return events


def _merge_div_same_pitch(events: list[dict], gap: int) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for event in events:
        if event["tuplet"]:
            groups.setdefault(("triplet", id(event)), []).append(event)
            continue
        key = (event["pitch"], event["note"].get("role") or "")
        groups.setdefault(key, []).append(dict(event))
    merged = []
    for group in groups.values():
        group.sort(key=lambda event: event["start"])
        current = None
        for event in group:
            if (
                current
                and not current["tuplet"]
                and not event["tuplet"]
                and event["start"] <= current["end"] + gap
            ):
                current["end"] = max(current["end"], event["end"])
                if float(event["note"].get("velocity") or 0) >= float(current["note"].get("velocity") or 0):
                    current["note"] = event["note"]
            else:
                if current:
                    merged.append(current)
                current = dict(event)
        if current:
            merged.append(current)
    return merged


def _regions(chords: list[dict], step: float, align: int, total: int, bar_len: int) -> list[tuple[int, int]]:
    """Chord spans snapped to a beat, so a held chord is not an off-beat tie chain."""
    raw = []
    for chord in chords or []:
        if chord.get("start") is None or chord.get("end") is None:
            continue
        start = _snap_div(_div_at(chord["start"], step), align)
        end = _snap_div(_div_at(chord["end"], step), align)
        start = max(0, min(total, start))
        end = max(0, min(total, end))
        if end <= start:
            end = min(total, start + align)
        if end > start:
            raw.append((start, end))
    raw.sort()
    merged: list[list[int]] = []
    for start, end in raw:
        if merged and start <= merged[-1][1] + align:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    # 简化 holds a chord for a bar where no symbol is known. 标准 uses a half note.
    bucket = bar_len if align >= QUARTER * 2 else max(align, (bar_len // 2) // align * align)
    filled: list[tuple[int, int]] = []
    cursor = 0
    for start, end in merged:
        while cursor + bucket <= start:
            filled.append((cursor, cursor + bucket))
            cursor += bucket
        if cursor < start:
            filled.append((cursor, start))
            cursor = start
        filled.append((start, end))
        cursor = max(cursor, end)
    while cursor + bucket <= total:
        filled.append((cursor, cursor + bucket))
        cursor += bucket
    if cursor < total:
        filled.append((cursor, total))
    return filled


def _as_pitched(events: list[dict]) -> list[dict]:
    return [
        {
            "start": event["start"],
            "end": event["end"],
            "pitches": [(event["pitch"], event["note"])],
            "tuplet": event["tuplet"],
            "tuplet_slot": event.get("tuplet_slot"),
            "tuplet_beat": event.get("tuplet_beat"),
        }
        for event in events
    ]


def _merge_repeated_chords(events: list[dict]) -> list[dict]:
    """The same chord sounding again on the next beat stays one longer note."""
    ordered = sorted(events, key=lambda event: (event["start"], event["end"]))
    merged: list[dict] = []
    for event in ordered:
        pitches = tuple(pitch for pitch, _note in event["pitches"])
        if (
            merged
            and not event["tuplet"]
            and not merged[-1]["tuplet"]
            and merged[-1]["end"] == event["start"]
            and tuple(pitch for pitch, _note in merged[-1]["pitches"]) == pitches
        ):
            merged[-1]["end"] = event["end"]
            continue
        merged.append(event)
    return merged


def _sustain(events: list[dict], chords: list[dict], step: float, level: dict, total: int, bar_len: int) -> list[dict]:
    """Hold the strongest harmony pitches for each chord, instead of restriking them."""
    melody = [event for event in events if event["note"].get("role") == "melody"]
    harmony = [event for event in events if event["note"].get("role") != "melody"]
    if not harmony:
        return _as_pitched(events)
    align = QUARTER * 2 if level["grid"] >= QUARTER else QUARTER
    sustained = []
    for start, end in _regions(chords, step, align, total, bar_len):
        overlapping = [event for event in harmony if event["end"] > start and event["start"] < end]
        if not overlapping:
            continue
        scores: dict[int, float] = {}
        sources: dict[int, dict] = {}
        for event in overlapping:
            overlap = min(event["end"], end) - max(event["start"], start)
            velocity = float(event["note"].get("velocity") or 70)
            pitch = event["pitch"]
            scores[pitch] = scores.get(pitch, 0.0) + overlap * velocity
            previous = sources.get(pitch)
            if previous is None or velocity >= float(previous.get("velocity") or 0):
                sources[pitch] = event["note"]
        ranked = sorted(scores, key=lambda pitch: (-scores[pitch], -pitch))[: level["max_tones"]]
        sustained.append(
            {
                "start": start,
                "end": end,
                "pitches": [(pitch, sources[pitch]) for pitch in sorted(ranked)],
                "tuplet": False,
                "tuplet_slot": None,
                "tuplet_beat": None,
            }
        )
    return _merge_repeated_chords(sustained) + _as_pitched(melody)


def _enforce_line(events: list[dict], prefer_low: bool = False) -> list[dict]:
    """Keep a single melodic line after quantization, so overlaps do not become extra voices."""
    ordered = sorted(
        events,
        key=lambda event: (event["start"], event["pitch"] if prefer_low else -event["pitch"], event["end"]),
    )
    line: list[dict] = []
    for event in ordered:
        item = dict(event)
        if line and item["start"] < line[-1]["end"]:
            if item["start"] <= line[-1]["start"]:
                continue
            line[-1]["end"] = item["start"]
            if line[-1]["end"] <= line[-1]["start"]:
                line.pop()
        if item["end"] > item["start"] and (not line or item["start"] >= line[-1]["end"]):
            line.append(item)
    return line


def _similar_len(left: int, right: int, level: dict) -> bool:
    short, long = (left, right) if left <= right else (right, left)
    if long - short <= level["slack"]:
        return True
    if long <= 0:
        return True
    return short / long >= level["ratio"]


def _dedupe_cap(pitches: list[tuple[int, dict]], limit: int) -> list[tuple[int, dict]]:
    best: dict[int, dict] = {}
    for pitch, note in pitches:
        previous = best.get(pitch)
        if previous is None or float(note.get("velocity") or 0) >= float(previous.get("velocity") or 0):
            best[pitch] = note
    pairs = list(best.items())
    if len(pairs) > limit:
        ranked = sorted(pairs, key=lambda item: (-(float(item[1].get("velocity") or 0)), -item[0]))
        pairs = ranked[:limit]
    return sorted(pairs, key=lambda item: item[0])


def _cluster(events: list[dict], level: dict) -> list[dict]:
    """Stack notes that share a grid slot and a similar length onto one stem."""
    hosts = [dict(event) for event in events if len(event["pitches"]) > 1]
    singles = [event for event in events if len(event["pitches"]) <= 1]
    by_start: dict[int, list[dict]] = {}
    for event in singles:
        by_start.setdefault(event["start"], []).append(event)
    clustered = []
    for start, group in by_start.items():
        group.sort(key=lambda event: -(event["end"] - event["start"]))
        buckets: list[dict] = []
        for event in group:
            duration = event["end"] - event["start"]
            placed = False
            for bucket in buckets:
                same_tuplet = (
                    event["tuplet"]
                    and bucket["tuplet"]
                    and event.get("tuplet_slot") == bucket.get("tuplet_slot")
                    and event.get("tuplet_beat") == bucket.get("tuplet_beat")
                )
                close = (
                    not event["tuplet"]
                    and not bucket["tuplet"]
                    and _similar_len(duration, bucket["end"] - bucket["start"], level)
                )
                if same_tuplet or close:
                    bucket["pitches"].extend(event["pitches"])
                    bucket["end"] = max(bucket["end"], event["end"])
                    placed = True
                    break
            if not placed:
                buckets.append(
                    {
                        "start": start,
                        "end": event["end"],
                        "pitches": list(event["pitches"]),
                        "tuplet": event["tuplet"],
                        "tuplet_slot": event.get("tuplet_slot"),
                        "tuplet_beat": event.get("tuplet_beat"),
                    }
                )
        clustered.extend(buckets)
    for cluster in clustered:
        duration = cluster["end"] - cluster["start"]
        absorbed = False
        for host in hosts:
            if host["tuplet"] or cluster["tuplet"]:
                continue
            if host["start"] == cluster["start"] and _similar_len(duration, host["end"] - host["start"], level):
                host["pitches"].extend(cluster["pitches"])
                host["end"] = max(host["end"], cluster["end"])
                absorbed = True
                break
        if not absorbed:
            hosts.append(cluster)
    for event in hosts:
        event["pitches"] = _dedupe_cap(event["pitches"], level["max_tones"])
    return [event for event in hosts if event["pitches"] and event["end"] > event["start"]]


def _fits(timeline: list[dict], event: dict) -> bool:
    return all(other["start"] >= event["end"] or event["start"] >= other["end"] for other in timeline)


def _clip_into(timeline: list[dict], event: dict) -> None:
    """A third rhythm would need another voice. Shorten the earlier note instead."""
    item = dict(event)
    for other in list(timeline):
        if other["start"] < item["end"] and item["start"] < other["end"]:
            if other["start"] <= item["start"]:
                other["end"] = item["start"]
            else:
                item["end"] = min(item["end"], other["start"])
            if other["end"] <= other["start"]:
                timeline.remove(other)
    if item["end"] > item["start"] and item["pitches"]:
        item["voice"] = 1
        timeline.append(item)


def _assign_voices(events: list[dict]) -> list[dict]:
    def melody_first(event: dict) -> tuple:
        melody = any(note.get("role") == "melody" for _pitch, note in event["pitches"])
        return (0 if melody else 1, event["start"], -(event["end"] - event["start"]))

    voices: list[list[dict]] = [[], []]
    for event in sorted(events, key=melody_first):
        placed = False
        for index, timeline in enumerate(voices):
            if _fits(timeline, event):
                item = dict(event)
                item["voice"] = index + 1
                timeline.append(item)
                placed = True
                break
        if not placed:
            _clip_into(voices[0], event)
    return voices[0] + voices[1]


def _pitched_events(
    notes: list[dict],
    *,
    step: float,
    level: dict,
    total: int,
    bar_len: int,
    beats: int,
    beat_type: int,
    monophonic: bool,
    prefer_low: bool,
    sustain: bool,
    chords: list[dict],
) -> list[dict]:
    prepared = _prepare_notes(notes)
    if monophonic:
        prepared = _monophonize(prepared, prefer_low=prefer_low)
    else:
        melody = _monophonize([note for note in prepared if note.get("role") == "melody"])
        rest = [note for note in prepared if note.get("role") != "melody"]
        prepared = melody + rest
    if level["triplets"] and monophonic and beats == 4 and beat_type == 4:
        _mark_triplets(prepared, step)
    prepared = _merge_same_pitch(prepared, level["gap"] * step)
    prepared = _simplify_lengths(prepared, step, level, total)
    events = _quantize_events(prepared, step, level, total)
    events = _merge_div_same_pitch(events, level["gap"])
    if monophonic:
        events = _enforce_line(events, prefer_low=prefer_low)
    if sustain:
        pitched = _sustain(events, chords, step, level, total, bar_len)
    else:
        pitched = _as_pitched(events)
    return _assign_voices(_cluster(pitched, level))


def _legal(size: int, pos: int, bar_len: int, beats: int, beat_type: int, allow_16: bool) -> bool:
    if size <= 0 or pos + size > bar_len:
        return False
    if size == 3 and (not allow_16 or pos % 3 != 0):
        return False
    if size == 6 and pos % 6 != 0:
        return False
    if size == 9 and pos % 6 != 0:
        return False
    if size == 12 and pos % 6 != 0:
        return False
    if size >= 18 and pos % QUARTER != 0:
        return False
    if beats == 4 and beat_type == 4:
        mid = bar_len // 2
        if pos < mid < pos + size:
            # A whole note, or a dotted half from beat 1, may cross the middle.
            if pos == 0 and size in {36, bar_len}:
                return True
            return False
    return True


def _spell_chunk(start: int, length: int, bar_len: int, beats: int, beat_type: int, allow_16: bool, tuplet: bool):
    if tuplet and length == 4:
        yield start, 4, "eighth", False
        return
    position = start
    left = length
    while left > 0:
        room = bar_len - (position % bar_len)
        if room <= 0:
            room = bar_len
        chunk = min(left, room)
        local = position
        remain = chunk
        while remain > 0:
            pos = local % bar_len
            picked = None
            for size, name, dotted in _VALUES:
                if size <= remain and _legal(size, pos, bar_len, beats, beat_type, allow_16):
                    picked = (size, name, dotted)
                    break
            if picked is None and allow_16:
                for size, name, dotted in ((3, "16th", False),):
                    if size <= remain and pos + size <= bar_len:
                        picked = (size, name, dotted)
                        break
            if picked is None:
                # Stay on the grid rather than inventing a 32nd. Borrow one slot.
                size = min(remain, 6 if not allow_16 else 3)
                size = max(1, size)
                picked = (size, "eighth" if size >= 6 else "16th", False)
            yield local, picked[0], picked[1], picked[2]
            local += picked[0]
            remain -= picked[0]
        position += chunk
        left -= chunk


def _span_pieces(start: int, length: int, pitches: list, meta: dict, bar_len: int, beats: int, beat_type: int, allow_16: bool) -> list[dict]:
    tuplet = bool(meta.get("tuplet")) and length == 4
    spelled = list(_spell_chunk(start, length, bar_len, beats, beat_type, allow_16, tuplet))
    pieces = []
    for index, (div, size, name, dotted) in enumerate(spelled):
        tie_stop = set()
        tie_start = set()
        if pitches and len(spelled) > 1:
            midi = {pitch for pitch, _note in pitches}
            if index:
                tie_stop = set(midi)
            if index < len(spelled) - 1:
                tie_start = set(midi)
        slot = meta.get("tuplet_slot")
        pieces.append(
            {
                "div": div,
                "duration": size,
                "type": name,
                "dotted": dotted,
                "pitches": list(pitches),
                "tie_stop": tie_stop,
                "tie_start": tie_start,
                "tuplet": tuplet,
                "tuplet_edge": "start" if tuplet and slot == 0 else "stop" if tuplet and slot == 2 else None,
            }
        )
    return pieces


def _spell_timeline(items: list[dict], total: int, bar_len: int, beats: int, beat_type: int, allow_16: bool) -> list[dict]:
    pieces: list[dict] = []
    cursor = 0
    for item in items:
        if item["start"] > cursor:
            pieces.extend(
                _span_pieces(cursor, item["start"] - cursor, [], {}, bar_len, beats, beat_type, allow_16)
            )
        pieces.extend(
            _span_pieces(
                item["start"],
                item["end"] - item["start"],
                item["pitches"],
                item,
                bar_len,
                beats,
                beat_type,
                allow_16,
            )
        )
        cursor = item["end"]
    if cursor < total:
        pieces.extend(_span_pieces(cursor, total - cursor, [], {}, bar_len, beats, beat_type, allow_16))
    return pieces


def _for_staff(pitches: list[tuple[int, dict]], staff: int, staves: int) -> list[tuple[int, dict]]:
    if staves == 1:
        return list(pitches)
    if staff == 1:
        return [(pitch, note) for pitch, note in pitches if pitch >= 60]
    return [(pitch, note) for pitch, note in pitches if pitch < 60]


def _timeline_for(events: list[dict], voice: int, staff: int, staves: int) -> list[dict]:
    items = []
    for event in events:
        if event.get("voice") != voice:
            continue
        pitches = _for_staff(event["pitches"], staff, staves)
        if not pitches:
            continue
        items.append({**event, "pitches": pitches})
    items.sort(key=lambda event: (event["start"], event["end"]))
    cleaned = []
    for item in items:
        if cleaned and item["start"] < cleaned[-1]["end"]:
            item = dict(item)
            item["start"] = cleaned[-1]["end"]
        if item["end"] > item["start"]:
            cleaned.append(item)
    return cleaned


def _chord_marks(chords: list[dict], step: float, grid: int) -> list[tuple[int, str]]:
    placed = []
    for chord in chords or []:
        symbol = chord.get("symbol")
        if not symbol or chord.get("start") is None:
            continue
        placed.append((_snap_div(_div_at(chord["start"], step), grid), str(symbol)))
    placed.sort()
    return placed


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


def _add_notations(parent, start: bool, stop: bool, tuplet_edge: str | None) -> None:
    if not start and not stop and not tuplet_edge:
        return
    notations = ET.SubElement(parent, "notations")
    if stop:
        ET.SubElement(notations, "tied").set("type", "stop")
    if start:
        ET.SubElement(notations, "tied").set("type", "start")
    if tuplet_edge:
        tuplet = ET.SubElement(notations, "tuplet")
        tuplet.set("type", tuplet_edge)
        tuplet.set("bracket", "yes")


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


def _emit_pitched(measure, piece, *, fifths, key_alters, state, staff, staves, voice, instrument_id, symbols) -> None:
    for symbol in symbols:
        _add_harmony(measure, symbol)
    pitches = piece["pitches"]
    if not pitches:
        note_el = ET.SubElement(measure, "note")
        rest = ET.SubElement(note_el, "rest")
        if piece.get("measure_rest"):
            rest.set("measure", "yes")
        ET.SubElement(note_el, "duration").text = str(piece["duration"])
        ET.SubElement(note_el, "voice").text = str(voice)
        ET.SubElement(note_el, "type").text = "whole" if piece.get("measure_rest") else piece["type"]
        if piece["dotted"] and not piece.get("measure_rest"):
            ET.SubElement(note_el, "dot")
        if staves == 2:
            ET.SubElement(note_el, "staff").text = str(staff)
        return
    for index, (midi, note) in enumerate(pitches):
        note_el = ET.SubElement(measure, "note")
        note_el.set("color", _color(note))
        if index:
            ET.SubElement(note_el, "chord")
        step, alter, octave = _spell_pitch(midi, fifths)
        _add_pitch(note_el, step, alter, octave)
        ET.SubElement(note_el, "duration").text = str(piece["duration"])
        start = midi in piece["tie_start"]
        stop = midi in piece["tie_stop"]
        _add_ties(note_el, start, stop)
        ET.SubElement(note_el, "instrument").set("id", instrument_id)
        ET.SubElement(note_el, "voice").text = str(voice)
        ET.SubElement(note_el, "type").text = piece["type"]
        if piece["dotted"]:
            ET.SubElement(note_el, "dot")
        _add_accidental(note_el, step, alter, octave, state, key_alters)
        if piece.get("tuplet"):
            modification = ET.SubElement(note_el, "time-modification")
            ET.SubElement(modification, "actual-notes").text = "3"
            ET.SubElement(modification, "normal-notes").text = "2"
        if staves == 2:
            ET.SubElement(note_el, "staff").text = str(staff)
        edge = piece.get("tuplet_edge") if index == 0 else None
        _add_notations(note_el, start, stop, edge)


def _drum_events(hits: list[dict], step: float, total: int) -> list[dict]:
    slots: dict[int, list[dict]] = {}
    for hit in hits or []:
        if hit.get("time") is None or not hit.get("kind"):
            continue
        kind = str(hit["kind"])
        if kind not in DRUM_LINE:
            continue
        div = _snap_div(_div_at(hit["time"], step), 3)
        if div < 0 or div >= total:
            continue
        current = slots.setdefault(div, [])
        if any(item["kind"] == kind for item in current):
            continue
        current.append(hit)
    events = []
    for div, group in sorted(slots.items()):
        events.append({"start": div, "end": min(total, div + 3), "hits": group, "voice": 1})
    return events


def _emit_drum(measure, piece, instrument_prefix: str, symbols: list[str]) -> None:
    for symbol in symbols:
        _add_harmony(measure, symbol)
    hits = piece.get("hits") or []
    if not hits:
        note_el = ET.SubElement(measure, "note")
        rest = ET.SubElement(note_el, "rest")
        if piece.get("measure_rest"):
            rest.set("measure", "yes")
        ET.SubElement(note_el, "duration").text = str(piece["duration"])
        ET.SubElement(note_el, "voice").text = "1"
        ET.SubElement(note_el, "type").text = "whole" if piece.get("measure_rest") else piece["type"]
        if piece["dotted"] and not piece.get("measure_rest"):
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


def _symbols_for(piece: dict, pending: list[tuple[int, str]]) -> list[str]:
    start = piece["div"]
    end = start + piece["duration"]
    found = []
    while pending and pending[0][0] < end:
        div, symbol = pending[0]
        if div < start:
            pending.pop(0)
            continue
        pending.pop(0)
        found.append(symbol)
    return found


def _emit_measure_pieces(measure, pieces, *, emit, symbols_pending, bar_len) -> None:
    if pieces and all(not piece.get("pitches") and not piece.get("hits") for piece in pieces):
        whole = {
            "div": pieces[0]["div"],
            "duration": bar_len,
            "type": "whole",
            "dotted": False,
            "pitches": [],
            "hits": [],
            "tie_start": set(),
            "tie_stop": set(),
            "measure_rest": True,
            "tuplet": False,
            "tuplet_edge": None,
        }
        emit(measure, whole, _symbols_for(whole, symbols_pending) if symbols_pending is not None else [])
        return
    for piece in pieces:
        symbols = _symbols_for(piece, symbols_pending) if symbols_pending is not None else []
        emit(measure, piece, symbols)


def _fill_part(
    part_el,
    part: dict,
    *,
    fifths,
    mode,
    beats,
    beat_type,
    bar_len,
    step,
    total,
    chord_marks,
    level,
    chord_source,
) -> None:
    clef = part.get("clef") or "treble"
    allow_16 = True if clef == "percussion" else level["allow_16"]
    if clef == "percussion":
        events = _drum_events(part.get("hits") or [], step, total)
        pieces = []
        cursor = 0
        for event in events:
            if event["start"] > cursor:
                pieces.extend(_span_pieces(cursor, event["start"] - cursor, [], {}, bar_len, beats, beat_type, True))
            hit_pieces = _span_pieces(event["start"], event["end"] - event["start"], [], {}, bar_len, beats, beat_type, True)
            for hit_piece in hit_pieces:
                hit_piece["hits"] = event["hits"]
            pieces.extend(hit_pieces)
            cursor = event["end"]
        if cursor < total:
            pieces.extend(_span_pieces(cursor, total - cursor, [], {}, bar_len, beats, beat_type, True))
        timelines = {(1, 1): pieces}
        staves = 1
        voice_ids = [1]
    else:
        sustain = bool(level["sustain"] and not part.get("monophonic") and part.get("kind") in CHORDAL_KINDS)
        events = _pitched_events(
            part.get("notes") or [],
            step=step,
            level=level,
            total=total,
            bar_len=bar_len,
            beats=beats,
            beat_type=beat_type,
            monophonic=bool(part.get("monophonic")),
            prefer_low=bool(part.get("prefer_low")),
            sustain=sustain,
            chords=chord_source if sustain else [],
        )
        staves = 2 if clef == "grand" else 1
        voice_ids = sorted({int(event.get("voice") or 1) for event in events}) or [1]
        if 1 not in voice_ids:
            voice_ids = [1, *voice_ids]
        timelines = {}
        for staff in range(1, staves + 1):
            for voice in voice_ids:
                timelines[(staff, voice)] = _spell_timeline(
                    _timeline_for(events, voice, staff, staves),
                    total,
                    bar_len,
                    beats,
                    beat_type,
                    allow_16,
                )
    measure_count = total // bar_len
    key_alters = _key_alters(fifths)
    pending = list(chord_marks if part.get("chords", True) else [])
    part_id = part_el.get("id") or "P1"
    for number in range(1, measure_count + 1):
        measure = ET.SubElement(part_el, "measure")
        measure.set("number", str(number))
        if number == 1:
            _attributes(measure, fifths=fifths, mode=mode, beats=beats, beat_type=beat_type, clef=clef)
        states = {1: {}, 2: {}}
        origin = (number - 1) * bar_len
        first = True
        for staff in range(1, staves + 1):
            for voice in voice_ids:
                group = [
                    piece
                    for piece in timelines[(staff, voice)]
                    if origin <= piece["div"] < origin + bar_len
                ]
                if voice != 1 and not any(piece.get("pitches") or piece.get("hits") for piece in group):
                    continue
                filled = sum(piece["duration"] for piece in group)
                if filled != bar_len:
                    raise ValueError("a notation measure ran past the barline" if filled > bar_len else "the last notation measure was left short")
                if not first:
                    backup = ET.SubElement(measure, "backup")
                    ET.SubElement(backup, "duration").text = str(bar_len)
                first = False
                symbols_pending = pending if staff == 1 and voice == 1 else None

                def emit(target, piece, symbols, staff=staff, voice=voice, state=states[staff]):
                    if clef == "percussion":
                        _emit_drum(target, piece, part_id, symbols)
                    else:
                        _emit_pitched(
                            target,
                            piece,
                            fifths=fifths,
                            key_alters=key_alters,
                            state=state,
                            staff=staff,
                            staves=staves,
                            voice=voice,
                            instrument_id=f"{part_id}-I1",
                            symbols=symbols,
                        )

                _emit_measure_pieces(
                    measure,
                    group,
                    emit=emit,
                    symbols_pending=symbols_pending,
                    bar_len=bar_len,
                )


def build_musicxml(
    parts: list[dict],
    bpm: float,
    time_signature: str = "4/4",
    key: dict | None = None,
    chords: list[dict] | None = None,
    duration: float = 4.0,
    simplify: str = DEFAULT_SIMPLIFY,
) -> str:
    """Return a MusicXML 3.1 score. `parts` are melody, harmony, or one instrument."""
    level = _LEVELS[normalize_simplify(simplify)]
    tempo = float(bpm) if bpm and float(bpm) > 0 else 120.0
    beats, beat_type = _parse_time(time_signature)
    bar_len = _bar_divisions(beats, beat_type)
    fifths, mode = key_fifths(key)
    step = (60.0 / tempo) / DIVISIONS
    total = int(round(float(duration) / step)) if duration else bar_len
    total = max(bar_len, total)
    if total % bar_len:
        total += bar_len - (total % bar_len)
    chord_source = list(chords or [])
    marks = _chord_marks(chord_source, step, level["grid"])
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
            chord_marks=marks if part.get("chords") else [],
            level=level,
            chord_source=chord_source,
        )
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode")


def _part(
    name: str,
    notes: list[dict],
    clef: str,
    program: int,
    monophonic: bool = False,
    kind: str = "",
    prefer_low: bool = False,
) -> dict:
    return {
        "name": name,
        "notes": notes,
        "clef": clef,
        "program": program,
        "monophonic": monophonic,
        "kind": kind,
        "prefer_low": prefer_low,
    }


def _stem_notes(stems: dict) -> dict[str, list]:
    notes = {}
    for name, value in (stems or {}).items():
        if isinstance(value, dict):
            notes[name] = list(value.get("notes") or [])
        elif isinstance(value, list):
            notes[name] = list(value)
    return notes


def parts_for_view(
    name: str,
    *,
    melody: list[dict] | None,
    harmony: list[dict] | None,
    stems: dict | None,
    hits: list[dict] | None,
) -> list[dict] | None:
    """The parts for one notation filename, or None when the name is unknown."""
    melody = list(melody or [])
    harmony = list(harmony or [])
    stem_notes = _stem_notes(stems or {})
    if name == "song":
        return [
            _part("旋律", melody, clef_for("melody", melody), 73, monophonic=True, kind="melody"),
            _part("和声", harmony, clef_for("harmony", harmony), 0, kind="harmony"),
        ]
    if name == "song_melody":
        return [_part("旋律", melody, clef_for("melody", melody), 73, monophonic=True, kind="melody")]
    if name == "song_harmony":
        return [_part("和声", harmony, clef_for("harmony", harmony), 0, kind="harmony")]
    if name == "drums":
        return [{"name": "鼓", "clef": "percussion", "hits": list(hits or []), "program": 0, "kind": "drums"}]
    kind = name
    role = None
    for suffix in ("_melody", "_harmony"):
        if name.endswith(suffix):
            kind = name[: -len(suffix)]
            role = suffix[1:]
            break
    if kind not in STEMS or kind == "drums":
        return None
    notes = stem_notes.get(kind) or []
    program = int(STEMS[kind]["program"])
    label = label_for(kind)
    if role == "melody":
        chosen = [note for note in notes if note.get("role") == "melody"]
        return [_part(f"{label}旋律", chosen, clef_for(name, chosen), program, monophonic=True, kind="melody")]
    if role == "harmony":
        chosen = [note for note in notes if note.get("role") == "harmony"]
        return [_part(f"{label}和声", chosen, clef_for(name, chosen), program, kind=kind)]
    if kind == "bass":
        return [_part(label, notes, "bass", program, monophonic=True, prefer_low=True, kind="bass")]
    return [_part(label, notes, clef_for(kind, notes), program, kind=kind)]


def musicxml_document(result: dict, name: str, simplify: str = DEFAULT_SIMPLIFY) -> str | None:
    """Rebuild one score from an analysis result. None if this view does not exist."""
    if not isinstance(result, dict) or not result.get("bpm"):
        return None
    stems = result.get("stems") or {}
    drum_hits = []
    drums = stems.get("drums")
    if isinstance(drums, dict):
        drum_hits = list(drums.get("hits") or [])
    parts = parts_for_view(
        name,
        melody=result.get("melody") or [],
        harmony=result.get("harmony") or [],
        stems=stems,
        hits=drum_hits,
    )
    if not parts:
        return None
    return build_musicxml(
        parts,
        bpm=float(result.get("bpm") or 120),
        time_signature=result.get("time_signature") or "4/4",
        key=result.get("key"),
        chords=result.get("chords") or [],
        duration=float(result.get("duration") or 4.0),
        simplify=simplify,
    )


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
    simplify: str = DEFAULT_SIMPLIFY,
) -> None:
    """Write one MusicXML file per view the window can show."""
    folder.mkdir(parents=True, exist_ok=True)
    names = ["song", "song_melody", "song_harmony", "drums"]
    for stem in stems:
        if stem == "drums" or stem not in STEMS:
            continue
        names.append(stem)
        if stem == "bass":
            continue
        names.append(f"{stem}_melody")
        names.append(f"{stem}_harmony")
    for name in names:
        parts = parts_for_view(name, melody=melody, harmony=harmony, stems=stems, hits=hits)
        if not parts:
            continue
        xml = build_musicxml(
            parts,
            bpm=bpm,
            time_signature=time_signature or "4/4",
            key=key,
            chords=chords or [],
            duration=duration,
            simplify=simplify,
        )
        (folder / f"{name}.musicxml").write_text(xml, encoding="utf-8")
