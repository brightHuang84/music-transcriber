"""Split a transcribed performance into a melody line and the harmony under it.

The line is the skyline of notes that start together: the highest note in each
onset, unless that highest note leaps and a slightly lower note continues the
line more smoothly. That is the same idea as the old skyline melody algorithms,
kept small enough to run on the notes Basic Pitch already wrote down.

Bass is its own part. A very quiet vocal stem is treated as bleed, not as the
song's melody. A clear vocal line is the song melody, and the other instruments
become the accompaniment.
"""

from __future__ import annotations

CLUSTER_SECONDS = 0.06
LEAP_SEMITONES = 9
SMOOTH_MARGIN = 3
OCTAVE = 12
# The owner's film-score vocal stem sits around 0.006 RMS and is only a little
# bleed. A real sung line is louder than this.
VOCAL_RMS_MIN = 0.02
VOCAL_MELODY_MIN = 8


def _tagged(note: dict, **extra) -> dict:
    item = dict(note)
    item.update(extra)
    return item


def _clusters(notes: list[dict]) -> list[list[dict]]:
    ordered = sorted(notes, key=lambda note: (float(note["start"]), -int(note["pitch"])))
    groups: list[list[dict]] = []
    for note in ordered:
        if not groups or float(note["start"]) - float(groups[-1][0]["start"]) > CLUSTER_SECONDS:
            groups.append([note])
        else:
            groups[-1].append(note)
    return groups


def _choose(cluster: list[dict], previous: dict | None) -> dict:
    ranked = sorted(cluster, key=lambda note: (-int(note["pitch"]), -int(note.get("velocity") or 0)))
    top = ranked[0]
    if previous is None:
        return top
    top_pitch = int(top["pitch"])
    previous_pitch = int(previous["pitch"])
    leap = abs(top_pitch - previous_pitch)
    if leap <= LEAP_SEMITONES:
        return top
    best = None
    best_distance = leap
    for note in ranked[1:]:
        pitch = int(note["pitch"])
        if top_pitch - pitch > OCTAVE:
            continue
        distance = abs(pitch - previous_pitch)
        if leap - distance >= SMOOTH_MARGIN and distance < best_distance:
            best = note
            best_distance = distance
    return best or top


def extract_melody(notes: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return `(melody, harmony)` covering every input note exactly once.

    The lists hold the same dictionaries that were passed in. Callers that need
    a role tag should copy them.
    """
    usable = [note for note in notes if note.get("pitch") is not None and note.get("start") is not None]
    if not usable:
        return [], []
    chosen_ids: set[int] = set()
    melody: list[dict] = []
    previous = None
    for cluster in _clusters(usable):
        chosen = _choose(cluster, previous)
        # A lower note that starts while the melody note is still sounding is
        # accompaniment, not a new tune. A higher note, or the same note struck
        # again, takes over the line.
        if (
            previous is not None
            and float(chosen["start"]) < float(previous["end"]) - 0.02
            and int(chosen["pitch"]) < int(previous["pitch"])
        ):
            continue
        melody.append(chosen)
        chosen_ids.add(id(chosen))
        previous = chosen
    harmony = [note for note in usable if id(note) not in chosen_ids]
    harmony.sort(key=lambda note: (float(note["start"]), int(note["pitch"])))
    return melody, harmony


def split_performance(notes_by_stem: dict[str, list[dict]], vocal_rms: float = 0.0) -> dict:
    """Tag each pitched stem, then build one song-level melody and harmony.

    Bass notes are marked ``bass`` and never enter the song melody or harmony.
    Other stems get ``melody`` or ``harmony``. The song melody is the vocal
    line when the vocal stem is loud enough and has a real phrase. Otherwise it
    is the skyline of every non-bass note, leaving a quiet vocal stem out so
    bleed does not become the tune.
    """
    stems: dict[str, dict] = {}
    for stem, notes in notes_by_stem.items():
        if stem == "bass":
            tagged = [_tagged(note, role="bass") for note in notes]
            tagged.sort(key=lambda note: (float(note["start"]), int(note["pitch"])))
            stems[stem] = {"notes": tagged, "melody": [], "harmony": []}
            continue
        if stem == "drums":
            continue
        melody, harmony = extract_melody(list(notes))
        melody_ids = {id(note) for note in melody}
        tagged = []
        for note in notes:
            if note.get("pitch") is None:
                continue
            role = "melody" if id(note) in melody_ids else "harmony"
            tagged.append(_tagged(note, role=role))
        tagged.sort(key=lambda note: (float(note["start"]), int(note["pitch"])))
        stems[stem] = {
            "notes": tagged,
            "melody": [note for note in tagged if note["role"] == "melody"],
            "harmony": [note for note in tagged if note["role"] == "harmony"],
        }

    vocal = stems.get("vocals")
    vocal_count = len(vocal["melody"]) if vocal else 0
    trusted_vocals = vocal is not None and float(vocal_rms) >= VOCAL_RMS_MIN and vocal_count >= VOCAL_MELODY_MIN
    if trusted_vocals:
        song_melody = [_tagged(note, source="vocals", role="melody") for note in vocal["melody"]]
        song_harmony = []
        for stem, pack in stems.items():
            if stem == "bass":
                continue
            for note in pack["notes"]:
                if stem == "vocals" and note.get("role") == "melody":
                    continue
                song_harmony.append(_tagged(note, source=stem, role="harmony"))
        melody_source = "vocals"
    else:
        pool = []
        for stem, pack in stems.items():
            if stem == "bass":
                continue
            if stem == "vocals" and float(vocal_rms) < VOCAL_RMS_MIN:
                continue
            for note in pack["notes"]:
                pool.append(_tagged(note, source=stem))
        melody, harmony = extract_melody(pool)
        for note in melody:
            note["role"] = "melody"
        for note in harmony:
            note["role"] = "harmony"
        song_melody = melody
        song_harmony = harmony
        melody_source = "skyline"

    song_melody.sort(key=lambda note: (float(note["start"]), int(note["pitch"])))
    song_harmony.sort(key=lambda note: (float(note["start"]), int(note["pitch"])))
    return {
        "stems": stems,
        "melody": song_melody,
        "harmony": song_harmony,
        "melody_source": melody_source,
    }
