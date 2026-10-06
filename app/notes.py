"""Pitch names for beginners, such as C4 and the solfege syllable mi."""

PITCH_CLASS_NAMES = [
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
]

SOLFEGE = ["do", "#do", "re", "#re", "mi", "fa", "#fa", "sol", "#sol", "la", "#la", "si"]


def midi_to_name(midi: int) -> str:
    """Return a scientific pitch name. MIDI 60 is C4."""
    midi = int(midi)
    return f"{PITCH_CLASS_NAMES[midi % 12]}{midi // 12 - 1}"


def midi_to_solfege(midi: int) -> str:
    return SOLFEGE[int(midi) % 12]


def describe_pitch(midi: int) -> dict:
    return {
        "pitch": int(midi),
        "name": midi_to_name(midi),
        "solfege": midi_to_solfege(midi),
    }
