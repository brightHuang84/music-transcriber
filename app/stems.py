"""Instrument names shared by separation, MIDI, and the window.

Demucs htdemucs (快速) returns four stems. htdemucs_6s (精细) adds guitar and
piano. Violin, cello, winds, brass, and synth are not stems of either model;
they stay inside 「其他乐器」.
"""

from __future__ import annotations

from app.errors import UserFacingError

# General MIDI program numbers. Piano is Acoustic Grand (0). Vocals use
# Voice Oohs so a file that also has piano does not give both the same sound.
STEMS: dict[str, dict] = {
    "vocals": {"label": "人声", "en": "Vocals", "track": "Vocals", "program": 53, "pitched": True},
    "drums": {"label": "鼓", "en": "Drums", "track": "Drums", "program": 0, "pitched": False},
    "bass": {"label": "贝斯", "en": "Bass", "track": "Bass", "program": 33, "pitched": True},
    "guitar": {"label": "吉他", "en": "Guitar", "track": "Guitar", "program": 25, "pitched": True},
    "piano": {"label": "钢琴", "en": "Piano", "track": "Piano", "program": 0, "pitched": True},
    "other": {"label": "其他乐器", "en": "Other", "track": "Other", "program": 48, "pitched": True},
}

# Shown in the window from left to right.
STEM_ORDER = ("vocals", "drums", "bass", "guitar", "piano", "other")

# Download sizes are the Hugging Face safetensors files, measured in MB (1e6 bytes).
MODES: dict[str, dict] = {
    "fast": {
        "id": "fast",
        "label": "快速",
        "model": "htdemucs",
        "stems": ("vocals", "drums", "bass", "other"),
        "download_mb": 84,
    },
    "fine": {
        "id": "fine",
        "label": "精细",
        "model": "htdemucs_6s",
        "stems": ("vocals", "drums", "bass", "guitar", "piano", "other"),
        "download_mb": 55,
    },
}

DEFAULT_MODE = "fine"


def mode_spec(mode: str | None) -> dict:
    chosen = mode or DEFAULT_MODE
    spec = MODES.get(chosen)
    if spec is None:
        raise UserFacingError("请选择「快速」或「精细」。")
    return spec


def label_for(stem: str) -> str:
    meta = STEMS.get(stem)
    if meta is None:
        return stem
    return str(meta["label"])


def pitched_names(stems: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    return tuple(name for name in stems if STEMS[name]["pitched"])
