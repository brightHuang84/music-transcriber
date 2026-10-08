"""Instrument names shared by separation, MIDI, and the window.

快速 is Demucs htdemucs (four stems). 精细 is htdemucs_6s, which also splits
piano and guitar. 最高质量 uses BS-RoFormer for drums, bass, piano, and
guitar, htdemucs_6s for vocals, and a bowed-strings model for 弦乐. Winds,
brass, and synth stay inside 「其他乐器」.
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
    # String Ensemble, same program as 「其他乐器」. In 最高质量 the bowed
    # parts live here; the leftover track keeps 48 so 精细, where violin is
    # still inside 「其他乐器」, does not change sound.
    "strings": {"label": "弦乐", "en": "Strings", "track": "Strings", "program": 48, "pitched": True},
    "other": {"label": "其他乐器", "en": "Other", "track": "Other", "program": 48, "pitched": True},
}

# Shown in the window from left to right.
STEM_ORDER = ("vocals", "drums", "bass", "guitar", "piano", "strings", "other")

# Download sizes are the weight files, measured in MB (1e6 bytes).
# 最高质量 is the six-stem RoFormer (699) plus bowed strings (303) plus the
# drum-kit model (438). htdemucs_6s (55MB) is also used for vocals.
MODES: dict[str, dict] = {
    "best": {
        "id": "best",
        "label": "最高质量",
        "model": "quality",
        "engine": "quality",
        "stems": ("vocals", "drums", "bass", "guitar", "piano", "strings", "other"),
        "download_mb": 1440,
        "drum_kit": True,
    },
    "fine": {
        "id": "fine",
        "label": "精细",
        "model": "htdemucs_6s",
        "engine": "demucs",
        "stems": ("vocals", "drums", "bass", "guitar", "piano", "other"),
        "download_mb": 55,
        "drum_kit": False,
    },
    "fast": {
        "id": "fast",
        "label": "快速",
        "model": "htdemucs",
        "engine": "demucs",
        "stems": ("vocals", "drums", "bass", "other"),
        "download_mb": 84,
        "drum_kit": False,
    },
}

DEFAULT_MODE = "best"


def mode_spec(mode: str | None) -> dict:
    chosen = mode or DEFAULT_MODE
    spec = MODES.get(chosen)
    if spec is None:
        raise UserFacingError("请选择「最高质量」、「精细」或「快速」。")
    return spec


def label_for(stem: str) -> str:
    meta = STEMS.get(stem)
    if meta is None:
        return stem
    return str(meta["label"])


def pitched_names(stems: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    return tuple(name for name in stems if STEMS[name]["pitched"])
