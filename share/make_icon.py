"""Draw share/tingyin-shipu.png without extra libraries."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path


def _png(width: int, height: int, pixels: bytes) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    raw = bytearray()
    stride = width * 4
    for y in range(height):
        raw.append(0)
        raw.extend(pixels[y * stride : (y + 1) * stride])
    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(bytes(raw), 9)) + chunk(b"IEND", b"")


def _blend(dst: tuple[int, int, int, int], src: tuple[int, int, int], alpha: float) -> tuple[int, int, int, int]:
    a = max(0.0, min(1.0, alpha))
    return tuple(int(dst[i] * (1 - a) + src[i] * a) for i in range(3)) + (255,)


def render(size: int = 256) -> bytes:
    ink = (36, 28, 22)
    paper = (244, 239, 228)
    cinnabar = (194, 75, 44)
    pixels = bytearray(size * size * 4)
    radius = int(size * 0.22)
    margin = int(size * 0.08)
    inner_r = int(size * 0.16)

    def put(x: int, y: int, color: tuple[int, int, int, int]) -> None:
        if 0 <= x < size and 0 <= y < size:
            i = (y * size + x) * 4
            pixels[i : i + 4] = bytes(color)

    def rounded(x: int, y: int, left: int, top: int, right: int, bottom: int, rad: int) -> bool:
        if not (left <= x < right and top <= y < bottom):
            return False
        cx = min(max(x, left + rad), right - rad - 1)
        cy = min(max(y, top + rad), bottom - rad - 1)
        if (x - cx) ** 2 + (y - cy) ** 2 <= rad * rad:
            return True
        return False

    for y in range(size):
        for x in range(size):
            if rounded(x, y, 0, 0, size, size, radius):
                put(x, y, ink + (255,))
            else:
                put(x, y, (0, 0, 0, 0))
            if rounded(x, y, margin, margin, size - margin, size - margin, inner_r):
                put(x, y, paper + (255,))

    def stroke_circle(cx: float, cy: float, rx: float, ry: float, width: float, color: tuple[int, int, int]) -> None:
        for y in range(int(cy - ry - width), int(cy + ry + width) + 1):
            for x in range(int(cx - rx - width), int(cx + rx + width) + 1):
                nx = (x - cx) / rx
                ny = (y - cy) / ry
                dist = abs((nx * nx + ny * ny) ** 0.5 - 1)
                # Approximate edge coverage in pixel space.
                edge = dist * min(rx, ry)
                if edge <= width / 2:
                    put(x, y, color + (255,))

    def fill_ellipse(cx: float, cy: float, rx: float, ry: float, color: tuple[int, int, int]) -> None:
        for y in range(int(cy - ry), int(cy + ry) + 1):
            for x in range(int(cx - rx), int(cx + rx) + 1):
                if ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1:
                    put(x, y, color + (255,))

    def fill_rect(x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
        for y in range(y0, y1):
            for x in range(x0, x1):
                put(x, y, color + (255,))

    # Eighth-note stem and head, matching the cinnabar accent in the UI.
    fill_rect(int(size * 0.46), int(size * 0.24), int(size * 0.51), int(size * 0.70), cinnabar)
    fill_ellipse(size * 0.42, size * 0.70, size * 0.14, size * 0.10, cinnabar)
    # Flag.
    for i, t in enumerate([i / 28 for i in range(29)]):
        x = int(size * (0.50 + 0.16 * (1 - (1 - t) ** 2)))
        y = int(size * (0.26 + 0.16 * t))
        fill_ellipse(x, y, size * 0.035, size * 0.028, cinnabar)
    return _png(size, size, bytes(pixels))


def main() -> None:
    path = Path(__file__).resolve().parent / "tingyin-shipu.png"
    path.write_bytes(render())
    print(path)


if __name__ == "__main__":
    main()
