"""Start the local server and open the browser."""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
import webbrowser

import uvicorn

from app.audio_io import ffmpeg_path


def _python_ok() -> bool:
    return (3, 10) <= sys.version_info[:2] < (3, 13)


def _free_port(preferred: int) -> int:
    if preferred:
        return preferred
    for port in range(8765, 8785):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise SystemExit("8765 到 8784 端口都被占用了。请关掉占用端口的程序后再试。")


def _open_browser(url: str) -> None:
    time.sleep(1.0)
    try:
        webbrowser.open(url)
    except Exception:
        print(f"浏览器没有自动打开。请手动访问 {url}")


def main() -> None:
    parser = argparse.ArgumentParser(description="听音识谱")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    if not _python_ok():
        raise SystemExit("请使用 Python 3.10、3.11 或 3.12。当前版本不适用。")
    if ffmpeg_path() is None:
        raise SystemExit(
            "没有找到 ffmpeg。\n"
            "Windows：在 PowerShell 里运行  winget install Gyan.FFmpeg  然后重新打开本程序。\n"
            "macOS：在终端运行  brew install ffmpeg  ，或到 https://ffmpeg.org/download.html 下载。\n"
            "装好后要重新打开一个终端，再运行 start 脚本。"
        )

    port = _free_port(args.port)
    url = f"http://127.0.0.1:{port}"
    print()
    print("听音识谱已启动")
    print(f"在浏览器打开：{url}")
    print("歌曲只在这台电脑上处理，不会上传。")
    print("关掉这个窗口就会退出程序。")
    print()
    if not args.no_browser:
        threading.Thread(target=_open_browser, args=(url,), daemon=True).start()
    uvicorn.run("app.server:app", host="127.0.0.1", port=port, log_level="info")


if __name__ == "__main__":
    main()
