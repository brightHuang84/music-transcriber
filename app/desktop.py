"""Native window for 听音识谱.

The interface stays the existing page (piano roll, drum grid, playback).
Qt 6 draws it with Chromium (Qt WebEngine) inside a normal application
window, so Ubuntu's menu can launch it without a browser or a terminal.

Chromium here is only the window. PyTorch still uses an NVIDIA GPU when
one is available; that choice is made later, when a song is separated.
"""

from __future__ import annotations

import logging
import os
import socket
import sys
import threading
import time
import urllib.request

# Set before Qt WebEngine starts. A pip-installed WebEngine often cannot use
# the Chromium sandbox, and software rendering avoids a blank window on
# machines where the display GPU and WebEngine disagree.
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--no-sandbox --disable-gpu")

logger = logging.getLogger(__name__)


def _free_port(preferred: int = 0) -> int:
    if preferred:
        return preferred
    for port in range(8765, 8785):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("8765 到 8784 端口都被占用了。请关掉其它听音识谱窗口后再打开。")


def _wait_until_up(port: int, timeout: float = 20.0) -> bool:
    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{port}/api/health"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=0.4) as response:
                if response.status == 200:
                    return True
        except Exception:
            time.sleep(0.1)
    return False


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from PySide6.QtCore import QUrl
    from PySide6.QtGui import QCloseEvent, QIcon
    from PySide6.QtWebEngineCore import QWebEngineDownloadRequest
    from PySide6.QtWebEngineWidgets import QWebEngineView
    from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

    class AppWindow(QWebEngineView):
        def closeEvent(self, event: QCloseEvent) -> None:
            event.accept()
            logger.info("窗口关闭，退出程序")
            # Quit here, not only after exec() returns. A hidden helper
            # window can keep the process alive after the main window is gone.
            os._exit(0)

    from app.audio_io import ffmpeg_path
    from app.install_desktop import APP_ID, repo_root
    from app.server import app as fastapi_app

    application = QApplication(sys.argv)
    application.setApplicationName("听音识谱")
    application.setDesktopFileName(APP_ID)
    icon = repo_root() / "share" / f"{APP_ID}.png"
    if icon.exists():
        application.setWindowIcon(QIcon(str(icon)))

    if ffmpeg_path() is None:
        QMessageBox.critical(
            None,
            "听音识谱",
            "没有找到 ffmpeg，所以还不能读取歌曲。\n请再运行一次 install.sh，它会帮忙安装。",
        )
        return 1

    try:
        port = _free_port()
    except RuntimeError as exc:
        QMessageBox.critical(None, "听音识谱", str(exc))
        return 1

    import uvicorn

    config = uvicorn.Config(
        fastapi_app,
        host="127.0.0.1",
        port=port,
        log_level="info",
        access_log=False,
    )
    server = uvicorn.Server(config)
    server.install_signal_handlers = False
    thread = threading.Thread(target=server.run, name="tingyin-http", daemon=True)
    thread.start()
    if not _wait_until_up(port):
        server.should_exit = True
        QMessageBox.critical(
            None,
            "听音识谱",
            "程序内部没有启动成功。\n请从应用程序菜单再打开一次。若仍然不行，把日志发给懂电脑的朋友。\n"
            f"日志：{Path_log()}",
        )
        return 1

    start_path = os.environ.get("TINGYIN_START_PATH", "/")
    if not start_path.startswith("/"):
        start_path = "/" + start_path

    window = AppWindow()
    window.setWindowTitle("听音识谱")
    window.resize(1180, 840)
    window.setMinimumSize(880, 640)
    if icon.exists():
        window.setWindowIcon(QIcon(str(icon)))

    downloads: list = []

    def on_download(item: QWebEngineDownloadRequest) -> None:
        suggested = item.suggestedFileName() or "听音识谱-下载"
        initial = str(Path_downloads() / suggested)
        path, _selected = QFileDialog.getSaveFileName(window, "保存文件", initial)
        if not path:
            item.cancel()
            return
        destination = __import__("pathlib").Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        item.setDownloadDirectory(str(destination.parent))
        item.setDownloadFileName(destination.name)
        downloads.append(item)
        item.accept()

        def release(state: QWebEngineDownloadRequest.DownloadState, current=item) -> None:
            done = {
                QWebEngineDownloadRequest.DownloadState.DownloadCompleted,
                QWebEngineDownloadRequest.DownloadState.DownloadCancelled,
                QWebEngineDownloadRequest.DownloadState.DownloadInterrupted,
            }
            if state in done and current in downloads:
                downloads.remove(current)

        item.stateChanged.connect(release)

    window.page().profile().downloadRequested.connect(on_download)

    def on_loaded(ok: bool) -> None:
        if ok:
            return
        QMessageBox.warning(window, "听音识谱", "界面没有加载出来。请关掉窗口，再从应用程序菜单重新打开。")

    window.loadFinished.connect(on_loaded)
    window.load(QUrl(f"http://127.0.0.1:{port}{start_path}"))
    window.show()
    logger.info("听音识谱窗口已打开，端口 %s", port)

    def leave() -> None:
        server.should_exit = True
        # WebEngine can stall while it shuts down, and the analysis worker
        # is not a daemon. Exit immediately so closing the window leaves
        # no hidden process, even if a song is still being processed.
        os._exit(0)

    application.aboutToQuit.connect(leave)
    application.exec()
    leave()


def Path_downloads():
    from pathlib import Path

    downloads = Path.home() / "Downloads"
    if not downloads.is_dir():
        downloads = Path.home()
    return downloads


def Path_log():
    from pathlib import Path

    return Path.home() / ".local" / "share" / "tingyin-shipu" / "app.log"


if __name__ == "__main__":
    sys.exit(main())
