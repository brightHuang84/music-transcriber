"""Desktop menu entry, without opening a window."""

from pathlib import Path

from app.install_desktop import desktop_entry_text, install_user_entry, uninstall_user_entry


def test_menu_entry_opens_a_window_without_a_terminal(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)

    root = tmp_path / "repo"
    share = root / "share"
    share.mkdir(parents=True)
    icon = Path(__file__).resolve().parents[1] / "share" / "tingyin-shipu.png"
    (share / "tingyin-shipu.png").write_bytes(icon.read_bytes())

    paths = install_user_entry(root)
    entry = paths["desktop"].read_text(encoding="utf-8")
    assert "Name=听音识谱" in entry
    assert "Terminal=false" in entry
    assert "StartupWMClass=tingyin-shipu" in entry
    assert "tingyin-shipu.png" in entry

    launcher = paths["launcher"].read_text(encoding="utf-8")
    assert "-m app.desktop" in launcher
    assert paths["launcher"].stat().st_mode & 0o111
    assert paths["icon"].is_file()

    text = desktop_entry_text(paths["launcher"], paths["icon"])
    assert "Terminal=false" in text

    removed = uninstall_user_entry()
    assert paths["desktop"] in removed
    assert not paths["desktop"].exists()
    assert not paths["icon"].exists()


def test_install_script_does_not_require_pip_inside_the_venv():
    source = (Path(__file__).resolve().parents[1] / "install.sh").read_text(encoding="utf-8")
    commands = "\n".join(
        line for line in source.splitlines() if not line.strip().startswith("#")
    )
    assert "uv pip install --python" in commands
    assert "-m pip" not in commands
    assert "UV_PYTHON_PREFERENCE=only-managed" in commands


def test_webengine_flags_are_set_before_pyside_imports():
    source = (Path(__file__).resolve().parents[1] / "app" / "desktop.py").read_text(encoding="utf-8")
    flags = source.index("QTWEBENGINE_CHROMIUM_FLAGS")
    pyside = source.index("PySide6")
    assert flags < pyside
    assert "--no-sandbox" in source
    assert "--disable-gpu" in source
