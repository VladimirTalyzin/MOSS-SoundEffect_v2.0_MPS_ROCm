"""Platform-specific helpers: audio preview, file manager, HiDPI.

The GUI itself is plain Tkinter and runs anywhere, but "play this file" and
"show me this folder" have no portable API. Everything OS-dependent lives here
so the rest of the code stays free of ``if sys.platform`` branches.

Supported: Windows (winsound), macOS (afplay), Linux (paplay/aplay/ffplay).
Whatever is left over falls back to the desktop's default handler.
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

IS_WINDOWS = sys.platform == "win32"
IS_MACOS = sys.platform == "darwin"

# Linux CLI players, in order of preference. paplay/aplay take WAV only, so a
# decoded temporary file is passed to them; ffplay handles MP3 directly.
_LINUX_PLAYERS = (
    ("ffplay", ("-nodisp", "-autoexit", "-loglevel", "quiet")),
    ("paplay", ()),
    ("aplay", ("-q",)),
)

_process: subprocess.Popen | None = None


def open_path(path: Path) -> None:
    """Opens a file or folder in the system's default application."""
    if IS_WINDOWS:
        os.startfile(str(path))  # type: ignore[attr-defined]
    elif IS_MACOS:
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def play(path: Path) -> None:
    """Plays an audio file without blocking the UI thread.

    Falls back to :func:`open_path` when no player is available, so the user
    still hears the result — just in another application.
    """
    stop()

    try:
        if IS_WINDOWS:
            _play_windows(path)
            return
        if IS_MACOS:
            _spawn(["afplay", str(path)])
            return
        _play_linux(path)
        return
    except Exception:
        pass

    open_path(path)


def stop() -> None:
    """Stops the preview started by :func:`play`, if it is still running."""
    global _process

    if IS_WINDOWS:
        import winsound

        winsound.PlaySound(None, winsound.SND_PURGE)
        return

    if _process is not None and _process.poll() is None:
        _process.terminate()
    _process = None


def enable_hidpi() -> None:
    """Asks Windows for per-monitor DPI awareness so text stays crisp.

    macOS and most Linux toolkits scale Tk themselves; there this is a no-op.
    """
    if not IS_WINDOWS:
        return
    try:
        from ctypes import windll

        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass


def preferred_theme() -> str:
    """Native-looking ttk theme for the current OS ("" = keep the default)."""
    if IS_WINDOWS:
        return "vista"
    if IS_MACOS:
        return "aqua"
    return ""


# ------------------------------------------------------------------ internals


def _spawn(command: list[str]) -> None:
    global _process
    _process = subprocess.Popen(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


def _play_windows(path: Path) -> None:
    import winsound

    target = path if path.suffix.lower() == ".wav" else _decode_to_wav(path)
    winsound.PlaySound(str(target), winsound.SND_FILENAME | winsound.SND_ASYNC)


def _play_linux(path: Path) -> None:
    is_wav = path.suffix.lower() == ".wav"
    for name, flags in _LINUX_PLAYERS:
        binary = shutil.which(name)
        if not binary:
            continue
        # Only ffplay decodes MP3; the ALSA/PulseAudio tools need WAV.
        target = path if (is_wav or name == "ffplay") else _decode_to_wav(path)
        _spawn([binary, *flags, str(target)])
        return
    raise RuntimeError("no CLI audio player found")


def _decode_to_wav(path: Path) -> Path:
    """Decodes to a temporary WAV for players that cannot read compressed audio."""
    import soundfile as sf

    data, sample_rate = sf.read(str(path))
    target = Path(tempfile.gettempdir()) / "moss_preview.wav"
    sf.write(str(target), data, sample_rate)
    return target
