"""Имена выходных файлов из текста промпта и хранение настроек."""
import json
import re
from pathlib import Path

# Запрещённые в именах файлов Windows символы + управляющие.
_FORBIDDEN = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_SEPARATORS = re.compile(r"[\s_]+")
_DASHES = re.compile(r"-{2,}")
# Зарезервированные имена устройств DOS: такой файл создать нельзя.
_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}

MAX_STEM = 80


def slugify(prompt: str, fallback: str = "sound") -> str:
    """Превращает промпт в основу имени файла.

    Пробелы становятся тире, запрещённые символы удаляются. Буквы других
    алфавитов (включая китайские иероглифы) сохраняются — NTFS их допускает.
    """
    text = _FORBIDDEN.sub("", prompt).strip()
    text = _SEPARATORS.sub("-", text)
    text = _DASHES.sub("-", text)
    text = text.strip(".-")  # точка в конце ломает имя в Windows

    if len(text) > MAX_STEM:
        # Обрезаем по границе слова, чтобы имя оставалось читаемым.
        text = text[:MAX_STEM].rsplit("-", 1)[0] or text[:MAX_STEM]
        text = text.strip(".-")

    if not text or text.upper() in _RESERVED:
        return fallback
    return text


def unique_path(directory: Path, stem: str, suffix: str) -> Path:
    """Возвращает свободный путь, добавляя -2, -3 ... при совпадении имён."""
    directory.mkdir(parents=True, exist_ok=True)
    candidate = directory / f"{stem}{suffix}"
    if not candidate.exists():
        return candidate

    index = 2
    while True:
        candidate = directory / f"{stem}-{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


class Settings:
    """Настройки в JSON рядом с приложением."""

    def __init__(self, path: Path, defaults: dict):
        self.path = path
        self.data = dict(defaults)
        self.load()

    def load(self) -> None:
        try:
            stored = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(stored, dict):
                self.data.update(stored)
        except (OSError, ValueError):
            pass  # первый запуск или повреждённый файл — остаются значения по умолчанию

    def save(self) -> None:
        try:
            self.path.write_text(
                json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError:
            pass  # нет прав на запись — не повод падать

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value) -> None:
        if self.data.get(key) != value:
            self.data[key] = value
            self.save()
