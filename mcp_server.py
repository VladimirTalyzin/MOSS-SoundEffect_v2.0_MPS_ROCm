"""MOSS-SoundEffect-v2.0 — MCP-сервер.

Экспортирует генерацию звуковых эффектов как инструмент MCP (Model Context
Protocol), чтобы модель можно было вызывать из любого MCP-клиента — Claude
Desktop, Claude Code, Cursor, Cline и т. п. Транспорт по умолчанию — stdio.

Запуск:
    python mcp_server.py                 # stdio (для Claude Desktop и др.)
    MOSS_MCP_TRANSPORT=sse python mcp_server.py   # HTTP/SSE на :8000

Модель весит ~11 ГБ и загружается лениво — при первом вызове инструмента, а не
при старте сервера, — и остаётся в памяти между вызовами. Устройство, dtype и
CPU-VAE определяются той же логикой, что и в generate.py / app.py, и
настраиваются теми же переменными окружения (MOSS_DEVICE, MOSS_DTYPE,
MOSS_CPU_VAE — см. README).
"""
import os
import sys
import threading
import time
from pathlib import Path

# Те же переменные окружения, что выставляет generate.py: без hardware-SDPA
# на ROCm генерация в ~2 раза медленнее, а TORCHDYNAMO ломает часть бэкендов.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import torch
import soundfile as sf

import mps_compat
import rocm_compat
from naming import slugify, unique_path

rocm_compat.apply()

from moss_soundeffect_v2 import MossSoundEffectPipeline

try:
    from mcp.server.fastmcp import FastMCP
except ModuleNotFoundError:  # pragma: no cover - подсказка вместо трейсбека
    sys.exit(
        "Не найден пакет MCP SDK. Установите его:\n"
        '    pip install "mcp[cli]"\n'
    )

MODEL_DIR = BASE_DIR / "models" / "MOSS-SoundEffect-v2.0"
OUTPUT_DIR = BASE_DIR / "outputs"
SAMPLE_RATE = 48000
MAX_SECONDS = 30.0  # предел, на который обучена модель

# Пайплайн грузится один раз и переиспользуется. Torch/MPS не любят
# параллельные вызовы с одного устройства, поэтому генерацию сериализуем.
_pipe = None
_pipe_lock = threading.Lock()

mcp = FastMCP("moss-soundeffect")


def _resolve_device_dtype():
    """Выбор устройства и dtype — та же логика, что в generate.py.

    ROCm-сборка PyTorch представляется как «cuda», поэтому auto работает и на
    AMD. bfloat16 на GPU: float16 переполняется в NaN (проверено на gfx1151).
    """
    device = os.environ.get("MOSS_DEVICE", "auto")
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    forced = os.environ.get("MOSS_DTYPE", "")
    if forced:
        dtype = getattr(torch, forced)
    else:
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
    return device, dtype


def _get_pipeline():
    """Лениво загружает пайплайн и кэширует его между вызовами."""
    global _pipe
    if _pipe is not None:
        return _pipe

    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Веса модели не найдены в {MODEL_DIR}. "
            "Скачайте их: python download_model.py"
        )

    torch.set_num_threads(os.cpu_count())
    device, dtype = _resolve_device_dtype()

    t0 = time.time()
    pipe = mps_compat.load_pipeline(
        MossSoundEffectPipeline, str(MODEL_DIR), dtype, device
    )
    print(
        f"[mcp] модель загружена: {device}/{str(dtype).split('.')[-1]} "
        f"за {time.time() - t0:.1f} c",
        file=sys.stderr,
    )

    # На ROCm декод VAE на GPU медленнее CPU и склонен к NaN (см. rocm_compat).
    fallback = "1" if rocm_compat.is_rocm() else "0"
    if device != "cpu" and os.environ.get("MOSS_CPU_VAE", fallback) == "1":
        rocm_compat.use_cpu_vae(pipe)

    _pipe = pipe
    return _pipe


def _save(audio: torch.Tensor, path: Path, as_mp3: bool) -> None:
    """Сохраняет (B, C, T) тензор с тем же пик-лимитингом, что и в app.py."""
    wav = audio[0].float().cpu().numpy().T  # (T, C)
    # Модель выдаёт сигнал впритык к 0 dBFS; MP3-кодек добавляет
    # интерсэмпловый звон, поэтому для lossy оставляем запас до -1 dBFS.
    ceiling = 0.89 if as_mp3 else 0.97
    peak = float(abs(wav).max())
    if peak > ceiling:
        wav = wav * (ceiling / peak)
    if as_mp3:
        sf.write(str(path), wav, SAMPLE_RATE, format="MP3", subtype="MPEG_LAYER_III")
    else:
        sf.write(str(path), wav, SAMPLE_RATE)


@mcp.tool()
def generate_sound_effect(
    prompt: str,
    seconds: float = 5.0,
    steps: int = 50,
    cfg: float = 4.0,
    seed: int | None = None,
    format: str = "wav",
) -> dict:
    """Сгенерировать звуковой эффект из текстового описания и сохранить в файл.

    Модель обучена на описаниях на английском и китайском. Файл сохраняется в
    каталог outputs/, имя формируется из промпта. При первом вызове модель
    загружается (~15-60 c), последующие вызовы быстрые.

    Args:
        prompt: Описание звука, напр. "A dog barking loudly in a park".
        seconds: Длительность клипа в секундах (0 < seconds <= 30).
        steps: Число шагов диффузии; больше — медленнее и обычно чище.
        cfg: Сила следования тексту; выше — буквальнее следует описанию.
        seed: Зерно ГПСЧ для воспроизводимости; None — случайное.
        format: "wav" (без потерь) или "mp3".

    Returns:
        Словарь с путём к файлу, фактическими параметрами и временем генерации.
    """
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("Промпт пустой.")

    fmt = format.lower().lstrip(".")
    if fmt not in ("wav", "mp3"):
        raise ValueError('format должен быть "wav" или "mp3".')

    if not 0 < seconds <= MAX_SECONDS:
        raise ValueError(f"seconds должно быть в диапазоне (0, {MAX_SECONDS:g}].")
    if steps < 1:
        raise ValueError("steps должно быть >= 1.")

    with _pipe_lock:
        pipe = _get_pipeline()

        kwargs = {}
        if seed is not None:
            kwargs["seed"] = int(seed)

        t0 = time.time()
        audio = pipe(
            prompt=prompt,
            seconds=seconds,
            num_inference_steps=steps,
            cfg_scale=cfg,
            **kwargs,
        )
        elapsed = time.time() - t0

        path = unique_path(OUTPUT_DIR, slugify(prompt), f".{fmt}")
        _save(audio, path, as_mp3=(fmt == "mp3"))

    return {
        "path": str(path),
        "prompt": prompt,
        "seconds": seconds,
        "steps": steps,
        "cfg": cfg,
        "seed": seed,
        "format": fmt,
        "generation_seconds": round(elapsed, 2),
    }


if __name__ == "__main__":
    transport = os.environ.get("MOSS_MCP_TRANSPORT", "stdio")
    mcp.run(transport=transport)
