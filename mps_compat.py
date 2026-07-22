"""Совместимость с Apple Silicon (MPS).

Metal-бэкенд PyTorch не поддерживает float64, а комплексные тензоры держит
только в complex64. Но DiT регистрирует буферы RoPE-частот ``freqs_cis_*`` в
complex128 (это пара float64). Поэтому прямой ``pipe.to("mps")`` падает с

    TypeError: Cannot convert a MPS Tensor to float64 dtype ...

ещё до первого шага генерации.

``rope_apply`` читает у этих буферов только ``.real`` и ``.imag`` (косинусы и
синусы в диапазоне [-1, 1]) и не делает комплексных умножений, так что сужение
complex128 -> complex64 (и float64 -> float32) практически без потерь. Поэтому
модель загружается на CPU, приводится к безопасным типам и только затем
переносится на MPS.

Порядок импорта не важен. На CUDA/ROCm/CPU ничего не делает.
"""
import warnings

import torch

# Типы, которых нет на MPS, и их безопасные замены.
_NARROW = {
    torch.float64: torch.float32,
    torch.complex128: torch.complex64,
}


def is_available() -> bool:
    """True, если доступен бэкенд MPS (Apple Silicon, macOS 14+)."""
    try:
        return torch.backends.mps.is_available()
    except Exception:
        return False


def sanitize(module: torch.nn.Module) -> bool:
    """Приводит float64/complex128 параметры и буферы к MPS-совместимым типам.

    Работает in-place по всем подмодулям. Возвращает True, если что-то меняли.
    """
    changed = False
    for m in module.modules():
        for name, buf in list(m._buffers.items()):
            if buf is not None and buf.dtype in _NARROW:
                m._buffers[name] = buf.to(_NARROW[buf.dtype])
                changed = True
        for name, param in list(m._parameters.items()):
            if param is not None and param.dtype in _NARROW:
                m._parameters[name] = torch.nn.Parameter(
                    param.to(_NARROW[param.dtype]),
                    requires_grad=param.requires_grad,
                )
                changed = True
    return changed


def _sinusoidal_embedding_1d_f32(dim, position):
    """float32-версия синусоидального эмбеддинга шага (замена float64-оригинала).

    Оригинал считает эмбеддинг в float64 ради точности, но MPS float64 не умеет.
    Шагов всего ~1000, а dim//2 небольшой — точности float32 с запасом хватает.
    """
    half = dim // 2
    sinusoid = torch.outer(
        position.float(),
        torch.pow(
            10000,
            -torch.arange(half, dtype=torch.float32, device=position.device).div(half),
        ),
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def patch_runtime() -> None:
    """Заменяет float64-эмбеддинг шага на float32-версию во всех модулях.

    ``sinusoidal_embedding_1d`` вызывается на каждом шаге денойзинга и жёстко
    кастует в float64 — на MPS это падает. Патчим имя в пространствах имён, где
    оно ищется во время выполнения (в т.ч. в wan_audio, куда оно импортировано).
    """
    import importlib

    module_names = (
        "moss_soundeffect_v2.diffsynth.pipelines.wan_audio",
        "moss_soundeffect_v2.diffsynth.models.wan_video_dit",
        "moss_soundeffect_v2.diffsynth.models.wan_audio_dit",
    )
    for name in module_names:
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        if hasattr(mod, "sinusoidal_embedding_1d"):
            mod.sinusoidal_embedding_1d = _sinusoidal_embedding_1d_f32


def quiet_known_warnings() -> None:
    """Гасит два безобидных, но пугающих предупреждения на пути MPS.

    Апстрим жёстко пишет ``torch.autocast("cuda", ...)`` в нескольких местах;
    на Mac это отключается с сообщением "CUDA is not available", которое звучит
    как ошибка, хотя таковой не является (autocast bfloat16 всё равно задаёт
    внешний контекст). Второе — fallback rms_norm при bf16-входе и fp32-весах:
    это лишь замечание о неслитом ядре, на корректность не влияет.
    """
    warnings.filterwarnings(
        "ignore", message=".*CUDA is not available.*Disabling autocast.*"
    )
    warnings.filterwarnings(
        "ignore", message=".*Mismatch dtype between input and weight.*"
    )


def load_pipeline(pipeline_cls, model_dir, torch_dtype, device):
    """Загружает пайплайн, обходя падение complex128 при переносе на MPS.

    Для MPS: грузим на CPU, чиним типы, патчим float64-эмбеддинг, переносим на
    устройство. Для остальных бэкендов — обычный ``from_pretrained``.
    """
    if str(device) == "mps":
        quiet_known_warnings()
        pipe = pipeline_cls.from_pretrained(
            model_dir, torch_dtype=torch_dtype, device="cpu"
        )
        sanitize(pipe.engine)
        patch_runtime()
        pipe.to("mps")
        return pipe

    return pipeline_cls.from_pretrained(
        model_dir, torch_dtype=torch_dtype, device=device
    )


if __name__ == "__main__":
    print("mps available:", is_available())
