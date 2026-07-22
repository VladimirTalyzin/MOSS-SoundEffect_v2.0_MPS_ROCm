"""Заглушки для ROCm-сборки PyTorch на Windows.

В ней вырезан torch.distributed, а descript-audiotools обращается к
``dist.ReduceOp.AVG`` на уровне модуля — импорт падает ещё до загрузки модели.
Distributed нам не нужен (инференс на одном устройстве), поэтому достаточно
подставить недостающие имена.

Импортировать ДО moss_soundeffect_v2. На CUDA/CPU-сборках ничего не делает.
"""
import torch
import torch.distributed as dist


def is_rocm() -> bool:
    """True для ROCm-сборки PyTorch — она выдаёт себя за CUDA."""
    return bool(getattr(torch.version, "hip", None))


def apply() -> bool:
    """Возвращает True, если заглушки понадобились."""
    patched = False

    if not hasattr(dist, "ReduceOp"):
        class ReduceOp:
            SUM = "sum"
            AVG = "avg"
            PRODUCT = "product"
            MIN = "min"
            MAX = "max"
            BAND = "band"
            BOR = "bor"
            BXOR = "bxor"

        dist.ReduceOp = ReduceOp
        patched = True

    # Одиночный процесс: распределённый режим всегда выключен.
    for name, value in (
        ("is_available", lambda: False),
        ("is_initialized", lambda: False),
        ("get_rank", lambda *a, **k: 0),
        ("get_world_size", lambda *a, **k: 1),
    ):
        if not hasattr(dist, name):
            setattr(dist, name, value)
            patched = True

    return patched


def use_cpu_vae(pipe) -> None:
    """Переносит DAC VAE на CPU в float32, оставляя DiT на GPU.

    Зачем: на Windows/gfx1151 декод VAE на GPU (а) даёт NaN, потому что веса
    остаются fp16, а обёртка autocast(float32) их не поднимает, и (б) работает
    ~20x медленнее CPU — MIOpen сваливается в fallback-солвер для свёрток.
    Декод вызывается один раз за генерацию, так что перенос почти бесплатен.
    """
    engine = pipe.engine
    vae = engine.vae.to(device="cpu", dtype=torch.float32)
    engine.vae = vae

    if getattr(vae, "_cpu_decode_patched", False):
        return

    original_decode = vae.decode

    def decode_on_cpu(latents, *args, **kwargs):
        return original_decode(
            latents.to(device="cpu", dtype=torch.float32), *args, **kwargs
        )

    vae.decode = decode_on_cpu
    vae._cpu_decode_patched = True


if __name__ == "__main__":
    print("rocm:", is_rocm())
    print("patched:", apply())
