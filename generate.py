"""MOSS-SoundEffect-v2.0 — генерация из командной строки.

Использование:
    python generate.py "A dog barking loudly in a park." out.wav --seconds 5 --steps 50
"""
import os, sys, time, argparse
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

import torch
import soundfile as sf

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
import rocm_compat

rocm_compat.apply()

from moss_soundeffect_v2 import MossSoundEffectPipeline

MODEL_DIR = BASE_DIR / "models" / "MOSS-SoundEffect-v2.0"
SAMPLE_RATE = 48000

# ROCm-сборка PyTorch представляется как "cuda", поэтому автовыбор работает и на AMD.
# Переопределить: MOSS_DEVICE=cuda / mps / cpu, MOSS_DTYPE=bfloat16 / float32
DEVICE = os.environ.get("MOSS_DEVICE", "auto")
DTYPE = os.environ.get("MOSS_DTYPE", "")


def resolve_device_dtype():
    device = DEVICE
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    if DTYPE:
        dtype = getattr(torch, DTYPE)
    else:
        # bfloat16 на GPU: float16 переполняется и даёт NaN (проверено на gfx1151).
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
    return device, dtype


def save_wav(audio: torch.Tensor, path: str, sr: int = SAMPLE_RATE) -> None:
    """Сохраняет (B, C, T) тензор в WAV через soundfile — без torchcodec/FFmpeg."""
    wav = audio[0].float().cpu().numpy().T  # (T, C)
    sf.write(path, wav, sr)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("prompt")
    p.add_argument("output", nargs="?", default="out.wav")
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    torch.set_num_threads(os.cpu_count())

    device, dtype = resolve_device_dtype()
    t0 = time.time()
    pipe = MossSoundEffectPipeline.from_pretrained(
        str(MODEL_DIR), torch_dtype=dtype, device=device
    )
    print(f"[load] {device}/{str(dtype).split('.')[-1]} {time.time() - t0:.1f}s", file=sys.stderr)

    # На ROCm декод VAE на GPU медленнее CPU и склонен к NaN, см. rocm_compat.
    fallback = "1" if rocm_compat.is_rocm() else "0"
    if device != "cpu" and os.environ.get("MOSS_CPU_VAE", fallback) == "1":
        rocm_compat.use_cpu_vae(pipe)

    kwargs = {}
    if args.seed is not None:
        kwargs["seed"] = args.seed

    t0 = time.time()
    audio = pipe(
        prompt=args.prompt,
        seconds=args.seconds,
        num_inference_steps=args.steps,
        cfg_scale=args.cfg,
        **kwargs,
    )
    print(f"[generate] {time.time() - t0:.1f}s", file=sys.stderr)

    save_wav(audio, args.output)
    print(f"[saved] {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
