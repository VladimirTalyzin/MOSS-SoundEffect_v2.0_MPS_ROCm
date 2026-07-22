"""Замер скорости MOSS-SoundEffect на текущем устройстве.

Печатает с/шаг и метрики звука, чтобы поймать случай "быстро, но мусор"
(известная проблема bf16 на Windows/gfx1151).

    python benchmark.py [--steps 30] [--dtype float16]
"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

import numpy as np
import soundfile as sf
import torch

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
import rocm_compat

rocm_compat.apply()

from moss_soundeffect_v2 import MossSoundEffectPipeline

MODEL_DIR = BASE_DIR / "models" / "MOSS-SoundEffect-v2.0"
DEFAULT_OUT = BASE_DIR / "outputs" / "bench.wav"
PROMPT = "A single loud glass shattering on a tile floor."
SEED = 7
SAMPLE_RATE = 48000


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--seconds", type=float, default=4.0)
    p.add_argument("--dtype", default="")
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args()

    device = args.device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    if args.dtype:
        dtype = getattr(torch, args.dtype)
    else:
        # bfloat16 на GPU: float16 переполняется и даёт NaN (проверено на gfx1151).
        dtype = torch.float32 if device == "cpu" else torch.bfloat16

    print(f"torch {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    if device.startswith("cuda"):
        print(f"device_name: {torch.cuda.get_device_name(0)}")
    print(f"running on: {device} / {str(dtype).split('.')[-1]}")
    print("-" * 60)

    torch.set_num_threads(os.cpu_count() or 8)

    t0 = time.time()
    pipe = MossSoundEffectPipeline.from_pretrained(
        str(MODEL_DIR), torch_dtype=dtype, device=device
    )
    print(f"load: {time.time() - t0:.1f}s")

    fallback = "1" if rocm_compat.is_rocm() else "0"
    if device != "cpu" and os.environ.get("MOSS_CPU_VAE", fallback) == "1":
        rocm_compat.use_cpu_vae(pipe)
        print("vae: moved to cpu/float32")

    times: list[float] = []

    def timed(iterable):
        last = time.time()
        for item in iterable:
            yield item
            now = time.time()
            times.append(now - last)
            last = now

    t0 = time.time()
    audio = pipe(
        prompt=PROMPT,
        seconds=args.seconds,
        num_inference_steps=args.steps,
        cfg_scale=4.0,
        seed=SEED,
        progress_bar_cmd=timed,
    )
    total = time.time() - t0

    wav = audio[0].float().cpu().numpy().T
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out, wav, SAMPLE_RATE)

    steady = times[1:] or times  # первый шаг включает прогрев
    peak = float(np.abs(wav).max())
    rms = float(np.sqrt((wav ** 2).mean()))
    bad = not np.isfinite(wav).all()

    print("-" * 60)
    print(f"total:        {total:.1f}s for {args.steps} steps")
    print(f"per step:     {sum(steady) / len(steady):.3f}s  (first: {times[0]:.2f}s)")
    print(f"audio peak:   {peak:.4f}")
    print(f"audio rms:    {rms:.4f}")
    print(f"nan/inf:      {bad}")
    if bad or peak < 0.01:
        print("!! ВЫХОД ПОХОЖ НА МУСОР — проверь dtype", file=sys.stderr)
    print(f"saved:        {args.out}")


if __name__ == "__main__":
    main()
