# MOSS-SoundEffect v2.0 — desktop app for AMD ROCm and Apple Silicon

A small, dependency-light desktop front-end for
[MOSS-SoundEffect-v2.0](https://huggingface.co/OpenMOSS-Team/MOSS-SoundEffect-v2.0),
the text-to-audio diffusion model from the OpenMOSS team. Type a description,
get a sound effect. Everything runs locally — no API keys, no uploads.

The upstream project ships CUDA-oriented scripts and a Gradio demo. This
repository adds what was missing for the hardware I actually use: a native
Tkinter window, a working **AMD ROCm** path on Windows, and the fixes needed to
get there. **Apple Silicon (MPS)** support is wired up but not yet verified —
see [Platform support](#platform-support).

---

## Features

- **Native window, no browser.** Tkinter only — no Gradio, no local web server.
- **Model stays loaded.** Weights load once in a background thread; the second
  generation starts instantly instead of paying the ~60 s load again.
- **Live progress and ETA.** Per-step timing measured on your machine, not
  guessed. Generation can be cancelled between steps.
- **Trilingual UI** — English, Russian, Chinese; auto-detected on first run.
- **MP3 or WAV output**, named after the prompt, with peak limiting tuned per
  format so MP3 encoding does not clip.
- **Device switching at runtime** — Auto / GPU / CPU, without restarting.
- **CLI and benchmark scripts** for scripted generation and hardware testing.

## Platform support

| Platform | Backend | Status |
|---|---|---|
| Windows + AMD Radeon (ROCm) | `cuda` (ROCm build) | Tested — primary development target |
| Windows / Linux + NVIDIA | `cuda` | Should work; untested |
| macOS + Apple Silicon | `mps` | Wired up, **not yet verified** |
| Any | `cpu` | Tested (slow — see [Performance](#performance)) |

Developed on an AMD Radeon 8060S (Strix Halo, `gfx1151`) under Windows 11 with
Python 3.12. Porting to a Mac mini M4 Pro is the next step, which is why the
MPS paths are already in place.

---

## Install

### 1. Clone and create a virtual environment

Python **3.12** is required by the upstream package.

```bash
git clone https://github.com/VladimirTalyzin/MOSS-SoundEffect_v2.0_MPS_ROCm.git
cd MOSS-SoundEffect_v2.0_MPS_ROCm
python -m venv venv
```

Activate it: `venv\Scripts\activate` (Windows) or `source venv/bin/activate`
(macOS/Linux).

### 2. Install PyTorch for your hardware

This has to come first and separately — the wheels are platform-specific.

**AMD ROCm on Windows** (needs Adrenalin driver 26.2.2 or newer):

```bash
pip install -f https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/ \
    "torch==2.9.1+rocm7.2.1" "torchaudio==2.9.1+rocm7.2.1"
```

**AMD ROCm on Linux:**

```bash
pip install --index-url https://download.pytorch.org/whl/rocm6.2 torch torchaudio
```

**Apple Silicon** — the default macOS wheels include MPS:

```bash
pip install torch torchaudio
```

**NVIDIA CUDA:**

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchaudio
```

**CPU only:**

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch torchaudio
```

### 3. Install the rest

```bash
pip install -r requirements.txt
```

This pulls the `moss_soundeffect_v2` package straight from the upstream
repository, pinned to the commit this app was built against. The upstream
checkout lands in `src/` (pip's default for editable VCS installs) and is
git-ignored here.

### 4. Download the weights (~11 GB)

```bash
python download_model.py
```

Files go to `models/MOSS-SoundEffect-v2.0/`, which is also git-ignored.

---

## Usage

### Desktop app

Windows: double-click **`MOSS SoundEffect.bat`**.
macOS: `chmod +x "MOSS SoundEffect.command"` once, then double-click it.
Or just run it directly:

```bash
python app.py
```

Prompts work in **English or Chinese** — those are the languages the model was
trained on. The UI language is separate and includes Russian.

Generated files land in `outputs/`, named after the prompt.

### Command line

```bash
python generate.py "A dog barking loudly in a park." out.wav --seconds 5 --steps 50
```

| Option | Default | Meaning |
|---|---|---|
| `--seconds` | `5.0` | Length of the clip (model maximum: 30) |
| `--steps` | `50` | Diffusion steps — more is slower and usually cleaner |
| `--cfg` | `4.0` | Prompt adherence; higher follows the text more literally |
| `--seed` | random | Fix it to reproduce a result exactly |

### Benchmark

Measures seconds per step and checks the output for NaN/silence, which catches
the "fast but garbage" failure mode described below.

```bash
python benchmark.py --steps 30
python benchmark.py --steps 30 --device cpu       # compare against CPU
python benchmark.py --steps 30 --dtype float16    # see it break on ROCm
```

### Environment variables

| Variable | Default | Effect |
|---|---|---|
| `MOSS_DEVICE` | `auto` | Force `cuda`, `mps` or `cpu` (CLI scripts only) |
| `MOSS_DTYPE` | auto | Force a dtype, e.g. `float32`. Overrides the safe default |
| `MOSS_CPU_VAE` | `1` on ROCm, else `0` | Run the VAE decoder on CPU (see below) |
| `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` | `1` | Hardware SDPA on ROCm — big speedup |

---

## Performance

Measured on AMD Radeon 8060S (Strix Halo, `gfx1151`), Windows 11, ROCm 7.2.1,
`bfloat16`, VAE on CPU:

| Configuration | Seconds per diffusion step |
|---|---|
| CPU (float32) | 4.15 |
| GPU, no AOTriton flag | 1.24 |
| GPU, AOTriton enabled | **0.54** |

That is a 7.6× speedup over CPU, and 2.3× of it comes from a single
environment variable.

---

## ROCm notes

Four things had to be worked out experimentally to get this model running on
`gfx1151`. All of them are handled automatically by the code; this section is
here so the reasoning is not lost.

**1. Use `bfloat16`, not `float16`.** `float16` overflows and produces NaN
non-deterministically — output is silence or noise, with no error raised. This
contradicts the common advice to prefer fp16 on Windows/gfx1151; for this model
the opposite holds. `bfloat16` and `float32` are both stable. This is why
`benchmark.py` prints peak/RMS/NaN stats instead of just timings: on this
hardware you have to check the audio, not just that the process exited zero.

**2. `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` is worth 2.3×.** Without it,
scaled dot-product attention silently falls back to the slow math path
(1.24 → 0.54 s/step). Both `app.py` and the CLI scripts set it on import; the
variable is ignored on CPU and NVIDIA.

**3. `torch.distributed` is stripped from the ROCm Windows build.**
`descript-audiotools` touches `dist.ReduceOp.AVG` at module level, so the
import fails before any weights are loaded. [`rocm_compat.py`](rocm_compat.py)
installs the handful of missing names — single-process inference never needs
the real thing.

**4. The VAE runs faster on CPU than on GPU.** MIOpen falls back to a generic
solver for the DAC decoder's convolutions, making GPU decode roughly 20×
slower than CPU, and leaving the weights in a dtype that can overflow. Since
decode happens once per generation, [`rocm_compat.use_cpu_vae()`](rocm_compat.py)
keeps the DiT on the GPU and moves just the VAE to CPU/float32. Enabled by
default on ROCm only; set `MOSS_CPU_VAE=1` to try it elsewhere.

---

## Project layout

```
app.py               Tkinter GUI — model loading, generation, playback
generate.py          Single-shot CLI generation
benchmark.py         Speed + output-sanity measurement
download_model.py    Fetches the weights from Hugging Face
i18n.py              UI strings for en / ru / zh, system language detection
naming.py            Prompt-to-filename slugs, JSON settings storage
platform_compat.py   Audio preview, file manager, HiDPI — per OS
rocm_compat.py       torch.distributed stubs, CPU-VAE fallback
```

Runtime state that is deliberately **not** in the repository: `models/`
(weights, ~11 GB), `outputs/` (generated audio), `venv*/`, and `settings.json`
(remembered language, device and format).

---

## Credits

- Model and inference pipeline: [OpenMOSS-Team/MOSS-SoundEffect-v2.0](https://huggingface.co/OpenMOSS-Team/MOSS-SoundEffect-v2.0)
  and [OpenMOSS/MOSS-TTS](https://github.com/OpenMOSS/MOSS-TTS), Apache-2.0.
- This desktop app and the ROCm/MPS integration: Vladimir Talyzin.

Licensed under the Apache License 2.0 — see [LICENSE](LICENSE). The model
weights carry their own licence terms; check the upstream model card before
using generated audio commercially.
