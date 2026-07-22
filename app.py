"""MOSS-SoundEffect-v2.0 — оконная оболочка (Tkinter).

Модель загружается один раз в фоновом потоке и остаётся в памяти,
поэтому повторные генерации не тратят время на загрузку весов.

Запуск:
    Windows: venv\\Scripts\\pythonw.exe app.py
    macOS/Linux: venv/bin/python app.py
"""
import os
import queue
import random
import sys
import threading
import time
import traceback
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
# Аппаратный SDPA на AMD-GPU: без этого флага attention идёт медленным
# math-фолбэком (замерено: 1.24 -> 0.54 с/шаг). На CPU/NVIDIA игнорируется.
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

import tkinter as tk
from tkinter import messagebox, ttk

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import i18n
import platform_compat
from i18n import t
from naming import Settings, slugify, unique_path

MODEL_DIR = BASE_DIR / "models" / "MOSS-SoundEffect-v2.0"
OUTPUT_DIR = BASE_DIR / "outputs"
SETTINGS_PATH = BASE_DIR / "settings.json"
SAMPLE_RATE = 48000

DEVICE_CHOICES = ("auto", "gpu", "cpu")


class Cancelled(Exception):
    """Генерация прервана пользователем."""


def detect_backend(torch) -> tuple[str, str, str]:
    """Возвращает (код, короткая метка, подробное описание) активного бэкенда.

    ROCm-сборка PyTorch выдаёт себя за CUDA, поэтому отличаем их по
    torch.version.hip — иначе AMD-карта отображалась бы как CUDA.
    """
    try:
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            hip = getattr(torch.version, "hip", None)
            if hip:
                return "cuda", f"ROCm {hip.split('-')[0]}", name
            cuda = getattr(torch.version, "cuda", None) or "?"
            return "cuda", f"CUDA {cuda}", name
    except Exception:
        pass

    try:
        if torch.backends.mps.is_available():
            return "mps", "MPS", "Apple Silicon"
    except Exception:
        pass

    import platform

    return "cpu", "CPU", platform.processor() or platform.machine()


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.pipe = None
        self.busy = False
        self.loading = False
        self.cancel_flag = threading.Event()
        self.events: "queue.Queue[tuple]" = queue.Queue()
        self.result_paths: list[Path] = []
        self.step_times: list[float] = []
        self.backend_text = ""
        self.load_token = 0

        self.settings = Settings(
            SETTINGS_PATH,
            {
                "language": i18n.detect_system_language(),
                "device": "auto",
                "mp3": True,
            },
        )
        self.lang = self.settings.get("language", i18n.DEFAULT_LANG)
        if self.lang not in i18n.STRINGS:
            self.lang = i18n.DEFAULT_LANG

        # Переменные переживают пересборку интерфейса при смене языка.
        self.seconds = tk.DoubleVar(value=5.0)
        self.steps = tk.IntVar(value=50)
        self.cfg = tk.DoubleVar(value=4.0)
        self.shift = tk.DoubleVar(value=5.0)
        self.seed = tk.IntVar(value=42)
        self.random_seed = tk.BooleanVar(value=True)
        self.mp3 = tk.BooleanVar(value=bool(self.settings.get("mp3", True)))
        self.device_pref = tk.StringVar(value=self.settings.get("device", "auto"))
        self.lang_var = tk.StringVar(value=i18n.LANGUAGE_NAMES[self.lang])

        self.mp3.trace_add("write", lambda *_: self.settings.set("mp3", self.mp3.get()))
        for var in (self.seconds, self.steps):
            var.trace_add("write", lambda *_: self._update_estimate())

        self._saved_prompt = i18n.PRESETS[self.lang][0]
        self._saved_negative = ""

        root.minsize(760, 700)
        self.container = ttk.Frame(root)
        self.container.pack(fill="both", expand=True)

        self._build_ui()
        self.root.after(80, self._drain_events)
        self._start_model_load()

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        self.root.title(t(self.lang, "title"))
        pad = {"padx": 10, "pady": 5}

        main = ttk.Frame(self.container, padding=12)
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=1)

        # --- Описание звука ---
        pf = ttk.LabelFrame(main, text=t(self.lang, "prompt_frame"), padding=10)
        pf.grid(row=0, column=0, sticky="ew", **pad)
        pf.columnconfigure(0, weight=1)

        self.prompt = tk.Text(pf, height=4, wrap="word", font=("Segoe UI", 10))
        self.prompt.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.prompt.insert("1.0", self._saved_prompt)

        ttk.Label(pf, text=t(self.lang, "prompt_hint")).grid(
            row=1, column=0, sticky="w", pady=(6, 0)
        )
        preset_btn = ttk.Menubutton(pf, text=t(self.lang, "examples"))
        menu = tk.Menu(preset_btn, tearoff=False)
        for text in i18n.PRESETS[self.lang]:
            label = text if len(text) <= 60 else text[:57] + "..."
            menu.add_command(label=label, command=lambda x=text: self._set_prompt(x))
        preset_btn["menu"] = menu
        preset_btn.grid(row=1, column=1, sticky="e", pady=(6, 0))

        ttk.Label(pf, text=t(self.lang, "negative_label")).grid(
            row=2, column=0, sticky="w", pady=(8, 2)
        )
        self.negative = ttk.Entry(pf)
        self.negative.grid(row=3, column=0, columnspan=2, sticky="ew")
        self.negative.insert(0, self._saved_negative)

        # --- Параметры ---
        gf = ttk.LabelFrame(main, text=t(self.lang, "params_frame"), padding=10)
        gf.grid(row=1, column=0, sticky="ew", **pad)
        for c in (1, 3):
            gf.columnconfigure(c, weight=1)

        self._spin(gf, 0, 0, t(self.lang, "duration"), self.seconds, 0.5, 30.0, 0.5)
        self._spin(gf, 0, 2, t(self.lang, "steps"), self.steps, 5, 200, 5)
        self._spin(gf, 1, 0, t(self.lang, "cfg"), self.cfg, 1.0, 15.0, 0.5)
        self._spin(gf, 1, 2, t(self.lang, "shift"), self.shift, 1.0, 10.0, 0.5)

        ttk.Label(gf, text=t(self.lang, "seed")).grid(row=2, column=0, sticky="w", pady=4)
        seed_box = ttk.Frame(gf)
        seed_box.grid(row=2, column=1, sticky="ew", pady=4)
        self.seed_entry = ttk.Spinbox(
            seed_box, from_=0, to=2**31 - 1, textvariable=self.seed, width=12
        )
        self.seed_entry.pack(side="left")
        ttk.Checkbutton(
            seed_box,
            text=t(self.lang, "random"),
            variable=self.random_seed,
            command=self._toggle_seed,
        ).pack(side="left", padx=8)
        self._toggle_seed()

        self.eta_label = ttk.Label(gf, text="", foreground="#555")
        self.eta_label.grid(row=2, column=2, columnspan=2, sticky="e", pady=4)

        # --- Сохранение ---
        of = ttk.LabelFrame(main, text=t(self.lang, "output_frame"), padding=10)
        of.grid(row=2, column=0, sticky="ew", **pad)
        ttk.Checkbutton(
            of, text=t(self.lang, "format_mp3"), variable=self.mp3
        ).grid(row=0, column=0, sticky="w")

        # --- Устройство и язык ---
        ef = ttk.LabelFrame(main, text=t(self.lang, "engine_frame"), padding=10)
        ef.grid(row=3, column=0, sticky="ew", **pad)
        ef.columnconfigure(4, weight=1)

        ttk.Label(ef, text=t(self.lang, "device_label")).grid(row=0, column=0, sticky="w")
        self.device_labels = {
            "auto": t(self.lang, "device_auto"),
            "gpu": t(self.lang, "device_gpu"),
            "cpu": t(self.lang, "device_cpu"),
        }
        self.device_combo = ttk.Combobox(
            ef,
            state="readonly",
            width=20,
            values=[self.device_labels[k] for k in DEVICE_CHOICES],
        )
        self.device_combo.set(self.device_labels[self.device_pref.get()])
        self.device_combo.grid(row=0, column=1, sticky="w", padx=(6, 16))
        self.device_combo.bind("<<ComboboxSelected>>", self._on_device_change)

        ttk.Label(ef, text=t(self.lang, "language_label")).grid(row=0, column=2, sticky="w")
        self.lang_combo = ttk.Combobox(
            ef,
            state="readonly",
            width=12,
            textvariable=self.lang_var,
            values=list(i18n.LANGUAGE_NAMES.values()),
        )
        self.lang_combo.grid(row=0, column=3, sticky="w", padx=6)
        self.lang_combo.bind("<<ComboboxSelected>>", self._on_language_change)

        self.backend_label = ttk.Label(
            ef, text=self.backend_text or t(self.lang, "status_detecting"),
            foreground="#0a6", font=("Segoe UI", 9, "bold"),
        )
        self.backend_label.grid(row=1, column=0, columnspan=5, sticky="w", pady=(8, 0))

        # --- Управление ---
        cf = ttk.Frame(main)
        cf.grid(row=4, column=0, sticky="ew", **pad)
        cf.columnconfigure(0, weight=1)

        self.progress = ttk.Progressbar(cf, mode="indeterminate")
        self.progress.grid(row=0, column=0, columnspan=4, sticky="ew", pady=(0, 8))

        self.generate_btn = ttk.Button(
            cf, text=t(self.lang, "generate"), command=self._on_generate
        )
        self.generate_btn.grid(row=1, column=0, sticky="w")
        self.cancel_btn = ttk.Button(
            cf, text=t(self.lang, "cancel"), command=self._on_cancel, state="disabled"
        )
        self.cancel_btn.grid(row=1, column=1, padx=6)
        self.play_btn = ttk.Button(
            cf, text=t(self.lang, "play"), command=self._on_play, state="disabled"
        )
        self.play_btn.grid(row=1, column=2, padx=6)
        ttk.Button(
            cf, text=t(self.lang, "open_folder"), command=self._open_folder
        ).grid(row=1, column=3, sticky="e")

        # --- Результаты ---
        rf = ttk.LabelFrame(main, text=t(self.lang, "results_frame"), padding=10)
        rf.grid(row=5, column=0, sticky="nsew", **pad)
        main.rowconfigure(5, weight=1)
        rf.columnconfigure(0, weight=1)
        rf.rowconfigure(0, weight=1)

        self.results = tk.Listbox(rf, height=6, font=("Segoe UI", 9))
        self.results.grid(row=0, column=0, sticky="nsew")
        self.results.bind("<Double-Button-1>", lambda _e: self._on_play())
        sb = ttk.Scrollbar(rf, orient="vertical", command=self.results.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.results.config(yscrollcommand=sb.set)
        for path in self.result_paths:
            self.results.insert("end", path.name)
        if self.result_paths:
            self.results.selection_set("end")
            self.results.see("end")
            self.play_btn.config(state="normal")

        self.status = ttk.Label(
            main, text="", relief="sunken", anchor="w", padding=4
        )
        self.status.grid(row=6, column=0, sticky="ew", padx=10, pady=(4, 0))

        self._update_estimate()
        self._apply_busy_state()

    def _spin(self, parent, row, col, label, var, lo, hi, inc) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=col, sticky="w", pady=4)
        ttk.Spinbox(
            parent, from_=lo, to=hi, increment=inc, textvariable=var, width=10
        ).grid(row=row, column=col + 1, sticky="w", pady=4, padx=(0, 12))

    def _rebuild_ui(self) -> None:
        """Пересобирает интерфейс на новом языке, сохраняя введённый текст."""
        self._saved_prompt = self.prompt.get("1.0", "end").strip()
        self._saved_negative = self.negative.get()
        status_text = self.status.cget("text")
        progress_mode = self.progress.cget("mode")
        progress_value = self.progress["value"]
        progress_max = self.progress["maximum"]

        for child in self.container.winfo_children():
            child.destroy()
        self._build_ui()

        self.status.config(text=status_text)
        self.progress.config(mode=progress_mode, maximum=progress_max)
        self.progress["value"] = progress_value
        if self.loading:
            self.progress.start(12)

    def _set_prompt(self, text: str) -> None:
        self.prompt.delete("1.0", "end")
        self.prompt.insert("1.0", text)

    def _toggle_seed(self) -> None:
        self.seed_entry.config(state="disabled" if self.random_seed.get() else "normal")

    def _apply_busy_state(self) -> None:
        blocked = self.busy or self.loading
        self.generate_btn.config(state="disabled" if blocked else "normal")
        self.cancel_btn.config(state="normal" if self.busy else "disabled")
        self.device_combo.config(state="disabled" if blocked else "readonly")

    def _update_estimate(self) -> None:
        """Оценка времени по среднему из уже измеренных шагов."""
        if not hasattr(self, "eta_label"):
            return
        try:
            steps = int(self.steps.get())
        except (tk.TclError, ValueError):
            return
        per_step = sum(self.step_times) / len(self.step_times) if self.step_times else 4.0
        key = "eta_measured" if self.step_times else "eta_estimate"
        self.eta_label.config(
            text=t(self.lang, key, total=self._fmt(steps * per_step), per=f"{per_step:.1f}")
        )

    def _fmt(self, seconds: float) -> str:
        seconds = int(seconds)
        if seconds >= 60:
            return t(self.lang, "min_sec", m=seconds // 60, s=seconds % 60)
        return t(self.lang, "sec", s=seconds)

    # ------------------------------------------------- Смена языка и устройства

    def _on_language_change(self, _event=None) -> None:
        chosen = self.lang_var.get()
        for code, name in i18n.LANGUAGE_NAMES.items():
            if name == chosen and code != self.lang:
                self.lang = code
                self.settings.set("language", code)
                self._rebuild_ui()
                return

    def _on_device_change(self, _event=None) -> None:
        chosen = self.device_combo.get()
        for code in DEVICE_CHOICES:
            if self.device_labels[code] == chosen and code != self.device_pref.get():
                self.device_pref.set(code)
                self.settings.set("device", code)
                self.pipe = None
                self.step_times = []
                self.status.config(text=t(self.lang, "status_reloading"))
                self._start_model_load()
                return

    # -------------------------------------------------------------- Модель

    def _start_model_load(self) -> None:
        self.loading = True
        self.load_token += 1
        self._apply_busy_state()
        self.progress.config(mode="indeterminate")
        self.progress.start(12)
        threading.Thread(
            target=self._load_model, args=(self.device_pref.get(), self.load_token),
            daemon=True,
        ).start()

    def _load_model(self, preference: str, token: int) -> None:
        try:
            if not MODEL_DIR.exists():
                raise FileNotFoundError(
                    f"{MODEL_DIR}\n\npython download_model.py"
                )
            import torch
            import rocm_compat

            rocm_compat.apply()  # заглушки torch.distributed для ROCm-сборки

            from moss_soundeffect_v2 import MossSoundEffectPipeline

            torch.set_num_threads(os.cpu_count() or 8)

            code, short, detail = detect_backend(torch)
            if preference == "cpu":
                import platform

                code, short = "cpu", "CPU"
                detail = platform.processor() or platform.machine()
            device = code

            self.events.put(("backend", token, f"{short} — {detail}"))
            self.events.put(("status", token, t(self.lang, "status_loading", device=short)))

            # bfloat16 на GPU: модель обучена в нём, и пайплайн всё равно
            # форсирует autocast(bfloat16). float16 здесь ПЕРЕПОЛНЯЕТСЯ и
            # недетерминированно выдаёт NaN — проверено на gfx1151.
            # То же верно для MPS: bf16 поддерживается с macOS 14.
            dtype_name = os.environ.get("MOSS_DTYPE", "")
            if dtype_name:
                dtype = getattr(torch, dtype_name)
            else:
                dtype = torch.float32 if device == "cpu" else torch.bfloat16

            t0 = time.time()
            pipe = MossSoundEffectPipeline.from_pretrained(
                str(MODEL_DIR), torch_dtype=dtype, device=device
            )

            # На ROCm VAE на GPU и медленнее CPU (MIOpen fallback-солвер), и
            # рискует переполнением — декод разовый, перенос почти бесплатен.
            # На CUDA/MPS по умолчанию не трогаем: MOSS_CPU_VAE=1 включает.
            fallback = "1" if rocm_compat.is_rocm() else "0"
            if device != "cpu" and os.environ.get("MOSS_CPU_VAE", fallback) == "1":
                rocm_compat.use_cpu_vae(pipe)

            self.events.put(("loaded", token, pipe, short, time.time() - t0))
        except Exception:
            self.events.put(("fatal", token, traceback.format_exc()))

    # ---------------------------------------------------------- Генерация

    def _on_generate(self) -> None:
        if self.busy or self.loading or self.pipe is None:
            return
        text = self.prompt.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning(
                t(self.lang, "empty_title"), t(self.lang, "empty_msg")
            )
            return
        try:
            params = {
                "prompt": text,
                "negative_prompt": self.negative.get().strip(),
                "seconds": float(self.seconds.get()),
                "num_inference_steps": int(self.steps.get()),
                "cfg_scale": float(self.cfg.get()),
                "sigma_shift": float(self.shift.get()),
                "seed": random.randint(0, 2**31 - 1)
                if self.random_seed.get()
                else int(self.seed.get()),
            }
        except (tk.TclError, ValueError) as exc:
            messagebox.showerror(t(self.lang, "bad_param_title"), str(exc))
            return

        if self.random_seed.get():
            self.seed.set(params["seed"])

        self.busy = True
        self.cancel_flag.clear()
        self._apply_busy_state()
        self.progress.config(mode="determinate", maximum=params["num_inference_steps"])
        self.progress["value"] = 0
        threading.Thread(
            target=self._generate, args=(params, self.mp3.get()), daemon=True
        ).start()

    def _generate(self, params: dict, as_mp3: bool) -> None:
        try:
            import soundfile as sf

            started = time.time()
            times: list[float] = []

            def progress_bar_cmd(iterable):
                """Замена tqdm: обновляет UI и позволяет прервать генерацию."""
                last = time.time()
                for i, item in enumerate(iterable):
                    if self.cancel_flag.is_set():
                        raise Cancelled
                    yield item
                    now = time.time()
                    times.append(now - last)
                    last = now
                    self.events.put(("progress", i + 1, times))

            audio = self.pipe(progress_bar_cmd=progress_bar_cmd, **params)

            suffix = ".mp3" if as_mp3 else ".wav"
            path = unique_path(OUTPUT_DIR, slugify(params["prompt"]), suffix)
            wav = audio[0].float().cpu().numpy().T

            # Модель выдаёт сигнал впритык к 0 dBFS. Кодек MP3 добавляет
            # интерсэмпловый звон (замерено: пик 0.97 -> 1.06 после декода),
            # который слышен как хрип, поэтому для lossy оставляем -1 dBFS.
            ceiling = 0.89 if as_mp3 else 0.97
            peak = float(abs(wav).max())
            if peak > ceiling:
                wav = wav * (ceiling / peak)
            if as_mp3:
                sf.write(str(path), wav, SAMPLE_RATE,
                         format="MP3", subtype="MPEG_LAYER_III")
            else:
                sf.write(str(path), wav, SAMPLE_RATE)

            self.events.put(("done", path, time.time() - started, times))
        except Cancelled:
            self.events.put(("cancelled",))
        except Exception:
            self.events.put(("error", traceback.format_exc()))

    def _on_cancel(self) -> None:
        self.cancel_flag.set()
        self.status.config(text=t(self.lang, "status_cancelling"))
        self.cancel_btn.config(state="disabled")

    # ----------------------------------------------------- События из UI

    def _drain_events(self) -> None:
        try:
            while True:
                self._handle(self.events.get_nowait())
        except queue.Empty:
            pass
        self.root.after(80, self._drain_events)

    def _handle(self, event: tuple) -> None:
        kind = event[0]

        # Ответы отменённой загрузки (пользователь успел сменить устройство).
        if kind in ("backend", "status", "loaded", "fatal") and event[1] != self.load_token:
            return

        if kind == "backend":
            self.backend_text = event[2]
            self.backend_label.config(text=self.backend_text)

        elif kind == "status":
            self.status.config(text=event[2])

        elif kind == "loaded":
            _, _, pipe, short, secs = event
            self.pipe = pipe
            self.loading = False
            self.progress.stop()
            self.progress.config(mode="determinate", value=0)
            self._apply_busy_state()
            self.status.config(
                text=t(self.lang, "status_ready", device=short, secs=f"{secs:.0f}")
            )

        elif kind == "progress":
            _, step, times = event
            self.progress["value"] = step
            total = int(self.steps.get())
            left = (total - step) * (sum(times) / len(times))
            self.status.config(
                text=t(self.lang, "status_step", step=step, total=total,
                       left=self._fmt(left))
            )

        elif kind == "done":
            _, path, elapsed, times = event
            self.step_times = times
            self._update_estimate()
            self.busy = False
            self._apply_busy_state()
            self.progress["value"] = self.progress["maximum"]
            self.result_paths.append(path)
            self.results.insert("end", path.name)
            self.results.selection_clear(0, "end")
            self.results.selection_set("end")
            self.results.see("end")
            self.play_btn.config(state="normal")
            self.status.config(
                text=t(self.lang, "status_done",
                       elapsed=self._fmt(elapsed), name=path.name)
            )

        elif kind == "cancelled":
            self.busy = False
            self._apply_busy_state()
            self.progress["value"] = 0
            self.status.config(text=t(self.lang, "status_cancelled"))

        elif kind == "error":
            self.busy = False
            self._apply_busy_state()
            self.progress["value"] = 0
            self.status.config(text=t(self.lang, "status_error"))
            messagebox.showerror(t(self.lang, "error_title"), event[1][-2000:])

        elif kind == "fatal":
            self.loading = False
            self.progress.stop()
            self._apply_busy_state()
            self.status.config(text=t(self.lang, "status_no_model"))
            messagebox.showerror(t(self.lang, "load_error_title"), event[2][-2000:])

    # ------------------------------------------------------ Воспроизведение

    def _on_play(self) -> None:
        sel = self.results.curselection()
        path = self.result_paths[sel[0]] if sel else (
            self.result_paths[-1] if self.result_paths else None
        )
        if not path or not path.exists():
            return
        platform_compat.play(path)

    def _open_folder(self) -> None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        platform_compat.open_path(OUTPUT_DIR)


def main() -> None:
    platform_compat.enable_hidpi()  # чёткий текст на HiDPI (Windows)
    root = tk.Tk()
    theme = platform_compat.preferred_theme()
    if theme:
        try:
            ttk.Style().theme_use(theme)
        except tk.TclError:
            pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
