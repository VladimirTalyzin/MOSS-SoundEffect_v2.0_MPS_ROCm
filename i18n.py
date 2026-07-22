"""Локализация интерфейса: английский, русский, китайский."""
import ctypes
import locale

DEFAULT_LANG = "en"

LANGUAGE_NAMES = {"en": "English", "ru": "Русский", "zh": "中文"}

# Первичные языки Windows (младшие 10 бит LANGID).
_PRIMARY_LANG = {0x09: "en", 0x19: "ru", 0x04: "zh"}

STRINGS = {
    "en": {
        "title": "MOSS-SoundEffect v2.0",
        "prompt_frame": "Sound description",
        "prompt_hint": "English or Chinese — the model was trained on these.",
        "examples": "Examples ▾",
        "negative_label": "Negative prompt (optional):",
        "params_frame": "Generation parameters",
        "duration": "Duration (sec):",
        "steps": "Diffusion steps:",
        "cfg": "CFG scale:",
        "shift": "Sigma shift:",
        "seed": "Seed:",
        "random": "random",
        "output_frame": "Output",
        "format_mp3": "Save as MP3 (for Telegram, messengers)",
        "engine_frame": "Compute device",
        "device_label": "Device:",
        "device_auto": "Auto",
        "device_gpu": "GPU",
        "device_cpu": "CPU (fallback)",
        "language_label": "Language:",
        "generate": "Generate",
        "cancel": "Cancel",
        "play": "▶ Play",
        "open_folder": "Open folder",
        "results_frame": "Generated files",
        "status_detecting": "Detecting device...",
        "status_loading": "Loading model on {device}...",
        "status_ready": "Model ready ({device}, {secs} s)",
        "status_reloading": "Switching device — reloading model...",
        "status_step": "Step {step}/{total} — about {left} left",
        "status_done": "Done in {elapsed} → {name}",
        "status_cancelled": "Generation cancelled.",
        "status_cancelling": "Cancelling — waiting for current step...",
        "status_error": "Generation failed.",
        "status_no_model": "Model not loaded.",
        "eta_estimate": "~{total} ({per} s/step, estimate)",
        "eta_measured": "~{total} ({per} s/step, measured)",
        "min_sec": "{m} min {s} s",
        "sec": "{s} s",
        "empty_title": "Empty prompt",
        "empty_msg": "Enter a sound description.",
        "bad_param_title": "Invalid parameter",
        "error_title": "Generation error",
        "load_error_title": "Could not load model",
        "backend_cpu": "CPU",
        "unknown": "unknown",
    },
    "ru": {
        "title": "MOSS-SoundEffect v2.0",
        "prompt_frame": "Описание звука",
        "prompt_hint": "Английский или китайский — модель обучена на них.",
        "examples": "Примеры ▾",
        "negative_label": "Negative prompt (необязательно):",
        "params_frame": "Параметры генерации",
        "duration": "Длительность (сек):",
        "steps": "Шагов диффузии:",
        "cfg": "CFG scale:",
        "shift": "Sigma shift:",
        "seed": "Seed:",
        "random": "случайный",
        "output_frame": "Сохранение",
        "format_mp3": "Сохранять в MP3 (для Telegram, мессенджеров)",
        "engine_frame": "Вычислительное устройство",
        "device_label": "Устройство:",
        "device_auto": "Автоматически",
        "device_gpu": "GPU",
        "device_cpu": "CPU (запасной вариант)",
        "language_label": "Язык:",
        "generate": "Сгенерировать",
        "cancel": "Отмена",
        "play": "▶ Прослушать",
        "open_folder": "Папка с файлами",
        "results_frame": "Сгенерированные файлы",
        "status_detecting": "Определение устройства...",
        "status_loading": "Загрузка модели на {device}...",
        "status_ready": "Модель готова ({device}, {secs} с)",
        "status_reloading": "Смена устройства — перезагрузка модели...",
        "status_step": "Шаг {step}/{total} — осталось ~{left}",
        "status_done": "Готово за {elapsed} → {name}",
        "status_cancelled": "Генерация отменена.",
        "status_cancelling": "Отмена — дождись завершения текущего шага...",
        "status_error": "Ошибка при генерации.",
        "status_no_model": "Модель не загружена.",
        "eta_estimate": "~{total} ({per} с/шаг, оценка)",
        "eta_measured": "~{total} ({per} с/шаг, замерено)",
        "min_sec": "{m} мин {s} с",
        "sec": "{s} с",
        "empty_title": "Пустой запрос",
        "empty_msg": "Введи описание звука.",
        "bad_param_title": "Неверный параметр",
        "error_title": "Ошибка генерации",
        "load_error_title": "Не удалось загрузить модель",
        "backend_cpu": "CPU",
        "unknown": "неизвестно",
    },
    "zh": {
        "title": "MOSS-SoundEffect v2.0",
        "prompt_frame": "音效描述",
        "prompt_hint": "支持英文或中文 —— 模型基于这两种语言训练。",
        "examples": "示例 ▾",
        "negative_label": "负面提示词（可选）：",
        "params_frame": "生成参数",
        "duration": "时长（秒）：",
        "steps": "扩散步数：",
        "cfg": "CFG 强度：",
        "shift": "Sigma shift：",
        "seed": "随机种子：",
        "random": "随机",
        "output_frame": "保存",
        "format_mp3": "保存为 MP3（适用于 Telegram 等）",
        "engine_frame": "计算设备",
        "device_label": "设备：",
        "device_auto": "自动",
        "device_gpu": "GPU",
        "device_cpu": "CPU（备用）",
        "language_label": "语言：",
        "generate": "生成",
        "cancel": "取消",
        "play": "▶ 播放",
        "open_folder": "打开文件夹",
        "results_frame": "已生成文件",
        "status_detecting": "正在检测设备……",
        "status_loading": "正在 {device} 上加载模型……",
        "status_ready": "模型就绪（{device}，{secs} 秒）",
        "status_reloading": "切换设备 —— 正在重新加载模型……",
        "status_step": "第 {step}/{total} 步 —— 剩余约 {left}",
        "status_done": "完成，用时 {elapsed} → {name}",
        "status_cancelled": "生成已取消。",
        "status_cancelling": "正在取消 —— 等待当前步骤结束……",
        "status_error": "生成失败。",
        "status_no_model": "模型未加载。",
        "eta_estimate": "约 {total}（{per} 秒/步，预估）",
        "eta_measured": "约 {total}（{per} 秒/步，实测）",
        "min_sec": "{m} 分 {s} 秒",
        "sec": "{s} 秒",
        "empty_title": "描述为空",
        "empty_msg": "请输入音效描述。",
        "bad_param_title": "参数无效",
        "error_title": "生成错误",
        "load_error_title": "模型加载失败",
        "backend_cpu": "CPU",
        "unknown": "未知",
    },
}

PRESETS = {
    "en": [
        "A cow mooing loudly in a barn.",
        "The crisp, rhythmic click-clack of fast typing on a mechanical keyboard.",
        "Heavy rain falling on a metal roof, distant thunder rumbling.",
        "Footsteps crunching slowly through fresh snow.",
        "A crackling campfire at night with occasional wood pops.",
    ],
    "zh": [
        "牛在牛棚里大声哞叫。",
        "机械键盘快速打字时清脆有节奏的咔哒声。",
        "大雨打在铁皮屋顶上，远处雷声隆隆。",
        "脚步缓慢地踩过新雪，发出嘎吱声。",
        "夜晚篝火噼啪作响，偶尔有木柴爆裂声。",
    ],
}
PRESETS["ru"] = PRESETS["en"]  # модель понимает только английский и китайский


def detect_system_language() -> str:
    """Определяет язык интерфейса из настроек Windows, иначе из локали Python."""
    try:
        langid = ctypes.windll.kernel32.GetUserDefaultUILanguage()
        lang = _PRIMARY_LANG.get(langid & 0x3FF)
        if lang:
            return lang
    except Exception:
        pass

    try:
        code = (locale.getlocale()[0] or "").lower()
        for prefix, lang in (
            ("english", "en"), ("russian", "ru"), ("chinese", "zh"),
            ("en", "en"), ("ru", "ru"), ("zh", "zh"),
        ):
            if code.startswith(prefix):
                return lang
    except Exception:
        pass

    return DEFAULT_LANG


def t(lang: str, key: str, **kwargs) -> str:
    """Переводит ключ, откатываясь на английский при отсутствии строки."""
    table = STRINGS.get(lang) or STRINGS[DEFAULT_LANG]
    text = table.get(key) or STRINGS[DEFAULT_LANG].get(key, key)
    return text.format(**kwargs) if kwargs else text
