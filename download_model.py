"""Скачивает веса MOSS-SoundEffect-v2.0 в models/ рядом с проектом (~11 ГБ)."""
import os
from pathlib import Path

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
from huggingface_hub import snapshot_download

REPO_ID = "OpenMOSS-Team/MOSS-SoundEffect-v2.0"
LOCAL_DIR = Path(__file__).resolve().parent / "models" / "MOSS-SoundEffect-v2.0"

path = snapshot_download(
    repo_id=REPO_ID,
    local_dir=str(LOCAL_DIR),
    max_workers=4,
)
print("downloaded to:", path)
