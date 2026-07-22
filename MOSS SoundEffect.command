#!/usr/bin/env bash
# Запуск оконной оболочки MOSS-SoundEffect на macOS и Linux.
# На macOS файл .command открывается двойным щелчком из Finder
# (один раз потребуется: chmod +x "MOSS SoundEffect.command").
set -euo pipefail
cd "$(dirname "$0")"

for env in venv venv-mps venv-moss; do
    if [ -x "$env/bin/python" ]; then
        exec "$env/bin/python" app.py
    fi
done

echo "Virtual environment not found. See README.md for setup instructions." >&2
exit 1
