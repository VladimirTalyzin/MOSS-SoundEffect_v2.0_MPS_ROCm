@echo off
rem Запуск оконной оболочки MOSS-SoundEffect без консольного окна.
rem Берётся первое найденное окружение: venv-rocm (GPU), venv, venv-moss (CPU).
setlocal
for %%V in (venv-rocm venv venv-moss) do (
    if exist "%~dp0%%V\Scripts\pythonw.exe" (
        start "" "%~dp0%%V\Scripts\pythonw.exe" "%~dp0app.py"
        exit /b 0
    )
)
echo Virtual environment not found. See README.md for setup instructions.
pause
exit /b 1
