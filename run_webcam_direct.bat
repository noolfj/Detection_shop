@echo off
echo ============================================
echo   AI CCTV Real-Time Camera Detection
echo ============================================
echo.

cd /d "%~dp0"
call venv\Scripts\activate

echo Запуск детекции в реальном времени с камеры 0...
echo Нажмите 'q' в окне камеры для выхода.
echo Нажмите 's' в окне камеры для сохранения скриншота тревоги.
echo.

python scripts\inference.py --camera 0 --threshold 0.4

pause
