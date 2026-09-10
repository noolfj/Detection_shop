@echo off
chcp 65001 >nul
echo =======================================================
echo   Установка PyTorch с поддержкой CUDA для видеокарты
echo =======================================================
echo.

cd /d "%~dp0"

echo [1/3] Скачивание torch (~2.4 ГБ) с поддержкой докачки...
curl.exe -C - -O "https://download.pytorch.org/whl/cu126/torch-2.14.0%%2Bcu126-cp314-cp314-win_amd64.whl"

echo.
echo [2/3] Скачивание torchvision (~6 МБ)...
curl.exe -C - -O "https://download.pytorch.org/whl/cu126/torchvision-0.29.0%%2Bcu126-cp314-cp314-win_amd64.whl"

echo.
echo [3/3] Установка в виртуальное окружение venv...
call venv\Scripts\activate
pip install --force-reinstall "torch-2.14.0+cu126-cp314-cp314-win_amd64.whl" "torchvision-0.29.0+cu126-cp314-cp314-win_amd64.whl"

echo.
echo Проверка видеокарты...
python -c "import torch; print('CUDA доступна:', torch.cuda.is_available()); print('Видеокарта:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'нет')"

echo.
echo =======================================================
echo Готово! Теперь проект будет работать на вашей видеокарте.
echo =======================================================
pause
