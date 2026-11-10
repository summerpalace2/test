@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo === MedMiner 打包（Windows，单文件版）===
where python >nul 2>nul || (echo 未找到 Python，请先安装 Python 3.10-3.13 并勾选 Add to PATH & pause & exit /b 1)
if not exist .venv ( python -m venv .venv )
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 ( pip install -r requirements.txt )
set MEDMINER_ONEFILE=1
pyinstaller medminer.spec --noconfirm --clean
if errorlevel 1 ( echo 打包失败，请把上面的报错发给组员排查 & pause & exit /b 1 )
echo.
echo 完成！单文件程序：dist\MedMiner.exe （首次启动需要解压，请耐心等 20-60 秒）
pause
