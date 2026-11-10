@echo off
chcp 65001 >nul
echo === MedMiner 打包（Windows）===
where python >nul 2>nul || (echo 未找到 Python，请先安装 Python 3.10-3.13 并勾选 Add to PATH & pause & exit /b 1)
if not exist .venv ( python -m venv .venv )
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
pyinstaller medminer.spec --noconfirm --clean
if errorlevel 1 ( echo 打包失败，请把上面的报错发给组员排查 & pause & exit /b 1 )
echo.
echo 完成！运行 dist\MedMiner\MedMiner.exe （把整个 dist\MedMiner 文件夹拷走即可）
pause
