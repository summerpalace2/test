@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ==== MedMiner 启动器 ====

where python >nul 2>nul
if errorlevel 1 (
  echo 未找到 Python。请先安装 Python 3.10 到 3.13（https://www.python.org/downloads/），
  echo 安装时务必勾选 "Add python.exe to PATH"，然后重新双击本文件。
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo 正在创建虚拟环境 .venv ...
  python -m venv .venv
  if errorlevel 1 ( echo 创建虚拟环境失败 & pause & exit /b 1 )
)
call ".venv\Scripts\activate.bat"

if not exist ".venv\.deps_ok" (
  echo 首次运行：正在安装依赖，体积较大，需要几分钟，请耐心等待 ...
  python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
  pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
  if errorlevel 1 (
    echo 清华镜像安装失败，改用官方源重试 ...
    pip install -r requirements.txt
    if errorlevel 1 (
      echo.
      echo 依赖安装失败。请把上面的报错信息截图发给组员排查。
      pause
      exit /b 1
    )
  )
  echo ok> ".venv\.deps_ok"
)

echo 正在启动 MedMiner ...
python run.py
if errorlevel 1 (
  echo.
  echo 程序异常退出，请把上面的报错信息截图发给组员排查。
  pause
)
