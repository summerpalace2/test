#!/usr/bin/env bash
# Linux/macOS 打包：./build.sh  → dist/MedMiner/MedMiner
set -e
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pyinstaller medminer.spec --noconfirm --clean
echo "完成：dist/MedMiner/MedMiner"
