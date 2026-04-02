#!/bin/bash

echo "==================================================="
echo "  开始在 Ubuntu 上配置带 CUDA 的强化学习环境"
echo "==================================================="

# 确保系统安装了 python3-venv (Ubuntu 默认可能没装)
# sudo apt-get update && sudo apt-get install -y python3-venv

# 1. 创建虚拟环境
echo "[1/4] 正在创建 .venv 虚拟环境..."
python3 -m venv .venv

# 2. 激活虚拟环境
echo "[2/4] 正在激活虚拟环境..."
source .venv/bin/activate

# 3. 升级基础工具
echo "[3/4] 正在升级 pip 和 wheel..."
python3 -m pip install --upgrade pip setuptools wheel

# 4. 强制安装带 CUDA 11.8 的 PyTorch
echo "[4/4] 正在安装 PyTorch (CUDA 11.8 版本)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 5. 安装其他项目依赖
echo "[5/5] 正在安装 requirements.txt 中的其他依赖..."
pip install -r requirements.txt

echo "==================================================="
echo "  环境配置完成！"
echo "  请在终端运行: source .venv/bin/activate 来激活环境"
echo "  然后运行: python check_cuda.py 测试显卡是否可用"
echo "==================================================="
