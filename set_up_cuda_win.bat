@echo off
chcp 65001 >nul
echo ===================================================
echo   开始在 Windows 上配置带 CUDA 的强化学习环境
echo ===================================================

:: 1. 创建虚拟环境
echo [1/4] 正在创建 .venv 虚拟环境...
python -m venv .venv

:: 2. 激活虚拟环境
echo [2/4] 正在激活虚拟环境...
call .venv\Scripts\activate

:: 3. 升级基础工具
echo [3/4] 正在升级 pip 和 wheel...
python -m pip install --upgrade pip setuptools wheel

:: 4. 强制安装带 CUDA 11.8 的 PyTorch
echo [4/4] 正在安装 PyTorch (CUDA 11.8 版本)...
:: 注意：这里使用官方的 cu118 源
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

:: 5. 安装其他项目依赖
echo [5/5] 正在安装 requirements.txt 中的其他依赖...
pip install -r requirements.txt

echo ===================================================
echo   环境配置完成！
echo   请在终端运行: .venv\Scripts\activate 来激活环境
echo   然后运行: python check_cuda.py 测试显卡是否可用
echo ===================================================
pause
