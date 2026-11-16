#!/usr/bin/env bash
# Linux / macOS setup:  bash setup.sh
set -e
cd "$(dirname "$0")"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "NVIDIA GPU detected - installing CUDA PyTorch"
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
else
  echo "No NVIDIA GPU - installing CPU PyTorch"
  if [ "$(uname)" = "Linux" ]; then
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  else
    pip install torch torchvision
  fi
fi
pip install -r requirements.txt
python -c "import torch,cv2,ultralytics;print('OK torch',torch.__version__,'cuda',torch.cuda.is_available())"
echo "Setup complete. Activate with: source .venv/bin/activate"
